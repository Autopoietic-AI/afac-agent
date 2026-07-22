# -*- coding: utf-8 -*-
"""B1 Data Intelligence core computations.

Deterministic, read-only audits of B1 node features, graph structure and
label-graph reliability.  No model training; no Test truth use.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.linalg import svds
from scipy.stats import entropy
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .task_adapter import NodeClassificationDataset


def _percentiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {}
    return {q: float(np.percentile(values, int(q[1:]))) for q in ["p5", "p25", "p50", "p75", "p95"]}


def integrity_audit(dataset: NodeClassificationDataset) -> dict[str, Any]:
    X = dataset.features
    adj = dataset.adj
    rows = np.diff(X.indptr)
    has_self_loop = False
    for i in range(dataset.n_nodes):
        cols = adj.indices[adj.indptr[i]:adj.indptr[i + 1]]
        if i in cols:
            has_self_loop = True
            break
    duplicate_edges = int(adj.nnz) - len(set(zip(*adj.nonzero())))
    return {
        "audit": "integrity",
        "n_nodes": dataset.n_nodes,
        "n_features": dataset.n_features,
        "n_classes": dataset.n_classes,
        "n_train": dataset.train_idx.size,
        "n_test": dataset.test_idx.size,
        "feature_shape": list(X.shape),
        "adj_shape": list(adj.shape),
        "feature_dtype": str(X.dtype),
        "adj_dtype": str(adj.dtype),
        "label_dtype": str(dataset.labels.dtype),
        "feature_finite": bool(np.isfinite(X.data).all()),
        "adj_finite": bool(np.isfinite(adj.data).all()),
        "train_test_overlap": len(set(dataset.train_idx.tolist()) & set(dataset.test_idx.tolist())),
        "test_truth_hidden": bool(np.all(dataset.labels[dataset.test_idx] == -1)),
        "train_labels_min": int(dataset.labels[dataset.train_idx].min()),
        "train_labels_max": int(dataset.labels[dataset.train_idx].max()),
        "duplicate_edges": duplicate_edges,
        "has_self_loop": has_self_loop,
        "feature_rows_nonzero_min": int(rows.min()),
        "feature_rows_nonzero_max": int(rows.max()),
        "feature_rows_nonzero_mean": float(rows.mean()),
    }


def feature_geometry(dataset: NodeClassificationDataset) -> dict[str, Any]:
    X = dataset.features.astype(np.float64)
    n, d = X.shape
    data = X.data
    nnz = X.nnz
    density = nnz / (n * d)
    col_means = np.asarray(X.mean(axis=0)).ravel()
    col_sumsq = np.asarray(X.multiply(X).mean(axis=0)).ravel()
    col_std = np.sqrt(np.maximum(col_sumsq - col_means ** 2, 0.0))
    row_norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    row_l1 = np.asarray(np.abs(X).sum(axis=1)).ravel()

    # PCA variance
    X_dense = X.toarray()
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X_dense)
    pca = PCA(n_components=min(d, n - 1), svd_solver="full")
    pca.fit(Xs)
    var = np.maximum(pca.explained_variance_ratio_, 0.0)
    cumvar = np.cumsum(var)
    thresholds = [0.50, 0.80, 0.90, 0.95, 0.99]

    def dim_for(th: float) -> int:
        idx = np.searchsorted(cumvar, th, side="right")
        return int(idx + 1)

    # Effective rank / participation ratio
    s = pca.singular_values_
    s = s[s > 0]
    s_norm = s / s.sum()
    effective_rank = float(np.exp(entropy(s_norm)))
    participation_ratio = float((s.sum() ** 2) / (s ** 2).sum())
    numeric_rank = int(np.sum(s > (s.max() * 1e-6)))

    # Feature shift: compare feature norms of train vs test
    train_mask = np.zeros(n, dtype=bool)
    train_mask[dataset.train_idx] = True
    test_mask = np.zeros(n, dtype=bool)
    test_mask[dataset.test_idx] = True
    train_norm_mean = float(row_norms[train_mask].mean())
    test_norm_mean = float(row_norms[test_mask].mean())
    norm_shift = test_norm_mean / (train_norm_mean + 1e-12) - 1.0

    # Highly correlated feature pairs (sample of columns)
    sample_cols = min(d, 200)
    rng = np.random.default_rng(2026)
    cols = rng.choice(d, sample_cols, replace=False)
    sub = Xs[:, cols]
    corr = np.corrcoef(sub.T)
    high_corr = []
    for i in range(sample_cols):
        for j in range(i + 1, sample_cols):
            if abs(corr[i, j]) > 0.95:
                high_corr.append((int(cols[i]), int(cols[j]), float(corr[i, j])))
        if len(high_corr) >= 20:
            break

    return {
        "audit": "feature_geometry",
        "density": float(density),
        "nonzero_ratio": float(nnz / (n * d)),
        "positive_value_ratio": float(np.sum(data > 0) / max(data.size, 1)),
        "negative_value_ratio": float(np.sum(data < 0) / max(data.size, 1)),
        "feature_value_min": float(data.min()) if data.size else None,
        "feature_value_max": float(data.max()) if data.size else None,
        "feature_value_mean": float(data.mean()) if data.size else None,
        "column_means": _percentiles(col_means),
        "column_stds": _percentiles(col_std),
        "row_l2_norms": _percentiles(row_norms),
        "row_l1_norms": _percentiles(row_l1),
        "pca_dimensions_for_variance": {f"{int(th * 100)}%": dim_for(th) for th in thresholds},
        "effective_rank": effective_rank,
        "participation_ratio": participation_ratio,
        "numeric_rank": numeric_rank,
        "feature_norm_train_mean": train_norm_mean,
        "feature_norm_test_mean": test_norm_mean,
        "feature_norm_shift": norm_shift,
        "high_correlation_pairs": high_corr[:20],
        "feature_regime": _feature_regime(density, effective_rank, numeric_rank, norm_shift),
    }


def _feature_regime(density: float, eff_rank: float, numeric_rank: int, norm_shift: float) -> str:
    if density > 0.5 and eff_rank > numeric_rank * 0.3:
        return "dense_low_rank"
    if density < 0.1 and eff_rank < numeric_rank * 0.15:
        return "sparse_high_rank"
    if abs(norm_shift) > 0.1:
        return "shifted_norm"
    return "moderate"


def graph_regime(dataset: NodeClassificationDataset) -> dict[str, Any]:
    n = dataset.n_nodes
    views = {}
    for view in ("directed_out", "directed_in", "undirected_union"):
        A = dataset.directed_adj(view)
        degrees = np.diff(A.indptr).astype(np.float64)
        in_deg = degrees if view != "directed_out" else np.diff(dataset.directed_adj("directed_in").indptr).astype(np.float64)
        out_deg = degrees if view != "directed_in" else np.diff(dataset.directed_adj("directed_out").indptr).astype(np.float64)
        und = np.diff(dataset.directed_adj("undirected_union").indptr).astype(np.float64)
        views[view] = {
            "n_edges": int(A.nnz),
            "mean_degree": float(degrees.mean()),
            "median_degree": float(np.median(degrees)),
            "max_degree": int(degrees.max()),
            "min_degree": int(degrees.min()),
            "isolated_count": int(np.sum(degrees == 0)),
            "degree_percentiles": _percentiles(degrees),
        }
    # weakly connected components on undirected view
    from scipy.sparse.csgraph import connected_components
    U = dataset.directed_adj("undirected_union")
    n_components, labels_cc = connected_components(U, directed=False)
    sizes = np.bincount(labels_cc)
    largest = int(sizes.max())

    # PageRank on undirected view (power iteration, deterministic)
    pr = _pagerank(U)

    # degree assortativity (undirected)
    deg_und = np.diff(U.indptr).astype(np.float64)
    src, dst = U.nonzero()
    mask = src != dst
    src_deg = deg_und[src[mask]]
    dst_deg = deg_und[dst[mask]]
    assort = float(np.corrcoef(src_deg, dst_deg)[0, 1]) if src_deg.size > 1 else 0.0

    # train/test degree propensity
    train_mask = np.zeros(n, dtype=bool)
    train_mask[dataset.train_idx] = True
    test_mask = np.zeros(n, dtype=bool)
    test_mask[dataset.test_idx] = True
    deg_shift = deg_und[test_mask].mean() / (deg_und[train_mask].mean() + 1e-12) - 1.0

    return {
        "audit": "graph_regime",
        "views": views,
        "n_weak_components": int(n_components),
        "largest_component_size": largest,
        "pagerank_percentiles": _percentiles(pr),
        "degree_assortativity": assort,
        "train_test_degree_shift": float(deg_shift),
        "graph_regime": _graph_regime(views, deg_und),
        "connectivity_regime": "single_large_component" if largest > 0.95 * n else "fragmented",
    }


def _pagerank(A: csr_matrix, alpha: float = 0.85, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    n = A.shape[0]
    out_deg = np.diff(A.indptr).astype(np.float64)
    out_deg[out_deg == 0] = 1.0
    # column-normalized transition matrix via row-normalized of A.T
    M = A.T.astype(np.float64)
    for i in range(n):
        M.data[M.indptr[i]:M.indptr[i + 1]] /= out_deg[i]
    r = np.ones(n) / n
    teleport = (1.0 - alpha) / n
    for _ in range(max_iter):
        r_next = alpha * (M @ r) + teleport
        if np.linalg.norm(r_next - r, 1) < tol:
            break
        r = r_next
    return r


def _graph_regime(views: dict, deg_und: np.ndarray) -> str:
    out = views["directed_out"]["n_edges"]
    inn = views["directed_in"]["n_edges"]
    und = views["undirected_union"]["n_edges"]
    sym = abs(out - inn) / max(out + inn, 1)
    if sym < 0.05:
        return "undirected_like"
    if und < out:
        return "directed_dominant"
    return "mixed_directed"


def label_graph_reliability(dataset: NodeClassificationDataset) -> dict[str, Any]:
    y = np.full(dataset.n_nodes, -1, dtype=np.int64)
    y[dataset.train_idx] = dataset.labels[dataset.train_idx]
    views = {}
    for view in ("directed_out", "directed_in", "undirected_union"):
        A = dataset.directed_adj(view)
        hom = _homophily(A, y, dataset.train_idx)
        class_hom = _class_conditional_homophily(A, y, dataset.train_idx)
        hop_rel = _hop_reliability(A, y, dataset.train_idx)
        deg_rel = _degree_bucket_reliability(A, y, dataset.train_idx)
        views[view] = {
            "global_homophily": hom,
            "class_conditional_homophily": class_hom,
            "hop_reliability": hop_rel,
            "degree_bucket_reliability": deg_rel,
        }
    # graph-feature disagreement proxy: graph LP accuracy vs feature-only baseline placeholder
    return {
        "audit": "label_graph_reliability",
        "views": views,
        "homophily_regime": "strong" if any(views[v]["global_homophily"] > 0.5 for v in views) else "weak",
        "directionality_recommendation": _directionality_recommendation(views),
    }


def _homophily(A: csr_matrix, y: np.ndarray, train_idx: np.ndarray) -> float:
    Y = np.zeros((A.shape[0], int(y[train_idx].max()) + 1), dtype=np.float64)
    for i in train_idx:
        if y[i] >= 0:
            Y[i, y[i]] = 1.0
    N = A.astype(np.float64) @ Y
    # remove self-loops contribution
    if np.any(A.diagonal()):
        for c in range(Y.shape[1]):
            N[:, c] -= Y[:, c] * A.diagonal()
    pred = N.argmax(axis=1)
    mask = N.sum(axis=1) > 0
    correct = ((pred == y) & mask)[train_idx].sum()
    total = mask[train_idx].sum()
    return correct / total if total else 0.0


def _class_conditional_homophily(A: csr_matrix, y: np.ndarray, train_idx: np.ndarray) -> dict[int, float]:
    Y = np.zeros((A.shape[0], int(y[train_idx].max()) + 1), dtype=np.float64)
    for i in train_idx:
        if y[i] >= 0:
            Y[i, y[i]] = 1.0
    N = A.astype(np.float64) @ Y
    if np.any(A.diagonal()):
        for c in range(Y.shape[1]):
            N[:, c] -= Y[:, c] * A.diagonal()
    pred = N.argmax(axis=1)
    mask = N.sum(axis=1) > 0
    out: dict[int, float] = {}
    for c in range(Y.shape[1]):
        m = (y == c) & mask
        m_train = m[train_idx]
        if m_train.any():
            out[c] = float(((pred == c) & m)[train_idx].sum() / m_train.sum())
        else:
            out[c] = None
    return out


def _hop_reliability(A: csr_matrix, y: np.ndarray, train_idx: np.ndarray, max_hop: int = 4) -> dict[str, float]:
    n_classes = int(y[train_idx].max()) + 1
    Y = np.zeros((A.shape[0], n_classes), dtype=np.float64)
    for i in train_idx:
        if y[i] >= 0:
            Y[i, y[i]] = 1.0
    N = A.astype(np.float64) @ Y
    if np.any(A.diagonal()):
        diag = A.diagonal()
        for c in range(n_classes):
            N[:, c] -= Y[:, c] * diag
    result = {}
    for hop in range(1, max_hop + 1):
        if hop > 1:
            N = A.astype(np.float64) @ N
        pred = N.argmax(axis=1)
        mask = N.sum(axis=1) > 0
        correct = ((pred == y) & mask)[train_idx].sum()
        count = mask[train_idx].sum()
        result[f"hop{hop}"] = correct / count if count else None
    return result


def _degree_bucket_reliability(A: csr_matrix, y: np.ndarray, train_idx: np.ndarray) -> dict[str, float]:
    n_classes = int(y[train_idx].max()) + 1
    Y = np.zeros((A.shape[0], n_classes), dtype=np.float64)
    for i in train_idx:
        if y[i] >= 0:
            Y[i, y[i]] = 1.0
    N = A.astype(np.float64) @ Y
    if np.any(A.diagonal()):
        diag = A.diagonal()
        for c in range(n_classes):
            N[:, c] -= Y[:, c] * diag
    pred = N.argmax(axis=1)
    mask = N.sum(axis=1) > 0
    deg = np.diff(A.indptr).astype(np.float64)
    buckets = ["0", "1", "2_5", "6_20", "21_plus"]
    cuts = [0, 1, 2, 6, 21, np.inf]
    out: dict[str, float] = {}
    for b, lo, hi in zip(buckets, cuts[:-1], cuts[1:]):
        in_bucket = (deg >= lo) & (deg < hi)
        m = in_bucket[train_idx] & mask[train_idx]
        if m.any():
            correct = ((pred == y) & in_bucket & mask)[train_idx].sum()
            out[b] = correct / m.sum()
        else:
            out[b] = None
    return out


def _directionality_recommendation(views: dict) -> str:
    scores = {v: views[v]["global_homophily"] for v in views}
    best = max(scores, key=scores.get)
    return best


def shift_audit(dataset: NodeClassificationDataset) -> dict[str, Any]:
    n = dataset.n_nodes
    train_mask = np.zeros(n, dtype=bool)
    train_mask[dataset.train_idx] = True
    test_mask = np.zeros(n, dtype=bool)
    test_mask[dataset.test_idx] = True

    deg_out = np.diff(dataset.adj.indptr).astype(np.float64)
    deg_in = np.diff(dataset.adj.T.tocsr().indptr).astype(np.float64)
    pr = _pagerank(dataset.directed_adj("undirected_union"))
    feat_norm = np.sqrt(np.asarray(dataset.features.multiply(dataset.features).sum(axis=1)).ravel())

    X_meta = np.column_stack([
        np.log1p(deg_out),
        np.log1p(deg_in),
        np.log1p(pr),
        feat_norm,
    ])
    y_meta = test_mask.astype(np.int64)

    # Propensity model
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_meta)
    model = LogisticRegression(max_iter=500, random_state=2026, solver="lbfgs")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    oof = np.zeros(n)
    for tr, va in skf.split(Xs, y_meta):
        model.fit(Xs[tr], y_meta[tr])
        oof[va] = model.predict_proba(Xs[va])[:, 1]
    combined_auc = float(roc_auc_score(y_meta, oof))

    # Degree-only AUC
    deg_auc = float(roc_auc_score(y_meta, deg_out + deg_in))
    # Feature-only AUC (logistic on standardized features)
    Xf = StandardScaler().fit_transform(dataset.features.toarray())
    oof_f = np.zeros(n)
    for tr, va in skf.split(Xf, y_meta):
        model.fit(Xf[tr], y_meta[tr])
        oof_f[va] = model.predict_proba(Xf[va])[:, 1]
    feature_auc = float(roc_auc_score(y_meta, oof_f))

    # Test-like validation subset: highest propensity train nodes, sized to ~20% of train
    prop_train = oof[train_mask]
    k = max(1, int(0.2 * dataset.train_idx.size))
    test_like_idx = dataset.train_idx[np.argsort(-prop_train)[:k]]
    low_deg_idx = dataset.train_idx[np.argsort(deg_out[train_mask] + deg_in[train_mask])[:k]]

    # Strongest shift features by logistic coef (fit fresh model on meta features)
    meta_model = LogisticRegression(max_iter=500, solver="lbfgs", random_state=2026)
    meta_model.fit(Xs, y_meta)
    strongest = np.argsort(-np.abs(meta_model.coef_[0]))[:4]
    feature_names = ["log1p_out_degree", "log1p_in_degree", "log1p_pagerank", "feature_norm"]
    strongest_clipped = [min(int(i), len(feature_names) - 1) for i in strongest]

    return {
        "audit": "shift_audit",
        "propensity_auc_combined": combined_auc,
        "propensity_auc_degree_only": deg_auc,
        "propensity_auc_feature_only": feature_auc,
        "shift_interpretation": "substantial" if combined_auc > 0.65 else "moderate" if combined_auc > 0.55 else "weak",
        "test_like_validation_subset_size": int(test_like_idx.size),
        "low_degree_validation_subset_size": int(low_deg_idx.size),
        "strongest_shift_meta_features": [feature_names[i] for i in strongest_clipped],
        "train_test_shift": combined_auc,
    }


def regime_identification(dataset: NodeClassificationDataset, audits: dict) -> dict[str, Any]:
    graph = audits["graph_regime"]
    feature = audits["feature_geometry"]
    label = audits["label_graph_reliability"]
    shift = audits["shift_audit"]
    graph_regime = graph["graph_regime"]
    feature_regime = feature["feature_regime"]
    hom = label["homophily_regime"]
    direction = label["directionality_recommendation"]
    primary = "graph_signal_present" if hom == "strong" else "feature_signal_primary"
    if shift["train_test_shift"] > 0.60:
        secondary = "train_test_distribution_shift"
    elif graph_regime == "directed_dominant":
        secondary = "directed_graph_view_matters"
    else:
        secondary = "multi_view_fusion_opportunity"
    avoid = ["A1 isolated expert", "A1 exact-2-hop oracle", "heavy deep GCN", "multi-seed brute force"]
    return {
        "audit": "regime_identification",
        "graph_regime": graph_regime,
        "feature_regime": feature_regime,
        "connectivity_regime": graph["connectivity_regime"],
        "homophily_regime": hom,
        "directionality_recommendation": direction,
        "train_test_shift": shift["shift_interpretation"],
        "feature_graph_alignment": primary,
        "primary_problem": "node_classification_with_moderate_graph_and_feature_signals",
        "secondary_problems": [secondary],
        "information_source_map": {
            "feature": "reusable_infrastructure",
            "directed_out": "requires_b1_revalidation",
            "directed_in": "requires_b1_revalidation",
            "undirected_union": "requires_b1_revalidation",
            "label_propagation": "requires_b1_revalidation",
        },
        "avoid_list": avoid,
        "model_search_prior": ["feature_mlp_or_lr", "low_strength_label_propagation", "shallow_graph_view", "cross_fit_fusion"],
        "initial_hypotheses": [
            "feature-only LR/MLP provides a reproducible baseline",
            f"{direction} graph view carries the strongest label correlation",
            "train/test propensity shift must be monitored via degree-matched and propensity-matched panels",
            "fusion of feature and low-strength graph signals likely outperforms either alone",
        ],
        "validation_risk": "moderate" if shift["train_test_shift"] > 0.55 else "low",
    }


def transferability_matrix() -> dict[str, Any]:
    return {
        "audit": "transferability_matrix",
        "cross_task_prior_mode": "advisory_only",
        "target_task_evidence_priority": "hard",
        "reusable_infrastructure": [
            "工程框架", "Schema", "路径管理", "Doctor", "OOF机制", "M4评价", "M5安全门",
            "M6B方案生成", "M6C Critic", "Research Memory结构", "Portfolio", "Fusion Controller",
            "通用节点分类Adapter", "日志/Hash/Resume/Trajectory",
        ],
        "requires_b1_revalidation": [
            "一跳信息", "Exact Two-hop", "高阶关系", "Directed-out/in", "Undirected-union",
            "Label Propagation", "Smooth", "GraphSAGE", "APPNP", "Feature-Graph融合",
            "节点级Gate", "类别级Gate", "类别权重", "分布重加权", "Feature-kNN", "补图", "多模型融合",
        ],
        "forbidden_direct_transfer": [
            "A1模型权重", "A1 Checkpoint", "A1 OOF概率", "A1 Test预测", "A1 Fold", "A1类别映射",
            "A1 Alpha", "A1 Gate阈值", "A1 isolated expert", "A1 Test-only Patch", "A1节点ID规则", "A1 Online Champion结果",
        ],
    }


def validation_protocol(dataset: NodeClassificationDataset) -> dict[str, Any]:
    return {
        "audit": "validation_protocol",
        "fold_identity": "AFAC_B1_FOLD_V1",
        "protocol": "stratified_5_fold",
        "n_splits": 5,
        "shuffle": True,
        "random_state": 2026,
        "stratified": True,
        "panels": [
            {"panel_id": "B1_STANDARD_PANEL", "type": "standard_stratified", "description": "default 5-fold stratified CV"},
            {"panel_id": "B1_DEGREE_MATCHED_PANEL", "type": "degree_matched", "description": "per-fold reweighting toward test degree distribution"},
            {"panel_id": "B1_PROPENSITY_MATCHED_PANEL", "type": "propensity_matched", "description": "validation subset with highest train/test propensity"},
            {"panel_id": "B1_LOW_DEGREE_PANEL", "type": "low_degree", "description": "validation subset of lowest-degree train nodes"},
            {"panel_id": "B1_TEST_LIKE_PANEL", "type": "test_like", "description": "validation subset most similar to test by propensity"},
        ],
    }
