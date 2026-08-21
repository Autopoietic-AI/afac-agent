# B2 Operator Performance Report — candidate_ranker real scale

- data_load: 3.388s; dataset: {'n_train_users': 40000, 'n_test_users': 10000, 'n_items': 14065, 'validation_status': 'passed'}
- operator: `candidate_ranker_experiment`; sources: ['popularity', 'history', 'pair_transition']
- folds: {"0": {"seconds": 3851.19, "n_eval_users": 8002, "candidate_hit_rate@10": 0.08535366158460385, "parent_hit_rate@10": 0.06123469132716821}, "1": {"seconds": 3918.504, "n_eval_users": 8002, "candidate_hit_rate@10": 0.08885278680329918, "parent_hit_rate@10": 0.06623344163959011}}
- stage timings (s): {"retriever_fit_seconds": 3.678, "retrieve_seconds": 6935.068, "union_merge_seconds": 266.708, "build_training_rows_seconds": 50.617, "ranker_fit_seconds": 214.94, "build_scoring_rows_seconds": 221.793, "rerank_seconds": 101.827}
- total 2-fold screen: 7769.696s (3884.848s/fold)
- estimated 3-fold confirm: 11654.544s
- estimated 5-fold: 19424.24s
- memory: {"rss_before_mb": 413.5, "peak_rss_mb": 413.5, "delta_mb": 0.0}
- candidate scale: {"n_eval_users_last_fold": 8002, "pool_recall": {"candidate_pool_recall@20": 0.21094726318420395, "candidate_pool_recall@50": 0.33366658335416144, "candidate_pool_recall@100": 0.5177455636090977, "candidate_pool_recall@200": 0.6497125718570358}, "top10_metrics": {"hit_rate@10": 0.08885278680329918, "ndcg@10": 0.05634098449621183, "mrr@10": 0.0464311501489707}}
- fits 2h formal budget (confirm + reserve + margin): **False**
