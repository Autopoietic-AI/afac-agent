# -*- coding: utf-8 -*-
"""Materialized hierarchical research views for M6R-A."""
from __future__ import annotations

import json, time
from pathlib import Path
from typing import Any

from .event_store import ResearchEventStore, json_dumps, load_json, make_event, rel_ref, sha256_file, stable_hash
from .taxonomy import BUCKETS, SCOPE_LEVELS

MEMORY_VERSION = "m6r_a_v2"
RESEARCH_CLASSES = [1, 8, 4]


def _as_list(v: Any) -> list[Any]: return v if isinstance(v, list) else []
def _n(d: dict[str, Any], *ks: str, default: Any = None) -> Any:
    cur: Any = d
    for k in ks:
        if not isinstance(cur, dict): return default
        cur = cur.get(k)
    return default if cur is None else cur

def _ref(fb: dict[str, Any]) -> dict[str, Any]:
    return {"feedback_id": fb.get("feedback_id", ""), "tool_name": fb.get("tool_name", ""), "evaluation_tier": fb.get("evaluation_tier", "")}

def _strength(level: str) -> float:
    return {"observed":1.0,"signal_evidence":0.75,"expert_scope":0.65,"structure_only":0.55,"artifact_integrity":0.45,"unavailable":0.1,"uncertain":0.2}.get(level,0.25)

class ResearchMemoryBuilder:
    def __init__(self, *, project_root: str | Path): self.project_root = Path(project_root).resolve()

    def build(self, *, problem_map_path: str|Path, feedback_paths: list[str|Path], deterministic_plan_path: str|Path, shadow_comparison_path: str|Path, project_state_path: str|Path, history_path: str|Path, research_policy_path: str|Path, out_root: str|Path, dry_run: bool=False, force_rebuild: bool=False) -> dict[str, Any]:
        paths = {"problem_map":Path(problem_map_path),"deterministic_plan":Path(deterministic_plan_path),"shadow_comparison":Path(shadow_comparison_path),"project_state":Path(project_state_path),"history":Path(history_path),"research_policy":Path(research_policy_path)}
        fpaths = [Path(p) for p in feedback_paths]
        missing = [k for k,p in paths.items() if not p.exists()]
        if not fpaths: missing.append("feedback")
        missing += [f"feedback[{i}]" for i,p in enumerate(fpaths) if not p.exists()]
        if missing: return {"status":"waiting_for_input","failure_reason":"missing_required_inputs","missing_inputs":missing,"artifacts":{}}
        out_root = Path(out_root); out_root = out_root if out_root.is_absolute() else self.project_root/out_root
        try:
            policy = self._load_policy(paths["research_policy"])
            problem, plan, shadow, state, history = [load_json(paths[k]) for k in ["problem_map","deterministic_plan","shadow_comparison","project_state","history"]]
            feedbacks = [load_json(p) for p in fpaths]; self._validate_feedbacks(feedbacks)
        except ValueError as exc:
            return {"status":"failed","failure_reason":"input_validation_failed","error":str(exc),"missing_inputs":[],"artifacts":{}}
        ih = {k: sha256_file(p) for k,p in paths.items()}; ih.update({f"feedback::{i}::{p.name}": sha256_file(p) for i,p in enumerate(fpaths)})
        memory_id = stable_hash({"memory_version":MEMORY_VERSION,"input_hashes":ih,"policy": {"levels":policy.get("analysis_levels"),"weights":policy.get("priority_weights")}})
        run_dir = out_root/memory_id; manifest_path = run_dir/"research_manifest.json"
        if dry_run: return {"status":"dry_run","memory_id":memory_id,"artifacts":{},"missing_inputs":[]}
        if manifest_path.exists() and not force_rebuild:
            m = load_json(manifest_path); return {"status":"duplicate","memory_id":memory_id,"view_hash":m.get("view_hash",""),"artifacts":m.get("artifacts",{}),"missing_inputs":[]}
        run_dir.mkdir(parents=True, exist_ok=True)
        events = self._build_events({"problem_map":problem,"plan":plan,"shadow":shadow,"project_state":state,"history":history,"feedbacks":feedbacks,"feedback_paths":fpaths,"input_hashes":ih})
        store = ResearchEventStore(run_dir/"research_events.jsonl"); appended = store.append_unique(events); loaded = store.load()
        views = self.materialize(loaded, policy=policy, memory_id=memory_id)
        artifacts = {"research_events": rel_ref(run_dir/"research_events.jsonl", self.project_root)}
        for name,payload in views.items():
            p = run_dir/f"{name}.json"; p.write_text(json_dumps(payload)+"\n", encoding="utf-8"); artifacts[name] = rel_ref(p, self.project_root)
        report = run_dir/"RESEARCH_MEMORY_REPORT.md"; report.write_text(render_memory_report(views), encoding="utf-8"); artifacts["research_report"] = rel_ref(report, self.project_root)
        vh = stable_hash({n:v.get("view_hash") for n,v in sorted(views.items())})
        manifest = {"memory_version":MEMORY_VERSION,"memory_id":memory_id,"created_at_epoch_seconds":time.time(),"baseline_commit":"af73875c7f6fe4aa083a5ac474f8d80f0f5262a8","read_only":True,"counts_as_experiment_round":False,"mutates_project_state":False,"mutates_predictions":False,"calls_llm":False,"calls_api":False,"uses_network":False,"executes_adapter":False,"trains_model":False,"generates_prediction":False,"event_count":len(loaded),"events_appended":appended,"view_hash":vh,"input_hashes":ih,"artifacts":artifacts}
        manifest_path.write_text(json_dumps(manifest)+"\n", encoding="utf-8"); artifacts["research_manifest"] = rel_ref(manifest_path, self.project_root)
        return {"status":"completed","memory_id":memory_id,"view_hash":vh,"event_count":len(loaded),"artifacts":artifacts,"missing_inputs":[]}

    def update(self, *, memory_root: str|Path, feedback_path: str|Path, experiment_manifest_path: str|Path, research_policy_path: str|Path) -> dict[str, Any]:
        mem = self._resolve_memory_dir(memory_root); missing=[n for n,p in {"feedback":Path(feedback_path),"experiment_manifest":Path(experiment_manifest_path),"research_policy":Path(research_policy_path)}.items() if not p.exists()]
        if missing: return {"status":"waiting_for_input","failure_reason":"missing_required_inputs","missing_inputs":missing,"artifacts":{}}
        policy = self._load_policy(research_policy_path); fb = load_json(feedback_path); self._validate_feedbacks([fb])
        event = self._event_for_feedback(fb, rel_ref(feedback_path, self.project_root), {"feedback":sha256_file(feedback_path)})
        store = ResearchEventStore(mem/"research_events.jsonl"); appended = store.append_unique([event])
        views = self.materialize(store.load(), policy=policy, memory_id=mem.name)
        for name,payload in views.items(): (mem/f"{name}.json").write_text(json_dumps(payload)+"\n", encoding="utf-8")
        manifest = load_json(mem/"research_manifest.json") if (mem/"research_manifest.json").exists() else {"artifacts":{}}
        manifest.update({"event_count":len(store.load()),"events_appended_last_update":appended,"view_hash":stable_hash({n:v.get("view_hash") for n,v in sorted(views.items())})})
        (mem/"research_manifest.json").write_text(json_dumps(manifest)+"\n", encoding="utf-8")
        return {"status":"completed" if appended else "duplicate","memory_id":mem.name,"events_appended":appended,"artifacts":{"research_manifest":rel_ref(mem/"research_manifest.json", self.project_root)}}

    def materialize(self, events: list[dict[str, Any]], *, policy: dict[str, Any], memory_id: str) -> dict[str, dict[str, Any]]:
        gv, bv, bcv, cv, mv = build_global_view(events), build_bucket_view(events, policy), build_bucket_class_view(events), build_cross_bucket_class_view(events), build_mechanism_view(events)
        ml, fl, sl = build_method_attempt_ledger(events), build_failure_ledger(events), build_success_ledger(events)
        idx, q = build_problem_method_index(ml, fl), build_research_queue(gv, bv, bcv, mv, policy)
        prof = {"profile_version":MEMORY_VERSION,"memory_id":memory_id,"scope_levels":SCOPE_LEVELS,"execution_blockers":gv["execution_blockers"],"problem_counts":{"global":len(gv["scientific_problems"]),"bucket":len(bv["buckets"]),"bucket_class":len(bcv["bucket_class_problems"]),"cross_bucket_class":len(cv["class_audits"]),"error_mechanism":len(mv["mechanisms"])}}
        views = {"global_problem_view":gv,"bucket_problem_view":bv,"bucket_class_problem_view":bcv,"cross_bucket_class_view":cv,"error_mechanism_view":mv,"method_attempt_ledger":ml,"failure_ledger":fl,"success_ledger":sl,"research_queue":q,"problem_method_index":idx,"research_problem_profile":prof}
        for payload in views.values(): payload["view_hash"] = stable_hash({k:v for k,v in payload.items() if k!="view_hash"})
        return views

    def _load_policy(self, path: str|Path) -> dict[str, Any]:
        p = load_json(path); req={"analysis_levels","bucket_taxonomy","mechanism_taxonomy","priority_weights","max_global_deep_research","max_bucket_deep_research","max_bucket_class_deep_research","allow_internal_model_knowledge_as_source","allow_unverified_method_promotion","allow_automatic_branch_reopen","allow_automatic_experiment_execution"}
        miss=sorted(req-set(p))
        if miss: raise ValueError(f"research policy missing required keys: {miss}")
        for k in ["allow_internal_model_knowledge_as_source","allow_unverified_method_promotion","allow_automatic_branch_reopen","allow_automatic_experiment_execution"]:
            if p.get(k) is not False: raise ValueError(f"research policy unsafe: {k} must be false")
        return p
    def _validate_feedbacks(self, feedbacks: list[dict[str, Any]]) -> None:
        for fb in feedbacks:
            for k in ["feedback_id","feedback_kind","evaluation_tier","tool_name","status","validity","metrics","evidence","safety"]:
                if k not in fb: raise ValueError(f"feedback missing required key: {k}")
            if fb.get("safety",{}).get("test_truth_metrics_emitted") is True: raise ValueError("feedback uses forbidden test truth metrics")
    def _resolve_memory_dir(self, memory_root: str|Path) -> Path:
        p=Path(memory_root); p=p if p.is_absolute() else self.project_root/p
        if (p/"research_events.jsonl").exists(): return p
        kids=sorted([x for x in p.iterdir() if (x/"research_events.jsonl").exists()]) if p.exists() else []
        if len(kids)==1: return kids[0]
        raise ValueError(f"cannot resolve memory root: {memory_root}")

    def _build_events(self, c: dict[str, Any]) -> list[dict[str, Any]]:
        events=[]; plan=c["plan"]; pm=c["problem_map"]; state=c["project_state"]; hist=c["history"]; ih=c["input_hashes"]
        missing=sorted(set(_as_list(plan.get("missing_inputs"))))
        if missing:
            events.append(make_event(event_type="problem_observed",task=state.get("task","A1"),scope_level="global",problem_ids=["blocker::missing_full_anchor_evaluation_inputs"],observations={"concept":"execution_blocker","blocker_id":"missing_full_anchor_evaluation_inputs","missing_inputs":missing},decision={"must_not_mask_scientific_problem":True,"auto_execution_allowed":False},outcome_status="blocked",evidence_level="planner_decision",confidence="high",failure_mode="missing_input",input_hashes=ih))
        events.append(make_event(event_type="problem_observed",task=state.get("task","A1"),scope_level="global",problem_ids=["scientific::heterogeneous_information_conditions"],observations={"concept":"scientific_problem","description":"Different structure buckets expose different information sources; one uniform propagation or representation strategy is not yet evidence-safe globally.","globalization_status":"unverified","supporting_bucket_count":len(_as_list(pm.get("problems"))),"execution_blocker_separate":bool(missing)},outcome_status="open",evidence_level="structure_only",confidence="medium",input_hashes=ih))
        for prob in _as_list(pm.get("problems")):
            bucket=str(prob.get("scope") or prob.get("bucket") or str(prob.get("problem_id","")).split("::")[-1])
            events.append(make_event(event_type="problem_observed",task=state.get("task","A1"),scope_level="bucket",scope_refs={"bucket_id":bucket},problem_ids=[f"bucket::{bucket}"],observations={"concept":"scientific_problem","bucket_id":bucket,"sample_count":prob.get("node_count"),"structure_evidence":prob.get("structure_evidence","unavailable"),"available_information_sources":_available_sources(bucket),"missing_information_sources":_missing_sources(bucket, prob),"closed_routes":prob.get("blocked_tool_families",[]),"open_questions":["Need fold-aware OOF bucket metrics before promotion."]},outcome_status="open",evidence_level="structure_only",confidence="medium",input_hashes=ih))
        for bucket in BUCKETS:
            for cls in RESEARCH_CLASSES:
                events.append(make_event(event_type="problem_deferred",task=state.get("task","A1"),scope_level="bucket_class",scope_refs={"bucket_id":bucket,"class_id":cls},problem_ids=[f"bucket_class::{bucket}::class_{cls}"],observations={"concept":"evidence_gap","bucket_id":bucket,"class_id":cls,"evidence_status":"unavailable","uncertainty":"No verified Bucket × Class OOF/confusion/rescue/damage metrics are available."},outcome_status="deferred",evidence_level="unavailable",confidence="low",input_hashes=ih))
        for cls in RESEARCH_CLASSES:
            events.append(make_event(event_type="problem_deferred",task=state.get("task","A1"),scope_level="cross_bucket_class",scope_refs={"class_id":cls},problem_ids=[f"cross_bucket_class::class_{cls}"],observations={"concept":"cross_bucket_class_audit","class_id":cls,"status":"uncertain","uncertainties":["Bucket-specific class metrics are unavailable; do not promote bucket-class evidence to global class claim."]},outcome_status="deferred",evidence_level="unavailable",confidence="low",input_hashes=ih))
        for mech in ["information_source_missing","neighbor_unreliability","uniform_smoothing_damage","multi_hop_signal_opportunity"]:
            events.append(make_event(event_type="hypothesis_created",task=state.get("task","A1"),scope_level="error_mechanism",scope_refs={"mechanism_id":mech},problem_ids=[f"mechanism::{mech}"],hypothesis_ids=[f"hypothesis::{mech}"],observations=_mechanism_obs(mech),outcome_status="open",evidence_level="unverified_self_contained_oof" if mech=="uniform_smoothing_damage" else "signal_evidence",confidence="medium" if mech in {"information_source_missing","multi_hop_signal_opportunity"} else "low",input_hashes=ih))
        for fb,path in zip(c["feedbacks"], c["feedback_paths"]):
            events.append(self._event_for_feedback(fb, rel_ref(path,self.project_root), ih)); events.extend(self._failure_events_for_feedback(fb, ih))
        for branch in sorted(set(map(str,_as_list(hist.get("closed_branches"))+_as_list(state.get("closed_branches"))))):
            events.append(make_event(event_type="branch_closed",task=state.get("task","A1"),branch_id=branch,observations={"closed_branch":branch,"reopen_requires_human":True},outcome_status="closed",evidence_level="confirmed_history",confidence="high",branch_effect="closed",input_hashes=ih))
        return events

    def _event_for_feedback(self, fb: dict[str, Any], path: str, ih: dict[str,str]) -> dict[str, Any]:
        tool=str(fb.get("tool_name","")); metrics=fb.get("metrics",{}); ev=fb.get("evidence",{})
        outcome="inconclusive"; et="method_inconclusive"; fail=""; succ=""; branch="none"
        if tool=="A1_V53Q1_PATCH_REPLAY_SAFE": outcome="materialized_reference"; succ="reference_verified"
        elif tool=="A1_V49A_EDGE_UTILITY_AUDIT": outcome="materialized_reference" if ev.get("champion_patch_coverage") else "inconclusive"; et="evidence_added"; succ="signal_evidence"
        elif tool=="A1_V46A1_ISOLATED_AUDIT": fail="missing_oof" if ev.get("oof_status",{}).get("status")=="unavailable" else ""
        elif tool=="A1_OOF_CANDIDATE_EVALUATOR":
            net=_n(metrics,"rescue_damage","net"); outcome="failure" if isinstance(net,(int,float)) and net<0 else "inconclusive"; fail="negative_net" if outcome=="failure" else "evaluation_unavailable"; branch="keep_closed"
        event_hashes={"feedback_id":str(fb.get("feedback_id","")),"execution_result_hash":str(fb.get("execution_result_hash","")),"execution_identity_hash":str(fb.get("execution_identity_hash",""))}
        return make_event(event_type=et,task=fb.get("task","A1"),experiment_id=str(fb.get("feedback_id","")),execution_identity=str(fb.get("execution_identity_hash","")),branch_id=tool,candidate_id=str(fb.get("adapter_id",tool)),scope_level="global",method_ids=[f"method::{tool}"],feedback_refs=[_ref(fb)],artifact_refs=[{"kind":"feedback","path":path}],observations={"feedback_kind":fb.get("feedback_kind"),"tool_name":tool,"recommendation":fb.get("recommendation"),"validity":fb.get("validity",{})},metrics_summary=metrics,outcome_status=outcome,evidence_level=str(fb.get("evaluation_tier","")),confidence="high" if fb.get("validity",{}).get("status")=="valid" else "low",failure_mode=fail,success_mode=succ,branch_effect=branch,input_hashes=event_hashes)

    def _failure_events_for_feedback(self, fb: dict[str, Any], ih: dict[str,str]) -> list[dict[str, Any]]:
        tool=str(fb.get("tool_name","")); metrics=fb.get("metrics",{}); ev=fb.get("evidence",{}); fs=[]
        if tool=="A1_V46A1_ISOLATED_AUDIT" and ev.get("oof_status",{}).get("status")=="unavailable": fs.append("missing_oof")
        if tool=="A1_OOF_CANDIDATE_EVALUATOR":
            if isinstance(_n(metrics,"overall","gain"),(int,float)) and _n(metrics,"overall","gain")<0: fs.append("negative_overall_gain")
            if isinstance(_n(metrics,"macro","gain"),(int,float)) and _n(metrics,"macro","gain")<0: fs.append("negative_macro_gain")
            if isinstance(_n(metrics,"rescue_damage","net"),(int,float)) and _n(metrics,"rescue_damage","net")<0: fs.append("negative_net")
            if ev.get("parent_identity_status")=="unverified": fs.append("parent_unverified")
        event_hashes={"feedback_id":str(fb.get("feedback_id","")),"execution_result_hash":str(fb.get("execution_result_hash","")),"execution_identity_hash":str(fb.get("execution_identity_hash",""))}
        out=[]
        for ft in sorted(set(fs)):
            out.append(make_event(event_type="failure_recorded",task=fb.get("task","A1"),experiment_id=str(fb.get("feedback_id","")),branch_id=tool,scope_level="global",method_ids=[f"method::{tool}"],feedback_refs=[_ref(fb)],observations={"failure_type":ft,"failure_summary":_failure_summary(ft),"root_cause_status":"hypothesis","method_vs_scope_distinction":_failure_scope(ft),"oracle_gain_does_not_reopen":True},metrics_summary=metrics,outcome_status="failure",evidence_level=str(fb.get("evaluation_tier","")),confidence="medium",failure_mode=ft,branch_effect="keep_closed" if ft in {"negative_net","parent_unverified"} else "none",input_hashes=event_hashes))
        return out


def _available_sources(bucket: str) -> list[str]:
    return {"isolated":["existing_node_attributes"],"one_hop_available":["existing_node_attributes","one_hop_topology"],"exact2_only":["existing_node_attributes","exact_two_hop_topology"],"exact3_4_only":["existing_node_attributes","higher_order_topology"],"graph_visible":["existing_node_attributes","graph_topology"]}.get(bucket,["existing_node_attributes"])
def _missing_sources(bucket: str, prob: dict[str, Any]) -> list[str]:
    sig=prob.get("signal_status",{}) if isinstance(prob.get("signal_status"),dict) else {}; miss=[]
    if sig.get("v53q1_anchor_oof")=="missing": miss.append("teacher_soft_targets")
    if sig.get("canonical_fold")=="missing": miss.append("fold_stability")
    if bucket=="isolated": miss.append("graph_topology")
    return sorted(set(miss))
def _mechanism_obs(mech: str) -> dict[str, Any]:
    m={"information_source_missing":{"affected_scope":"isolated and no-visible-train buckets","supporting_evidence":["M2 structure buckets show nodes without visible train supervision."],"open_hypotheses":["Need non-topological information source or teacher signal for isolated nodes."]},"neighbor_unreliability":{"affected_scope":"one_hop_available class-sensitive neighborhoods","supporting_evidence":["v49A edge utility gate identified selected edge cases."],"open_hypotheses":["One-hop reliability may vary by class; requires Bucket × Class OOF evidence."]},"uniform_smoothing_damage":{"affected_scope":"global/self-contained OOF candidate","supporting_evidence":["Observed negative overall/macro/net in self-contained OOF feedback."],"contradictory_evidence":["Oracle gain exists but is not sufficient to reopen a route."],"open_hypotheses":["Uniform smoothing can damage heterogeneous neighborhoods."]},"multi_hop_signal_opportunity":{"affected_scope":"exact2_only and exact3_4_only","supporting_evidence":["M2 exact-hop structure buckets are observed."],"open_hypotheses":["Exact-hop signals should be audited separately from one-hop propagation."]}}
    return {"mechanism_id":mech, **m.get(mech,{})}
def _failure_summary(ft: str) -> str:
    return {"negative_overall_gain":"Candidate overall OOF gain is below parent in self-contained comparison.","negative_macro_gain":"Candidate macro OOF gain is below parent in self-contained comparison.","negative_net":"Rescue/damage net is negative; oracle gain must not reopen automatically.","parent_unverified":"Parent identity is unverified, so result cannot promote against frozen champion.","missing_oof":"Scope evidence exists but final OOF is missing."}.get(ft,ft)
def _failure_scope(ft: str) -> str:
    return "evaluation_unavailable" if ft in {"missing_oof","parent_unverified"} else "configuration_failed" if ft.startswith("negative") else "method_failed"

def build_global_view(events: list[dict[str, Any]]) -> dict[str, Any]:
    blockers=[]; sci=[]
    for e in events:
        obs=e.get("observations",{})
        if obs.get("concept")=="execution_blocker": blockers.append({"blocker_id":obs.get("blocker_id"),"missing_inputs":obs.get("missing_inputs",[]),"evidence_level":e.get("evidence_level")})
        if e.get("scope_level")=="global" and obs.get("concept")=="scientific_problem": sci.append({"problem_id":e.get("problem_ids",[""])[0],"description":obs.get("description"),"globalization_status":obs.get("globalization_status","unverified"),"confidence":e.get("confidence")})
    return {"view_version":MEMORY_VERSION,"scope_level":"global","execution_blockers":blockers,"scientific_problems":sci,"globalization_status":"unverified","primary_contradiction":sci[0]["description"] if sci else "unavailable","secondary_contradiction":"Full anchor inputs remain unavailable but are tracked separately from scientific problems."}

def build_bucket_view(events: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any]:
    by={}
    for e in events:
        if e.get("scope_level")!="bucket": continue
        obs=e.get("observations",{}); b=str(obs.get("bucket_id",e.get("scope_refs",{}).get("bucket_id","")))
        by[b]={"bucket_id":b,"sample_count":obs.get("sample_count"),"parent_metric":None,"candidate_metric":None,"rescue":None,"damage":None,"net":None,"main_contradiction":"information availability differs by structure bucket","secondary_contradiction":"OOF bucket metrics unavailable","available_information_sources":obs.get("available_information_sources",[]),"missing_information_sources":obs.get("missing_information_sources",[]),"known_effective_methods":[],"known_failed_methods":[],"closed_routes":obs.get("closed_routes",[]),"open_questions":obs.get("open_questions",[]),"evidence_status":obs.get("structure_evidence","unavailable")}
    for b in policy.get("bucket_taxonomy",BUCKETS): by.setdefault(b,{"bucket_id":b,"sample_count":None,"parent_metric":None,"candidate_metric":None,"rescue":None,"damage":None,"net":None,"main_contradiction":"insufficient bucket evidence","secondary_contradiction":"not present in current problem map","available_information_sources":_available_sources(b),"missing_information_sources":["bucket_oof_metrics"],"known_effective_methods":[],"known_failed_methods":[],"closed_routes":[],"open_questions":["Need explicit bucket evidence."],"evidence_status":"unavailable"})
    return {"view_version":MEMORY_VERSION,"scope_level":"bucket","buckets":[by[k] for k in sorted(by)]}

def build_bucket_class_view(events: list[dict[str, Any]]) -> dict[str, Any]:
    rows=[]
    for e in events:
        if e.get("scope_level")!="bucket_class": continue
        r=e.get("scope_refs",{}); rows.append({"bucket_id":r.get("bucket_id"),"class_id":r.get("class_id"),"sample_count":None,"parent_metric":None,"candidate_metric":None,"gain":None,"rescue":None,"damage":None,"net":None,"main_confusion_targets":[],"confidence_shift":None,"fold_stability":"unavailable","evidence_status":e.get("evidence_level","unavailable"),"uncertainty":e.get("observations",{}).get("uncertainty","unavailable")})
    return {"view_version":MEMORY_VERSION,"scope_level":"bucket_class","bucket_class_problems":sorted(rows,key=lambda x:(str(x.get("bucket_id")),int(x.get("class_id") or -1)))}

def build_cross_bucket_class_view(events: list[dict[str, Any]]) -> dict[str, Any]:
    rows=[]
    for e in events:
        if e.get("scope_level")!="cross_bucket_class": continue
        cid=e.get("scope_refs",{}).get("class_id"); rows.append({"class_id":cid,"bucket_comparison":[],"shared_failure_pattern":"uncertain","bucket_specific_pattern":"uncertain","class_global_problem_status":"uncertain","evidence_refs":e.get("evidence_refs",[]),"uncertainties":e.get("observations",{}).get("uncertainties",[])})
    return {"view_version":MEMORY_VERSION,"scope_level":"cross_bucket_class","class_audits":sorted(rows,key=lambda x:int(x.get("class_id") or -1))}

def build_mechanism_view(events: list[dict[str, Any]]) -> dict[str, Any]:
    rows=[]
    for e in events:
        if e.get("scope_level")!="error_mechanism": continue
        obs=e.get("observations",{}); rows.append({"mechanism_id":obs.get("mechanism_id"),"supporting_evidence":obs.get("supporting_evidence",[]),"contradictory_evidence":obs.get("contradictory_evidence",[]),"confidence":e.get("confidence"),"affected_scope":obs.get("affected_scope",""),"known_methods":[],"failed_methods":[],"open_hypotheses":obs.get("open_hypotheses",[]),"evidence_level":e.get("evidence_level")})
    return {"view_version":MEMORY_VERSION,"scope_level":"error_mechanism","mechanisms":sorted(rows,key=lambda x:str(x.get("mechanism_id")))}

def build_method_attempt_ledger(events: list[dict[str, Any]]) -> dict[str, Any]:
    attempts=[]
    for e in events:
        if e.get("event_type") not in {"method_inconclusive","evidence_added"}: continue
        tool=e.get("branch_id",""); metrics=e.get("metrics_summary",{})
        if not tool: continue
        attempts.append({"attempt_version":MEMORY_VERSION,"attempt_id":stable_hash({"event_id":e.get("event_id"),"ledger":"method_attempt"}),"experiment_id":e.get("experiment_id"),"branch_id":tool,"parent_branch_id":e.get("parent_branch_id",""),"method_id":f"method::{tool}","method_family":_method_family(tool),"target_scope_level":_target_scope(tool),"target_scope_refs":{},"target_problem_ids":e.get("problem_ids",[]),"target_mechanism_ids":_target_mechanisms(tool),"hypothesis":_attempt_hypothesis(tool),"information_source_type":_information_source(tool),"new_information_status":_new_info(tool),"parent_candidate":e.get("parent_candidate_id",""),"candidate_identity":e.get("candidate_id",""),"configuration_identity":e.get("execution_identity",""),"tool_identity":tool,"adapter_identity":e.get("candidate_id",""),"metrics_before":{},"metrics_after":metrics,"metric_deltas":{"overall_gain":_n(metrics,"overall","gain"),"macro_gain":_n(metrics,"macro","gain")},"rescue":_n(metrics,"rescue_damage","rescue"),"damage":_n(metrics,"rescue_damage","damage"),"net":_n(metrics,"rescue_damage","net"),"change_precision":_n(metrics,"rescue_damage","change_precision"),"fold_status":_n(metrics,"fold","status",default="unavailable"),"scope_status":"valid" if tool=="A1_V46A1_ISOLATED_AUDIT" else "not_applicable","leakage_status":"safe","parent_identity_status":e.get("observations",{}).get("validity",{}).get("parent_identity_status","unavailable"),"outcome":e.get("outcome_status"),"verdict":_verdict(e),"reason_codes":_reason_codes(e),"success_conditions":[],"failure_conditions":[],"stop_conditions":["do not reopen closed branch automatically"],"feedback_refs":e.get("feedback_refs",[]),"evidence_refs":e.get("evidence_refs",[]),"attempt_hash":stable_hash({"event_id":e.get("event_id"),"metrics":metrics})})
    return {"ledger_version":MEMORY_VERSION,"attempts":sorted(attempts,key=lambda x:str(x.get("attempt_id")))}

def build_failure_ledger(events: list[dict[str, Any]]) -> dict[str, Any]:
    failures=[]
    for e in events:
        if e.get("event_type")!="failure_recorded": continue
        obs=e.get("observations",{})
        failures.append({"failure_id":stable_hash({"event_id":e.get("event_id"),"ledger":"failure"}),"attempt_id":stable_hash({"event_id":e.get("event_id"),"ledger":"method_attempt"}),"experiment_id":e.get("experiment_id"),"scope_level":e.get("scope_level"),"scope_refs":e.get("scope_refs",{}),"failure_type":obs.get("failure_type"),"failure_summary":obs.get("failure_summary"),"root_cause_status":obs.get("root_cause_status"),"root_cause_hypotheses":[obs.get("failure_summary")],"reproducibility":"deterministic_from_feedback","evidence_refs":e.get("feedback_refs",[]),"should_close_branch":e.get("branch_effect")=="keep_closed","reopen_conditions":["human approval","verified new information","canonical Fold and final OOF available"],"transferable_lesson":obs.get("method_vs_scope_distinction"),"affected_future_methods":[],"oracle_gain_does_not_reopen":obs.get("oracle_gain_does_not_reopen",False)})
    return {"ledger_version":MEMORY_VERSION,"failures":sorted(failures,key=lambda x:str(x.get("failure_id")))}

def build_success_ledger(events: list[dict[str, Any]]) -> dict[str, Any]:
    rows=[]
    for e in events:
        if e.get("success_mode"): rows.append({"success_id":stable_hash({"event_id":e.get("event_id"),"ledger":"success"}),"experiment_id":e.get("experiment_id"),"success_mode":e.get("success_mode"),"scope_level":e.get("scope_level"),"evidence_level":e.get("evidence_level"),"feedback_refs":e.get("feedback_refs",[])})
    return {"ledger_version":MEMORY_VERSION,"successes":sorted(rows,key=lambda x:str(x.get("success_id")))}

def build_problem_method_index(ml: dict[str,Any], fl: dict[str,Any]) -> dict[str,Any]:
    return {"index_version":MEMORY_VERSION,"methods_by_problem":{},"failures_by_method":{f["failure_type"]:f for f in fl.get("failures",[])},"attempt_count":len(ml.get("attempts",[]))}

def build_research_queue(gv: dict[str,Any], bv: dict[str,Any], bcv: dict[str,Any], mv: dict[str,Any], policy: dict[str,Any]) -> dict[str,Any]:
    w=policy.get("priority_weights",{}); items=[]
    for p in gv.get("scientific_problems",[]): items.append(_queue_item("global",{},[p.get("problem_id")],"Is a unified strategy unsafe across heterogeneous structure buckets?",0,"structure_only",w,"global"))
    for b in bv.get("buckets",[]): items.append(_queue_item("bucket",{"bucket_id":b.get("bucket_id")},[f"bucket::{b.get('bucket_id')}"],f"What new information source is needed for bucket {b.get('bucket_id')}?",b.get("sample_count") if isinstance(b.get("sample_count"),int) else 0,b.get("evidence_status","unavailable"),w,"bucket"))
    for r in bcv.get("bucket_class_problems",[]):
        if r.get("class_id") in RESEARCH_CLASSES and r.get("bucket_id") in {"one_hop_available","exact2_only","isolated"}: items.append(_queue_item("bucket_class",{"bucket_id":r.get("bucket_id"),"class_id":r.get("class_id")},[f"bucket_class::{r.get('bucket_id')}::class_{r.get('class_id')}"],f"Does class {r.get('class_id')} fail specifically in {r.get('bucket_id')}?",0,r.get("evidence_status","unavailable"),w,"bucket_class"))
    for m in mv.get("mechanisms",[]): items.append(_queue_item("error_mechanism",{"mechanism_id":m.get("mechanism_id")},[f"mechanism::{m.get('mechanism_id')}"],f"How should mechanism {m.get('mechanism_id')} be tested safely?",0,m.get("evidence_level","unavailable"),w,"mechanism"))
    ranked=sorted(items,key=lambda x:(-x["priority_score"],x["scope_level"],json.dumps(x["scope_refs"],sort_keys=True)))
    limits={"global":int(policy.get("max_global_deep_research",0)),"bucket":int(policy.get("max_bucket_deep_research",0)),"bucket_class":int(policy.get("max_bucket_class_deep_research",0))}; used={k:0 for k in limits}
    for i,item in enumerate(ranked,1):
        item["priority_rank"]=i; lvl=item["scope_level"]; item["deep_research_selected"]=lvl in used and used[lvl]<limits[lvl]
        if item["deep_research_selected"]: used[lvl]+=1
    return {"queue_version":MEMORY_VERSION,"policy_limits":limits,"items":ranked}

def _queue_item(level, refs, problems, question, count, evidence, weights, family):
    comps={"affected_count":min(float(count)/11001.0,1.0) if count else 0.0,"evidence_strength":_strength(evidence),"method_availability":0.5 if family in {"bucket","bucket_class","mechanism"} else 0.25,"risk":0.1}
    score=sum(float(weights.get(k,0.0))*v for k,v in comps.items())
    return {"queue_item_id":stable_hash({"level":level,"refs":refs,"problems":problems}),"problem_ids":problems,"scope_level":level,"scope_refs":refs,"primary_question":question,"secondary_questions":[],"affected_count":count,"evidence_strength":evidence,"estimated_score_impact":"unknown_without_full_anchor_oof","macro_importance":"unknown","fold_stability":"unavailable","novel_information_gap":"required","method_availability":comps["method_availability"],"experiment_cost":"low_read_only_research","research_cost":"medium","risk":comps["risk"],"priority_components":comps,"priority_score":round(score,12),"priority_rank":0,"recommended_action":"generate_research_brief","required_evidence":["canonical Fold","final v53Q-1 OOF proba","Bucket × Class OOF metrics"],"missing_evidence":["full anchor OOF","bucket-class metrics"],"status":"open","reason_codes":["policy_ranked","no_auto_execution"]}

def _method_family(t): return "edge_utility_signal" if "V49A" in t else "isolated_expert_scope" if "V46A" in t else "oof_candidate_evaluation" if "OOF" in t else "champion_replay" if "REPLAY" in t else "artifact_audit"
def _target_scope(t): return "bucket" if "V46A" in t else "error_mechanism" if "V49A" in t else "global"
def _target_mechanisms(t): return ["neighbor_unreliability","multi_hop_signal_opportunity"] if "V49A" in t else ["information_source_missing"] if "V46A" in t else ["uniform_smoothing_damage"] if "OOF" in t else []
def _attempt_hypothesis(t): return {"A1_V49A_EDGE_UTILITY_AUDIT":"Edge utility signal may isolate safe transition-stable patches.","A1_V46A1_ISOLATED_AUDIT":"Isolated expert changes should be scoped to isolated nodes.","A1_V53Q1_PATCH_REPLAY_SAFE":"Replay verifies reference identity, not new gain.","A1_OOF_CANDIDATE_EVALUATOR":"Self-contained OOF candidate must beat embedded parent before promotion."}.get(t,"Audit feedback can inform research memory.")
def _information_source(t): return "directed_path_signal" if "V49A" in t else "existing_node_attributes" if "V46A" in t else "teacher_soft_targets" if "OOF" in t else "other"
def _new_info(t): return "new_information" if "V49A" in t else "new_combination_of_existing_information" if "V46A" in t else "new_representation_only" if "OOF" in t else "same_information_as_failed_route" if "REPLAY" in t else "unknown"
def _verdict(e): return "do_not_promote" if e.get("outcome_status")=="failure" else "reference_only" if e.get("success_mode")=="reference_verified" else "informational_only"
def _reason_codes(e): return sorted(set(([e.get("failure_mode")] if e.get("failure_mode") else [])+([e.get("success_mode")] if e.get("success_mode") else [])+(["candidate_already_materialized"] if e.get("outcome_status")=="materialized_reference" else [])))
def render_memory_report(views: dict[str,Any]) -> str:
    gv=views["global_problem_view"]; q=views["research_queue"]
    lines=["# M6R-A v2 Research Memory Report","","## Global","",f"Primary: {gv.get('primary_contradiction')}","","## Top Research Queue",""]
    for item in q.get("items",[])[:5]: lines.append(f"- rank {item['priority_rank']}: {item['primary_question']} score={item['priority_score']}")
    lines += ["","No LLM/API/Adapter/training/prediction was executed.",""]
    return "\n".join(lines)

# ------------------------- M6R-A v2.1 multi-axis correction -----------------
import csv as _csv

AXIS_CONNECTIVITY = "connectivity_visibility"
AXIS_REACHABILITY = "train_label_reachability"
AXIS_DEGREE = "degree_band"
AXIS_CLASS = "class_id"


def _axis_value_from_row(row: dict[str, Any], axis_id: str) -> str:
    if axis_id == AXIS_CONNECTIVITY:
        if row.get("graph_visible") == "true" and row.get("isolated") != "true":
            return "graph_visible"
        if row.get("isolated") == "true" and row.get("graph_visible") != "true":
            return "isolated"
        return "unknown"
    if axis_id == AXIS_REACHABILITY:
        return row.get("primary_supervision_bucket") or "unknown"
    if axis_id == AXIS_DEGREE:
        return row.get("degree_bucket") or "unknown"
    return "unknown"


def make_scope_signature(bucket_axes: list[dict[str, str]] | None = None, class_id: int | None = None, mechanism_id: str | None = None) -> dict[str, Any]:
    axes = sorted(bucket_axes or [], key=lambda x: (str(x.get("axis_id")), str(x.get("value_id"))))
    sig = {"bucket_axes": axes, "class_id": class_id, "mechanism_id": mechanism_id}
    sig["scope_id"] = stable_hash(sig)
    return sig


def _single_axis(axis_id: str, value_id: str) -> dict[str, Any]:
    return make_scope_signature([{"axis_id": axis_id, "value_id": value_id}])


def _load_node_bucket_rows(problem_map_path: Path) -> list[dict[str, Any]]:
    candidate = problem_map_path.parent / "a1_node_buckets.csv"
    if not candidate.exists():
        return []
    with candidate.open("r", encoding="utf-8-sig", newline="") as file:
        return list(_csv.DictReader(file))


def _axis_registry_and_overlap(rows: list[dict[str, Any]], root: Path, problem_map_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    axes = [
        {"axis_id": AXIS_CONNECTIVITY, "display_name": "Connectivity visibility", "axis_type": "partition", "allowed_values": ["graph_visible", "isolated", "unknown"], "mutually_exclusive_within_axis": True, "can_intersect_with_axes": [AXIS_REACHABILITY, AXIS_DEGREE], "source_field": "graph_visible/isolated", "evidence_refs": [rel_ref(problem_map_path.parent / "a1_node_buckets.csv", root)]},
        {"axis_id": AXIS_REACHABILITY, "display_name": "Train-label reachability", "axis_type": "partition", "allowed_values": ["one_hop_available", "exact2_only", "exact3_4_only", "no_visible_train_within_4_hops", "unknown"], "mutually_exclusive_within_axis": True, "can_intersect_with_axes": [AXIS_CONNECTIVITY, AXIS_DEGREE], "source_field": "primary_supervision_bucket", "evidence_refs": [rel_ref(problem_map_path.parent / "a1_node_buckets.csv", root)]},
        {"axis_id": AXIS_DEGREE, "display_name": "Degree band", "axis_type": "partition", "allowed_values": ["degree_1", "degree_2_5", "degree_6p", "isolated", "unknown"], "mutually_exclusive_within_axis": True, "can_intersect_with_axes": [AXIS_CONNECTIVITY, AXIS_REACHABILITY], "source_field": "degree_bucket", "evidence_refs": [rel_ref(problem_map_path.parent / "a1_node_buckets.csv", root)]},
        {"axis_id": AXIS_CLASS, "display_name": "Class id", "axis_type": "analysis_dimension", "allowed_values": [str(i) for i in range(10)], "mutually_exclusive_within_axis": True, "can_intersect_with_axes": [AXIS_CONNECTIVITY, AXIS_REACHABILITY, AXIS_DEGREE], "source_field": "true_label/train_only", "evidence_refs": [rel_ref(problem_map_path.parent / "a1_node_buckets.csv", root)]},
    ]
    axis_counts: dict[str, dict[str, int]] = {a["axis_id"]: {} for a in axes if a["axis_id"] != AXIS_CLASS}
    within = {a["axis_id"]: {"duplicate_count": 0, "uncovered_count": 0, "unknown_count": 0} for a in axes if a["axis_id"] != AXIS_CLASS}
    intersections: dict[str, int] = {}
    for row in rows:
        values = {axis: _axis_value_from_row(row, axis) for axis in [AXIS_CONNECTIVITY, AXIS_REACHABILITY, AXIS_DEGREE]}
        # raw duplicate/uncovered audit for connectivity only needs both bools; other axes are single source fields.
        if row.get("graph_visible") == "true" and row.get("isolated") == "true":
            within[AXIS_CONNECTIVITY]["duplicate_count"] += 1
        if row.get("graph_visible") != "true" and row.get("isolated") != "true":
            within[AXIS_CONNECTIVITY]["uncovered_count"] += 1
        for axis, value in values.items():
            axis_counts[axis][value] = axis_counts[axis].get(value, 0) + 1
            if value == "unknown":
                within[axis]["unknown_count"] += 1
        for a1, a2 in [(AXIS_CONNECTIVITY, AXIS_REACHABILITY), (AXIS_CONNECTIVITY, AXIS_DEGREE), (AXIS_REACHABILITY, AXIS_DEGREE)]:
            key = f"{a1}={values[a1]}|{a2}={values[a2]}"
            intersections[key] = intersections.get(key, 0) + 1
    registry = {"registry_version": "m6r_a_v2_1", "bucket_axis_registry": axes, "axis_value_counts": axis_counts, "notes": ["Values within an axis are mutually exclusive; values across axes may intersect.", "class_id is an independent analysis axis and is not a bucket value."]}
    isolated = axis_counts.get(AXIS_CONNECTIVITY, {}).get("isolated", 0)
    no_visible = axis_counts.get(AXIS_REACHABILITY, {}).get("no_visible_train_within_4_hops", 0)
    iso_no = intersections.get(f"{AXIS_CONNECTIVITY}=isolated|{AXIS_REACHABILITY}=no_visible_train_within_4_hops", 0)
    gv_no = intersections.get(f"{AXIS_CONNECTIVITY}=graph_visible|{AXIS_REACHABILITY}=no_visible_train_within_4_hops", 0)
    overlap = {"audit_version": "m6r_a_v2_1", "node_count": len(rows), "axis_within_audit": within, "axis_value_counts": axis_counts, "cross_axis_intersections": intersections, "illegal_intersections": [], "unknown_counts": {axis: info["unknown_count"] for axis, info in within.items()}, "specific_relations": {"isolated_vs_no_visible_train_within_4_hops": {"intersection_count": iso_no, "isolated_count": isolated, "no_visible_train_within_4_hops_count": no_visible, "isolated_coverage_ratio": round(iso_no / isolated, 12) if isolated else None, "no_visible_coverage_ratio": round(iso_no / no_visible, 12) if no_visible else None, "scope_overlap_status": _overlap_status(iso_no, isolated, no_visible)}, "graph_visible_and_no_visible_train_within_4_hops": {"intersection_count": gv_no, "status": "observed" if gv_no else "none"}}, "coverage_hash": stable_hash({"within": within, "counts": axis_counts, "intersections": intersections})}
    registry["coverage_hash"] = stable_hash(registry)
    return registry, overlap


def _overlap_status(intersection: int, count_a: int, count_b: int) -> str:
    if count_a == 0 or count_b == 0:
        return "unknown"
    if intersection == 0:
        return "disjoint"
    if intersection == count_a == count_b:
        return "identical"
    if intersection == count_a:
        return "subset"
    if intersection == count_b:
        return "superset"
    ratio = intersection / min(count_a, count_b)
    if ratio >= 0.8:
        return "high_overlap"
    if ratio >= 0.2:
        return "partial_overlap"
    return "low_overlap"


def _scope_count_from_audit(audit: dict[str, Any], sig: dict[str, Any]) -> int | None:
    axes = sig.get("bucket_axes") or []
    if len(axes) == 1:
        axis, value = axes[0]["axis_id"], axes[0]["value_id"]
        return audit.get("axis_value_counts", {}).get(axis, {}).get(value)
    if len(axes) == 2:
        a, b = axes
        key = f"{a['axis_id']}={a['value_id']}|{b['axis_id']}={b['value_id']}"
        rkey = f"{b['axis_id']}={b['value_id']}|{a['axis_id']}={a['value_id']}"
        inter = audit.get("cross_axis_intersections", {})
        return inter.get(key, inter.get(rkey))
    return None


def _m21_build(self, *, problem_map_path: str|Path, feedback_paths: list[str|Path], deterministic_plan_path: str|Path, shadow_comparison_path: str|Path, project_state_path: str|Path, history_path: str|Path, research_policy_path: str|Path, out_root: str|Path, dry_run: bool=False, force_rebuild: bool=False) -> dict[str, Any]:
    paths={"problem_map":Path(problem_map_path),"deterministic_plan":Path(deterministic_plan_path),"shadow_comparison":Path(shadow_comparison_path),"project_state":Path(project_state_path),"history":Path(history_path),"research_policy":Path(research_policy_path)}
    fpaths=[Path(p) for p in feedback_paths]
    missing=[k for k,p in paths.items() if not p.exists()] + [f"feedback[{i}]" for i,p in enumerate(fpaths) if not p.exists()]
    if not fpaths: missing.append("feedback")
    if missing: return {"status":"waiting_for_input","failure_reason":"missing_required_inputs","missing_inputs":missing,"artifacts":{}}
    out=Path(out_root); out=out if out.is_absolute() else self.project_root/out
    try:
        policy=self._load_policy(paths["research_policy"]); pm=load_json(paths["problem_map"]); plan=load_json(paths["deterministic_plan"]); shadow=load_json(paths["shadow_comparison"]); state=load_json(paths["project_state"]); hist=load_json(paths["history"]); feedbacks=[load_json(p) for p in fpaths]; self._validate_feedbacks(feedbacks)
    except ValueError as exc:
        return {"status":"failed","failure_reason":"input_validation_failed","error":str(exc),"missing_inputs":[],"artifacts":{}}
    rows=_load_node_bucket_rows(paths["problem_map"])
    registry,audit=_axis_registry_and_overlap(rows, self.project_root, paths["problem_map"])
    ih={k:sha256_file(p) for k,p in paths.items()}; ih.update({f"feedback::{i}::{p.name}":sha256_file(p) for i,p in enumerate(fpaths)})
    ih["bucket_axis_coverage"] = registry.get("coverage_hash", "")
    memory_id=stable_hash({"memory_version":"m6r_a_v2_1","input_hashes":ih,"policy":{"levels":policy.get("analysis_levels"),"weights":policy.get("priority_weights"),"priority_component_weights":policy.get("priority_component_weights")}})
    run=out/memory_id; manifest_path=run/"research_manifest.json"
    if dry_run: return {"status":"dry_run","memory_id":memory_id,"artifacts":{},"missing_inputs":[]}
    if manifest_path.exists() and not force_rebuild:
        m=load_json(manifest_path); return {"status":"duplicate","memory_id":memory_id,"view_hash":m.get("view_hash",""),"artifacts":m.get("artifacts",{}),"missing_inputs":[]}
    run.mkdir(parents=True, exist_ok=True)
    ctx={"problem_map":pm,"plan":plan,"shadow":shadow,"project_state":state,"history":hist,"feedbacks":feedbacks,"feedback_paths":fpaths,"input_hashes":ih,"bucket_axis_registry":registry,"bucket_overlap_audit":audit}
    events=self._build_events(ctx)
    store=ResearchEventStore(run/"research_events.jsonl"); appended=store.append_unique(events); loaded=store.load()
    views=self.materialize(loaded, policy=policy, memory_id=memory_id)
    artifacts={"research_events":rel_ref(run/"research_events.jsonl", self.project_root)}
    for name,payload in views.items():
        p=run/f"{name}.json"; p.write_text(json_dumps(payload)+"\n", encoding="utf-8"); artifacts[name]=rel_ref(p,self.project_root)
    report=run/"RESEARCH_MEMORY_REPORT.md"; report.write_text(render_memory_report(views), encoding="utf-8"); artifacts["research_report"]=rel_ref(report,self.project_root)
    vh=stable_hash({n:v.get("view_hash") for n,v in sorted(views.items())})
    manifest={"memory_version":"m6r_a_v2_1","memory_id":memory_id,"created_at_epoch_seconds":time.time(),"baseline_commit":"af73875c7f6fe4aa083a5ac474f8d80f0f5262a8","read_only":True,"counts_as_experiment_round":False,"mutates_project_state":False,"mutates_predictions":False,"calls_llm":False,"calls_api":False,"uses_network":False,"executes_adapter":False,"trains_model":False,"generates_prediction":False,"event_count":len(loaded),"events_appended":appended,"view_hash":vh,"input_hashes":ih,"artifacts":artifacts}
    manifest_path.write_text(json_dumps(manifest)+"\n", encoding="utf-8"); artifacts["research_manifest"]=rel_ref(manifest_path,self.project_root)
    return {"status":"completed","memory_id":memory_id,"view_hash":vh,"event_count":len(loaded),"artifacts":artifacts,"missing_inputs":[]}


def _m21_build_events(self, c: dict[str, Any]) -> list[dict[str, Any]]:
    events = ResearchMemoryBuilder._m6ra_original_build_events(self, c) if hasattr(ResearchMemoryBuilder, "_m6ra_original_build_events") else []
    ih=c.get("input_hashes", {})
    events.append(make_event(event_type="taxonomy_registered", task=c.get("project_state",{}).get("task","A1"), scope_level="global", problem_ids=["taxonomy::bucket_axis_registry"], observations={"bucket_axis_registry": c.get("bucket_axis_registry", {}), "bucket_overlap_audit": c.get("bucket_overlap_audit", {}), "multi_axis_scope_enabled": True}, outcome_status="completed", evidence_level="dataset_profile", confidence="high", input_hashes={"bucket_axis_coverage": ih.get("bucket_axis_coverage", "")}))
    for axis in c.get("bucket_axis_registry", {}).get("bucket_axis_registry", []):
        events.append(make_event(event_type="scope_reclassified", task=c.get("project_state",{}).get("task","A1"), scope_level="bucket", scope_refs={"axis_id":axis.get("axis_id")}, problem_ids=[f"axis::{axis.get('axis_id')}"], observations={"axis_id":axis.get("axis_id"), "allowed_values":axis.get("allowed_values", []), "within_axis_mutual_exclusion": axis.get("mutually_exclusive_within_axis"), "can_intersect_with_axes": axis.get("can_intersect_with_axes", [])}, outcome_status="completed", evidence_level="dataset_profile", confidence="high", input_hashes={"bucket_axis_coverage": ih.get("bucket_axis_coverage", "")}))
    events.append(make_event(event_type="view_superseded", task=c.get("project_state",{}).get("task","A1"), scope_level="global", observations={"superseded_view": "single_bucket_id_identity", "replacement": "multi_axis_scope_signature", "reason": "graph_visible/isolated and train-label reachability are orthogonal axes"}, outcome_status="completed", evidence_level="design_correction", confidence="high", input_hashes={"bucket_axis_coverage": ih.get("bucket_axis_coverage", "")}))
    return events

if not hasattr(ResearchMemoryBuilder, "_m6ra_original_build_events"):
    ResearchMemoryBuilder._m6ra_original_build_events = ResearchMemoryBuilder._build_events
ResearchMemoryBuilder._build_events = _m21_build_events
ResearchMemoryBuilder.build = _m21_build

def _scope_record(sig: dict[str, Any], affected_count: int | None, parent_ids: list[str] | None = None, child_ids: list[str] | None = None) -> dict[str, Any]:
    return {"scope_id": sig["scope_id"], "scope_signature": sig, "bucket_axes": sig.get("bucket_axes", []), "class_id": sig.get("class_id"), "mechanism_id": sig.get("mechanism_id"), "scope_parent_ids": parent_ids or [], "scope_child_ids": child_ids or [], "affected_count": affected_count, "coverage_hash": stable_hash({"signature": sig, "affected_count": affected_count})}


def _latest_taxonomy(events: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    for e in reversed(events):
        obs=e.get("observations", {})
        if obs.get("multi_axis_scope_enabled"):
            return obs.get("bucket_axis_registry", {}), obs.get("bucket_overlap_audit", {})
    return {"bucket_axis_registry": [], "axis_value_counts": {}}, {"axis_value_counts": {}, "cross_axis_intersections": {}}


def _m21_materialize(self, events: list[dict[str, Any]], *, policy: dict[str, Any], memory_id: str) -> dict[str, dict[str, Any]]:
    registry, audit = _latest_taxonomy(events)
    gv = build_global_view(events)
    bv = _m21_bucket_view(events, policy, audit)
    bcv = _m21_bucket_class_view(events, audit)
    cv = _m21_cross_bucket_class_view(events, audit)
    mv = _m21_mechanism_view(events, audit)
    ml, fl, sl = build_method_attempt_ledger(events), build_failure_ledger(events), build_success_ledger(events)
    idx = _m21_problem_method_index(ml, fl, bv, bcv, mv)
    q = _m21_research_queue(gv, bv, bcv, mv, policy, audit, fl)
    prof = {"profile_version":"m6r_a_v2_1","memory_id":memory_id,"scope_levels":SCOPE_LEVELS,"execution_blockers":gv["execution_blockers"],"bucket_axis_registry_ref":"bucket_axis_registry.json","bucket_overlap_audit_ref":"bucket_overlap_audit.json","problem_counts":{"global":len(gv["scientific_problems"]),"bucket":len(bv["buckets"]),"bucket_class":len(bcv["bucket_class_problems"]),"cross_bucket_class":len(cv["class_audits"]),"error_mechanism":len(mv["mechanisms"])}}
    views={"bucket_axis_registry":registry,"bucket_overlap_audit":audit,"global_problem_view":gv,"bucket_problem_view":bv,"bucket_class_problem_view":bcv,"cross_bucket_class_view":cv,"error_mechanism_view":mv,"method_attempt_ledger":ml,"failure_ledger":fl,"success_ledger":sl,"research_queue":q,"problem_method_index":idx,"research_problem_profile":prof}
    for payload in views.values(): payload["view_hash"] = stable_hash({k:v for k,v in payload.items() if k!="view_hash"})
    return views


def _m21_bucket_view(events: list[dict[str, Any]], policy: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    buckets=[]
    for axis in [AXIS_CONNECTIVITY, AXIS_REACHABILITY, AXIS_DEGREE]:
        for value,count in sorted(audit.get("axis_value_counts", {}).get(axis, {}).items()):
            sig=_single_axis(axis, value); rec=_scope_record(sig, count)
            rec.update({"bucket_id": value, "axis_id": axis, "sample_count": count, "main_contradiction": "information availability differs by bucket axis value", "secondary_contradiction": "OOF bucket metrics unavailable", "parent_metric": None, "candidate_metric": None, "rescue": None, "damage": None, "net": None, "available_information_sources": _available_sources(value), "missing_information_sources": ["bucket_oof_metrics"], "known_effective_methods": [], "known_failed_methods": [], "closed_routes": [], "open_questions": ["Need axis-aware bucket OOF metrics before promotion."], "evidence_status": "observed" if value != "unknown" else "unavailable"})
            buckets.append(rec)
    return {"view_version":"m6r_a_v2_1","scope_level":"bucket","bucket_identity_policy":"multi_axis_scope_signature","buckets":buckets}


def _m21_bucket_class_view(events: list[dict[str, Any]], audit: dict[str, Any]) -> dict[str, Any]:
    rows=[]
    axis_values=[]
    for axis in [AXIS_CONNECTIVITY, AXIS_REACHABILITY]:
        axis_values += [(axis,v) for v in sorted(audit.get("axis_value_counts", {}).get(axis, {})) if v != "unknown"]
    for axis,value in axis_values:
        for cls in RESEARCH_CLASSES:
            sig=make_scope_signature([{"axis_id":axis,"value_id":value}], class_id=cls)
            rec=_scope_record(sig, audit.get("axis_value_counts", {}).get(axis, {}).get(value))
            rec.update({"bucket_id": value, "axis_id": axis, "sample_count": None, "parent_metric": None, "candidate_metric": None, "gain": None, "rescue": None, "damage": None, "net": None, "main_confusion_targets": [], "confidence_shift": None, "fold_stability": "unavailable", "evidence_status": "unavailable", "uncertainty": "No verified Bucket × Class OOF/confusion/rescue/damage metrics are available."})
            rows.append(rec)
    return {"view_version":"m6r_a_v2_1","scope_level":"bucket_class","bucket_class_problems":sorted(rows,key=lambda x:(x.get("axis_id"),str(x.get("bucket_id")),int(x.get("class_id") or -1)))}


def _m21_cross_bucket_class_view(events: list[dict[str, Any]], audit: dict[str, Any]) -> dict[str, Any]:
    rows=[]
    for cls in RESEARCH_CLASSES:
        sig=make_scope_signature([], class_id=cls)
        rec=_scope_record(sig, None)
        rec.update({"bucket_comparison": [], "shared_failure_pattern": "uncertain", "bucket_specific_pattern": "uncertain", "class_global_problem_status": "uncertain", "evidence_refs": [], "uncertainties": ["Bucket-specific class metrics are unavailable; do not promote bucket-class evidence to global class claim."]})
        rows.append(rec)
    return {"view_version":"m6r_a_v2_1","scope_level":"cross_bucket_class","class_audits":rows}


def _m21_mechanism_view(events: list[dict[str, Any]], audit: dict[str, Any]) -> dict[str, Any]:
    rows=[]; seen=set()
    for e in events:
        if e.get("scope_level") != "error_mechanism": continue
        obs=e.get("observations",{}); mech=obs.get("mechanism_id")
        if not mech or mech in seen: continue
        seen.add(mech); sig=make_scope_signature([], mechanism_id=mech); rec=_scope_record(sig, None)
        rec.update({"mechanism_id":mech,"supporting_evidence":obs.get("supporting_evidence",[]),"contradictory_evidence":obs.get("contradictory_evidence",[]),"confidence":e.get("confidence"),"affected_scope":obs.get("affected_scope",""),"known_methods":[],"failed_methods":[],"open_hypotheses":obs.get("open_hypotheses",[]),"evidence_level":e.get("evidence_level")})
        rows.append(rec)
    return {"view_version":"m6r_a_v2_1","scope_level":"error_mechanism","mechanisms":sorted(rows,key=lambda x:str(x.get("mechanism_id")))}


def _m21_problem_method_index(ml: dict[str,Any], fl: dict[str,Any], bv: dict[str,Any], bcv: dict[str,Any], mv: dict[str,Any]) -> dict[str,Any]:
    return {"index_version":"m6r_a_v2_1","multi_axis_scope_support":True,"methods_by_problem":{},"failures_by_method":{f["failure_type"]:f for f in fl.get("failures",[])},"scope_index":{"bucket":[b["scope_id"] for b in bv.get("buckets",[])],"bucket_class":[b["scope_id"] for b in bcv.get("bucket_class_problems",[])],"mechanism":[m["scope_id"] for m in mv.get("mechanisms",[])]},"attempt_count":len(ml.get("attempts",[]))}


def _transform_affected(count: int, total: int, policy: dict[str, Any]) -> float | None:
    if total <= 0: return None
    ratio = max(0.0, min(float(count)/float(total), 1.0))
    mode = policy.get("affected_count_transform", "capped_ratio")
    cap = float(policy.get("affected_count_cap_ratio", 0.25))
    if mode == "linear": return ratio
    if mode == "sqrt": return ratio ** 0.5
    if mode == "log1p":
        import math
        return math.log1p(count) / math.log1p(total)
    return min(ratio, cap) / cap if cap > 0 else ratio


def _scope_overlap_ratio(a: dict[str, Any], b: dict[str, Any], audit: dict[str, Any]) -> tuple[str, float | None]:
    ca, cb = a.get("affected_count"), b.get("affected_count")
    axa, axb = a.get("bucket_axes") or [], b.get("bucket_axes") or []
    if not axa or not axb or not isinstance(ca,int) or not isinstance(cb,int): return "unknown", None
    if axa == axb: return "identical", 1.0
    if len(axa)==1 and len(axb)==1:
        x,y=axa[0],axb[0]
        if x["axis_id"] == y["axis_id"]: return "disjoint", 0.0
        key=f"{x['axis_id']}={x['value_id']}|{y['axis_id']}={y['value_id']}"; rkey=f"{y['axis_id']}={y['value_id']}|{x['axis_id']}={x['value_id']}"
        inter=audit.get("cross_axis_intersections",{}).get(key, audit.get("cross_axis_intersections",{}).get(rkey, 0))
        ratio=inter/min(ca,cb) if min(ca,cb) else None
        return _overlap_status(inter, ca, cb), round(ratio,12) if ratio is not None else None
    return "unknown", None


def _m21_research_queue(gv: dict[str,Any], bv: dict[str,Any], bcv: dict[str,Any], mv: dict[str,Any], policy: dict[str,Any], audit: dict[str,Any], fl: dict[str,Any]) -> dict[str,Any]:
    weights=policy.get("priority_component_weights", policy.get("priority_weights", {})); total=audit.get("node_count", 13752) or 13752
    items=[]
    for b in bv.get("buckets",[]):
        if b.get("axis_id") == AXIS_DEGREE: continue
        items.append(_m21_queue_item("bucket", b, f"What new information source is needed for {b.get('axis_id')}={b.get('bucket_id')}?", weights, policy, total, fl))
    for m in mv.get("mechanisms",[]): items.append(_m21_queue_item("error_mechanism", m, f"How should mechanism {m.get('mechanism_id')} be tested safely?", weights, policy, total, fl))
    for p in gv.get("scientific_problems",[]):
        sig=make_scope_signature([]); rec=_scope_record(sig, None); rec.update({"evidence_status":"structure_only"})
        items.append(_m21_queue_item("global", rec, "Is a unified strategy unsafe across heterogeneous structure buckets?", weights, policy, total, fl))
    old_rank = {"one_hop_available": 1, "no_visible_train_within_4_hops": 2, "exact2_only": 3}
    items=sorted(items,key=lambda x:(-x["raw_priority_score"],x["scope_level"],json.dumps(x["scope_refs"],sort_keys=True)))
    selected=[]; limits={"global":int(policy.get("max_global_deep_research",0)),"bucket":int(policy.get("max_bucket_deep_research",0)),"bucket_class":int(policy.get("max_bucket_class_deep_research",0))}; used={k:0 for k in limits}
    final=[]
    for item in items:
        penalty=0.0; reasons=[]; overlaps=[]
        for s in selected:
            status,ratio=_scope_overlap_ratio(item,s,audit); overlaps.append({"other_queue_item_id":s["queue_item_id"],"scope_overlap_status":status,"overlap_ratio":ratio})
            if status in {"high_overlap","subset","superset","identical"} and item["scope_level"] == s["scope_level"]:
                penalty += float(policy.get("high_overlap_topk_penalty", 0.35)); reasons.append("high_overlap_budget_suppressed")
        item["scope_overlaps"] = overlaps
        item["priority_components"]["overlap_penalty_component"] = -penalty if penalty else 0.0
        item["penalty_total"] = round(penalty + abs(float(item["priority_components"].get("closed_branch_penalty_component") or 0.0)), 12)
        item["final_priority_score"] = round(item["raw_priority_score"] - penalty, 12)
        item["priority_score"] = item["final_priority_score"]
        item["selection_reason"] = "policy_top_k_candidate" if not reasons else ";".join(reasons)
        final.append(item)
        lvl=item["scope_level"]
        item["deep_research_selected"] = lvl in used and used[lvl] < limits[lvl] and not reasons
        if item["deep_research_selected"]:
            used[lvl]+=1; selected.append(item)
        item["old_priority_rank"] = old_rank.get(item.get("scope_refs",{}).get("value_id") or item.get("scope_refs",{}).get("bucket_id"))
    final=sorted(final,key=lambda x:(-x["final_priority_score"],x["scope_level"],json.dumps(x["scope_refs"],sort_keys=True)))
    for i,item in enumerate(final,1): item["priority_rank"] = i
    return {"queue_version":"m6r_a_v2_1","policy_limits":limits,"priority_policy":{"affected_count_transform":policy.get("affected_count_transform"),"weights":weights},"items":final}


def _m21_queue_item(level: str, rec: dict[str,Any], question: str, weights: dict[str,Any], policy: dict[str,Any], total: int, fl: dict[str,Any]) -> dict[str,Any]:
    count=rec.get("affected_count") if isinstance(rec.get("affected_count"),int) else 0
    evidence=rec.get("evidence_status") or rec.get("evidence_level") or "unavailable"
    affected_ratio = count/total if total and count else 0.0
    affected_component = _transform_affected(count,total,policy) if count else 0.0
    new_info = 1.0 if rec.get("mechanism_id") in {"information_source_missing","multi_hop_signal_opportunity"} or rec.get("axis_id") == AXIS_REACHABILITY else 0.45
    researchability = 0.8 if level in {"bucket","error_mechanism"} else 0.35
    method_avail = 0.55 if level in {"bucket","error_mechanism"} else 0.25
    components={
        "affected_count_component": affected_component,
        "affected_ratio_component": round(affected_ratio,12),
        "error_headroom_component": None,
        "evidence_strength_component": _strength(evidence),
        "fold_stability_component": None,
        "macro_importance_component": None,
        "novelty_information_gap_component": new_info,
        "expected_score_impact_component": None,
        "researchability_component": researchability,
        "method_availability_component": method_avail,
        "experiment_cost_component": -0.15,
        "research_cost_component": -0.10,
        "risk_component": -0.10,
        "overlap_penalty_component": 0.0,
        "closed_branch_penalty_component": 0.0,
    }
    raw=sum(float(weights.get(k,0.0))*v for k,v in components.items() if isinstance(v,(int,float)))
    sig=rec.get("scope_signature") or {"bucket_axes":rec.get("bucket_axes",[]),"class_id":rec.get("class_id"),"mechanism_id":rec.get("mechanism_id"),"scope_id":rec.get("scope_id")}
    return {"queue_item_id":stable_hash({"level":level,"sig":sig,"question":question}),"problem_ids":[f"{level}::{rec.get('scope_id') or rec.get('mechanism_id') or rec.get('bucket_id','global')}"] ,"scope_level":level,"scope_refs":{"scope_id":rec.get("scope_id"),"axis_id":rec.get("axis_id"),"value_id":rec.get("bucket_id"),"mechanism_id":rec.get("mechanism_id")},"scope_signature":sig,"bucket_axes":rec.get("bucket_axes",[]),"class_id":rec.get("class_id"),"mechanism_id":rec.get("mechanism_id"),"scope_parent_ids":rec.get("scope_parent_ids",[]),"scope_child_ids":rec.get("scope_child_ids",[]),"affected_count":count,"coverage_hash":rec.get("coverage_hash"),"primary_question":question,"secondary_questions":[],"evidence_strength":evidence,"estimated_score_impact":"unavailable_without_full_anchor_oof","macro_importance":"unavailable","fold_stability":"unavailable","novel_information_gap":"required","method_availability":method_avail,"experiment_cost":"low_read_only_research","research_cost":"medium","risk":0.1,"priority_components":components,"raw_priority_score":round(raw,12),"penalty_total":0.0,"final_priority_score":round(raw,12),"priority_score":round(raw,12),"priority_rank":0,"recommended_action":"generate_research_brief","required_evidence":["canonical Fold","final v53Q-1 OOF proba","Bucket × Class OOF metrics"],"missing_evidence":["error headroom","expected score impact","macro importance","full anchor OOF","bucket-class metrics"],"status":"open","reason_codes":["policy_ranked","no_auto_execution","headroom_unavailable_not_imputed"],"selection_reason":"policy_ranked"}

ResearchMemoryBuilder.materialize = _m21_materialize
