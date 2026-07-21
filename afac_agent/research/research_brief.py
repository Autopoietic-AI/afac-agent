# -*- coding: utf-8 -*-
"""Deterministic Method Research Brief builder for Top-K queue items."""
from __future__ import annotations

import json, time
from pathlib import Path
from typing import Any

from .event_store import json_dumps, load_json, rel_ref, stable_hash

BRIEF_VERSION = "m6r_a_v2"

class ResearchBriefBuilder:
    def __init__(self, *, project_root: str | Path): self.project_root = Path(project_root).resolve()
    def build(self, *, memory_root: str|Path, queue_item_id: str, out_root: str|Path, dry_run: bool=False, force_rebuild: bool=False) -> dict[str, Any]:
        mem = self._resolve_memory_dir(memory_root); queue_path = mem/"research_queue.json"
        if not queue_path.exists(): return {"status":"waiting_for_input","failure_reason":"missing_required_inputs","missing_inputs":["research_queue"],"artifacts":{}}
        queue=load_json(queue_path); items=queue.get("items",[]); item=next((x for x in items if x.get("queue_item_id")==queue_item_id), None)
        if not item: return {"status":"waiting_for_input","failure_reason":"unknown_queue_item","missing_inputs":["queue_item"],"artifacts":{}}
        if not item.get("deep_research_selected"):
            return {"status":"blocked","failure_reason":"queue_item_not_top_k","queue_item_id":queue_item_id,"artifacts":{}}
        brief=self._brief(mem.name, item); brief_id=brief["brief_id"]
        out=Path(out_root); out=out if out.is_absolute() else self.project_root/out; run=out/brief_id
        if dry_run: return {"status":"dry_run","brief_id":brief_id,"artifacts":{}}
        path=run/"research_brief.json"
        if path.exists() and not force_rebuild:
            return {"status":"duplicate","brief_id":brief_id,"artifacts":{"research_brief":rel_ref(path,self.project_root)}}
        run.mkdir(parents=True, exist_ok=True); path.write_text(json_dumps(brief)+"\n", encoding="utf-8")
        md=run/"RESEARCH_BRIEF.md"; md.write_text(self._markdown(brief), encoding="utf-8")
        manifest={"brief_version":BRIEF_VERSION,"brief_id":brief_id,"created_at_epoch_seconds":time.time(),"memory_id":mem.name,"queue_item_id":queue_item_id,"calls_llm":False,"uses_network":False,"executes_adapter":False,"artifacts":{"research_brief":rel_ref(path,self.project_root),"research_brief_md":rel_ref(md,self.project_root)}}
        mp=run/"brief_manifest.json"; mp.write_text(json_dumps(manifest)+"\n", encoding="utf-8")
        return {"status":"completed","brief_id":brief_id,"artifacts":manifest["artifacts"] | {"brief_manifest":rel_ref(mp,self.project_root)}}
    def _resolve_memory_dir(self, memory_root: str|Path) -> Path:
        p=Path(memory_root); p=p if p.is_absolute() else self.project_root/p
        if (p/"research_queue.json").exists(): return p
        kids=sorted([x for x in p.iterdir() if (x/"research_queue.json").exists()]) if p.exists() else []
        if len(kids)==1: return kids[0]
        raise ValueError(f"cannot resolve memory root: {memory_root}")
    def _brief(self, memory_id: str, item: dict[str, Any]) -> dict[str, Any]:
        scope=item.get("scope_level")
        btype={"global":"global_research_brief","bucket":"bucket_research_brief","bucket_class":"bucket_class_research_brief","error_mechanism":"mechanism_research_brief"}.get(scope,"bucket_research_brief")
        core={"brief_version":BRIEF_VERSION,"brief_type":btype,"memory_id":memory_id,"target_problem_ids":item.get("problem_ids",[]),"scope_level":scope,"scope_refs":item.get("scope_refs",{}),"primary_research_question":item.get("primary_question"),"secondary_research_questions":item.get("secondary_questions",[]),"current_evidence":{"evidence_strength":item.get("evidence_strength"),"priority_components":item.get("priority_components",{})},"evidence_gaps":item.get("missing_evidence",[]),"current_parent":"v53Q-1 frozen champion","current_best_solution":"v53Q-1 Transition-Stable Edge-H2 patch","successful_local_methods":[],"failed_local_methods":[],"closed_routes":[],"required_new_information":item.get("required_evidence",[]),"allowed_information_sources":["papers","public methods","verified project artifacts"],"excluded_information_sources":["test truth","raw node-level labels","unverified internal model knowledge"],"allowed_method_families":["read-only method research","leakage-safe hypothesis design"],"excluded_method_families":["automatic training","automatic prediction","automatic submission","closed-branch reopening"],"search_keywords":[str(item.get("primary_question","")), str(scope), json.dumps(item.get("scope_refs",{}), ensure_ascii=False)],"negative_keywords":["test labels","leaderboard probing","oracle test truth"],"source_requirements":["cite external method sources in later M6R-B", "do not use LLM internal knowledge as verified source"],"maturity_requirements":["must map to Bucket ? Class ? Mechanism before experiment"],"compute_constraints":["no GPU in research brief generation"],"evaluation_constraints":["requires canonical Fold and final OOF for promotion"],"leakage_constraints":["no Test truth metrics"],"minimal_experiment_requirements":["explicit inputs", "read-only audit first", "human approval before mutating artifacts"],"success_conditions":["new information source identified", "local conflict check passes"],"failure_conditions":["same information as failed route", "missing evidence remains"],"stop_conditions":["closed branch would need reopening", "requires test truth"]}
        core["brief_id"]=stable_hash(core); return core
    def _markdown(self, b: dict[str, Any]) -> str:
        return f"# {b['brief_type']}\n\nQuestion: {b['primary_research_question']}\n\nNo LLM/API/Adapter was executed.\n"
