# Per-episode experiment record for the TimedHome evaluation. Everything needed to re-score an episode
# against any temporal formula offline (run-length encoded per-proposition truth over every sim step),
# plus the compact planner trace. Written next to the standard PARTNR outputs.
import json
import os
from typing import Any, Dict, List

import numpy as np

SCHEMA_VERSION = 1


def _rle(bools: List[bool]) -> List[List[Any]]:
    out: List[List[Any]] = []
    for t, b in enumerate(bools):
        if not out or out[-1][1] != bool(b):
            out.append([t, bool(b)])
    return out


def _jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return str(x)


def build_record(env_interface, info: Dict[str, Any], planner_infos: List[Dict[str, Any]],
                 runtime: float, instruction: str, run_name: str) -> Dict[str, Any]:
    ep = env_interface.env.env.env._env.current_episode
    log = info.get("task_evaluation_log") or {}
    seq = log.get("state_sequence") or []
    props = log.get("propositions") or []
    sat_at = log.get("proposition_satisfied_at") or []
    prop_records = []
    for i, p in enumerate(props):
        truth = [bool(step[i].is_satisfied) for step in seq] if seq else []
        prop_records.append({"idx": i, "function_name": p.function_name,
                             "args": {k: v for k, v in p.args.items()},
                             "satisfied_at": int(sat_at[i]) if i < len(sat_at) else -1,
                             "rle": _rle(truth)})
    iters = []
    prev_actions, prev_thought = None, None
    for i, pi in enumerate(planner_infos):
        # one loop iteration happens per sim step; keep only iterations where something happened
        has_response = any(pi.get("responses", {}).values()) if isinstance(pi.get("responses"), dict) else bool(pi.get("responses"))
        changed = pi.get("high_level_actions") != prev_actions or pi.get("thought") != prev_thought
        prev_actions, prev_thought = pi.get("high_level_actions"), pi.get("thought")
        done = pi.get("is_done")
        done = any(done.values()) if isinstance(done, dict) else bool(done)
        if not (changed or has_response or done or i == len(planner_infos) - 1):
            continue
        iters.append({"i": i, "sim_step_count": pi.get("sim_step_count"),
                      "high_level_actions": pi.get("high_level_actions"),
                      "responses": pi.get("responses"), "thought": pi.get("thought"),
                      "replan_required": pi.get("replan_required"), "replanned": pi.get("replanned"),
                      "is_done": pi.get("is_done"),
                      "agent_positions": pi.get("agent_positions")})
    perception = env_interface.perception
    try:
        sim_conf = env_interface.conf.habitat.simulator
        freq = {"ctrl_freq": sim_conf.ctrl_freq, "ac_freq_ratio": sim_conf.ac_freq_ratio}
    except Exception:
        freq = {}
    return {
        "schema_version": SCHEMA_VERSION,
        "run_name": run_name,
        "episode_id": ep.episode_id,
        "scene_id": ep.scene_id,
        "instruction": instruction,
        "episode_info": ep.info,
        "metrics": {
            "task_percent_complete": info.get("task_percent_complete"),
            "task_state_success": info.get("task_state_success"),
            "sim_step_count": info.get("num_steps"),
            "total_step_count": info.get("total_step_count"),
            "runtime_s": runtime,
            **freq,
        },
        "task_explanation": info.get("task_explanation"),
        "n_state_steps": len(seq),
        "propositions": prop_records,
        "constraint_satisfaction": log.get("constraint_satisfaction"),
        "sim_handle_to_name": dict(getattr(perception, "sim_handle_to_name", {})),
        "iterations": iters,
    }


def write_record(record: Dict[str, Any], output_dir: str) -> str:
    d = os.path.join(output_dir, "timedhome")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"record-{record['run_name']}.json")
    with open(path, "w") as f:
        json.dump(record, f, default=_jsonable)
    return path
