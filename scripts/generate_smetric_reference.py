#!/usr/bin/env python3
"""Generate Rust differential fixtures from ssched's Python SMetric.

Run with SSCHED_ROOT pointing at the ssched checkout.  This is a development
tool only; the Router has no Python runtime dependency.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import sys
from pathlib import Path

root = Path(os.environ.get("SSCHED_ROOT", "../ssched")).resolve()
sys.path.insert(0, str(root / "src"))

from ssched.scheduler.core import (  # noqa: E402
    ClusterSnapshot,
    DecodeContract,
    InstanceView,
    RequestContext,
)
from ssched.scheduler.policy.smetric import SMetric  # noqa: E402
from ssched.state.schema import EngineState  # noqa: E402


def inst(idx: int, **kw) -> InstanceView:
    return InstanceView(engine_id=f"engine_{idx}", idx=idx, **kw)


def request(input_length: int, turn_depth: int) -> RequestContext:
    return RequestContext(
        request_id="fixture", session_id=None, input_length=input_length,
        block_hashes=(), turn_depth=turn_depth,
    )


def case(name: str, config: dict, req: dict, instances: list[dict]) -> dict:
    counter = itertools.count(0)
    py_instances = []
    for idx, raw in enumerate(instances):
        values = dict(raw)
        contracts = values.pop("decode_contracts", [])
        values["decode_contracts"] = tuple(DecodeContract(**c) for c in contracts)
        real_pending = values.pop("real_pending_prefill_tokens", None)
        real_decode = values.pop("real_ongoing_decode_tokens", None)
        real_inflight = values.pop("real_inflight", None)
        if any(value is not None for value in (real_pending, real_decode, real_inflight)):
            values["real_state"] = EngineState(
                engine_id=f"engine_{idx}", ts=0.0,
                num_running=real_inflight or 0, num_waiting=0,
                gpu_blocks_total=1, gpu_blocks_free=1, gpu_kv_used_frac=0.0,
                pending_prefill_tokens=real_pending or 0,
                ongoing_decode_tokens=real_decode or 0,
                num_prefilling=0, max_prefill_remaining=0,
            )
        py_instances.append(inst(idx, **values))
    decision = SMetric(**config).route(
        request(**req),
        ClusterSnapshot(instances=py_instances, next_rr=lambda: next(counter)),
    )
    return {
        "name": name,
        "config": config,
        "request": req,
        "instances": instances,
        "expected_position": decision.instance_idx,
        "expected_reason": decision.reason,
    }


healthy = {"elapsed_s": 0.5, "input_length": 1000, "emitted_tokens": 10}
cases = [
    case("overload_boundary_inclusive", {"overload": 2.0},
         {"input_length": 16000, "turn_depth": 2}, [
             {"cache_hit_tokens": 16000, "pending_prefill_tokens": 32000},
             {"cache_hit_tokens": 0, "pending_prefill_tokens": 8000},
             {"cache_hit_tokens": 0, "pending_prefill_tokens": 8000},
         ]),
    case("overload_spill_load_rr", {"overload": 2.0},
         {"input_length": 16000, "turn_depth": 2}, [
             {"cache_hit_tokens": 16000, "pending_prefill_tokens": 48000},
             {"cache_hit_tokens": 0}, {"cache_hit_tokens": 0},
         ]),
    case("budget_raw_fits", {"gate": "budget", "drain_tps": 1000},
         {"input_length": 16000, "turn_depth": 2}, [
             {"cache_hit_tokens": 15000, "pending_prefill_tokens": 1000},
             {"cache_hit_tokens": 0},
         ]),
    case("budget_raw_spills", {"gate": "budget", "drain_tps": 1000},
         {"input_length": 16000, "turn_depth": 2}, [
             {"cache_hit_tokens": 15000, "pending_prefill_tokens": 1001},
             {"cache_hit_tokens": 0},
         ]),
    case("budget_attention_spills", {"gate": "budget_attention", "drain_tps": 21400},
         {"input_length": 64000, "turn_depth": 2}, [
             {"cache_hit_tokens": 60000, "pending_prefill_tokens": 100000,
              "pending_prefill_attention": 0.0},
             {"cache_hit_tokens": 0},
         ]),
    case("lmetric", {"fallback": "lmetric"},
         {"input_length": 1000, "turn_depth": 1}, [
             {"num_requests": 4},
             {"pending_prefill_tokens": 100, "num_requests": 1},
         ]),
    case("prefill_work_attention", {"fallback": "prefill_work_attention"},
         {"input_length": 4000, "turn_depth": 1}, [
             {"pending_prefill_tokens": 1000,
              "pending_prefill_attention": 50_000_000.0},
             {"pending_prefill_tokens": 3000},
         ]),
    case("lmetric_attention", {"fallback": "lmetric_attention"},
         {"input_length": 4000, "turn_depth": 1}, [
             {"pending_prefill_tokens": 1000,
              "pending_prefill_attention": 50_000_000.0, "num_requests": 1},
             {"pending_prefill_tokens": 3000, "num_requests": 3},
         ]),
    case("dynamo", {"fallback": "dynamo", "overlap_score_credit_decay": 1.0},
         {"input_length": 1600, "turn_depth": 1}, [
             {"cache_hit_tokens": 1200, "pending_prefill_tokens": 3200,
              "ongoing_decode_tokens": 0},
             {"cache_hit_tokens": 0, "pending_prefill_tokens": 0,
              "ongoing_decode_tokens": 160},
         ]),
    case("dynamo_logit_alias", {"fallback": "dynamo_logit"},
         {"input_length": 1600, "turn_depth": 1}, [
             {"cache_hit_tokens": 1200}, {"cache_hit_tokens": 0},
         ]),
    case("contract_safe_filters", {
             "gate": "budget_attention", "attention_l_eq": 13568,
             "drain_tps": 13100, "fallback": "lmetric_attention",
             "contract_safe": True,
         }, {"input_length": 40000, "turn_depth": 1}, [
             {"pending_prefill_tokens": 1000, "num_requests": 1,
              "decode_contracts": [healthy]},
             {"pending_prefill_tokens": 90000, "num_requests": 1},
         ]),
    case("contract_safe_no_candidate", {
             "gate": "budget_attention", "attention_l_eq": 13568,
             "drain_tps": 13100, "fallback": "lmetric_attention",
             "contract_safe": True,
         }, {"input_length": 40000, "turn_depth": 1}, [
             {"pending_prefill_tokens": 1000, "num_requests": 1,
              "decode_contracts": [healthy]},
             {"pending_prefill_tokens": 90000, "num_requests": 1,
              "decode_contracts": [healthy]},
         ]),
    case("store_rescue", {
             "gate": "budget_attention", "fallback": "prefill_work_attention",
             "drain_tps": 21400, "store_rescue": True,
         }, {"input_length": 64000, "turn_depth": 2}, [
             {"cache_hit_tokens": 60000, "store_hit_tokens": 60000,
              "pending_prefill_tokens": 100000},
             {"cache_hit_tokens": 0, "store_hit_tokens": 60000},
         ]),
    case("measured_attention_rate", {
             "gate": "budget_attention", "drain_tps": 1000,
             "drain_source": "measured",
         }, {"input_length": 16000, "turn_depth": 2}, [
             {"cache_hit_tokens": 15000, "pending_prefill_tokens": 3000,
              "est_prefill_work_tps": 10000},
             {"cache_hit_tokens": 0},
         ]),
    case("service_time_cold", {"service_time_routing": True},
         {"input_length": 8000, "turn_depth": 1}, [
             {"cache_hit_tokens": 0}, {"cache_hit_tokens": 4000},
         ]),
    case("hit_ratio_is_strict", {"hit_ratio": 0.5},
         {"input_length": 1000, "turn_depth": 2}, [
             {"cache_hit_tokens": 500, "pending_prefill_tokens": 100},
             {"cache_hit_tokens": 0},
         ]),
    case("prefill_scale_does_not_scale_decode", {"prefill_load_scale": 4.0},
         {"input_length": 100, "turn_depth": 1}, [
             {"pending_prefill_tokens": 16},
             {"ongoing_decode_tokens": 48},
         ]),
    case("decode_active_request_weight", {"decode_active_request_weight": 1.0},
         {"input_length": 100, "turn_depth": 1}, [
             {"num_requests": 3},
             {"pending_prefill_tokens": 16},
         ]),
    case("store_pricing_changes_own_work", {
             "fallback": "prefill_work_attention", "store_pricing": True,
             "attention_l_eq": 1_000_000_000,
         }, {"input_length": 1000, "turn_depth": 1}, [
             {"store_hit_tokens": 800},
             {"cache_hit_tokens": 500, "store_hit_tokens": 800,
              "pending_prefill_tokens": 50},
         ]),
    case("queue_store_pricing", {
             "gate": "budget_attention", "queue_store_pricing": True,
             "drain_tps": 1000, "store_load_tps": 100_000,
         }, {"input_length": 1000, "turn_depth": 2}, [
             {"cache_hit_tokens": 1000, "pending_prefill_tokens": 1000,
              "pending_store_tokens": 1000},
             {},
         ]),
    case("engine_feed_scales_shadow_attention", {
             "gate": "budget_attention", "drain_tps": 1000,
             "attention_l_eq": 1000,
         }, {"input_length": 1000, "turn_depth": 2}, [
             {"cache_hit_tokens": 1000, "pending_prefill_tokens": 100,
              "pending_prefill_attention": 100_000,
              "real_pending_prefill_tokens": 1000},
             {},
         ]),
    case("engine_feed_blends_decode_and_requests", {
             "prefill_load_scale": 3.0, "decode_active_request_weight": 2.0,
         }, {"input_length": 100, "turn_depth": 1}, [
             {"pending_prefill_tokens": 16, "ongoing_decode_tokens": 16,
              "num_requests": 1, "real_ongoing_decode_tokens": 64,
              "real_inflight": 3},
             {"pending_prefill_tokens": 64},
         ]),
    case("dynamo_decay_ignores_decode", {
             "fallback": "dynamo", "overlap_score_credit_decay": 2.0,
         }, {"input_length": 1600, "turn_depth": 1}, [
             {"cache_hit_tokens": 1200, "pending_prefill_tokens": 1600,
              "ongoing_decode_tokens": 16000},
             {"pending_prefill_tokens": 0, "ongoing_decode_tokens": 0},
         ]),
    case("dynamo_track_prefill_disabled", {
             "fallback": "dynamo", "track_prefill_tokens": False,
         }, {"input_length": 1600, "turn_depth": 1}, [
             {"cache_hit_tokens": 1200, "pending_prefill_tokens": 16000},
             {"cache_hit_tokens": 0},
         ]),
    case("dynamo_host_credit", {
             "fallback": "dynamo", "host_cache_hit_weight": 1.0,
         }, {"input_length": 1600, "turn_depth": 1}, [
             {"store_hit_tokens": 1200},
             {},
         ]),
    case("dynamo_rr_precedes_contract_suffix", {
             "fallback": "dynamo", "contract_safe": True,
         }, {"input_length": 100, "turn_depth": 1}, [
             {"decode_contracts": [healthy]},
             {}, {},
         ]),
    case("home_quiet_uses_fresh_engine_only", {
             "gate": "budget", "drain_tps": 1,
             "home_quiet_stick": 1,
         }, {"input_length": 100, "turn_depth": 2}, [
             {"cache_hit_tokens": 100, "pending_prefill_tokens": 1000,
              "real_inflight": 1},
             {},
         ]),
    case("budget_gamma_zero_inclusive", {
             "gate": "budget", "budget_gamma": 0.0,
         }, {"input_length": 100, "turn_depth": 2}, [
             {"cache_hit_tokens": 100}, {},
         ]),
    case("measured_token_rate", {
             "gate": "budget", "drain_tps": 1000,
             "drain_source": "measured",
         }, {"input_length": 1000, "turn_depth": 2}, [
             {"cache_hit_tokens": 1000, "pending_prefill_tokens": 3000,
              "est_prefill_tps": 10000},
             {},
         ]),
    case("contract_already_blown_is_exempt", {
             "contract_safe": True,
         }, {"input_length": 1000, "turn_depth": 1}, [
             {"decode_contracts": [{"elapsed_s": 10.0,
                                     "input_length": 100,
                                     "emitted_tokens": 0}]},
             {"pending_prefill_tokens": 100},
         ]),
    case("store_rescue_no_feasible_store_target", {
             "gate": "budget_attention", "fallback": "prefill_work_attention",
             "drain_tps": 1000, "store_rescue": True,
         }, {"input_length": 64_000, "turn_depth": 2}, [
             {"cache_hit_tokens": 60_000, "pending_prefill_tokens": 100_000},
             {},
         ]),
]

source = root / "src/ssched/scheduler/policy/smetric.py"
source_paths = [
    "src/ssched/scheduler/policy/smetric.py",
    "src/ssched/scheduler/policy/external.py",
    "src/ssched/scheduler/policy/service_cost.py",
    "src/ssched/scheduler/policy/baselines.py",
    "src/ssched/scheduler/core.py",
    "src/ssched/state/schema.py",
]
payload = {
    "source": str(source.relative_to(root)),
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "sources_sha256": {
        path: hashlib.sha256((root / path).read_bytes()).hexdigest()
        for path in source_paths
    },
    "cases": cases,
}
destination = Path(__file__).resolve().parents[1] / "tests/fixtures/smetric_reference.json"
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(destination)
