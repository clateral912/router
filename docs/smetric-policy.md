# Native SMetric policy

This port is based on the live ssched working tree, including its uncommitted
changes. The frozen inputs used for the differential fixtures are:

- ssched Git `8a83db6a797d1d7547e33fdf352f63b91fc1b079`
- `src/ssched/scheduler/policy/smetric.py` SHA-256
  `7871badc7d4be2bb83565400f6227b51f40b251e939695cc628c782982d01659`
- `src/ssched/scheduler/policy/external.py` SHA-256
  `e871dce0f4751c3a530fcc28c4c80cfd375ba17be037c3fda52aeb6952a5ba89`
- `src/ssched/scheduler/policy/service_cost.py` SHA-256
  `2b620a4857a901ed072308240c90fdc0ba509dc928e44d94262c9592a69e9a50`
- `src/ssched/scheduler/policy/baselines.py` (registered `lmetric`) SHA-256
  `31c347d40eacd10dc2c3b8900dde6502b2d8260266422d209d865d75bec9c819`
- `src/ssched/scheduler/core.py` SHA-256
  `662264cb1f60f24a67fc6b78638031881b67dcd3935fbbb3e08c2867f354da89`
- `src/ssched/state/schema.py` SHA-256
  `8cf8a3b0b2d086d84533d23e0a8960c748c17903666ab2593ebe4edcd382ef47`
- Router main `b87c7776968ee1daf0e5eec03fec5660ec201682`

The community default uses the overload gate (`2.0`), hit ratio `0.5`, and
the Dynamo-style fallback without a service dependency. This differs from the
Python Figure 13 prototype's load fallback; the reference fixtures explicitly
retain that prototype default. Research presets remain opt-in:

| Preset | Required overrides from the prototype |
|---|---|
| 30B-PD | `gate=budget_attention`, `drain_tps=21400`, `fallback=lmetric_attention`, `contract_safe=true`, `slo_input_tokens_per_s=8000`, `slo_tpot_s=.020` |
| 235B-PD | 30B-PD plus `drain_tps=13100`, `attention_l_eq=13568`, `slo_tpot_s=.033` |
| 30B-PO | `gate=budget_attention`, `drain_tps=21400`, `fallback=prefill_work_attention`, `store_rescue=true`, `store_load_tps=162000` |

## Option and behavior matrix

| Option | Python-equivalent behavior |
|---|---|
| `gate=overload` | Stick when `load[home] <= overload_factor * mean(load)`; infinity always passes, including an idle cluster. |
| `gate=budget` | Stick when `(pending_prefill + own_uncached) / drain <= gamma * (base + input/rate)`. Decode backlog is excluded. |
| `gate=budget_attention` | The budget gate in cold-token-equivalent attention work. Queue store pricing splits store onload from model work when enabled. |
| `overload_factor` | Relative-load threshold for the overload gate; non-negative and infinity is the guard-removed endpoint. |
| `budget_gamma` | Multiplier on the absolute TTFT budget; zero and infinity retain their Python boundary semantics. |
| `budget_base_s`, `budget_input_tokens_per_s` | Define the gate budget as `base + input/rate`. |
| `drain_tps` | Positive cold-start/configured per-worker rate in the active gate's units. |
| `fallback=load` | Request-independent `scale*pending_prefill/block + decode/block + weight*active_requests`. Only prefill is scaled. |
| `fallback=lmetric` | `(pending_prefill + own_uncached) * active_requests`. |
| `fallback=prefill_work_attention` | Pending attention work plus this request's attention work. |
| `fallback=lmetric_attention` | Attention fallback multiplied by active requests. |
| `fallback=dynamo` / `dynamo_logit` | Dynamo raw prefill, per-tier credit, active-prefill-only decay, decode cost, and active-request weight. The two names are aliases. |
| `hit_ratio` | Strict GPU-prefix test: `gpu_hit > hit_ratio * input_length`. Store hits never satisfy the home/eviction guard. |
| `drain_source` | `config` always uses `drain_tps`; `measured` uses per-worker token or attention-work samples after warmup and falls back before then. |
| `store_pricing` | Lets ordinary own-work ranking use a global store hit. It never changes home selection. |
| `queue_store_pricing` | Prices the pending store-served share at `store_load_tps`. |
| `store_rescue` | For continuations only, and only if every cold candidate misses the request budget, select the fastest store-priced candidate that fits. Invalid with Dynamo fallback. |
| `contract_safe` | Removes candidates where this prefill exceeds any still-viable live decode's remaining contract; uses the whole fleet if the pool is empty. |
| `slo_base_s`, `slo_input_tokens_per_s`, `slo_tpot_s` | Define each live decode's remaining contract as `base + input/rate + emitted*tpot - elapsed`. Already-blown contracts are exempt. |
| `home_quiet_stick` | Overrides a failed queue gate only when a fresh engine observation reports at most this many running plus waiting requests. |
| `session_home_depth` | Pins a session once to the first fallback result using the requested prompt-block hash depth. The gate still runs every turn. |
| `service_time_routing` | Runs the separate serial service-cost model before the gate/fallback path, including store-loss Beta prior, age bins, queue head aging, and completion learning. |
| `service_fixed_s` | Fixed latency term in both the store-hit and loss-adjusted service estimates. |
| `attention_l_eq` | Divisor converting `n*(L-n/2)` attention moment into cold-token equivalents. |
| `block_size` | Dynamo score unit only; default 16 tokens and unrelated to the 512-token session hash block. |
| `prefill_load_scale` | Multiplies pending prefill in load/Dynamo scores; decode remains unscaled. |
| `decode_active_request_weight` | Adds the configured cost per effective active request. |
| `overlap_score_credit` | Dynamo device-cache credit per block. |
| `overlap_score_credit_decay` | Reduces device credit by active-prefill excess over the least-loaded eligible candidate; decode is excluded. |
| `host_cache_hit_weight` | Dynamo host/store-tier credit for the hit beyond the local GPU hit. |
| `track_prefill_tokens` | Disables Dynamo's raw prefill and overlap-credit terms when false, retaining decode/request terms. |
| `eviction_interval_secs`, `max_tree_size` | Stock Router adapter Tree maintenance; these do not alter the pure decision core. |
| `drain_window_secs`, `drain_min_samples` | Stock Router adapter's sliding p90 online-rate window and warmup count; Python's defaults are 180 seconds and five samples. |

Exact score ties consume one policy-local round-robin step. Strict orderings do
not advance it. A first turn is `X-Session-Turn: 1`; a missing or invalid header
uses Python's `RequestContext.turn_depth=1` default.

## Router observation adapter

The policy is a native Rust policy and has no Redis, Python process, controller,
or KVEvents dependency. `SMetricRequest` and `SMetricInstance` form the complete
native observation interface used by the decision core. The stock main adapter
fills the fields it can observe from the Tree and the HTTP request lifecycle.
`update_external_observation` can supply the remaining fields from a future
native metrics source without changing the algorithm.

Main Tree is character-based, while ssched uses exact token counts and
512-token block hashes. The adapter queries Tree separately for every worker,
but its units are characters. `session_home_depth` likewise uses deterministic
512-character blocks in the stock adapter. Its lifecycle shadow therefore also
prices request lengths in characters because main Router has no tokenizer in
the routing path. The complete decision core accepts exact token counts and
exact block hashes, and the differential fixtures use those exact units.

The current main Router does not expose these ssched observations:

- fresh engine `pending_prefill`, `ongoing_decode`, running/waiting counts, or
  the freshness boundary used by `eff_*`; without a native publisher, the
  gate and fallback see only requests admitted by this router and
  `home_quiet_stick` cannot fire;
- global store prefix hit/age and pending store-served queue share;
  `store_pricing`, queue store pricing, store rescue, and service-cost loss
  learning therefore remain inactive without those optional inputs;
- the engine backlog's context profile when no router reservation exists;
  attention work then degrades to the observed token count, matching Python's
  documented empty-shadow behavior;
- emitted-token counts for every live decode and actual recovered cached tokens;
  local decode contracts carry `emitted_tokens=0`, and service-cost completion
  cannot learn store-loss outcomes unless a native integration calls
  `note_cache_result` with the recovered count;
- normalized realized response tokens/text for Tree maintenance; successful
  completion inserts the routed request text, but cannot append the generated
  output as ssched's exact token-block shadow does. This can understate the
  next turn's local hit across the previous assistant response.

Router-native lifecycle tracking in regular and P/D routing supplies shadow
pending prefill, attention moment, active requests, decode context, p90
per-request TTFT rate samples, elapsed decode contracts, and exact cleanup on success,
error, retry, stream end, and cancellation. Dynamic and discovery-backed P/D
workers initialize the same policy Tree before selection. Missing optional
observations remain zero or `None`. Therefore the decision core is equivalent
for identical complete inputs, while stock-main runtime behavior can differ
from ssched when the missing fields are material.

## Differential validation

`scripts/generate_smetric_reference.py` imports the frozen Python implementation
only during development and writes `tests/fixtures/smetric_reference.json`.
Rust tests instantiate a fresh native policy for each of 31 fixtures and compare
both the selected position and reason. The fixture set covers all gates and
fallbacks, inclusive boundaries, no-safe-candidate fallback, exact ties,
measured-rate cold start/use, store rescue, contract filtering, fresh-engine
blending and attention rescaling, Dynamo credit/decay switches, and the separate
service-time path. Additional Rust lifecycle tests cover reservation phase
changes, online calibration, cleanup, completion-only Tree insertion, stable
homes, service-loss learning, and the fresh-engine-only quiet-home switch.
