//! Native SMetric routing policy.
//!
//! The decision core mirrors `ssched.scheduler.policy.smetric.SMetric` and is
//! deliberately independent from Router's observation adapter.  In particular,
//! missing engine/store observations remain missing; the adapter never invents
//! them.  `SMetricRequest` and `SMetricInstance` are the optional native state
//! interface for deployments that can publish richer observations.

use super::{get_healthy_worker_indices, normalize_model_key, LoadBalancingPolicy, RequestHeaders};
use crate::config::{SMetricDrainSource, SMetricFallback, SMetricGate, SMetricPolicyConfig};
use crate::core::Worker;
use crate::metrics::RouterMetrics;
use crate::tree::Tree;
use dashmap::DashMap;
use std::collections::{HashMap, VecDeque};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::ThreadId;
use std::time::{Duration, Instant};
use tracing::debug;

pub type SMetricConfig = SMetricPolicyConfig;

const HOME_CAP: usize = 200_000;
const SERVICE_LOSS_WINDOW: usize = 256;

#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DecodeContract {
    pub elapsed_s: f64,
    pub input_length: usize,
    pub emitted_tokens: usize,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct SMetricRequest {
    pub request_key: u64,
    pub input_length: usize,
    pub turn_depth: usize,
    pub block_hashes: Vec<u64>,
    /// Monotonic seconds used only by the service-time arm.
    pub now_s: f64,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct SMetricInstance {
    pub idx: usize,
    pub cache_hit_tokens: usize,
    pub store_hit_tokens: usize,
    pub store_prefix_age_s: f64,
    /// Effective values: callers with an engine feed should supply max(feed,
    /// router shadow), exactly as Python's `eff_*` accessors do.
    pub pending_prefill_tokens: f64,
    pub pending_prefill_attention: f64,
    pub pending_prefill_compute_attention: f64,
    pub pending_store_tokens: f64,
    pub ongoing_decode_tokens: f64,
    pub num_requests: f64,
    /// Fresh engine fields. `None` means there is no fresh engine feed; the
    /// `eff_*` rules then use only the router shadow fields above.
    pub real_pending_prefill_tokens: Option<f64>,
    pub real_ongoing_decode_tokens: Option<f64>,
    pub real_num_requests: Option<f64>,
    /// Fresh engine running+waiting count used by `home_quiet_stick`.
    pub real_inflight: Option<usize>,
    pub est_prefill_tps: Option<f64>,
    pub est_prefill_work_tps: Option<f64>,
    pub decode_contracts: Vec<DecodeContract>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SMetricDecision {
    pub position: usize,
    pub instance_idx: usize,
    pub reason: String,
}

#[derive(Debug, Clone, Default)]
pub struct SMetricExternalObservation {
    pub store_hit_tokens: Option<usize>,
    pub store_prefix_age_s: Option<f64>,
    pub pending_prefill_tokens: Option<f64>,
    pub pending_prefill_attention: Option<f64>,
    pub pending_prefill_compute_attention: Option<f64>,
    pub pending_store_tokens: Option<f64>,
    pub ongoing_decode_tokens: Option<f64>,
    pub num_requests: Option<f64>,
    pub real_inflight: Option<usize>,
    pub est_prefill_tps: Option<f64>,
    pub est_prefill_work_tps: Option<f64>,
    pub decode_contracts: Option<Vec<DecodeContract>>,
}

#[derive(Debug)]
struct Reservation {
    request_text: String,
    input_tokens: f64,
    prefill_tokens: f64,
    prefill_attention: f64,
    prefill_compute_attention: f64,
    store_tokens: f64,
    work_units: f64,
    started_at: Instant,
    response_started: bool,
}

#[derive(Debug, Default)]
struct WorkerState {
    pending_prefill_tokens: f64,
    pending_prefill_attention: f64,
    pending_prefill_compute_attention: f64,
    pending_store_tokens: f64,
    ongoing_decode_tokens: f64,
    active_requests: usize,
    reservations: HashMap<u64, Reservation>,
    token_rate_samples: VecDeque<(Instant, f64)>,
    work_rate_samples: VecDeque<(Instant, f64)>,
}

#[derive(Debug, Clone)]
struct ServicePending {
    idx: usize,
    input_length: usize,
    gpu_hit: usize,
    store_hit: usize,
    age_bin: usize,
    loss_probability: f64,
    hit_seconds: f64,
    miss_seconds: f64,
    started_at: f64,
}

#[derive(Debug, Default)]
struct ServiceCostState {
    pending: HashMap<u64, ServicePending>,
    queues: HashMap<usize, VecDeque<u64>>,
    losses: [VecDeque<f64>; 5],
    loss_severity: [VecDeque<f64>; 5],
}

#[derive(Debug)]
pub struct SMetricPolicy {
    config: SMetricConfig,
    trees: DashMap<String, Arc<Tree>>,
    worker_models: DashMap<String, String>,
    state: Mutex<HashMap<String, WorkerState>>,
    external: DashMap<String, SMetricExternalObservation>,
    rr: AtomicUsize,
    next_reservation: AtomicU64,
    started_at: Instant,
    last_eviction: Mutex<Instant>,
    homes: Mutex<(HashMap<u64, usize>, VecDeque<u64>)>,
    service: Mutex<ServiceCostState>,
    /// Route reserves service work before lifecycle start; preserve the key
    /// selected for each worker until `on_request_start` claims it.
    service_selections: Mutex<HashMap<(ThreadId, String), VecDeque<u64>>>,
}

impl SMetricPolicy {
    pub fn new() -> Self {
        Self::with_config(SMetricConfig::default())
    }

    pub fn with_config(config: SMetricConfig) -> Self {
        Self {
            config,
            trees: DashMap::new(),
            worker_models: DashMap::new(),
            state: Mutex::new(HashMap::new()),
            external: DashMap::new(),
            rr: AtomicUsize::new(0),
            next_reservation: AtomicU64::new(1),
            started_at: Instant::now(),
            last_eviction: Mutex::new(Instant::now()),
            homes: Mutex::new((HashMap::new(), VecDeque::new())),
            service: Mutex::new(ServiceCostState::default()),
            service_selections: Mutex::new(HashMap::new()),
        }
    }

    pub fn config(&self) -> &SMetricConfig {
        &self.config
    }

    /// Optional native input for observations not exposed by main Router.
    pub fn update_external_observation(
        &self,
        worker_url: impl Into<String>,
        observation: SMetricExternalObservation,
    ) {
        self.external.insert(worker_url.into(), observation);
    }

    #[inline]
    fn attention_moment(uncached: f64, input: f64) -> f64 {
        uncached * (input - uncached / 2.0).max(0.0)
    }

    #[inline]
    fn work_units(&self, uncached: f64, input: f64) -> f64 {
        uncached + Self::attention_moment(uncached, input) / self.config.attention_l_eq
    }

    fn next_rr(&self, count: usize) -> usize {
        self.rr.fetch_add(1, Ordering::Relaxed) % count
    }

    fn choose_min(&self, candidates: &[(usize, f64)]) -> Option<usize> {
        self.choose_min_detail(candidates)
            .map(|(position, _)| position)
    }

    fn choose_min_detail(&self, candidates: &[(usize, f64)]) -> Option<(usize, bool)> {
        let best = candidates
            .iter()
            .map(|(_, score)| *score)
            .reduce(f64::min)?;
        let tied: Vec<usize> = candidates
            .iter()
            .filter_map(|(pos, score)| (*score == best).then_some(*pos))
            .collect();
        let used_rr = tied.len() > 1;
        Some((
            if used_rr {
                tied[self.next_rr(tied.len())]
            } else {
                tied[0]
            },
            used_rr,
        ))
    }

    fn load(&self, inst: &SMetricInstance) -> f64 {
        let block = self.config.block_size as f64;
        // Python scales only pending prefill. Decode is deliberately outside
        // `prefill_load_scale`.
        self.config.prefill_load_scale * Self::eff_pending_prefill(inst) / block
            + Self::eff_ongoing_decode(inst) / block
            + self.config.decode_active_request_weight * Self::eff_num_requests(inst)
    }

    fn eff_pending_prefill(inst: &SMetricInstance) -> f64 {
        inst.real_pending_prefill_tokens
            .map_or(inst.pending_prefill_tokens.max(0.0), |real| {
                real.max(0.0).max(inst.pending_prefill_tokens.max(0.0))
            })
    }

    fn eff_ongoing_decode(inst: &SMetricInstance) -> f64 {
        inst.real_ongoing_decode_tokens
            .map_or(inst.ongoing_decode_tokens.max(0.0), |real| {
                real.max(0.0).max(inst.ongoing_decode_tokens.max(0.0))
            })
    }

    fn eff_num_requests(inst: &SMetricInstance) -> f64 {
        inst.real_num_requests
            .map_or(inst.num_requests.max(0.0), |real| {
                real.max(0.0).max(inst.num_requests.max(0.0))
            })
    }

    fn pending_work(&self, inst: &SMetricInstance) -> f64 {
        let tokens = Self::eff_pending_prefill(inst);
        let shadow_tokens = inst.pending_prefill_tokens.max(0.0);
        let mut attention = inst.pending_prefill_attention.max(0.0);
        if attention > 0.0 && shadow_tokens > 0.0 && tokens > shadow_tokens {
            attention *= tokens / shadow_tokens;
        }
        tokens + attention / self.config.attention_l_eq.max(1.0)
    }

    fn pending_work_split(&self, inst: &SMetricInstance) -> (f64, f64) {
        let tokens = Self::eff_pending_prefill(inst);
        let shadow_tokens = inst.pending_prefill_tokens.max(0.0);
        let store = inst.pending_store_tokens.max(0.0).min(tokens);
        let mut attention = inst.pending_prefill_compute_attention.max(0.0);
        if attention > 0.0 && shadow_tokens > 0.0 && tokens > shadow_tokens {
            attention *= tokens / shadow_tokens;
        }
        (
            tokens - store + attention / self.config.attention_l_eq.max(1.0),
            store,
        )
    }

    fn hit_for_price(&self, req: &SMetricRequest, inst: &SMetricInstance) -> usize {
        let gpu = inst.cache_hit_tokens.min(req.input_length);
        if self.config.store_pricing {
            gpu.max(inst.store_hit_tokens.min(req.input_length))
        } else {
            gpu
        }
    }

    fn own_work(&self, req: &SMetricRequest, inst: &SMetricInstance) -> f64 {
        let uncached = req
            .input_length
            .saturating_sub(self.hit_for_price(req, inst)) as f64;
        self.work_units(uncached, req.input_length as f64)
    }

    fn measured_drain(&self, inst: &SMetricInstance) -> Option<f64> {
        if self.config.drain_source != SMetricDrainSource::Measured {
            return None;
        }
        let value = match self.config.gate {
            SMetricGate::BudgetAttention => inst.est_prefill_work_tps,
            _ => inst.est_prefill_tps,
        };
        value.filter(|v| *v > 0.0)
    }

    fn drain(&self, inst: &SMetricInstance) -> f64 {
        self.measured_drain(inst).unwrap_or(self.config.drain_tps)
    }

    fn budget_s(&self, req: &SMetricRequest) -> f64 {
        self.config.budget_gamma
            * (self.config.budget_base_s
                + req.input_length as f64 / self.config.budget_input_tokens_per_s)
    }

    fn queue_fits_budget(&self, req: &SMetricRequest, inst: &SMetricInstance) -> bool {
        let drain = self.drain(inst);
        let predicted = match self.config.gate {
            SMetricGate::BudgetAttention => {
                if self.config.queue_store_pricing {
                    let (pending, store) = self.pending_work_split(inst);
                    (pending + self.own_work(req, inst)) / drain
                        + store / self.config.store_load_tps
                } else {
                    (self.pending_work(inst) + self.own_work(req, inst)) / drain
                }
            }
            SMetricGate::Budget => {
                let uncached = req.input_length.saturating_sub(inst.cache_hit_tokens) as f64;
                (Self::eff_pending_prefill(inst) + uncached) / drain
            }
            SMetricGate::Overload => unreachable!(),
        };
        predicted <= self.budget_s(req)
    }

    fn home_quiet(&self, inst: &SMetricInstance) -> bool {
        self.config
            .home_quiet_stick
            .zip(inst.real_inflight)
            .is_some_and(|(limit, actual)| actual <= limit)
    }

    fn contract_balance_s(&self, contract: &DecodeContract) -> f64 {
        self.config.slo_base_s
            + contract.input_length as f64 / self.config.slo_input_tokens_per_s
            + contract.emitted_tokens as f64 * self.config.slo_tpot_s
            - contract.elapsed_s
    }

    fn own_seconds(&self, req: &SMetricRequest, inst: &SMetricInstance) -> f64 {
        let own = if self.config.gate == SMetricGate::BudgetAttention {
            self.own_work(req, inst)
        } else {
            req.input_length.saturating_sub(inst.cache_hit_tokens) as f64
        };
        own / self.drain(inst)
    }

    fn contract_safe(&self, req: &SMetricRequest, inst: &SMetricInstance) -> bool {
        let own = self.own_seconds(req, inst);
        inst.decode_contracts.iter().all(|contract| {
            let balance = self.contract_balance_s(contract);
            balance < 0.0 || own <= balance
        })
    }

    fn predicted_s(&self, req: &SMetricRequest, inst: &SMetricInstance, store: bool) -> f64 {
        let drain = self.drain(inst);
        let (pending, pending_store) = if self.config.queue_store_pricing {
            self.pending_work_split(inst)
        } else {
            (self.pending_work(inst), 0.0)
        };
        if !store {
            return (pending + self.own_work(req, inst)) / drain
                + pending_store / self.config.store_load_tps;
        }
        let gpu = self.hit_for_price(req, inst);
        let store_hit = inst.store_hit_tokens.min(req.input_length);
        let onload = store_hit.saturating_sub(gpu) as f64;
        let uncached = req.input_length.saturating_sub(gpu.max(store_hit)) as f64;
        (pending + self.work_units(uncached, req.input_length as f64)) / drain
            + (onload + pending_store) / self.config.store_load_tps
    }

    fn store_rescue_pick(
        &self,
        req: &SMetricRequest,
        insts: &[SMetricInstance],
        candidates: &[usize],
    ) -> Option<usize> {
        let budget = self.budget_s(req);
        let cold_best = candidates
            .iter()
            .map(|&pos| self.predicted_s(req, &insts[pos], false))
            .reduce(f64::min)?;
        if cold_best <= budget {
            return None;
        }
        candidates
            .iter()
            .filter_map(|&pos| {
                let predicted = self.predicted_s(req, &insts[pos], true);
                (predicted <= budget).then_some((pos, predicted))
            })
            .min_by(|(pa, sa), (pb, sb)| sa.total_cmp(sb).then(pa.cmp(pb)))
            .map(|(pos, _)| pos)
    }

    fn fallback_scores(
        &self,
        req: &SMetricRequest,
        insts: &[SMetricInstance],
        load: &[f64],
        candidates: &[usize],
    ) -> Vec<(usize, f64)> {
        match self.config.fallback {
            SMetricFallback::Load => candidates.iter().map(|&p| (p, load[p])).collect(),
            SMetricFallback::Lmetric => candidates
                .iter()
                .map(|&p| {
                    let i = &insts[p];
                    let uncached = req.input_length.saturating_sub(i.cache_hit_tokens) as f64;
                    (
                        p,
                        (Self::eff_pending_prefill(i) + uncached) * Self::eff_num_requests(i),
                    )
                })
                .collect(),
            SMetricFallback::PrefillWorkAttention => candidates
                .iter()
                .map(|&p| {
                    (
                        p,
                        self.pending_work(&insts[p]) + self.own_work(req, &insts[p]),
                    )
                })
                .collect(),
            SMetricFallback::LmetricAttention => candidates
                .iter()
                .map(|&p| {
                    (
                        p,
                        (self.pending_work(&insts[p]) + self.own_work(req, &insts[p]))
                            * Self::eff_num_requests(&insts[p]),
                    )
                })
                .collect(),
            SMetricFallback::Dynamo | SMetricFallback::DynamoLogit => {
                self.dynamo_scores(req, insts, candidates)
            }
        }
    }

    fn dynamo_scores(
        &self,
        req: &SMetricRequest,
        insts: &[SMetricInstance],
        candidates: &[usize],
    ) -> Vec<(usize, f64)> {
        let block = self.config.block_size as f64;
        let request_blocks = (req.input_length as f64 / block).max(1.0);
        let min_active = candidates
            .iter()
            .map(|&p| Self::eff_pending_prefill(&insts[p]))
            .reduce(f64::min)
            .unwrap_or(0.0);
        candidates
            .iter()
            .map(|&p| {
                let inst = &insts[p];
                let device = inst.cache_hit_tokens.min(req.input_length) as f64;
                let store_total = inst.store_hit_tokens.min(req.input_length) as f64;
                let host = (store_total - device).max(0.0);
                let active_prefill = Self::eff_pending_prefill(inst);
                let cached = device + host;
                let raw_tokens = if self.config.track_prefill_tokens {
                    active_prefill
                        + (req.input_length as f64 - cached.min(req.input_length as f64))
                        + cached
                } else {
                    0.0
                };
                // Decay is based only on active prefill excess, never decode.
                let decay = if self.config.track_prefill_tokens
                    && self.config.overlap_score_credit_decay > 0.0
                {
                    let excess_blocks = (active_prefill - min_active).max(0.0) / block;
                    1.0 / (1.0
                        + self.config.overlap_score_credit_decay * excess_blocks / request_blocks)
                } else {
                    1.0
                };
                let credit = self.config.overlap_score_credit * decay * device / block
                    + self.config.host_cache_hit_weight * host / block;
                let adjusted = (raw_tokens / block - credit).max(0.0);
                let score = self.config.prefill_load_scale * adjusted
                    + Self::eff_ongoing_decode(inst) / block
                    + self.config.decode_active_request_weight * Self::eff_num_requests(inst);
                (p, score)
            })
            .collect()
    }

    fn home_key(&self, req: &SMetricRequest) -> Option<u64> {
        let depth = self.config.session_home_depth?;
        req.block_hashes
            .get(depth.min(req.block_hashes.len().saturating_sub(1)))
            .copied()
    }

    fn remembered_home(&self, key: Option<u64>, insts: &[SMetricInstance]) -> Option<usize> {
        let key = key?;
        let homes = self.homes.lock().unwrap();
        let idx = *homes.0.get(&key)?;
        insts.iter().position(|inst| inst.idx == idx)
    }

    fn remember_home(&self, key: Option<u64>, idx: usize) {
        let Some(key) = key else { return };
        let mut homes = self.homes.lock().unwrap();
        if homes.0.contains_key(&key) {
            return;
        }
        if homes.0.len() >= HOME_CAP {
            if let Some(oldest) = homes.1.pop_front() {
                homes.0.remove(&oldest);
            }
        }
        homes.0.insert(key, idx);
        homes.1.push_back(key);
    }

    fn service_work_s(&self, n: usize, input: usize) -> f64 {
        self.work_units(n as f64, input as f64) / self.config.drain_tps
    }

    fn age_bin(age: f64) -> usize {
        [10.0, 30.0, 60.0, 120.0]
            .iter()
            .take_while(|&&bound| age >= bound)
            .count()
    }

    fn service_estimate(
        &self,
        state: &ServiceCostState,
        req: &SMetricRequest,
        inst: &SMetricInstance,
    ) -> (f64, f64, f64, usize) {
        let gpu = inst.cache_hit_tokens.min(req.input_length);
        let store = gpu.max(inst.store_hit_tokens.min(req.input_length));
        let age_bin = Self::age_bin(inst.store_prefix_age_s);
        let outcomes = &state.losses[age_bin];
        let loss = if store > gpu {
            (1.0 + outcomes.iter().sum::<f64>()) / (2.0 + outcomes.len() as f64)
        } else {
            0.0
        };
        let hit_s = self.config.service_fixed_s
            + self.service_work_s(req.input_length - store, req.input_length)
            + (store - gpu) as f64 / self.config.store_load_tps;
        let severity = &state.loss_severity[age_bin];
        let lost_fraction = (1.0 + severity.iter().sum::<f64>()) / (1.0 + severity.len() as f64);
        let hit_work = self.service_work_s(req.input_length - store, req.input_length);
        let cold_work = self.service_work_s(req.input_length - gpu, req.input_length);
        let miss_s = hit_s + lost_fraction * (cold_work - hit_work);
        (hit_s, miss_s, loss, age_bin)
    }

    fn survival_remaining(mean: f64, age: f64) -> (f64, f64) {
        let (lo, hi) = (0.5 * mean, 1.5 * mean);
        if age < lo {
            (1.0, mean - age)
        } else if age < hi {
            ((hi - age) / (hi - lo), (hi - age) / 2.0)
        } else {
            (0.0, 0.0)
        }
    }

    fn service_queue_seconds(
        &self,
        state: &ServiceCostState,
        inst: &SMetricInstance,
        now: f64,
    ) -> f64 {
        let Some(ids) = state.queues.get(&inst.idx) else {
            return inst
                .real_pending_prefill_tokens
                .filter(|value| *value > 0.0)
                .map(|remaining| self.service_work_s(remaining as usize, remaining as usize))
                .unwrap_or(0.0);
        };
        let mut total = 0.0;
        for (position, id) in ids.iter().enumerate() {
            let Some(pending) = state.pending.get(id) else {
                continue;
            };
            if position > 0 {
                total += (1.0 - pending.loss_probability) * pending.hit_seconds
                    + pending.loss_probability * pending.miss_seconds;
                continue;
            }
            let age = (now - pending.started_at).max(0.0);
            let (sh, rh) = Self::survival_remaining(pending.hit_seconds, age);
            let (sm, rm) = Self::survival_remaining(pending.miss_seconds, age);
            let (ph, pm) = (
                (1.0 - pending.loss_probability) * sh,
                pending.loss_probability * sm,
            );
            if ph + pm > 0.0 {
                total += (ph * rh + pm * rm) / (ph + pm);
            } else {
                let remaining = inst
                    .real_pending_prefill_tokens
                    .unwrap_or(pending.input_length.saturating_sub(pending.gpu_hit) as f64);
                total += self.config.service_fixed_s.max(self.service_work_s(
                    remaining.max(0.0).min(pending.input_length as f64) as usize,
                    pending.input_length,
                ));
            }
        }
        if ids.is_empty() {
            if let Some(remaining) = inst
                .real_pending_prefill_tokens
                .filter(|value| *value > 0.0)
            {
                total = self.service_work_s(remaining as usize, remaining as usize);
            }
        }
        total
    }

    fn route_service(
        &self,
        req: &SMetricRequest,
        insts: &[SMetricInstance],
    ) -> Option<SMetricDecision> {
        let mut state = self.service.lock().unwrap();
        let estimates: Vec<_> = insts
            .iter()
            .map(|inst| self.service_estimate(&state, req, inst))
            .collect();
        let queues: Vec<_> = insts
            .iter()
            .map(|inst| self.service_queue_seconds(&state, inst, req.now_s))
            .collect();
        let predictions: Vec<f64> = queues
            .iter()
            .zip(&estimates)
            .map(|(q, e)| q + (1.0 - e.2) * e.0 + e.2 * e.1)
            .collect();
        let home = (0..insts.len()).max_by(|&a, &b| {
            insts[a]
                .cache_hit_tokens
                .cmp(&insts[b].cache_hit_tokens)
                .then_with(|| predictions[b].total_cmp(&predictions[a]))
                .then_with(|| b.cmp(&a))
        })?;
        let fits = req.turn_depth > 1
            && insts[home].cache_hit_tokens as f64
                > self.config.hit_ratio * req.input_length as f64
            && predictions[home] <= self.budget_s(req);
        let selected = if fits {
            home
        } else {
            self.choose_min(
                &predictions
                    .iter()
                    .enumerate()
                    .map(|(p, &s)| (p, s))
                    .collect::<Vec<_>>(),
            )?
        };
        if let Some(old) = state.pending.remove(&req.request_key) {
            if let Some(queue) = state.queues.get_mut(&old.idx) {
                queue.retain(|id| *id != req.request_key);
            }
        }
        let estimate = estimates[selected];
        state.pending.insert(
            req.request_key,
            ServicePending {
                idx: insts[selected].idx,
                input_length: req.input_length,
                gpu_hit: insts[selected].cache_hit_tokens,
                store_hit: insts[selected].store_hit_tokens,
                age_bin: estimate.3,
                loss_probability: estimate.2,
                hit_seconds: estimate.0,
                miss_seconds: estimate.1,
                started_at: req.now_s,
            },
        );
        state
            .queues
            .entry(insts[selected].idx)
            .or_default()
            .push_back(req.request_key);
        Some(SMetricDecision {
            position: selected,
            instance_idx: insts[selected].idx,
            reason: if fits {
                "smetric_service_stick"
            } else {
                "smetric_service_move"
            }
            .into(),
        })
    }

    pub fn note_cache_result(&self, request_key: u64, cached_tokens: Option<usize>) {
        let mut state = self.service.lock().unwrap();
        let Some(pending) = state.pending.remove(&request_key) else {
            return;
        };
        let was_head = state
            .queues
            .get(&pending.idx)
            .and_then(|q| q.front())
            .is_some_and(|id| *id == request_key);
        if let Some(queue) = state.queues.get_mut(&pending.idx) {
            queue.retain(|id| *id != request_key);
        }
        if was_head {
            let next = state
                .queues
                .get(&pending.idx)
                .and_then(|q| q.front())
                .copied();
            if let Some(next_pending) = next.and_then(|id| state.pending.get_mut(&id)) {
                next_pending.started_at = self.started_at.elapsed().as_secs_f64();
            }
        }
        if let Some(cached) = cached_tokens.filter(|_| pending.store_hit > pending.gpu_hit) {
            let hit_work = self.service_work_s(
                pending.input_length - pending.store_hit,
                pending.input_length,
            );
            let miss_work =
                self.service_work_s(pending.input_length - pending.gpu_hit, pending.input_length);
            let actual_work = self.service_work_s(
                pending.input_length.saturating_sub(cached),
                pending.input_length,
            );
            let fraction =
                ((actual_work - hit_work) / (miss_work - hit_work).max(1e-9)).clamp(0.0, 1.0);
            let lost = cached.saturating_add(512) < pending.store_hit;
            Self::push_bounded(&mut state.losses[pending.age_bin], lost as u8 as f64);
            if lost {
                Self::push_bounded(&mut state.loss_severity[pending.age_bin], fraction);
            }
        }
    }

    fn push_bounded(queue: &mut VecDeque<f64>, value: f64) {
        if queue.len() == SERVICE_LOSS_WINDOW {
            queue.pop_front();
        }
        queue.push_back(value);
    }

    fn clear_unclaimed_service_selections(&self) {
        let thread_id = std::thread::current().id();
        let abandoned: Vec<u64> = {
            let mut selections = self.service_selections.lock().unwrap();
            let keys: Vec<_> = selections
                .keys()
                .filter(|(owner, _)| *owner == thread_id)
                .cloned()
                .collect();
            keys.into_iter()
                .flat_map(|key| selections.remove(&key).unwrap_or_default())
                .collect()
        };
        for request_key in abandoned {
            self.note_cache_result(request_key, None);
        }
    }

    /// Exact decision core. Candidate order is semantically significant for
    /// first-index and round-robin tie behavior, as in Python.
    pub fn route_observations(
        &self,
        req: &SMetricRequest,
        insts: &[SMetricInstance],
    ) -> Option<SMetricDecision> {
        if insts.is_empty() {
            return None;
        }
        if self.config.service_time_routing {
            return self.route_service(req, insts);
        }
        let hits: Vec<usize> = insts
            .iter()
            .map(|i| i.cache_hit_tokens.min(req.input_length))
            .collect();
        let load: Vec<f64> = insts.iter().map(|i| self.load(i)).collect();
        let mut home = (0..insts.len()).max_by_key(|&p| (hits[p], std::cmp::Reverse(p)))?;
        let home_key = self.home_key(req);
        if let Some(pinned) = self.remembered_home(home_key, insts) {
            home = pinned;
        }

        let mean = load.iter().sum::<f64>() / load.len() as f64;
        let mut fits = match self.config.gate {
            SMetricGate::Overload => {
                self.config.overload_factor.is_infinite()
                    || load[home] <= self.config.overload_factor * mean
            }
            SMetricGate::Budget | SMetricGate::BudgetAttention => {
                self.queue_fits_budget(req, &insts[home])
            }
        };
        if !fits && self.home_quiet(&insts[home]) {
            fits = true;
        }
        let safe: Option<Vec<usize>> = self.config.contract_safe.then(|| {
            (0..insts.len())
                .filter(|&p| self.contract_safe(req, &insts[p]))
                .collect()
        });
        let stick = req.turn_depth != 1
            && fits
            && hits[home] as f64 > self.config.hit_ratio * req.input_length as f64
            && safe.as_ref().is_none_or(|pool| pool.contains(&home));
        let decision = if stick {
            SMetricDecision {
                position: home,
                instance_idx: insts[home].idx,
                reason: "smetric_stick".into(),
            }
        } else {
            let candidates: Vec<usize> = match safe.as_ref() {
                Some(pool) if !pool.is_empty() => pool.clone(),
                _ => (0..insts.len()).collect(),
            };
            let rescue = if self.config.store_rescue && req.turn_depth > 1 {
                self.store_rescue_pick(req, insts, &candidates)
            } else {
                None
            };
            let (position, used_rr) = if let Some(position) = rescue {
                (position, false)
            } else {
                self.choose_min_detail(&self.fallback_scores(req, insts, &load, &candidates))?
            };
            let fallback = match self.config.fallback {
                SMetricFallback::Load => "load",
                SMetricFallback::Dynamo | SMetricFallback::DynamoLogit => "dynamo_prefill_router",
                SMetricFallback::Lmetric => "lmetric",
                SMetricFallback::PrefillWorkAttention => "prefill_work_attention",
                SMetricFallback::LmetricAttention => "lmetric_attention",
            };
            let mut reason = format!("smetric_fallback_{fallback}");
            let dynamo = matches!(
                self.config.fallback,
                SMetricFallback::Dynamo | SMetricFallback::DynamoLogit
            );
            if dynamo && used_rr {
                reason.push_str("_rr");
            }
            if let Some(pool) = safe.as_ref() {
                if pool.is_empty() {
                    reason.push_str("_no_safe_engine");
                } else if pool.len() < insts.len() {
                    reason.push_str("_contract_safe");
                }
            }
            if rescue.is_some() {
                reason.push_str("_store_rescue");
            } else if used_rr && !dynamo {
                reason.push_str("_rr");
            }
            SMetricDecision {
                position,
                instance_idx: insts[position].idx,
                reason,
            }
        };
        if self.config.session_home_depth.is_some() {
            self.remember_home(home_key, decision.instance_idx);
        }
        Some(decision)
    }

    fn maybe_evict(&self) {
        if self.config.eviction_interval_secs == 0 {
            return;
        }
        let mut last = self.last_eviction.lock().unwrap();
        if last.elapsed() < Duration::from_secs(self.config.eviction_interval_secs) {
            return;
        }
        for tree in self.trees.iter() {
            tree.value().evict_tenant_by_size(self.config.max_tree_size);
        }
        *last = Instant::now();
    }

    fn prune_rates(config: &SMetricConfig, samples: &mut VecDeque<(Instant, f64)>, now: Instant) {
        let window = Duration::from_secs(config.drain_window_secs);
        while samples
            .front()
            .is_some_and(|(at, _)| now.duration_since(*at) > window)
        {
            samples.pop_front();
        }
    }

    fn percentile_rate(config: &SMetricConfig, samples: &VecDeque<(Instant, f64)>) -> Option<f64> {
        if samples.len() < config.drain_min_samples {
            return None;
        }
        let mut values: Vec<f64> = samples.iter().map(|(_, rate)| *rate).collect();
        values.sort_by(f64::total_cmp);
        Some(values[(0.9 * (values.len() - 1) as f64) as usize])
    }

    fn hash_block(text: &str, depth: usize) -> Option<u64> {
        const APPROX_BLOCK_CHARS: usize = 512;
        let block: String = text
            .chars()
            .skip(depth * APPROX_BLOCK_CHARS)
            .take(APPROX_BLOCK_CHARS)
            .collect();
        if block.is_empty() {
            return None;
        }
        let mut hasher = DefaultHasher::new();
        block.hash(&mut hasher);
        Some(hasher.finish())
    }

    fn runtime_observations(
        &self,
        workers: &[Arc<dyn Worker>],
        healthy: &[usize],
        text: &str,
    ) -> Vec<SMetricInstance> {
        let model = normalize_model_key(workers[healthy[0]].model_id());
        let tree = self.trees.get(model).map(|value| Arc::clone(value.value()));
        let mut states = self.state.lock().unwrap();
        let now = Instant::now();
        healthy
            .iter()
            .map(|&worker_pos| {
                let url = workers[worker_pos].url();
                let state = states.entry(url.to_string()).or_default();
                Self::prune_rates(&self.config, &mut state.token_rate_samples, now);
                Self::prune_rates(&self.config, &mut state.work_rate_samples, now);
                let external = self
                    .external
                    .get(url)
                    .map(|value| value.clone())
                    .unwrap_or_default();
                let hit = tree
                    .as_ref()
                    .map(|tree| tree.prefix_match_tenant(text, url).chars().count())
                    .unwrap_or(0)
                    .min(text.chars().count());
                let elapsed =
                    |reservation: &Reservation| reservation.started_at.elapsed().as_secs_f64();
                let local_contracts: Vec<_> = state
                    .reservations
                    .values()
                    .filter(|reservation| reservation.response_started)
                    .map(|reservation| DecodeContract {
                        elapsed_s: elapsed(reservation),
                        input_length: reservation.input_tokens as usize,
                        emitted_tokens: 0,
                    })
                    .collect();
                SMetricInstance {
                    idx: worker_pos,
                    cache_hit_tokens: hit,
                    store_hit_tokens: external.store_hit_tokens.unwrap_or(0),
                    store_prefix_age_s: external.store_prefix_age_s.unwrap_or(0.0),
                    pending_prefill_tokens: state.pending_prefill_tokens,
                    pending_prefill_attention: external
                        .pending_prefill_attention
                        .unwrap_or(state.pending_prefill_attention),
                    pending_prefill_compute_attention: external
                        .pending_prefill_compute_attention
                        .unwrap_or(state.pending_prefill_compute_attention),
                    pending_store_tokens: external
                        .pending_store_tokens
                        .unwrap_or(state.pending_store_tokens),
                    ongoing_decode_tokens: state.ongoing_decode_tokens,
                    num_requests: state.active_requests as f64,
                    real_pending_prefill_tokens: external.pending_prefill_tokens,
                    real_ongoing_decode_tokens: external.ongoing_decode_tokens,
                    real_num_requests: external
                        .num_requests
                        .or_else(|| external.real_inflight.map(|value| value as f64)),
                    real_inflight: external.real_inflight,
                    est_prefill_tps: external
                        .est_prefill_tps
                        .or_else(|| Self::percentile_rate(&self.config, &state.token_rate_samples)),
                    est_prefill_work_tps: external
                        .est_prefill_work_tps
                        .or_else(|| Self::percentile_rate(&self.config, &state.work_rate_samples)),
                    decode_contracts: external.decode_contracts.unwrap_or(local_contracts),
                }
            })
            .collect()
    }
}

impl Default for SMetricPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl LoadBalancingPolicy for SMetricPolicy {
    fn select_worker_with_headers(
        &self,
        workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
        headers: Option<&RequestHeaders>,
    ) -> Option<usize> {
        let healthy = get_healthy_worker_indices(workers);
        if healthy.is_empty() {
            return None;
        }
        let text = request_text.unwrap_or("");
        let input = text.chars().count();
        let turn_depth = headers
            .and_then(|h| h.get("x-session-turn"))
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(1)
            .max(1);
        if self.config.service_time_routing {
            // Selection and lifecycle start are synchronous in Router. Any
            // unclaimed selection owned by this thread came from an aborted
            // pair-selection path and must not remain in the serial queue.
            self.clear_unclaimed_service_selections();
        }
        let request_key = self.next_reservation.fetch_add(1, Ordering::Relaxed);
        let block_hashes = self
            .config
            .session_home_depth
            .map(|depth| {
                (0..=depth)
                    .filter_map(|d| Self::hash_block(text, d))
                    .collect()
            })
            .unwrap_or_default();
        let req = SMetricRequest {
            request_key,
            input_length: input,
            turn_depth,
            block_hashes,
            now_s: self.started_at.elapsed().as_secs_f64(),
        };
        let observations = self.runtime_observations(workers, &healthy, text);
        let decision = self.route_observations(&req, &observations)?;
        let selected = decision.instance_idx;
        if self.config.service_time_routing {
            self.service_selections
                .lock()
                .unwrap()
                .entry((
                    std::thread::current().id(),
                    workers[selected].url().to_string(),
                ))
                .or_default()
                .push_back(request_key);
        }
        workers[selected].increment_processed();
        RouterMetrics::record_processed_request(workers[selected].url());
        RouterMetrics::record_policy_decision(self.name(), workers[selected].url());
        debug!(
            worker = workers[selected].url(),
            reason = decision.reason,
            "SMetric decision"
        );
        Some(selected)
    }

    fn on_request_start(
        &self,
        worker_url: &str,
        request_text: Option<&str>,
        _headers: Option<&RequestHeaders>,
    ) -> Option<u64> {
        let text = request_text.unwrap_or("");
        let input = text.chars().count();
        let model = self.worker_models.get(worker_url)?.clone();
        let tree = self.trees.get(model.as_str())?.clone();
        let gpu_hit = tree
            .prefix_match_tenant(text, worker_url)
            .chars()
            .count()
            .min(input);
        let external = self
            .external
            .get(worker_url)
            .map(|value| value.clone())
            .unwrap_or_default();
        let store_hit = external.store_hit_tokens.unwrap_or(0).min(input);
        let prefill = input.saturating_sub(gpu_hit) as f64;
        let store_tokens = input
            .saturating_sub(gpu_hit)
            .min(store_hit.saturating_sub(gpu_hit)) as f64;
        let moment = Self::attention_moment(prefill, input as f64);
        let compute_moment = Self::attention_moment(prefill - store_tokens, input as f64);
        let work = self.work_units(prefill, input as f64);
        let reservation_id = if self.config.service_time_routing {
            self.service_selections
                .lock()
                .unwrap()
                .entry((std::thread::current().id(), worker_url.to_string()))
                .or_default()
                .pop_back()
                .unwrap_or_else(|| self.next_reservation.fetch_add(1, Ordering::Relaxed))
        } else {
            self.next_reservation.fetch_add(1, Ordering::Relaxed)
        };

        let mut states = self.state.lock().unwrap();
        let worker = states.entry(worker_url.to_string()).or_default();
        worker.pending_prefill_tokens += prefill;
        worker.pending_prefill_attention += moment;
        worker.pending_prefill_compute_attention += compute_moment;
        worker.pending_store_tokens += store_tokens;
        worker.active_requests += 1;
        worker.reservations.insert(
            reservation_id,
            Reservation {
                request_text: text.to_string(),
                input_tokens: input as f64,
                prefill_tokens: prefill,
                prefill_attention: moment,
                prefill_compute_attention: compute_moment,
                store_tokens,
                work_units: work,
                started_at: Instant::now(),
                response_started: false,
            },
        );
        drop(states);
        Some(reservation_id)
    }

    fn on_request_first_response(
        &self,
        worker_url: &str,
        reservation_id: Option<u64>,
        elapsed: Duration,
    ) {
        let Some(id) = reservation_id else { return };
        let mut states = self.state.lock().unwrap();
        let Some(worker) = states.get_mut(worker_url) else {
            return;
        };
        let Some(reservation) = worker.reservations.get_mut(&id) else {
            return;
        };
        if reservation.response_started {
            return;
        }
        reservation.response_started = true;
        worker.pending_prefill_tokens =
            (worker.pending_prefill_tokens - reservation.prefill_tokens).max(0.0);
        worker.pending_prefill_attention =
            (worker.pending_prefill_attention - reservation.prefill_attention).max(0.0);
        // With no store observation compute==total; when store is present the
        // aggregate is conservatively reduced by the total moment here. Rich
        // integrations can overwrite it through the external observation.
        worker.pending_prefill_compute_attention = (worker.pending_prefill_compute_attention
            - reservation.prefill_compute_attention)
            .max(0.0);
        worker.pending_store_tokens =
            (worker.pending_store_tokens - reservation.store_tokens).max(0.0);
        worker.ongoing_decode_tokens += reservation.input_tokens;
        if elapsed.is_zero() {
            return;
        }
        let now = Instant::now();
        let model_tokens = (reservation.prefill_tokens - reservation.store_tokens).max(0.0);
        if model_tokens >= 1024.0 {
            worker
                .token_rate_samples
                .push_back((now, model_tokens / elapsed.as_secs_f64()));
        }
        if reservation.work_units > 0.0 {
            worker
                .work_rate_samples
                .push_back((now, reservation.work_units / elapsed.as_secs_f64()));
        }
    }

    fn on_request_finish(
        &self,
        worker_url: &str,
        reservation_id: Option<u64>,
        success: bool,
        elapsed: Duration,
    ) {
        let Some(id) = reservation_id else { return };
        if success {
            self.on_request_first_response(worker_url, Some(id), elapsed);
        }
        let mut states = self.state.lock().unwrap();
        let Some(worker) = states.get_mut(worker_url) else {
            return;
        };
        let mut completed_text = None;
        if let Some(reservation) = worker.reservations.remove(&id) {
            if reservation.response_started {
                worker.ongoing_decode_tokens =
                    (worker.ongoing_decode_tokens - reservation.input_tokens).max(0.0);
            } else {
                worker.pending_prefill_tokens =
                    (worker.pending_prefill_tokens - reservation.prefill_tokens).max(0.0);
                worker.pending_prefill_attention =
                    (worker.pending_prefill_attention - reservation.prefill_attention).max(0.0);
                worker.pending_prefill_compute_attention = (worker
                    .pending_prefill_compute_attention
                    - reservation.prefill_compute_attention)
                    .max(0.0);
                worker.pending_store_tokens =
                    (worker.pending_store_tokens - reservation.store_tokens).max(0.0);
            }
            worker.active_requests = worker.active_requests.saturating_sub(1);
            if success {
                completed_text = Some(reservation.request_text);
            }
        }
        drop(states);
        if let Some(text) = completed_text {
            if let Some(model) = self.worker_models.get(worker_url) {
                if let Some(tree) = self.trees.get(model.as_str()) {
                    tree.insert(&text, worker_url);
                    self.maybe_evict();
                }
            }
        }
        if self.config.service_time_routing {
            self.note_cache_result(id, None);
        }
    }

    fn tracks_worker_load(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "smetric"
    }

    fn needs_request_text(&self) -> bool {
        true
    }

    fn needs_headers(&self) -> bool {
        true
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn requires_initialization(&self) -> bool {
        true
    }

    fn init_workers(&self, workers: &[Arc<dyn Worker>]) {
        let mut states = self.state.lock().unwrap();
        for worker in workers {
            let model = normalize_model_key(worker.model_id()).to_string();
            let tree = self
                .trees
                .entry(model.clone())
                .or_insert_with(|| Arc::new(Tree::new()))
                .clone();
            tree.insert("", worker.url());
            self.worker_models.insert(worker.url().to_string(), model);
            states.entry(worker.url().to_string()).or_default();
        }
    }

    fn remove_worker_by_url(&self, url: &str) {
        for tree in self.trees.iter() {
            tree.value().remove_tenant(url);
        }
        self.worker_models.remove(url);
        self.external.remove(url);
        self.state.lock().unwrap().remove(url);
        self.service_selections
            .lock()
            .unwrap()
            .retain(|(_, worker_url), _| worker_url != url);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorker, WorkerType};
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct FixtureFile {
        source_sha256: String,
        sources_sha256: HashMap<String, String>,
        cases: Vec<FixtureCase>,
    }

    #[derive(Deserialize)]
    struct FixtureCase {
        name: String,
        config: FixtureConfig,
        request: FixtureRequest,
        instances: Vec<FixtureInstance>,
        expected_position: usize,
        expected_reason: String,
    }

    #[derive(Default, Deserialize)]
    struct FixtureConfig {
        overload: Option<f64>,
        gate: Option<String>,
        budget_gamma: Option<f64>,
        drain_tps: Option<f64>,
        drain_source: Option<String>,
        fallback: Option<String>,
        attention_l_eq: Option<f64>,
        overlap_score_credit_decay: Option<f64>,
        overlap_score_credit: Option<f64>,
        host_cache_hit_weight: Option<f64>,
        track_prefill_tokens: Option<bool>,
        prefill_load_scale: Option<f64>,
        decode_active_request_weight: Option<f64>,
        hit_ratio: Option<f64>,
        store_pricing: Option<bool>,
        queue_store_pricing: Option<bool>,
        store_load_tps: Option<f64>,
        home_quiet_stick: Option<usize>,
        contract_safe: Option<bool>,
        store_rescue: Option<bool>,
        service_time_routing: Option<bool>,
    }

    #[derive(Deserialize)]
    struct FixtureRequest {
        input_length: usize,
        turn_depth: usize,
    }

    #[derive(Default, Deserialize)]
    struct FixtureInstance {
        #[serde(default)]
        cache_hit_tokens: usize,
        #[serde(default)]
        store_hit_tokens: usize,
        #[serde(default)]
        pending_prefill_tokens: f64,
        #[serde(default)]
        pending_prefill_attention: f64,
        #[serde(default)]
        pending_prefill_compute_attention: f64,
        #[serde(default)]
        pending_store_tokens: f64,
        #[serde(default)]
        ongoing_decode_tokens: f64,
        #[serde(default)]
        num_requests: f64,
        real_pending_prefill_tokens: Option<f64>,
        real_ongoing_decode_tokens: Option<f64>,
        real_inflight: Option<usize>,
        est_prefill_tps: Option<f64>,
        est_prefill_work_tps: Option<f64>,
        #[serde(default)]
        decode_contracts: Vec<DecodeContract>,
    }

    fn config(raw: FixtureConfig) -> SMetricConfig {
        let mut config = SMetricConfig::default();
        // Reference fixtures retain the Python prototype's load fallback.
        config.fallback = SMetricFallback::Load;
        if let Some(value) = raw.overload {
            config.overload_factor = value;
        }
        if let Some(value) = raw.drain_tps {
            config.drain_tps = value;
        }
        if let Some(value) = raw.budget_gamma {
            config.budget_gamma = value;
        }
        if let Some(value) = raw.attention_l_eq {
            config.attention_l_eq = value;
        }
        if let Some(value) = raw.overlap_score_credit_decay {
            config.overlap_score_credit_decay = value;
        }
        if let Some(value) = raw.overlap_score_credit {
            config.overlap_score_credit = value;
        }
        if let Some(value) = raw.host_cache_hit_weight {
            config.host_cache_hit_weight = value;
        }
        if let Some(value) = raw.track_prefill_tokens {
            config.track_prefill_tokens = value;
        }
        if let Some(value) = raw.prefill_load_scale {
            config.prefill_load_scale = value;
        }
        if let Some(value) = raw.decode_active_request_weight {
            config.decode_active_request_weight = value;
        }
        if let Some(value) = raw.hit_ratio {
            config.hit_ratio = value;
        }
        if let Some(value) = raw.store_pricing {
            config.store_pricing = value;
        }
        if let Some(value) = raw.queue_store_pricing {
            config.queue_store_pricing = value;
        }
        if let Some(value) = raw.store_load_tps {
            config.store_load_tps = value;
        }
        if let Some(value) = raw.home_quiet_stick {
            config.home_quiet_stick = Some(value);
        }
        if let Some(value) = raw.contract_safe {
            config.contract_safe = value;
        }
        if let Some(value) = raw.store_rescue {
            config.store_rescue = value;
        }
        if let Some(value) = raw.service_time_routing {
            config.service_time_routing = value;
        }
        if let Some(value) = raw.gate.as_deref() {
            config.gate = match value {
                "overload" => SMetricGate::Overload,
                "budget" => SMetricGate::Budget,
                "budget_attention" => SMetricGate::BudgetAttention,
                other => panic!("unknown fixture gate {other}"),
            };
        }
        if let Some(value) = raw.drain_source.as_deref() {
            config.drain_source = match value {
                "config" => SMetricDrainSource::Config,
                "measured" => SMetricDrainSource::Measured,
                other => panic!("unknown fixture drain source {other}"),
            };
        }
        if let Some(value) = raw.fallback.as_deref() {
            config.fallback = match value {
                "load" => SMetricFallback::Load,
                "dynamo" => SMetricFallback::Dynamo,
                "dynamo_logit" => SMetricFallback::DynamoLogit,
                "lmetric" => SMetricFallback::Lmetric,
                "prefill_work_attention" => SMetricFallback::PrefillWorkAttention,
                "lmetric_attention" => SMetricFallback::LmetricAttention,
                other => panic!("unknown fixture fallback {other}"),
            };
        }
        config
    }

    #[test]
    fn matches_python_reference_fixtures() {
        let fixtures: FixtureFile = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/smetric_reference.json"
        )))
        .unwrap();
        assert_eq!(
            fixtures.source_sha256,
            "7871badc7d4be2bb83565400f6227b51f40b251e939695cc628c782982d01659"
        );
        for (path, expected) in [
            (
                "src/ssched/scheduler/policy/external.py",
                "e871dce0f4751c3a530fcc28c4c80cfd375ba17be037c3fda52aeb6952a5ba89",
            ),
            (
                "src/ssched/scheduler/policy/service_cost.py",
                "2b620a4857a901ed072308240c90fdc0ba509dc928e44d94262c9592a69e9a50",
            ),
            (
                "src/ssched/scheduler/policy/baselines.py",
                "31c347d40eacd10dc2c3b8900dde6502b2d8260266422d209d865d75bec9c819",
            ),
            (
                "src/ssched/scheduler/core.py",
                "662264cb1f60f24a67fc6b78638031881b67dcd3935fbbb3e08c2867f354da89",
            ),
            (
                "src/ssched/state/schema.py",
                "8cf8a3b0b2d086d84533d23e0a8960c748c17903666ab2593ebe4edcd382ef47",
            ),
        ] {
            assert_eq!(
                fixtures.sources_sha256.get(path).map(String::as_str),
                Some(expected)
            );
        }
        for fixture in fixtures.cases {
            let policy = SMetricPolicy::with_config(config(fixture.config));
            let request = SMetricRequest {
                request_key: 1,
                input_length: fixture.request.input_length,
                turn_depth: fixture.request.turn_depth,
                now_s: 0.0,
                ..Default::default()
            };
            let instances: Vec<_> = fixture
                .instances
                .into_iter()
                .enumerate()
                .map(|(idx, raw)| SMetricInstance {
                    idx,
                    cache_hit_tokens: raw.cache_hit_tokens,
                    store_hit_tokens: raw.store_hit_tokens,
                    pending_prefill_tokens: raw.pending_prefill_tokens,
                    pending_prefill_attention: raw.pending_prefill_attention,
                    pending_prefill_compute_attention: raw.pending_prefill_compute_attention,
                    pending_store_tokens: raw.pending_store_tokens,
                    ongoing_decode_tokens: raw.ongoing_decode_tokens,
                    num_requests: raw.num_requests,
                    real_pending_prefill_tokens: raw.real_pending_prefill_tokens,
                    real_ongoing_decode_tokens: raw.real_ongoing_decode_tokens,
                    real_num_requests: raw.real_inflight.map(|value| value as f64),
                    real_inflight: raw.real_inflight,
                    est_prefill_tps: raw.est_prefill_tps,
                    est_prefill_work_tps: raw.est_prefill_work_tps,
                    decode_contracts: raw.decode_contracts,
                    ..Default::default()
                })
                .collect();
            let actual = policy.route_observations(&request, &instances).unwrap();
            assert_eq!(
                actual.position, fixture.expected_position,
                "fixture {}: Python reason {}, Rust reason {}",
                fixture.name, fixture.expected_reason, actual.reason
            );
            assert_eq!(
                actual.reason, fixture.expected_reason,
                "fixture {} reason",
                fixture.name
            );
        }
    }

    #[test]
    fn stable_home_and_gate_are_separate() {
        let policy = SMetricPolicy::with_config(SMetricConfig {
            session_home_depth: Some(0),
            overload_factor: f64::INFINITY,
            ..Default::default()
        });
        let mut request = SMetricRequest {
            request_key: 1,
            input_length: 100,
            turn_depth: 1,
            block_hashes: vec![7],
            ..Default::default()
        };
        let cold = vec![
            SMetricInstance {
                idx: 10,
                ..Default::default()
            },
            SMetricInstance {
                idx: 20,
                ..Default::default()
            },
        ];
        let first = policy.route_observations(&request, &cold).unwrap();
        request.turn_depth = 2;
        let warm_elsewhere = vec![
            SMetricInstance {
                idx: 10,
                cache_hit_tokens: 60,
                ..Default::default()
            },
            SMetricInstance {
                idx: 20,
                cache_hit_tokens: 100,
                ..Default::default()
            },
        ];
        let second = policy
            .route_observations(&request, &warm_elsewhere)
            .unwrap();
        assert_eq!(second.instance_idx, first.instance_idx);
    }

    #[test]
    fn empty_candidate_set_returns_none() {
        assert!(SMetricPolicy::new()
            .route_observations(&SMetricRequest::default(), &[])
            .is_none());
    }

    #[test]
    fn round_robin_advances_only_on_exact_ties() {
        let policy = SMetricPolicy::new();
        let request = SMetricRequest {
            input_length: 100,
            turn_depth: 1,
            ..Default::default()
        };
        let tied = vec![
            SMetricInstance {
                idx: 0,
                ..Default::default()
            },
            SMetricInstance {
                idx: 1,
                ..Default::default()
            },
            SMetricInstance {
                idx: 2,
                ..Default::default()
            },
        ];
        assert_eq!(
            policy.route_observations(&request, &tied).unwrap().position,
            0
        );

        let strict = vec![
            SMetricInstance {
                idx: 0,
                pending_prefill_tokens: 10.0,
                ..Default::default()
            },
            SMetricInstance {
                idx: 1,
                ..Default::default()
            },
        ];
        assert_eq!(
            policy
                .route_observations(&request, &strict)
                .unwrap()
                .position,
            1
        );
        assert_eq!(
            policy.route_observations(&request, &tied).unwrap().position,
            1
        );
        assert_eq!(
            policy.route_observations(&request, &tied).unwrap().position,
            2
        );
        assert_eq!(
            policy.route_observations(&request, &tied).unwrap().position,
            0
        );
    }

    #[test]
    fn home_quiet_requires_real_engine_observation() {
        let policy = SMetricPolicy::with_config(SMetricConfig {
            gate: SMetricGate::Budget,
            drain_tps: 1.0,
            home_quiet_stick: Some(1),
            ..Default::default()
        });
        let request = SMetricRequest {
            input_length: 100,
            turn_depth: 2,
            ..Default::default()
        };
        let base = SMetricInstance {
            idx: 0,
            cache_hit_tokens: 100,
            pending_prefill_tokens: 1000.0,
            ..Default::default()
        };
        let other = SMetricInstance {
            idx: 1,
            ..Default::default()
        };
        assert_eq!(
            policy
                .route_observations(&request, &[base.clone(), other.clone()])
                .unwrap()
                .position,
            1
        );
        let mut quiet = base;
        quiet.real_inflight = Some(1);
        assert_eq!(
            policy
                .route_observations(&request, &[quiet, other])
                .unwrap()
                .position,
            0
        );
    }

    #[test]
    fn lifecycle_moves_work_calibrates_and_cleans_up() {
        let policy = SMetricPolicy::with_config(SMetricConfig {
            drain_source: SMetricDrainSource::Measured,
            drain_min_samples: 1,
            ..Default::default()
        });
        let worker: Arc<dyn Worker> = Arc::new(BasicWorker::new(
            "http://worker".to_string(),
            WorkerType::Regular,
        ));
        let workers = vec![Arc::clone(&worker)];
        policy.init_workers(&workers);
        let text = "a".repeat(2_048);

        let cold = policy.runtime_observations(&workers, &[0], &text);
        assert_eq!(cold[0].cache_hit_tokens, 0);
        assert_eq!(cold[0].est_prefill_tps, None);

        let id = policy
            .on_request_start(worker.url(), Some(&text), None)
            .unwrap();
        {
            let states = policy.state.lock().unwrap();
            let state = states.get(worker.url()).unwrap();
            assert_eq!(state.pending_prefill_tokens, 2_048.0);
            assert_eq!(state.active_requests, 1);
            assert_eq!(state.ongoing_decode_tokens, 0.0);
        }

        policy.on_request_first_response(worker.url(), Some(id), Duration::from_secs(1));
        // The phase transition is idempotent.
        policy.on_request_first_response(worker.url(), Some(id), Duration::from_secs(1));
        {
            let states = policy.state.lock().unwrap();
            let state = states.get(worker.url()).unwrap();
            assert_eq!(state.pending_prefill_tokens, 0.0);
            assert_eq!(state.ongoing_decode_tokens, 2_048.0);
            assert_eq!(state.token_rate_samples.len(), 1);
            assert_eq!(state.work_rate_samples.len(), 1);
        }

        policy.on_request_finish(worker.url(), Some(id), true, Duration::from_secs(2));
        // Finish is likewise idempotent and cannot underflow the ledger.
        policy.on_request_finish(worker.url(), Some(id), true, Duration::from_secs(2));
        {
            let states = policy.state.lock().unwrap();
            let state = states.get(worker.url()).unwrap();
            assert_eq!(state.active_requests, 0);
            assert_eq!(state.ongoing_decode_tokens, 0.0);
            assert!(state.reservations.is_empty());
        }
        let warm = policy.runtime_observations(&workers, &[0], &text);
        assert_eq!(warm[0].cache_hit_tokens, 2_048);
        assert_eq!(warm[0].est_prefill_tps, Some(2_048.0));
        assert!(warm[0].est_prefill_work_tps.unwrap() > 2_048.0);

        let failed_text = "b".repeat(100);
        let failed = policy
            .on_request_start(worker.url(), Some(&failed_text), None)
            .unwrap();
        policy.on_request_finish(worker.url(), Some(failed), false, Duration::from_millis(10));
        let failed_view = policy.runtime_observations(&workers, &[0], &failed_text);
        assert_eq!(failed_view[0].cache_hit_tokens, 0);
    }

    #[test]
    fn service_cost_completion_learns_store_loss() {
        let policy = SMetricPolicy::with_config(SMetricConfig {
            service_time_routing: true,
            ..Default::default()
        });
        let request = SMetricRequest {
            request_key: 9,
            input_length: 2_000,
            turn_depth: 1,
            now_s: 1.0,
            ..Default::default()
        };
        let instance = SMetricInstance {
            idx: 3,
            cache_hit_tokens: 0,
            store_hit_tokens: 1_500,
            store_prefix_age_s: 30.0,
            ..Default::default()
        };
        policy.route_observations(&request, &[instance]).unwrap();
        policy.note_cache_result(9, Some(0));
        let service = policy.service.lock().unwrap();
        assert_eq!(
            service.losses[2].iter().copied().collect::<Vec<_>>(),
            vec![1.0]
        );
        assert_eq!(service.loss_severity[2].len(), 1);
        assert!(service.pending.is_empty());
    }

    #[test]
    fn abandoned_service_selection_is_retired_before_next_route() {
        let policy = SMetricPolicy::with_config(SMetricConfig {
            service_time_routing: true,
            ..Default::default()
        });
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(BasicWorker::new(
                "http://worker-0".to_string(),
                WorkerType::Regular,
            )),
            Arc::new(BasicWorker::new(
                "http://worker-1".to_string(),
                WorkerType::Regular,
            )),
        ];
        policy.init_workers(&workers);
        policy.select_worker(&workers, Some("first")).unwrap();
        assert_eq!(policy.service.lock().unwrap().pending.len(), 1);

        let selected = policy.select_worker(&workers, Some("second")).unwrap();
        assert_eq!(policy.service.lock().unwrap().pending.len(), 1);
        let id = policy
            .on_request_start(workers[selected].url(), Some("second"), None)
            .unwrap();
        policy.on_request_finish(
            workers[selected].url(),
            Some(id),
            false,
            Duration::from_millis(1),
        );
        assert!(policy.service.lock().unwrap().pending.is_empty());
    }
}
