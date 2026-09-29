//! SMetric: session cache affinity with a prefill-work fallback.
use super::{LoadBalancingPolicy, RequestHeaders};
use crate::config::SMetricConfig;
use crate::core::{PrefillCharge, Worker};
use crate::metrics::RouterMetrics;
use crate::protocols::spec::SMetricPrompt;
use crate::tree::Tree;
use dashmap::DashMap;
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Debug)]
pub struct SMetricPolicy {
    config: SMetricConfig,
    trees: Arc<DashMap<String, Arc<Tree>>>,
    round_robin: AtomicUsize,
    rates: DashMap<String, Arc<Mutex<RateHistory>>>,
    decision_log: bool,
}

impl SMetricPolicy {
    pub fn new(config: SMetricConfig) -> Self {
        let trees: Arc<DashMap<String, Arc<Tree>>> = Arc::new(DashMap::new());
        let eviction_trees = Arc::downgrade(&trees);
        let max_size = config.max_tree_size;
        // Like cache_aware, bound each worker's historical text without scanning per request.
        std::thread::spawn(move || loop {
            std::thread::sleep(Duration::from_secs(30));
            let Some(trees) = eviction_trees.upgrade() else {
                break;
            };
            for entry in trees.iter() {
                entry.value().evict_tenant_by_size(max_size);
            }
        });
        Self {
            config,
            trees,
            rates: DashMap::new(),
            round_robin: AtomicUsize::new(0),
            decision_log: std::env::var_os("SMETRIC_DECISION_LOG").is_some(),
        }
    }

    pub fn select_prefill(
        &self,
        workers: &[Arc<dyn Worker>],
        prompt: &SMetricPrompt,
        passes_turn_gate: bool,
        colocated: bool,
    ) -> Option<(usize, SMetricPrefill)> {
        let l = prompt.text.chars().count();
        let start = self.round_robin.fetch_add(1, Ordering::Relaxed) % workers.len().max(1);
        let mut min: Option<(usize, u64, u64)> = None; // index, score, own cost
        let mut prev: Option<(usize, usize, u64, bool)> = None; // index, hit, own cost, TTFT
        let mut any_meets_ttft = false;
        let mut candidates = self.decision_log.then(|| Vec::with_capacity(workers.len()));
        for offset in 0..workers.len() {
            let idx = (start + offset) % workers.len();
            let worker = &workers[idx];
            if !worker.is_available() {
                continue;
            }
            let hit = self
                .trees
                .get(worker.model_id())
                .as_ref()
                .map_or(0, |tree| {
                    tree.prefix_match_tenant_char_count(&prompt.text, worker.url())
                });
            let n = (l - hit) as f64;
            let cost = (self.config.c_lin * n + self.config.c_att * n * (l as f64 - n / 2.0)).ceil()
                as u64;
            let pending = worker.pending_prefill_work();
            let queued = pending.saturating_add(cost);
            let rate = self.config.prefill_rate.or_else(|| {
                self.rates
                    .get(worker.url())
                    .and_then(|history| history.lock().rate)
            });
            let meets = rate.is_none_or(|rate| {
                (queued as f64) / rate
                    <= self.config.slack
                        * (self.config.ttft_slo_base + self.config.ttft_slo_per_char * l as f64)
            });
            any_meets_ttft |= meets;
            let load = if colocated { worker.load() as u64 } else { 0 };
            let score = if colocated {
                queued.saturating_mul(load)
            } else {
                queued
            };
            if let Some(candidates) = &mut candidates {
                candidates.push(serde_json::json!({
                    "worker": worker.url(), "hit": hit, "own_work": cost,
                    "pending_work": pending, "load": load, "rate": rate,
                    "meets": meets, "score": score,
                }));
            }
            if min.is_none_or(|(_, best, _)| score < best) {
                min = Some((idx, score, cost));
            }
            if prev.is_none_or(|(_, best, _, _)| hit > best) {
                prev = Some((idx, hit, cost, meets));
            }
        }
        let (prev_idx, hit, prev_cost, prev_meets) = prev?;
        let (min_idx, _, min_cost) = min?;
        let (idx, cost) = if passes_turn_gate
            && (hit as f64) > self.config.hit_ratio * prompt.est_hit_chars as f64
            && (prev_meets || !any_meets_ttft)
        {
            (prev_idx, prev_cost)
        } else {
            (min_idx, min_cost)
        };
        if !prompt.text.is_empty() {
            self.trees
                .entry(workers[idx].model_id().to_string())
                .or_insert_with(|| Arc::new(Tree::new()))
                .insert(&prompt.text, workers[idx].url());
        }
        RouterMetrics::record_processed_request(workers[idx].url());
        RouterMetrics::record_policy_decision(self.name(), workers[idx].url());
        let rates = if self.config.prefill_rate.is_none() {
            Some(
                self.rates
                    .entry(workers[idx].url().to_string())
                    .or_insert_with(|| Arc::new(Mutex::new(RateHistory::default())))
                    .clone(),
            )
        } else {
            None
        };
        Some((
            idx,
            SMetricPrefill {
                charge: Some(PrefillCharge::new(workers[idx].clone(), cost)),
                work: cost,
                started: Instant::now(),
                rates,
                decision: candidates.map(|candidates| serde_json::json!({
                    "selected_worker": workers[idx].url(),
                    "home_worker": workers[prev_idx].url(),
                    "fallback_worker": workers[min_idx].url(),
                    "reason": if !passes_turn_gate { "first_turn" }
                        else if (hit as f64) <= self.config.hit_ratio * prompt.est_hit_chars as f64 { "low_hit" }
                        else if prev_meets { "home_meets" }
                        else if !any_meets_ttft { "none_meets_stick" }
                        else { "home_rejected" },
                    "any_meets": any_meets_ttft, "home_meets": prev_meets,
                    "chars": l, "est_hit_chars": prompt.est_hit_chars,
                    "gate_budget": self.config.slack * (self.config.ttft_slo_base + self.config.ttft_slo_per_char * l as f64),
                    "colocated": colocated, "candidates": candidates,
                })),
                context: None,
            },
        ))
    }
}

const RATE_WINDOW: usize = 64;
const RATE_MIN_SAMPLES: usize = 8;

#[derive(Debug, Default)]
struct RateHistory {
    samples: VecDeque<f64>,
    rate: Option<f64>,
}

impl RateHistory {
    fn record(&mut self, rate: f64) {
        if self.samples.len() == RATE_WINDOW {
            self.samples.pop_front();
        }
        self.samples.push_back(rate);
        if self.samples.len() >= RATE_MIN_SAMPLES {
            let mut sorted: Vec<_> = self.samples.iter().copied().collect();
            sorted.sort_by(f64::total_cmp);
            self.rate = Some(sorted[((sorted.len() - 1) as f64 * 0.9).round() as usize]);
        }
    }
}

pub struct SMetricPrefill {
    charge: Option<PrefillCharge>,
    work: u64,
    started: Instant,
    rates: Option<Arc<Mutex<RateHistory>>>,
    decision: Option<serde_json::Value>,
    context: Option<(String, String)>,
}

impl SMetricPrefill {
    pub fn record_decision(&mut self, request_id: &str, session_id: &str, turn: &str) {
        if let Some(mut decision) = self.decision.take() {
            let worker = decision["selected_worker"].as_str().unwrap_or("").to_owned();
            decision["request_id"] = request_id.into();
            decision["session_id"] = session_id.into();
            decision["turn"] = turn.into();
            self.context = Some((request_id.to_owned(), worker));
            eprintln!("SMETRIC_DECISION_JSON {}", decision);
        }
    }

    pub fn on_first_token(&mut self) {
        let Some(charge) = self.charge.take() else {
            return;
        };
        drop(charge);
        let elapsed = self.started.elapsed().as_secs_f64();
        let mut learned_rate = None;
        let mut samples = 0;
        if let Some(rates) = &self.rates {
            if self.work > 0 && elapsed > 0.0 {
                let mut history = rates.lock();
                history.record(self.work as f64 / elapsed);
                learned_rate = history.rate;
                samples = history.samples.len();
            }
        }
        if let Some((request_id, worker)) = &self.context {
            eprintln!("SMETRIC_RATE_JSON {}", serde_json::json!({
                "request_id": request_id, "worker": worker, "work": self.work,
                "elapsed_s": elapsed, "sample_rate": self.work as f64 / elapsed,
                "learned_rate": learned_rate, "samples": samples,
            }));
        }
    }
}

impl LoadBalancingPolicy for SMetricPolicy {
    fn select_worker_with_headers(
        &self,
        _workers: &[Arc<dyn Worker>],
        _request_text: Option<&str>,
        _headers: Option<&RequestHeaders>,
    ) -> Option<usize> {
        None
    }

    fn name(&self) -> &'static str {
        "smetric"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorker, PrefillCharge, WorkerType};
    fn prompt(text: &str, est_hit_chars: usize) -> SMetricPrompt {
        SMetricPrompt {
            text: text.to_owned(),
            est_hit_chars,
        }
    }

    fn select(
        policy: &SMetricPolicy,
        workers: &[Arc<dyn Worker>],
        prompt: &SMetricPrompt,
        turn_gate: bool,
        colocated: bool,
    ) -> (usize, SMetricPrefill) {
        policy
            .select_prefill(workers, prompt, turn_gate, colocated)
            .unwrap()
    }

    #[test]
    fn cache_affinity_yields_to_a_feasible_worker_but_survives_when_none_meet_ttft() {
        let config = SMetricConfig {
            c_lin: 1.0,
            c_att: 0.0,
            prefill_rate: Some(1.0),
            slack: 1.0,
            hit_ratio: 0.5,
            ttft_slo_base: 20.0,
            ttft_slo_per_char: 0.0,
            max_tree_size: 1000,
        };
        let policy = SMetricPolicy::new(config);
        let workers: Vec<Arc<dyn Worker>> = ["http://one", "http://two"]
            .into_iter()
            .map(|url| {
                Arc::new(BasicWorker::new(url.into(), WorkerType::Regular)) as Arc<dyn Worker>
            })
            .collect();
        assert_eq!(
            select(&policy, &workers, &prompt("shared", 0), false, false).0,
            0
        );
        let first = PrefillCharge::new(workers[0].clone(), 100);
        assert_eq!(
            select(&policy, &workers, &prompt("shared plus", 6), true, false).0,
            1
        );
        let second = PrefillCharge::new(workers[1].clone(), 1000);
        // Both exceed TTFT. Worker 1 has the longest history even though its q is higher.
        assert_eq!(
            select(
                &policy,
                &workers,
                &prompt("shared plus more", 11),
                true,
                false,
            )
            .0,
            1
        );
        drop(first);
        drop(second);
        assert_eq!(workers[0].pending_prefill_work(), 0);
        assert_eq!(workers[1].pending_prefill_work(), 0);
    }

    #[test]
    fn colocated_balance_counts_inflight_requests_but_pd_prefill_does_not() {
        let config = SMetricConfig {
            c_lin: 1.0,
            c_att: 0.0,
            prefill_rate: Some(1.0),
            slack: 1.0,
            hit_ratio: 0.5,
            ttft_slo_base: 10.0,
            ttft_slo_per_char: 0.0,
            max_tree_size: 1000,
        };
        let workers: Vec<Arc<dyn Worker>> = ["http://one", "http://two"]
            .into_iter()
            .map(|url| {
                Arc::new(BasicWorker::new(url.into(), WorkerType::Regular)) as Arc<dyn Worker>
            })
            .collect();
        for _ in 0..3 {
            workers[0].increment_load();
        }
        workers[1].increment_load();
        let prompt = prompt("first turn", 0);
        assert_eq!(
            select(
                &SMetricPolicy::new(config.clone()),
                &workers,
                &prompt,
                false,
                true
            )
            .0,
            1
        );
        assert_eq!(
            select(&SMetricPolicy::new(config), &workers, &prompt, false, false).0,
            0
        );
    }

    #[test]
    fn learned_rate_is_used_only_without_a_configured_rate() {
        let config = SMetricConfig {
            c_lin: 1.0,
            c_att: 0.0,
            prefill_rate: None,
            slack: 1.0,
            hit_ratio: 0.5,
            ttft_slo_base: 20.0,
            ttft_slo_per_char: 0.0,
            max_tree_size: 1000,
        };
        let workers: Vec<Arc<dyn Worker>> = ["http://one", "http://two"]
            .into_iter()
            .map(|url| {
                Arc::new(BasicWorker::new(url.into(), WorkerType::Regular)) as Arc<dyn Worker>
            })
            .collect();
        let dynamic = SMetricPolicy::new(config.clone());
        assert_eq!(
            select(&dynamic, &workers, &prompt("shared", 0), false, false).0,
            0
        );
        let queued = PrefillCharge::new(workers[0].clone(), 100);
        let follow_up = prompt("shared plus", 6);
        assert_eq!(select(&dynamic, &workers, &follow_up, true, false).0, 0);
        let rate_history = dynamic.rates.get(workers[0].url()).unwrap();
        for _ in 0..RATE_MIN_SAMPLES {
            rate_history.lock().record(1.0);
        }
        drop(rate_history);
        assert_eq!(select(&dynamic, &workers, &follow_up, true, false).0, 1);
        drop(queued);

        let fixed = SMetricPolicy::new(SMetricConfig {
            prefill_rate: Some(1000.0),
            ..config
        });
        assert_eq!(
            select(&fixed, &workers, &prompt("shared", 0), false, false).0,
            0
        );
        let fixed_queue = PrefillCharge::new(workers[0].clone(), 100);
        // The configured rate remains authoritative even if observed rates exist.
        fixed.rates.insert(
            workers[0].url().into(),
            Arc::new(Mutex::new(RateHistory {
                samples: VecDeque::new(),
                rate: Some(1.0),
            })),
        );
        assert_eq!(select(&fixed, &workers, &follow_up, true, false).0, 0);
        drop(fixed_queue);
    }

    #[test]
    fn first_token_releases_pending_work_and_trains_only_unconfigured_rates() {
        let config = SMetricConfig {
            c_lin: 1.0,
            c_att: 0.0,
            prefill_rate: None,
            slack: 1.0,
            hit_ratio: 0.5,
            ttft_slo_base: 20.0,
            ttft_slo_per_char: 0.0,
            max_tree_size: 1000,
        };
        let worker: Arc<dyn Worker> =
            Arc::new(BasicWorker::new("http://one".into(), WorkerType::Regular));
        let workers = vec![worker.clone()];
        let learned = SMetricPolicy::new(config.clone());
        for i in 0..RATE_MIN_SAMPLES {
            let sample = prompt(&format!("prefill-{i}"), 0);
            let (_, mut tracker) = select(&learned, &workers, &sample, false, true);
            assert!(worker.pending_prefill_work() > 0);
            tracker.on_first_token();
            tracker.on_first_token();
            assert_eq!(worker.pending_prefill_work(), 0);
            assert_eq!(
                learned
                    .rates
                    .get(worker.url())
                    .unwrap()
                    .lock()
                    .rate
                    .is_some(),
                i + 1 == RATE_MIN_SAMPLES
            );
        }

        let fixed = SMetricPolicy::new(SMetricConfig {
            prefill_rate: Some(1000.0),
            ..config
        });
        let (_, mut tracker) = select(&fixed, &workers, &prompt("fixed prefill", 0), false, true);
        tracker.on_first_token();
        assert_eq!(worker.pending_prefill_work(), 0);
        assert!(fixed.rates.is_empty());

        let (_, tracker) = select(
            &learned,
            &workers,
            &prompt("cancelled prefill", 0),
            false,
            true,
        );
        drop(tracker);
        assert_eq!(worker.pending_prefill_work(), 0);
    }
}
