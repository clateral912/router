//! SMetric Figure 13: session cache affinity with a prefill-work fallback.
use super::{LoadBalancingPolicy, RequestHeaders};
use crate::config::SMetricConfig;
use crate::core::Worker;
use crate::metrics::RouterMetrics;
use crate::protocols::spec::SMetricPrompt;
use crate::tree::Tree;
use dashmap::DashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

#[derive(Debug)]
pub struct SMetricPolicy {
    config: SMetricConfig,
    trees: Arc<DashMap<String, Arc<Tree>>>,
    round_robin: AtomicUsize,
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
            round_robin: AtomicUsize::new(0),
        }
    }

    pub fn select_prefill(
        &self,
        workers: &[Arc<dyn Worker>],
        prompt: &SMetricPrompt,
        passes_turn_gate: bool,
    ) -> Option<(usize, u64)> {
        let l = prompt.text.chars().count();
        let start = self.round_robin.fetch_add(1, Ordering::Relaxed) % workers.len().max(1);
        let mut min: Option<(usize, u64, u64)> = None; // index, score, own cost
        let mut prev: Option<(usize, usize, u64, bool)> = None; // index, hit, own cost, TTFT
        let mut any_meets_ttft = false;
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
            let score = worker.pending_prefill_work().saturating_add(cost);
            let meets = (score as f64) / self.config.prefill_rate
                <= self.config.slack
                    * (self.config.ttft_slo_base + self.config.ttft_slo_per_char * l as f64);
            any_meets_ttft |= meets;
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
        Some((idx, cost))
    }
}

impl LoadBalancingPolicy for SMetricPolicy {
    fn select_worker_with_headers(
        &self,
        _workers: &[Arc<dyn Worker>],
        _request_text: Option<&str>,
        _headers: Option<&RequestHeaders>,
    ) -> Option<usize> {
        // Generic routing lacks the request's historical boundary and turn gate.
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

    #[test]
    fn cache_affinity_yields_to_a_feasible_worker_but_survives_when_none_meet_ttft() {
        let config = SMetricConfig {
            c_lin: 1.0,
            c_att: 0.0,
            prefill_rate: 1.0,
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
            policy
                .select_prefill(&workers, &prompt("shared", 0), false)
                .unwrap()
                .0,
            0
        );
        let first = PrefillCharge::new(workers[0].clone(), 100);
        assert_eq!(
            policy
                .select_prefill(&workers, &prompt("shared plus", 6), true)
                .unwrap()
                .0,
            1
        );
        let second = PrefillCharge::new(workers[1].clone(), 1000);
        // Both exceed TTFT. Worker 1 has the longest history even though its q is higher.
        assert_eq!(
            policy
                .select_prefill(&workers, &prompt("shared plus more", 11), true)
                .unwrap()
                .0,
            1
        );
        drop(first);
        drop(second);
        assert_eq!(workers[0].pending_prefill_work(), 0);
        assert_eq!(workers[1].pending_prefill_work(), 0);
    }
}
