use crate::kv_index::{Locality, StorageTier};
use crate::policies::CandidateView;

#[derive(Clone, Copy, Debug, Default)]
pub(in crate::policies) struct PrefixMatch {
    pub chars: usize,
    pub tokens: usize,
    pub observed: bool,
}

pub(in crate::policies) fn prefix_match(
    candidate: &CandidateView<'_>,
    block_size: u32,
    chars: usize,
) -> PrefixMatch {
    let n = candidate.input_tokens;
    let b = block_size as usize;
    if n == 0 || b == 0 {
        return PrefixMatch::default();
    }
    let blocks = candidate
        .matches
        .iter()
        .filter(|hit| {
            hit.target == *candidate.target
                && hit.tier == StorageTier::Device
                && hit.locality == Locality::Local
        })
        .flat_map(|hit| hit.segments.iter())
        .filter(|(start, _)| *start == 0)
        .map(|(_, end)| *end as usize)
        .max()
        .unwrap_or(0);
    let tokens = blocks.min(n.saturating_sub(1) / b) * b;
    PrefixMatch {
        chars: ((chars as u128 * tokens as u128) / n as u128) as usize,
        tokens,
        observed: blocks > 0,
    }
}
