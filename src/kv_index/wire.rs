//! msgpack wire types mirroring vLLM's KV cache events.

use serde::{Deserialize, Serialize};

/// Engine-emitted block hash — `bytes` or `int` on the wire. Opaque key for
/// parent chaining and `remove` lookup only.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ExternalBlockHash {
    Bytes(Vec<u8>),
    Int(u64),
}

impl<'de> Deserialize<'de> for ExternalBlockHash {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::{self, Visitor};
        use std::fmt;

        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = ExternalBlockHash;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("bytes or int")
            }
            fn visit_bytes<E: de::Error>(self, v: &[u8]) -> Result<Self::Value, E> {
                Ok(ExternalBlockHash::Bytes(v.to_vec()))
            }
            fn visit_borrowed_bytes<E: de::Error>(self, v: &'de [u8]) -> Result<Self::Value, E> {
                Ok(ExternalBlockHash::Bytes(v.to_vec()))
            }
            fn visit_byte_buf<E: de::Error>(self, v: Vec<u8>) -> Result<Self::Value, E> {
                Ok(ExternalBlockHash::Bytes(v))
            }
            fn visit_u64<E: de::Error>(self, v: u64) -> Result<Self::Value, E> {
                Ok(ExternalBlockHash::Int(v))
            }
            fn visit_i64<E: de::Error>(self, v: i64) -> Result<Self::Value, E> {
                // Negative would alias (−1 → u64::MAX collides in the anchor
                // map); reject as a wire error rather than silently alias.
                if v < 0 {
                    return Err(E::custom(format!("negative block hash: {v}")));
                }
                Ok(ExternalBlockHash::Int(v as u64))
            }
        }
        deserializer.deserialize_any(V)
    }
}

impl Serialize for ExternalBlockHash {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            ExternalBlockHash::Bytes(b) => serializer.serialize_bytes(b),
            ExternalBlockHash::Int(i) => serializer.serialize_u64(*i),
        }
    }
}

/// A KV cache event. vLLM tags each with the class name under `"type"` (map mode)
/// or as the leading element of a positional `msgspec.Struct(array_like=True, tag=True)` array.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum KVEvent {
    #[serde(rename = "BlockStored")]
    BlockStored(BlockStored),
    #[serde(rename = "BlockRemoved")]
    BlockRemoved(BlockRemoved),
    #[serde(rename = "AllBlocksCleared")]
    AllBlocksCleared(AllBlocksCleared),
}

struct WireKVEvent(KVEvent);

impl<'de> Deserialize<'de> for WireKVEvent {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::{self, MapAccess, SeqAccess, Visitor};
        use std::fmt;

        struct KVEventVisitor;
        impl<'de> Visitor<'de> for KVEventVisitor {
            type Value = WireKVEvent;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a tagged KVEvent map or vLLM msgspec array-like KVEvent")
            }

            fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<Self::Value, M::Error> {
                let mut entries = Vec::new();
                while let Some((k, v)) = map.next_entry::<rmpv::Value, rmpv::Value>()? {
                    if !v.is_nil() {
                        entries.push((k, v));
                    }
                }
                let cleaned = rmpv::Value::Map(entries);
                let ev = rmpv::ext::from_value::<KVEvent>(cleaned).map_err(de::Error::custom)?;
                Ok(WireKVEvent(ev))
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let tag: rmpv::Value = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::custom("empty KVEvent array"))?;
                let is_stored = tag.as_str() == Some("BlockStored") || tag.as_u64() == Some(0);
                let is_removed = tag.as_str() == Some("BlockRemoved") || tag.as_u64() == Some(1);
                let is_cleared =
                    tag.as_str() == Some("AllBlocksCleared") || tag.as_u64() == Some(2);
                if is_stored {
                    let block_hashes: Vec<ExternalBlockHash> = seq
                        .next_element()?
                        .ok_or_else(|| de::Error::custom("missing block_hashes"))?;
                    let parent_block_hash: Option<ExternalBlockHash> =
                        seq.next_element()?.unwrap_or(None);
                    let token_ids: Vec<u32> = seq
                        .next_element()?
                        .ok_or_else(|| de::Error::custom("missing token_ids"))?;
                    let block_size: u32 = seq
                        .next_element()?
                        .ok_or_else(|| de::Error::custom("missing block_size"))?;
                    let lora_id: Option<i64> = seq.next_element()?.unwrap_or(None);
                    let medium: Option<String> = seq.next_element()?.unwrap_or(None);
                    let lora_name: Option<String> = seq.next_element()?.unwrap_or(None);
                    let extra_keys: Option<Vec<rmpv::Value>> = seq.next_element()?.unwrap_or(None);
                    while let Some(_ignored) = seq.next_element::<rmpv::Value>()? {}
                    Ok(WireKVEvent(KVEvent::BlockStored(BlockStored {
                        extra_keys,
                        block_hashes,
                        parent_block_hash,
                        token_ids,
                        block_size,
                        lora_id,
                        medium,
                        lora_name,
                        group_idx: None,
                        kv_cache_spec_kind: None,
                        kv_cache_spec_sliding_window: None,
                        locality: None,
                        ownership: None,
                        session_id: None,
                    })))
                } else if is_removed {
                    let block_hashes: Vec<ExternalBlockHash> = seq
                        .next_element()?
                        .ok_or_else(|| de::Error::custom("missing block_hashes"))?;
                    let medium: Option<String> = seq.next_element()?.unwrap_or(None);
                    while let Some(_ignored) = seq.next_element::<rmpv::Value>()? {}
                    Ok(WireKVEvent(KVEvent::BlockRemoved(BlockRemoved {
                        block_hashes,
                        medium,
                        group_idx: None,
                        locality: None,
                        ownership: None,
                    })))
                } else if is_cleared {
                    while let Some(_ignored) = seq.next_element::<rmpv::Value>()? {}
                    Ok(WireKVEvent(KVEvent::AllBlocksCleared(AllBlocksCleared {})))
                } else {
                    Err(de::Error::custom(format!("unknown KVEvent tag: {tag:?}")))
                }
            }
        }

        deserializer.deserialize_any(KVEventVisitor)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockStored {
    /// Opaque, lossless MessagePack values. Structural acceptance does not
    /// establish extension semantics or device eligibility for consumers.
    #[serde(default)]
    pub extra_keys: Option<Vec<rmpv::Value>>,
    pub block_hashes: Vec<ExternalBlockHash>,
    pub parent_block_hash: Option<ExternalBlockHash>,
    pub token_ids: Vec<u32>,
    pub block_size: u32,
    pub lora_id: Option<i64>,
    pub medium: Option<String>,
    pub lora_name: Option<String>,
    #[serde(default)]
    pub group_idx: Option<u32>,
    #[serde(default)]
    pub kv_cache_spec_kind: Option<String>,
    #[serde(default)]
    pub kv_cache_spec_sliding_window: Option<u32>,
    #[serde(default)]
    pub locality: Option<String>,
    #[serde(default)]
    pub ownership: Option<String>,
    #[serde(default)]
    pub session_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockRemoved {
    pub block_hashes: Vec<ExternalBlockHash>,
    pub medium: Option<String>,
    #[serde(default)]
    pub group_idx: Option<u32>,
    #[serde(default)]
    pub locality: Option<String>,
    #[serde(default)]
    pub ownership: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AllBlocksCleared {}

/// A batch of events: a positional array `[ts, events, (data_parallel_rank)?]`
/// (`array_like=True`); `seq` rides in its own ZMQ frame. Derived `Deserialize`
/// reads maps, so a custom positional impl is required. `Serialize` is
/// intentionally absent — inbound-only, and a derived `Serialize` would emit
/// a map, not the positional array.
#[derive(Debug, Clone)]
pub struct KVEventBatch {
    pub ts: f64,
    pub events: Vec<KVEvent>,
    pub data_parallel_rank: Option<u32>,
}

impl<'de> Deserialize<'de> for KVEventBatch {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::{self, SeqAccess, Visitor};
        use std::fmt;

        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = KVEventBatch;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a 2- or 3-element array [ts, events, (rank)?]")
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let ts: f64 = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::custom("missing ts"))?;
                let wire_events: Vec<WireKVEvent> = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::custom("missing events"))?;
                let events = wire_events.into_iter().map(|w| w.0).collect();
                let data_parallel_rank: Option<u32> = seq.next_element()?;
                Ok(KVEventBatch {
                    ts,
                    events,
                    data_parallel_rank,
                })
            }
        }
        deserializer.deserialize_seq(V)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rmp_serde::{from_slice, to_vec};

    #[test]
    fn block_stored_with_int_hashes_round_trips() {
        let ev = KVEvent::BlockStored(BlockStored {
            extra_keys: None,
            block_hashes: vec![ExternalBlockHash::Int(42), ExternalBlockHash::Int(43)],
            parent_block_hash: Some(ExternalBlockHash::Int(42)),
            token_ids: vec![1, 2, 3, 4],
            block_size: 4,
            lora_id: None,
            medium: Some("device".into()),
            lora_name: None,
            group_idx: Some(0),
            kv_cache_spec_kind: Some("full".into()),
            kv_cache_spec_sliding_window: None,
            locality: Some("LOCAL".into()),
            ownership: None,
            session_id: None,
        });
        let buf = to_vec(&ev).unwrap();
        match from_slice::<KVEvent>(&buf).unwrap() {
            KVEvent::BlockStored(b) => {
                assert_eq!(b.block_hashes.len(), 2);
                assert_eq!(b.block_size, 4);
                assert_eq!(b.token_ids, vec![1, 2, 3, 4]);
                assert_eq!(b.group_idx, Some(0));
            }
            _ => panic!("expected BlockStored"),
        }
    }

    #[test]
    fn block_stored_with_byte_hashes_round_trips() {
        let ev = KVEvent::BlockStored(BlockStored {
            extra_keys: None,
            block_hashes: vec![ExternalBlockHash::Bytes(vec![0u8; 32])],
            parent_block_hash: None,
            token_ids: vec![5],
            block_size: 16,
            lora_id: None,
            medium: None,
            lora_name: None,
            group_idx: None,
            kv_cache_spec_kind: None,
            kv_cache_spec_sliding_window: None,
            locality: None,
            ownership: None,
            session_id: None,
        });
        let buf = to_vec(&ev).unwrap();
        match from_slice::<KVEvent>(&buf).unwrap() {
            KVEvent::BlockStored(b) => match &b.block_hashes[0] {
                ExternalBlockHash::Bytes(b) => assert_eq!(b.len(), 32),
                _ => panic!("expected bytes hash"),
            },
            _ => panic!("expected BlockStored"),
        }
    }

    #[test]
    fn block_removed_optionals_default_when_missing() {
        #[derive(Serialize)]
        #[serde(tag = "type")]
        enum Tag<'a> {
            #[serde(rename = "BlockRemoved")]
            BlockRemoved(&'a BlockRemoved),
        }
        let minimal = BlockRemoved {
            block_hashes: vec![ExternalBlockHash::Int(42)],
            medium: None,
            group_idx: None,
            locality: None,
            ownership: None,
        };
        let buf = to_vec(&Tag::BlockRemoved(&minimal)).unwrap();
        match from_slice::<KVEvent>(&buf).unwrap() {
            KVEvent::BlockRemoved(b) => {
                assert_eq!(b.block_hashes.len(), 1);
                assert!(b.group_idx.is_none());
                assert!(b.locality.is_none());
                assert!(b.ownership.is_none());
            }
            _ => panic!("expected BlockRemoved"),
        }
    }

    #[test]
    fn all_blocks_cleared_round_trips() {
        let ev = KVEvent::AllBlocksCleared(AllBlocksCleared {});
        let buf = to_vec(&ev).unwrap();
        assert!(matches!(
            from_slice::<KVEvent>(&buf).unwrap(),
            KVEvent::AllBlocksCleared(_)
        ));
    }

    #[test]
    fn batch_decodes_two_element_array() {
        let buf = to_vec(&(1.5f64, Vec::<KVEvent>::new())).unwrap();
        let back: KVEventBatch = from_slice(&buf).unwrap();
        assert_eq!(back.ts, 1.5);
        assert!(back.events.is_empty());
        assert!(back.data_parallel_rank.is_none());
    }

    #[test]
    fn batch_decodes_three_element_array() {
        let buf = to_vec(&(2.5f64, Vec::<KVEvent>::new(), 1u32)).unwrap();
        let back: KVEventBatch = from_slice(&buf).unwrap();
        assert_eq!(back.ts, 2.5);
        assert_eq!(back.data_parallel_rank, Some(1));
    }

    #[test]
    fn vllm_018_positional_array_batch_decodes() {
        // Exact msgspec.Struct(array_like=True, tag=True) bytes produced by vLLM 0.18.1:
        // [1.0, [["BlockStored", [123], None, [1, 2, 3], 16, None, "GPU", None, None],
        //        ["BlockRemoved", [123], "GPU"],
        //        ["AllBlocksCleared"]], 0]
        let raw: &[u8] = &[
            0x93, 0xcb, 0x3f, 0xf0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x93, 0x99, 0xab, 0x42,
            0x6c, 0x6f, 0x63, 0x6b, 0x53, 0x74, 0x6f, 0x72, 0x65, 0x64, 0x91, 0x7b, 0xc0, 0x93,
            0x01, 0x02, 0x03, 0x10, 0xc0, 0xa3, 0x47, 0x50, 0x55, 0xc0, 0xc0, 0x93, 0xac, 0x42,
            0x6c, 0x6f, 0x63, 0x6b, 0x52, 0x65, 0x6d, 0x6f, 0x76, 0x65, 0x64, 0x91, 0x7b, 0xa3,
            0x47, 0x50, 0x55, 0x91, 0xb0, 0x41, 0x6c, 0x6c, 0x42, 0x6c, 0x6f, 0x63, 0x6b, 0x73,
            0x43, 0x6c, 0x65, 0x61, 0x72, 0x65, 0x64, 0x00,
        ];
        let batch: KVEventBatch = from_slice(raw).unwrap();
        assert_eq!(batch.ts, 1.0);
        assert_eq!(batch.data_parallel_rank, Some(0));
        assert_eq!(batch.events.len(), 3);
        match &batch.events[0] {
            KVEvent::BlockStored(b) => {
                assert_eq!(b.block_hashes, vec![ExternalBlockHash::Int(123)]);
                assert!(b.parent_block_hash.is_none());
                assert_eq!(b.token_ids, vec![1, 2, 3]);
                assert_eq!(b.block_size, 16);
                assert_eq!(b.medium.as_deref(), Some("GPU"));
            }
            other => panic!("expected BlockStored, got {other:?}"),
        }
        match &batch.events[1] {
            KVEvent::BlockRemoved(b) => {
                assert_eq!(b.block_hashes, vec![ExternalBlockHash::Int(123)]);
                assert_eq!(b.medium.as_deref(), Some("GPU"));
            }
            other => panic!("expected BlockRemoved, got {other:?}"),
        }
        assert!(matches!(batch.events[2], KVEvent::AllBlocksCleared(_)));
    }
}
