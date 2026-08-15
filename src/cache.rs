//! # Response cache — pure-function call deduplication
//!
//! Embedding endpoints
//! (`text → vector` is a function), image generation with deterministic
//! seeds, and chat with `temperature: 0` + no tools all return the
//! same bytes for the same inputs. The cache keys on the canonical
//! request shape and serves repeat calls without hitting upstream.
//!
//! ## Trait surface
//!
//! [`ResponseCache`] is the gateway-side surface; bindings reach for
//! it through the engine, never directly. The default implementation
//! is [`LruResponseCache`] — `Arc<RwLock<LruCache<…>>>` with byte
//! cap + per-entry TTL. Future Redis / memcached backends fit the
//! same trait.
//!
//! ## Key composition
//!
//! Cache keys are BLAKE3 hashes of all cache-relevant inputs.
//! [`CacheKey::for_chat`] hashes the binding identity, model, full
//! rendered system + user prompts, the tools signature, and sampling
//! settings — anything that would change the upstream's deterministic
//! output. [`CacheKey::for_embedding`] hashes binding identity, model,
//! input text, and dimensions.

use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use bytes::Bytes;
use lru::LruCache;
use tokio::sync::RwLock;

use crate::chat_config::SamplingSpec;

/// Composed BLAKE3 hash of all cache-relevant inputs. The hash is
/// what's stored as the underlying map key; the wrapper exists so the
/// caller can't accidentally cache against a non-canonical string.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheKey {
    hash: String,
}

impl CacheKey {
    /// Compose from a chat binding's request shape. Hashes binding
    /// identity, model, the *fully rendered* system + user text, the
    /// tools signature (sorted list of allowed names), and the
    /// sampling spec. Any change in any of those produces a different
    /// key, so the cache is sound for deterministic chat
    /// (`temperature: 0` + no tools); operators must opt out for
    /// non-deterministic configurations.
    pub fn for_chat(
        backend_name: &str,
        provider: &str,
        model: &str,
        rendered_system: &str,
        rendered_user: &str,
        tools_signature: &str,
        sampling: &SamplingSpec,
    ) -> Self {
        let mut h = blake3::Hasher::new();
        h.update(b"chat\0");
        h.update(backend_name.as_bytes());
        h.update(b"\0");
        h.update(provider.as_bytes());
        h.update(b"\0");
        h.update(model.as_bytes());
        h.update(b"\0");
        h.update(rendered_system.as_bytes());
        h.update(b"\0");
        h.update(rendered_user.as_bytes());
        h.update(b"\0");
        h.update(tools_signature.as_bytes());
        h.update(b"\0");
        // Sampling settings — finalize-into-bytes manually so we don't
        // depend on the serde shape. `0_f32::to_le_bytes` is stable
        // across Rust releases; `Option::None` hashes to a sentinel
        // byte distinct from any concrete value.
        write_opt_f32(&mut h, sampling.temperature);
        write_opt_f32(&mut h, sampling.top_p);
        write_opt_u32(&mut h, sampling.max_completion_tokens);
        write_opt_i64(&mut h, sampling.seed);
        Self {
            hash: hex::encode(h.finalize().as_bytes()),
        }
    }

    /// Compose from an embedding binding's request shape. The
    /// embedding endpoint is a pure function `text → vector`, so the
    /// only inputs are the binding identity, model, the literal input
    /// text, and the optional dimensionality reduction.
    pub fn for_embedding(
        backend_name: &str,
        provider: &str,
        model: &str,
        input_text: &str,
        dimensions: Option<u32>,
    ) -> Self {
        let mut h = blake3::Hasher::new();
        h.update(b"embed\0");
        h.update(backend_name.as_bytes());
        h.update(b"\0");
        h.update(provider.as_bytes());
        h.update(b"\0");
        h.update(model.as_bytes());
        h.update(b"\0");
        h.update(input_text.as_bytes());
        h.update(b"\0");
        write_opt_u32(&mut h, dimensions);
        Self {
            hash: hex::encode(h.finalize().as_bytes()),
        }
    }

    /// Test/admin escape hatch: build a key from a pre-computed hash
    /// string. Most callers should use the `for_*` constructors so
    /// composition is uniform.
    pub fn from_hash(hash: impl Into<String>) -> Self {
        Self { hash: hash.into() }
    }

    pub fn as_str(&self) -> &str {
        &self.hash
    }
}

fn write_opt_f32(h: &mut blake3::Hasher, v: Option<f32>) {
    match v {
        Some(x) => {
            h.update(b"\x01");
            h.update(&x.to_le_bytes());
        }
        None => {
            h.update(b"\x00");
        }
    };
}
fn write_opt_u32(h: &mut blake3::Hasher, v: Option<u32>) {
    match v {
        Some(x) => {
            h.update(b"\x01");
            h.update(&x.to_le_bytes());
        }
        None => {
            h.update(b"\x00");
        }
    };
}
fn write_opt_i64(h: &mut blake3::Hasher, v: Option<i64>) {
    match v {
        Some(x) => {
            h.update(b"\x01");
            h.update(&x.to_le_bytes());
        }
        None => {
            h.update(b"\x00");
        }
    };
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CacheStats {
    pub item_count: u64,
    pub byte_count: u64,
    pub max_bytes: u64,
    /// Lifetime totals — increment on every get/put, never reset.
    /// Operators wanting per-window rates derive them from Prometheus
    /// counter increments.
    pub hits: u64,
    pub misses: u64,
}

#[async_trait]
pub trait ResponseCache: Send + Sync + std::fmt::Debug {
    /// Fetch by key. Returns `None` on miss / expired / evicted —
    /// callers re-issue upstream and (typically) `put` the result.
    async fn get(&self, key: &CacheKey) -> Option<Bytes>;

    /// Store. `ttl` is a hint: the implementation may evict earlier
    /// under memory pressure (LRU) and may keep entries longer when
    /// content is hash-deduplicated.
    async fn put(&self, key: CacheKey, value: Bytes, ttl: Duration);

    /// Evict explicitly. Idempotent.
    async fn invalidate(&self, key: &CacheKey);

    /// Snapshot of utilisation. Surfaced as Prometheus gauges by the
    /// engine.
    fn stats(&self) -> CacheStats;
}

// ---------------------------------------------------------------------------
// In-process LRU + TTL implementation
// ---------------------------------------------------------------------------

/// Default in-process cache. Same shape as
/// [`crate::content_store::InProcessContentStore`] — `Arc<RwLock<LruCache<…>>>`
/// with byte cap + per-entry TTL evicted lazily on read. Volatile —
/// loses contents on process restart. Suitable for single-node
/// deployments without persistence requirements.
pub struct LruResponseCache {
    inner: Arc<RwLock<Inner>>,
    max_bytes: usize,
}

impl std::fmt::Debug for LruResponseCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LruResponseCache")
            .field("max_bytes", &self.max_bytes)
            .finish()
    }
}

struct Inner {
    map: LruCache<String, Entry>,
    byte_count: usize,
    hits: u64,
    misses: u64,
}

struct Entry {
    bytes: Bytes,
    expires_at: Option<Instant>,
}

impl LruResponseCache {
    pub fn new(max_bytes: usize) -> Arc<Self> {
        // Same entry-cap rationale as the content store: large enough
        // that byte cap is the operative limit.
        let entry_cap = NonZeroUsize::new(1_048_576).expect("1 << 20 is non-zero");
        Arc::new(Self {
            inner: Arc::new(RwLock::new(Inner {
                map: LruCache::new(entry_cap),
                byte_count: 0,
                hits: 0,
                misses: 0,
            })),
            max_bytes,
        })
    }

    fn evict_to_fit(inner: &mut Inner, max_bytes: usize) {
        if max_bytes == 0 {
            return;
        }
        while inner.byte_count > max_bytes {
            let Some((_, entry)) = inner.map.pop_lru() else {
                break;
            };
            inner.byte_count = inner.byte_count.saturating_sub(entry.bytes.len());
        }
    }
}

#[async_trait]
impl ResponseCache for LruResponseCache {
    async fn get(&self, key: &CacheKey) -> Option<Bytes> {
        let mut inner = self.inner.write().await;
        // Treat reads as access for LRU purposes: hot keys stay
        // resident, cold ones drift toward eviction.
        let Some(entry) = inner.map.get_mut(key.as_str()) else {
            inner.misses = inner.misses.saturating_add(1);
            return None;
        };
        if let Some(t) = entry.expires_at
            && t <= Instant::now()
        {
            // Lazy expire — drop now so a subsequent put can refresh.
            let entry = inner.map.pop(key.as_str()).expect("just observed");
            inner.byte_count = inner.byte_count.saturating_sub(entry.bytes.len());
            inner.misses = inner.misses.saturating_add(1);
            return None;
        }
        let bytes = entry.bytes.clone();
        inner.hits = inner.hits.saturating_add(1);
        Some(bytes)
    }

    async fn put(&self, key: CacheKey, value: Bytes, ttl: Duration) {
        if self.max_bytes > 0 && value.len() > self.max_bytes {
            // Single entry larger than the whole cap — refuse rather
            // than evict everything. Caller treats as a cache miss.
            return;
        }
        let mut inner = self.inner.write().await;
        let expires_at = if ttl.is_zero() {
            None
        } else {
            Some(Instant::now() + ttl)
        };
        let entry = Entry {
            bytes: value.clone(),
            expires_at,
        };
        if let Some(prior) = inner.map.put(key.hash.clone(), entry) {
            inner.byte_count = inner.byte_count.saturating_sub(prior.bytes.len());
        }
        inner.byte_count = inner.byte_count.saturating_add(value.len());
        Self::evict_to_fit(&mut inner, self.max_bytes);
    }

    async fn invalidate(&self, key: &CacheKey) {
        let mut inner = self.inner.write().await;
        if let Some(entry) = inner.map.pop(key.as_str()) {
            inner.byte_count = inner.byte_count.saturating_sub(entry.bytes.len());
        }
    }

    fn stats(&self) -> CacheStats {
        match self.inner.try_read() {
            Ok(inner) => CacheStats {
                item_count: inner.map.len() as u64,
                byte_count: inner.byte_count as u64,
                max_bytes: self.max_bytes as u64,
                hits: inner.hits,
                misses: inner.misses,
            },
            Err(_) => CacheStats {
                max_bytes: self.max_bytes as u64,
                ..Default::default()
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sampling() -> SamplingSpec {
        SamplingSpec::default()
    }

    #[test]
    fn for_chat_keys_diverge_on_inputs() {
        let a = CacheKey::for_chat("b", "openai", "gpt-4o-mini", "sys", "u1", "[]", &sampling());
        let b = CacheKey::for_chat("b", "openai", "gpt-4o-mini", "sys", "u2", "[]", &sampling());
        assert_ne!(a, b);

        // Different model
        let c = CacheKey::for_chat("b", "openai", "gpt-4o", "sys", "u1", "[]", &sampling());
        assert_ne!(a, c);

        // Different binding name
        let d = CacheKey::for_chat("c", "openai", "gpt-4o-mini", "sys", "u1", "[]", &sampling());
        assert_ne!(a, d);
    }

    #[test]
    fn for_chat_same_inputs_same_key() {
        let a = CacheKey::for_chat("b", "openai", "gpt-4o-mini", "sys", "u1", "[]", &sampling());
        let b = CacheKey::for_chat("b", "openai", "gpt-4o-mini", "sys", "u1", "[]", &sampling());
        assert_eq!(a, b);
    }

    #[test]
    fn for_chat_diverges_on_temperature() {
        let s_a = SamplingSpec {
            temperature: Some(0.0),
            ..Default::default()
        };
        let s_b = SamplingSpec {
            temperature: Some(0.7),
            ..Default::default()
        };
        let a = CacheKey::for_chat("b", "openai", "gpt-4o-mini", "sys", "u1", "[]", &s_a);
        let b = CacheKey::for_chat("b", "openai", "gpt-4o-mini", "sys", "u1", "[]", &s_b);
        assert_ne!(a, b);
    }

    #[test]
    fn for_embedding_keys_diverge_on_text_and_model() {
        let a = CacheKey::for_embedding("b", "openai", "ada-002", "hello", None);
        let b = CacheKey::for_embedding("b", "openai", "ada-002", "world", None);
        assert_ne!(a, b);
        let c = CacheKey::for_embedding("b", "openai", "3-small", "hello", None);
        assert_ne!(a, c);
    }

    #[test]
    fn for_embedding_dimensions_change_key() {
        let a = CacheKey::for_embedding("b", "openai", "3-small", "x", None);
        let b = CacheKey::for_embedding("b", "openai", "3-small", "x", Some(512));
        assert_ne!(a, b);
    }

    #[tokio::test]
    async fn put_get_round_trip() {
        let c = LruResponseCache::new(1024);
        let k = CacheKey::from_hash("k1");
        c.put(
            k.clone(),
            Bytes::from_static(b"hello"),
            Duration::from_secs(60),
        )
        .await;
        let got = c.get(&k).await.unwrap();
        assert_eq!(got.as_ref(), b"hello");
    }

    #[tokio::test]
    async fn miss_returns_none_and_increments_counter() {
        let c = LruResponseCache::new(1024);
        let stats0 = c.stats();
        let k = CacheKey::from_hash("nope");
        assert!(c.get(&k).await.is_none());
        let stats1 = c.stats();
        assert_eq!(stats1.misses, stats0.misses + 1);
        assert_eq!(stats1.hits, stats0.hits);
    }

    #[tokio::test]
    async fn hit_increments_counter() {
        let c = LruResponseCache::new(1024);
        let k = CacheKey::from_hash("hit");
        c.put(k.clone(), Bytes::from_static(b"v"), Duration::from_secs(60))
            .await;
        let _ = c.get(&k).await;
        assert_eq!(c.stats().hits, 1);
    }

    #[tokio::test]
    async fn ttl_expiry_returns_miss_after_deadline() {
        let c = LruResponseCache::new(1024);
        let k = CacheKey::from_hash("k");
        c.put(
            k.clone(),
            Bytes::from_static(b"v"),
            Duration::from_millis(50),
        )
        .await;
        assert!(c.get(&k).await.is_some());
        tokio::time::sleep(Duration::from_millis(80)).await;
        assert!(c.get(&k).await.is_none());
    }

    #[tokio::test]
    async fn ttl_zero_means_no_expiry() {
        let c = LruResponseCache::new(1024);
        let k = CacheKey::from_hash("k");
        c.put(k.clone(), Bytes::from_static(b"v"), Duration::ZERO)
            .await;
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(c.get(&k).await.is_some());
    }

    #[tokio::test]
    async fn invalidate_drops_entry() {
        let c = LruResponseCache::new(1024);
        let k = CacheKey::from_hash("k");
        c.put(k.clone(), Bytes::from_static(b"v"), Duration::from_secs(60))
            .await;
        assert!(c.get(&k).await.is_some());
        c.invalidate(&k).await;
        assert!(c.get(&k).await.is_none());
    }

    #[tokio::test]
    async fn lru_evicts_on_byte_cap_exceeded() {
        let c = LruResponseCache::new(20);
        let k1 = CacheKey::from_hash("a");
        let k2 = CacheKey::from_hash("b");
        let k3 = CacheKey::from_hash("c");
        c.put(
            k1.clone(),
            Bytes::from_static(b"AAAAAAAA"),
            Duration::from_secs(60),
        )
        .await;
        c.put(
            k2.clone(),
            Bytes::from_static(b"BBBBBBBB"),
            Duration::from_secs(60),
        )
        .await;
        // Touch k1 so k2 becomes the LRU candidate.
        let _ = c.get(&k1).await;
        c.put(
            k3.clone(),
            Bytes::from_static(b"CCCCCCCC"),
            Duration::from_secs(60),
        )
        .await;
        assert!(c.get(&k1).await.is_some());
        assert!(c.get(&k2).await.is_none());
        assert!(c.get(&k3).await.is_some());
    }

    #[tokio::test]
    async fn put_too_large_for_cap_is_dropped() {
        let c = LruResponseCache::new(8);
        let k = CacheKey::from_hash("k");
        // 16-byte payload, 8-byte cap → store refuses (treated as miss).
        c.put(
            k.clone(),
            Bytes::from_static(b"too big to fit!!"),
            Duration::from_secs(60),
        )
        .await;
        assert!(c.get(&k).await.is_none());
    }

    #[tokio::test]
    async fn stats_track_size_and_count() {
        let c = LruResponseCache::new(1024);
        c.put(
            CacheKey::from_hash("a"),
            Bytes::from_static(b"hello"),
            Duration::from_secs(60),
        )
        .await;
        c.put(
            CacheKey::from_hash("b"),
            Bytes::from_static(b"world!"),
            Duration::from_secs(60),
        )
        .await;
        let s = c.stats();
        assert_eq!(s.item_count, 2);
        assert_eq!(s.byte_count, 11);
        assert_eq!(s.max_bytes, 1024);
    }
}
