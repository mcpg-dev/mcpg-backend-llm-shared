//! # ContentStore — gateway-managed binary blob storage
//!
//! Solves the "where do we put
//! generated images / audio / large tool outputs?" problem without
//! forcing every binding to embed binary content inline in the JSON
//! response.
//!
//! ## Trait
//!
//! [`ContentStore`] is the gateway-side surface. Implementations:
//!
//! - [`InProcessContentStore`] — `Arc<RwLock<LruCache<…>>>`, content-
//!   addressed (BLAKE3), volatile, lost on restart. Default for
//!   single-node deployments.
//! - `FileSystemContentStore` — planned follow-up. SQLite metadata
//!   index + on-disk blobs + Ed25519-signed presigned URLs.
//! - `S3ContentStore` — planned follow-up. AWS-SDK / S3-compatible
//!   backends with native pre-signed URLs.
//!
//! ## Bindings reach this through `BackendHost`
//!
//! Plugins do not hold an `Arc<dyn ContentStore>` directly — they
//! call through [`mcpg_plugin_protocol::BackendHost::store_content`]
//! / `fetch_content`. The `GatewayBackendHost` impl plumbs both
//! child-tool dispatch AND the operator-configured store behind one
//! handle, so the trait surface stays narrow.
//!
//! ## Resource URI scheme
//!
//! Returned `ResourceHandle.uri` is `mcpg-resource://<id>` where
//! `<id>` is one of:
//!
//! - `hash:<blake3-256-hex>` — anonymous, content-addressed.
//! - `alias:<session-id>:<operator-name>` — operator-aliased within
//!   a session. The hash is still recorded internally.
//!
//! The `<id>` is opaque to clients — they pass it to `resources/read`.

use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use lru::LruCache;
use tokio::sync::RwLock;

// The ContentStore entity trait + its value types now live in the protocol
// crate (the canonical entity-class home, mirroring catalog/secret/etc.);
// re-exported here for source compatibility with existing importers.
// `InProcessContentStore` below implements the protocol trait.
pub use mcpg_plugin_protocol::content_store::{
    ContentStore, ContentStoreError, ContentStorePlugin, ContentStoreStats, ContentToStore,
    ResourceContent, ResourceHandle,
};

// ---------------------------------------------------------------------------
// In-process implementation
// ---------------------------------------------------------------------------

/// Default in-process LRU + TTL store. Content-addressed by BLAKE3.
///
/// - Capacity is bounded by `max_bytes` (LRU eviction when exceeded).
/// - Per-entry TTL is enforced eagerly on read (lazy sweep — no
///   background task at this layer; the gateway runs a periodic
///   sweep across all stores via [`InProcessContentStore::sweep_expired`]).
/// - Volatile — lost on process restart. Suitable for single-node
///   deployments without persistence requirements.
/// - `signed_url` returns [`ContentStoreError::SignedUrlNotSupported`]
///   so callers fall back to MCP `resources/read`.
pub struct InProcessContentStore {
    inner: Arc<RwLock<Inner>>,
    max_bytes: usize,
}

impl std::fmt::Debug for InProcessContentStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InProcessContentStore")
            .field("max_bytes", &self.max_bytes)
            .finish()
    }
}

struct Inner {
    /// Hash-keyed primary store. The LRU's eviction order is the
    /// access order tracked by the `lru` crate.
    by_hash: LruCache<String, Entry>,
    /// Alias → hash redirect. An alias is a stable name within a
    /// session (e.g. `incident-screenshot`) that resolves to whatever
    /// content was last stored under it — so callers can re-store
    /// under the same alias and not pile up duplicates.
    alias_to_hash: HashMap<String, String>,
    /// Live byte total. Updated on insert/evict/delete; checked
    /// against `max_bytes` after every put to drive LRU eviction.
    byte_count: usize,
}

#[derive(Clone)]
struct Entry {
    bytes: bytes::Bytes,
    mime_type: String,
    session_id: Option<String>,
    tenant_id: Option<String>,
    stored_at: chrono::DateTime<chrono::Utc>,
    expires_at: Option<chrono::DateTime<chrono::Utc>>,
}

impl InProcessContentStore {
    /// Build a new in-process store with the given byte cap.
    /// `max_bytes` of 0 is treated as "unlimited" — useful for tests;
    /// production deployments always pass a positive cap.
    pub fn new(max_bytes: usize) -> Arc<Self> {
        // The LRU `cap` controls *entry count*, not byte count, so we
        // pick something high enough that byte-cap eviction is the
        // operative limit. `usize::MAX` would work but the underlying
        // `LruCache::new(NonZeroUsize)` constructor needs a value, and
        // we want the cap reachable so eviction stays well-defined.
        // 1M entries is ample (256 MiB ÷ 256 bytes ≈ 1M entries; most
        // blobs are several KB+).
        let entry_cap = NonZeroUsize::new(1_048_576).expect("1 << 20 is non-zero");
        Arc::new(Self {
            inner: Arc::new(RwLock::new(Inner {
                by_hash: LruCache::new(entry_cap),
                alias_to_hash: HashMap::new(),
                byte_count: 0,
            })),
            max_bytes,
        })
    }

    fn evict_to_fit(inner: &mut Inner, max_bytes: usize) {
        if max_bytes == 0 {
            return;
        }
        while inner.byte_count > max_bytes {
            let Some((_, entry)) = inner.by_hash.pop_lru() else {
                break;
            };
            inner.byte_count = inner.byte_count.saturating_sub(entry.bytes.len());
        }
    }

    fn resolve_to_hash(inner: &Inner, id: &str) -> Option<String> {
        if let Some(rest) = id.strip_prefix("hash:") {
            // Stored hashes are bare hex; the prefix is a public
            // namespace marker.
            return Some(rest.to_owned());
        }
        if id.starts_with("alias:") {
            return inner.alias_to_hash.get(id).cloned();
        }
        // Bare hex (no namespace) is also accepted as a hash —
        // forgiving of legacy callers, but not encouraged.
        if !id.is_empty() && id.chars().all(|c| c.is_ascii_hexdigit()) {
            return Some(id.to_owned());
        }
        None
    }
}

#[async_trait]
impl ContentStore for InProcessContentStore {
    async fn put(&self, content: ContentToStore) -> Result<ResourceHandle, ContentStoreError> {
        let size = content.bytes.len();
        if self.max_bytes > 0 && size > self.max_bytes {
            return Err(ContentStoreError::SizeLimit {
                limit_bytes: self.max_bytes,
                actual_bytes: size,
            });
        }

        let hash_hex = hex::encode(blake3::hash(&content.bytes).as_bytes());
        let stored_at = chrono::Utc::now();
        let expires_at = content
            .ttl
            .and_then(|d| chrono::Duration::from_std(d).ok())
            .map(|d| stored_at + d);

        let entry = Entry {
            bytes: content.bytes.clone(),
            mime_type: content.mime_type.clone(),
            session_id: content.session_id.clone(),
            tenant_id: content.tenant_id.clone(),
            stored_at,
            expires_at,
        };

        let mut inner = self.inner.write().await;

        // Insert / replace by hash. If the hash already existed (dedup
        // hit), the byte_count delta is zero net — but we still
        // refresh the access time so the entry rides the LRU.
        let prior = inner.by_hash.put(hash_hex.clone(), entry);
        if let Some(prior) = prior {
            inner.byte_count = inner.byte_count.saturating_sub(prior.bytes.len());
        }
        inner.byte_count = inner.byte_count.saturating_add(size);

        // Wire up alias → hash redirect if the operator supplied one.
        // Aliases are session-scoped to avoid cross-session collisions;
        // a missing session_id falls back to an unscoped alias (rare).
        let id = if let Some(alias) = content.alias.as_deref() {
            let session = content.session_id.as_deref().unwrap_or("__no_session__");
            let alias_id = format!("alias:{session}:{alias}");
            inner
                .alias_to_hash
                .insert(alias_id.clone(), hash_hex.clone());
            alias_id
        } else {
            format!("hash:{hash_hex}")
        };

        Self::evict_to_fit(&mut inner, self.max_bytes);

        Ok(ResourceHandle {
            id: id.clone(),
            uri: format!("mcpg-resource://{id}"),
            size_bytes: size,
            mime_type: content.mime_type,
            expires_at,
            content_hash: format!("blake3:{hash_hex}"),
        })
    }

    async fn get(&self, id: &str) -> Result<Option<ResourceContent>, ContentStoreError> {
        let mut inner = self.inner.write().await;
        let Some(hash) = Self::resolve_to_hash(&inner, id) else {
            return Ok(None);
        };
        // `get_mut` updates LRU position; `peek_mut` would not. We
        // treat reads as access for LRU purposes since the content is
        // hot enough to be re-served.
        let Some(entry) = inner.by_hash.get_mut(&hash) else {
            return Ok(None);
        };
        if let Some(t) = entry.expires_at
            && t <= chrono::Utc::now()
        {
            // Lazy-expire: drop right now so subsequent calls hit the
            // empty path. Keeps us correct between background sweeps.
            let entry = inner.by_hash.pop(&hash).expect("just observed");
            inner.byte_count = inner.byte_count.saturating_sub(entry.bytes.len());
            return Ok(None);
        }
        let entry = entry.clone();
        Ok(Some(ResourceContent {
            bytes: entry.bytes,
            mime_type: entry.mime_type,
            session_id: entry.session_id,
            tenant_id: entry.tenant_id,
            stored_at: entry.stored_at,
            expires_at: entry.expires_at,
        }))
    }

    async fn delete(&self, id: &str) -> Result<(), ContentStoreError> {
        let mut inner = self.inner.write().await;
        let Some(hash) = Self::resolve_to_hash(&inner, id) else {
            return Ok(());
        };
        if let Some(entry) = inner.by_hash.pop(&hash) {
            inner.byte_count = inner.byte_count.saturating_sub(entry.bytes.len());
        }
        // Reap any aliases that pointed at the removed hash.
        inner.alias_to_hash.retain(|_, h| h != &hash);
        Ok(())
    }

    async fn signed_url(
        &self,
        _id: &str,
        _ttl: Duration,
    ) -> Result<Option<String>, ContentStoreError> {
        Err(ContentStoreError::SignedUrlNotSupported)
    }

    fn stats(&self) -> ContentStoreStats {
        // `try_read` so reading metrics never blocks the storage path
        // under contention. A momentarily stale snapshot is acceptable
        // for gauge emission.
        match self.inner.try_read() {
            Ok(inner) => ContentStoreStats {
                item_count: inner.by_hash.len() as u64,
                byte_count: inner.byte_count as u64,
                max_bytes: self.max_bytes as u64,
            },
            Err(_) => ContentStoreStats {
                item_count: 0,
                byte_count: 0,
                max_bytes: self.max_bytes as u64,
            },
        }
    }

    async fn sweep_expired(&self) -> usize {
        let now = chrono::Utc::now();
        let mut inner = self.inner.write().await;
        let mut to_remove = Vec::new();
        for (hash, entry) in inner.by_hash.iter() {
            if entry.expires_at.as_ref().is_some_and(|t| t <= &now) {
                to_remove.push(hash.clone());
            }
        }
        let removed = to_remove.len();
        for hash in to_remove {
            if let Some(entry) = inner.by_hash.pop(&hash) {
                inner.byte_count = inner.byte_count.saturating_sub(entry.bytes.len());
            }
        }
        // Drop alias entries whose target hash was just evicted.
        let live_hashes: std::collections::HashSet<String> =
            inner.by_hash.iter().map(|(k, _)| k.clone()).collect();
        inner
            .alias_to_hash
            .retain(|_, hash| live_hashes.contains(hash));
        removed
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn store_bytes(bytes: &[u8]) -> ContentToStore {
        ContentToStore {
            bytes: bytes::Bytes::copy_from_slice(bytes),
            mime_type: "application/octet-stream".into(),
            alias: None,
            session_id: None,
            tenant_id: None,
            ttl: None,
        }
    }

    #[tokio::test]
    async fn put_returns_hash_handle_and_uri() {
        let store = InProcessContentStore::new(1024);
        let h = store.put(store_bytes(b"hello world")).await.unwrap();
        assert!(h.id.starts_with("hash:"));
        assert_eq!(h.uri, format!("mcpg-resource://{}", h.id));
        assert!(h.content_hash.starts_with("blake3:"));
        assert_eq!(h.size_bytes, 11);
    }

    #[tokio::test]
    async fn get_returns_stored_bytes() {
        let store = InProcessContentStore::new(1024);
        let h = store.put(store_bytes(b"hello")).await.unwrap();
        let got = store.get(&h.id).await.unwrap().unwrap();
        assert_eq!(got.bytes.as_ref(), b"hello");
        assert_eq!(got.mime_type, "application/octet-stream");
    }

    #[tokio::test]
    async fn put_dedups_on_content_hash() {
        let store = InProcessContentStore::new(1024);
        let h1 = store.put(store_bytes(b"same content")).await.unwrap();
        let h2 = store.put(store_bytes(b"same content")).await.unwrap();
        // Same bytes → same hash → same id.
        assert_eq!(h1.id, h2.id);
        let stats = store.stats();
        assert_eq!(stats.item_count, 1);
    }

    #[tokio::test]
    async fn alias_resolves_to_same_content() {
        let store = InProcessContentStore::new(1024);
        let mut content = store_bytes(b"screenshot bytes");
        content.alias = Some("incident-screenshot".into());
        content.session_id = Some("sess-123".into());
        let h = store.put(content).await.unwrap();
        assert_eq!(h.id, "alias:sess-123:incident-screenshot");
        let got = store.get(&h.id).await.unwrap().unwrap();
        assert_eq!(got.bytes.as_ref(), b"screenshot bytes");
        // Hash form still resolves to the same blob.
        let hash_id = h.content_hash.replace("blake3:", "hash:");
        let got2 = store.get(&hash_id).await.unwrap().unwrap();
        assert_eq!(got.bytes, got2.bytes);
    }

    #[tokio::test]
    async fn delete_is_idempotent() {
        let store = InProcessContentStore::new(1024);
        let h = store.put(store_bytes(b"x")).await.unwrap();
        store.delete(&h.id).await.unwrap();
        store.delete(&h.id).await.unwrap(); // Second delete is a no-op.
        assert!(store.get(&h.id).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn signed_url_unsupported_on_in_process() {
        let store = InProcessContentStore::new(1024);
        let err = store
            .signed_url("hash:abc", Duration::from_secs(60))
            .await
            .unwrap_err();
        assert!(matches!(err, ContentStoreError::SignedUrlNotSupported));
    }

    #[tokio::test]
    async fn over_size_limit_rejects() {
        let store = InProcessContentStore::new(8);
        let err = store.put(store_bytes(b"too big to fit")).await.unwrap_err();
        match err {
            ContentStoreError::SizeLimit {
                limit_bytes,
                actual_bytes,
            } => {
                assert_eq!(limit_bytes, 8);
                assert_eq!(actual_bytes, 14);
            }
            other => panic!("expected SizeLimit, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn lru_evicts_oldest_when_over_byte_cap() {
        // 3 entries × 8 bytes each = 24 bytes max-cap allows 3
        // simultaneously; storing a 4th evicts the LRU.
        let store = InProcessContentStore::new(24);
        let h1 = store.put(store_bytes(b"aaaaaaaa")).await.unwrap();
        let h2 = store.put(store_bytes(b"bbbbbbbb")).await.unwrap();
        let h3 = store.put(store_bytes(b"cccccccc")).await.unwrap();
        // Touch h1 so h2 becomes the LRU candidate.
        let _ = store.get(&h1.id).await.unwrap();
        let _ = store.put(store_bytes(b"dddddddd")).await.unwrap();
        // h2 should now be gone; h1, h3, d4 remain.
        assert!(store.get(&h1.id).await.unwrap().is_some());
        assert!(store.get(&h2.id).await.unwrap().is_none());
        assert!(store.get(&h3.id).await.unwrap().is_some());
    }

    #[tokio::test]
    async fn ttl_expiry_returns_none_after_deadline() {
        let store = InProcessContentStore::new(1024);
        let mut content = store_bytes(b"expires soon");
        content.ttl = Some(Duration::from_millis(50));
        let h = store.put(content).await.unwrap();
        assert!(store.get(&h.id).await.unwrap().is_some());
        tokio::time::sleep(Duration::from_millis(80)).await;
        assert!(store.get(&h.id).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn sweep_expired_removes_stale_entries() {
        let store = InProcessContentStore::new(1024);
        let mut a = store_bytes(b"a");
        a.ttl = Some(Duration::from_millis(20));
        let _ = store.put(a).await.unwrap();
        let _ = store.put(store_bytes(b"b")).await.unwrap(); // no TTL.
        tokio::time::sleep(Duration::from_millis(60)).await;
        let removed = store.sweep_expired().await;
        assert_eq!(removed, 1);
        assert_eq!(store.stats().item_count, 1);
    }

    #[tokio::test]
    async fn unknown_id_returns_none_not_error() {
        let store = InProcessContentStore::new(1024);
        assert!(store.get("hash:doesnotexist").await.unwrap().is_none());
        assert!(store.get("alias:nope:nada").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn stats_track_size_and_count() {
        let store = InProcessContentStore::new(1024);
        let _ = store.put(store_bytes(b"hello")).await.unwrap();
        let _ = store.put(store_bytes(b"world!")).await.unwrap();
        let stats = store.stats();
        assert_eq!(stats.item_count, 2);
        assert_eq!(stats.byte_count, 11);
        assert_eq!(stats.max_bytes, 1024);
    }
}
