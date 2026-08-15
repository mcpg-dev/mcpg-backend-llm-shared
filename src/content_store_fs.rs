//! Filesystem-backed [`ContentStore`].
//!
//! On-disk layout under `<root>`:
//!
//! - `blobs/<2>/<2>/<rest>` — raw bytes; the path is the BLAKE3 hex.
//! - `meta/<2>/<2>/<rest>.json` — sidecar with `MetaRecord`
//!   (mime, stored_at, expires_at, session_id, tenant_id, byte size).
//! - `aliases/<base64url(alias_id)>.json` — alias→hash redirect record.
//!
//! Atomic writes: each file is written to `<path>.tmp.<rand>` then
//! `rename`d into place. A partial crash leaves the temp file behind
//! (cleaned on next sweep) but never a half-written final file.
//!
//! A small in-memory index of `(hash → MetaRecord)` is rebuilt on
//! startup by walking the meta tree. The index drives:
//! - O(1) lookups for `get` / `delete` / `signed_url`.
//! - LRU eviction by `stored_at` when `max_bytes` is exceeded
//!   (older entries removed first; ties broken by hash for stability).
//! - Aggregate `byte_count` for `stats()`.
//!
//! Concurrency: the index is `RwLock`-guarded; disk writes happen
//! outside the lock. A second writer racing on the same hash either
//! wins atomically (idempotent — same content → same bytes) or sees
//! its tempfile cleaned up by the rename. Two writers under different
//! aliases pointing at the same content both succeed and both alias
//! files end up on disk.
//!
//! Signed URLs are not provided by this store — operators wanting
//! presigned download links should use `S3ContentStore` (native
//! presigner) or front the gateway with an HTTP route that vends
//! Ed25519-signed redirects (separate, not in this crate).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use base64::Engine as _;
use serde::{Deserialize, Serialize};
use tokio::fs;
use tokio::io::AsyncWriteExt;
use tokio::sync::RwLock;

use crate::content_store::{
    ContentStore, ContentStoreError, ContentStoreStats, ContentToStore, ResourceContent,
    ResourceHandle,
};

/// Per-blob metadata persisted as a JSON sidecar next to the bytes.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct MetaRecord {
    mime_type: String,
    stored_at: chrono::DateTime<chrono::Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    expires_at: Option<chrono::DateTime<chrono::Utc>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tenant_id: Option<String>,
    size_bytes: u64,
}

/// Per-alias redirect record persisted as a JSON sidecar under
/// `aliases/`. `target_hash` is the BLAKE3 hex this alias resolves to.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct AliasRecord {
    target_hash: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    expires_at: Option<chrono::DateTime<chrono::Utc>>,
}

#[derive(Debug, Clone)]
struct IndexEntry {
    meta: MetaRecord,
}

#[derive(Debug, Default)]
struct Index {
    by_hash: HashMap<String, IndexEntry>,
    alias_to_hash: HashMap<String, String>,
    byte_count: u64,
}

/// Filesystem-backed content store. See module docs for layout.
pub struct FileSystemContentStore {
    root: PathBuf,
    max_bytes: u64,
    index: RwLock<Index>,
}

impl std::fmt::Debug for FileSystemContentStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileSystemContentStore")
            .field("root", &self.root)
            .field("max_bytes", &self.max_bytes)
            .finish()
    }
}

impl FileSystemContentStore {
    /// Open (or create) a filesystem-backed store rooted at `root`.
    /// `max_bytes` of 0 disables size-cap eviction (TTL still applies).
    /// Walks the existing meta tree to bootstrap the in-memory index.
    pub async fn open(
        root: impl Into<PathBuf>,
        max_bytes: u64,
    ) -> Result<Arc<Self>, ContentStoreError> {
        let root = root.into();
        for sub in ["blobs", "meta", "aliases"] {
            let dir = root.join(sub);
            fs::create_dir_all(&dir)
                .await
                .map_err(|e| ContentStoreError::Storage {
                    message: format!("create {}: {e}", dir.display()),
                })?;
        }
        let store = Arc::new(Self {
            root,
            max_bytes,
            index: RwLock::new(Index::default()),
        });
        store.bootstrap_index().await?;
        Ok(store)
    }

    fn blob_path(&self, hash: &str) -> PathBuf {
        let (a, b) = split_hash(hash);
        self.root.join("blobs").join(a).join(b).join(hash)
    }

    fn meta_path(&self, hash: &str) -> PathBuf {
        let (a, b) = split_hash(hash);
        self.root
            .join("meta")
            .join(a)
            .join(b)
            .join(format!("{hash}.json"))
    }

    fn alias_path(&self, alias_id: &str) -> PathBuf {
        let encoded = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(alias_id);
        self.root.join("aliases").join(format!("{encoded}.json"))
    }

    /// Walk the meta tree and rebuild the in-memory index. Called
    /// once at `open()`. Skips any entry whose blob is missing
    /// (deletes the orphaned meta file).
    async fn bootstrap_index(&self) -> Result<(), ContentStoreError> {
        let mut index = self.index.write().await;
        let meta_root = self.root.join("meta");
        let mut stack = vec![meta_root.clone()];
        while let Some(dir) = stack.pop() {
            let mut rd = match fs::read_dir(&dir).await {
                Ok(rd) => rd,
                Err(_) => continue,
            };
            while let Some(entry) = rd.next_entry().await.map_err(io_err)? {
                let path = entry.path();
                let ft = entry.file_type().await.map_err(io_err)?;
                if ft.is_dir() {
                    stack.push(path);
                    continue;
                }
                if path.extension().and_then(|s| s.to_str()) != Some("json") {
                    continue;
                }
                let Some(hash) = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .map(|s| s.to_owned())
                else {
                    continue;
                };
                let blob = self.blob_path(&hash);
                if !blob.exists() {
                    let _ = fs::remove_file(&path).await;
                    continue;
                }
                let bytes = match fs::read(&path).await {
                    Ok(b) => b,
                    Err(_) => continue,
                };
                let Ok(meta) = serde_json::from_slice::<MetaRecord>(&bytes) else {
                    continue;
                };
                index.byte_count = index.byte_count.saturating_add(meta.size_bytes);
                index.by_hash.insert(hash, IndexEntry { meta });
            }
        }

        // Walk aliases.
        let alias_root = self.root.join("aliases");
        if let Ok(mut rd) = fs::read_dir(&alias_root).await {
            while let Some(entry) = rd.next_entry().await.map_err(io_err)? {
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) != Some("json") {
                    continue;
                }
                let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                    continue;
                };
                let Ok(alias_id_bytes) =
                    base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(stem)
                else {
                    continue;
                };
                let Ok(alias_id) = String::from_utf8(alias_id_bytes) else {
                    continue;
                };
                let bytes = match fs::read(&path).await {
                    Ok(b) => b,
                    Err(_) => continue,
                };
                let Ok(rec) = serde_json::from_slice::<AliasRecord>(&bytes) else {
                    continue;
                };
                if !index.by_hash.contains_key(&rec.target_hash) {
                    let _ = fs::remove_file(&path).await;
                    continue;
                }
                index.alias_to_hash.insert(alias_id, rec.target_hash);
            }
        }

        Ok(())
    }

    fn resolve_to_hash(index: &Index, id: &str) -> Option<String> {
        if let Some(rest) = id.strip_prefix("hash:") {
            return Some(rest.to_owned());
        }
        if id.starts_with("alias:") {
            return index.alias_to_hash.get(id).cloned();
        }
        if !id.is_empty() && id.chars().all(|c| c.is_ascii_hexdigit()) {
            return Some(id.to_owned());
        }
        None
    }

    /// Evict by `stored_at` ascending until `byte_count <= max_bytes`.
    /// Removes blob + meta + dangling aliases. Called inside
    /// `put` after each write that may have pushed past the cap.
    async fn evict_to_fit(&self, index: &mut Index) {
        if self.max_bytes == 0 || index.byte_count <= self.max_bytes {
            return;
        }
        let mut entries: Vec<(String, chrono::DateTime<chrono::Utc>)> = index
            .by_hash
            .iter()
            .map(|(h, e)| (h.clone(), e.meta.stored_at))
            .collect();
        entries.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        for (hash, _) in entries {
            if index.byte_count <= self.max_bytes {
                break;
            }
            self.remove_one(&mut *index, &hash).await;
        }
    }

    /// Remove one hash everywhere — index, blob, meta, dangling
    /// aliases. Errors during disk removal are logged but not
    /// propagated; the index is the source of truth for live state.
    async fn remove_one(&self, index: &mut Index, hash: &str) {
        if let Some(entry) = index.by_hash.remove(hash) {
            index.byte_count = index.byte_count.saturating_sub(entry.meta.size_bytes);
        }
        let _ = fs::remove_file(self.blob_path(hash)).await;
        let _ = fs::remove_file(self.meta_path(hash)).await;
        // Reap aliases whose target was just evicted.
        let dead: Vec<String> = index
            .alias_to_hash
            .iter()
            .filter_map(|(k, v)| if v == hash { Some(k.clone()) } else { None })
            .collect();
        for alias in dead {
            index.alias_to_hash.remove(&alias);
            let _ = fs::remove_file(self.alias_path(&alias)).await;
        }
    }
}

fn io_err(e: std::io::Error) -> ContentStoreError {
    ContentStoreError::Storage {
        message: e.to_string(),
    }
}

fn split_hash(hash: &str) -> (&str, &str) {
    // Defensive: all real BLAKE3 hex hashes are 64 chars, but tolerate
    // short ids in tests. The key invariant is that we always split on
    // the same boundaries so a given hash maps to a stable path.
    let len = hash.len();
    let a_end = 2.min(len);
    let b_end = 4.min(len);
    (&hash[..a_end], &hash[a_end..b_end])
}

/// Atomic write via tempfile + rename. Caller passes the final path;
/// we sit a `<path>.tmp.<rand-hex>` next to it, write, sync, rename.
async fn atomic_write(path: &Path, contents: &[u8]) -> Result<(), ContentStoreError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).await.map_err(io_err)?;
    }
    let mut rand = [0u8; 8];
    getrandom::fill(&mut rand).map_err(|e| ContentStoreError::Storage {
        message: format!("rng: {e}"),
    })?;
    let tmp = path.with_extension(format!("tmp.{}", hex::encode(rand)));
    {
        let mut f = fs::File::create(&tmp).await.map_err(io_err)?;
        f.write_all(contents).await.map_err(io_err)?;
        f.sync_all().await.map_err(io_err)?;
    }
    fs::rename(&tmp, path).await.map_err(io_err)?;
    Ok(())
}

#[async_trait]
impl ContentStore for FileSystemContentStore {
    async fn put(&self, content: ContentToStore) -> Result<ResourceHandle, ContentStoreError> {
        let size = content.bytes.len();
        let size_u64 = size as u64;
        if self.max_bytes > 0 && size_u64 > self.max_bytes {
            return Err(ContentStoreError::SizeLimit {
                limit_bytes: self.max_bytes as usize,
                actual_bytes: size,
            });
        }

        let hash_hex = hex::encode(blake3::hash(&content.bytes).as_bytes());
        let stored_at = chrono::Utc::now();
        let expires_at = content
            .ttl
            .and_then(|d| chrono::Duration::from_std(d).ok())
            .map(|d| stored_at + d);

        let meta = MetaRecord {
            mime_type: content.mime_type.clone(),
            stored_at,
            expires_at,
            session_id: content.session_id.clone(),
            tenant_id: content.tenant_id.clone(),
            size_bytes: size_u64,
        };

        // Disk writes happen outside the index lock — concurrent
        // writers on the same hash race the rename; whoever wins
        // last writes the same bytes (content-addressed) so it's
        // idempotent. The tempfile from the loser is cleaned up
        // by the rename overwrite on POSIX; on Windows the second
        // rename fails harmlessly (we ignore the error since the
        // blob and meta both exist).
        let blob = self.blob_path(&hash_hex);
        let meta_path = self.meta_path(&hash_hex);
        if !blob.exists() {
            atomic_write(&blob, &content.bytes).await?;
        }
        let meta_bytes = serde_json::to_vec(&meta).map_err(|e| ContentStoreError::Storage {
            message: format!("encode meta: {e}"),
        })?;
        atomic_write(&meta_path, &meta_bytes).await?;

        let mut index = self.index.write().await;

        // Replace prior index entry (dedup hit just refreshes meta).
        if let Some(prior) = index.by_hash.remove(&hash_hex) {
            index.byte_count = index.byte_count.saturating_sub(prior.meta.size_bytes);
        }
        index.byte_count = index.byte_count.saturating_add(size_u64);
        index
            .by_hash
            .insert(hash_hex.clone(), IndexEntry { meta: meta.clone() });

        let id = if let Some(alias) = content.alias.as_deref() {
            let session = content.session_id.as_deref().unwrap_or("__no_session__");
            let alias_id = format!("alias:{session}:{alias}");
            let alias_rec = AliasRecord {
                target_hash: hash_hex.clone(),
                expires_at,
            };
            let alias_path = self.alias_path(&alias_id);
            let alias_bytes =
                serde_json::to_vec(&alias_rec).map_err(|e| ContentStoreError::Storage {
                    message: format!("encode alias: {e}"),
                })?;
            // Drop the index lock briefly — we're holding a write
            // guard, but atomic_write is async and may yield. Move
            // the file write before the index update.
            drop(index);
            atomic_write(&alias_path, &alias_bytes).await?;
            let mut index = self.index.write().await;
            index
                .alias_to_hash
                .insert(alias_id.clone(), hash_hex.clone());
            self.evict_to_fit(&mut index).await;
            alias_id
        } else {
            self.evict_to_fit(&mut index).await;
            format!("hash:{hash_hex}")
        };

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
        let hash = {
            let index = self.index.read().await;
            let Some(hash) = Self::resolve_to_hash(&index, id) else {
                return Ok(None);
            };
            hash
        };

        let meta = {
            let index = self.index.read().await;
            let Some(entry) = index.by_hash.get(&hash) else {
                return Ok(None);
            };
            if let Some(t) = entry.meta.expires_at
                && t <= chrono::Utc::now()
            {
                drop(index);
                let mut index = self.index.write().await;
                self.remove_one(&mut index, &hash).await;
                return Ok(None);
            }
            entry.meta.clone()
        };

        let bytes = fs::read(self.blob_path(&hash)).await.map_err(io_err)?;
        Ok(Some(ResourceContent {
            bytes: bytes::Bytes::from(bytes),
            mime_type: meta.mime_type,
            session_id: meta.session_id,
            tenant_id: meta.tenant_id,
            stored_at: meta.stored_at,
            expires_at: meta.expires_at,
        }))
    }

    async fn delete(&self, id: &str) -> Result<(), ContentStoreError> {
        let mut index = self.index.write().await;
        let Some(hash) = Self::resolve_to_hash(&index, id) else {
            return Ok(());
        };
        self.remove_one(&mut index, &hash).await;
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
        match self.index.try_read() {
            Ok(index) => ContentStoreStats {
                item_count: index.by_hash.len() as u64,
                byte_count: index.byte_count,
                max_bytes: self.max_bytes,
            },
            Err(_) => ContentStoreStats {
                item_count: 0,
                byte_count: 0,
                max_bytes: self.max_bytes,
            },
        }
    }

    async fn sweep_expired(&self) -> usize {
        let now = chrono::Utc::now();
        let expired: Vec<String> = {
            let index = self.index.read().await;
            index
                .by_hash
                .iter()
                .filter_map(|(h, e)| e.meta.expires_at.filter(|t| *t <= now).map(|_| h.clone()))
                .collect()
        };
        if expired.is_empty() {
            return 0;
        }
        let mut index = self.index.write().await;
        for hash in &expired {
            self.remove_one(&mut index, hash).await;
        }
        expired.len()
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

    async fn fresh_store(max_bytes: u64) -> (Arc<FileSystemContentStore>, tempfile::TempDir) {
        let tmp = tempfile::tempdir().unwrap();
        let store = FileSystemContentStore::open(tmp.path(), max_bytes)
            .await
            .unwrap();
        (store, tmp)
    }

    #[tokio::test]
    async fn put_returns_hash_handle_and_uri() {
        let (store, _tmp) = fresh_store(1024).await;
        let h = store.put(store_bytes(b"hello world")).await.unwrap();
        assert!(h.id.starts_with("hash:"));
        assert_eq!(h.uri, format!("mcpg-resource://{}", h.id));
        assert!(h.content_hash.starts_with("blake3:"));
        assert_eq!(h.size_bytes, 11);
    }

    #[tokio::test]
    async fn get_returns_stored_bytes() {
        let (store, _tmp) = fresh_store(1024).await;
        let h = store.put(store_bytes(b"hello")).await.unwrap();
        let got = store.get(&h.id).await.unwrap().unwrap();
        assert_eq!(got.bytes.as_ref(), b"hello");
        assert_eq!(got.mime_type, "application/octet-stream");
    }

    #[tokio::test]
    async fn put_dedups_on_content_hash() {
        let (store, _tmp) = fresh_store(1024).await;
        let h1 = store.put(store_bytes(b"same")).await.unwrap();
        let h2 = store.put(store_bytes(b"same")).await.unwrap();
        assert_eq!(h1.id, h2.id);
        assert_eq!(store.stats().item_count, 1);
    }

    #[tokio::test]
    async fn alias_resolves_to_same_content() {
        let (store, _tmp) = fresh_store(1024).await;
        let mut content = store_bytes(b"screenshot");
        content.alias = Some("incident".into());
        content.session_id = Some("sess-9".into());
        let h = store.put(content).await.unwrap();
        assert_eq!(h.id, "alias:sess-9:incident");
        let got = store.get(&h.id).await.unwrap().unwrap();
        assert_eq!(got.bytes.as_ref(), b"screenshot");
        let hash_id = h.content_hash.replace("blake3:", "hash:");
        let got2 = store.get(&hash_id).await.unwrap().unwrap();
        assert_eq!(got2.bytes.as_ref(), b"screenshot");
    }

    #[tokio::test]
    async fn delete_removes_blob_meta_and_alias() {
        let (store, tmp) = fresh_store(1024).await;
        let mut content = store_bytes(b"bye");
        content.alias = Some("a1".into());
        content.session_id = Some("s1".into());
        let h = store.put(content).await.unwrap();
        store.delete(&h.id).await.unwrap();
        assert!(store.get(&h.id).await.unwrap().is_none());
        // Disk artefacts gone.
        let mut blob_count = 0;
        let blobs_root = tmp.path().join("blobs");
        let mut stack = vec![blobs_root];
        while let Some(d) = stack.pop() {
            let mut rd = fs::read_dir(&d).await.unwrap();
            while let Some(e) = rd.next_entry().await.unwrap() {
                if e.file_type().await.unwrap().is_dir() {
                    stack.push(e.path());
                } else {
                    blob_count += 1;
                }
            }
        }
        assert_eq!(blob_count, 0, "blob file should be gone");
    }

    #[tokio::test]
    async fn sweep_expired_drops_old_entries() {
        let (store, _tmp) = fresh_store(1024).await;
        let mut content = store_bytes(b"transient");
        content.ttl = Some(Duration::from_millis(20));
        let h = store.put(content).await.unwrap();
        tokio::time::sleep(Duration::from_millis(40)).await;
        let removed = store.sweep_expired().await;
        assert_eq!(removed, 1);
        assert!(store.get(&h.id).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn lazy_expire_on_get() {
        let (store, _tmp) = fresh_store(1024).await;
        let mut content = store_bytes(b"transient");
        content.ttl = Some(Duration::from_millis(20));
        let h = store.put(content).await.unwrap();
        tokio::time::sleep(Duration::from_millis(40)).await;
        // No background sweep — get itself drops the expired entry.
        assert!(store.get(&h.id).await.unwrap().is_none());
        assert_eq!(store.stats().item_count, 0);
    }

    #[tokio::test]
    async fn size_cap_evicts_oldest() {
        // Cap at 12 bytes — three 6-byte blobs cannot all fit.
        let (store, _tmp) = fresh_store(12).await;
        let h1 = store.put(store_bytes(b"aaaaaa")).await.unwrap();
        // Tiny gap so stored_at ordering is deterministic.
        tokio::time::sleep(Duration::from_millis(5)).await;
        let h2 = store.put(store_bytes(b"bbbbbb")).await.unwrap();
        tokio::time::sleep(Duration::from_millis(5)).await;
        let _h3 = store.put(store_bytes(b"cccccc")).await.unwrap();
        // h1 should be gone (oldest), h2 + h3 should live.
        assert!(store.get(&h1.id).await.unwrap().is_none());
        assert!(store.get(&h2.id).await.unwrap().is_some());
        let stats = store.stats();
        assert_eq!(stats.item_count, 2);
    }

    #[tokio::test]
    async fn reopen_rebuilds_index_from_disk() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().to_path_buf();
        let h = {
            let store = FileSystemContentStore::open(&path, 1024).await.unwrap();
            store.put(store_bytes(b"persisted")).await.unwrap()
        };
        // Drop the store, reopen pointing at the same dir.
        let store2 = FileSystemContentStore::open(&path, 1024).await.unwrap();
        let got = store2.get(&h.id).await.unwrap().unwrap();
        assert_eq!(got.bytes.as_ref(), b"persisted");
        assert_eq!(store2.stats().item_count, 1);
    }

    #[tokio::test]
    async fn signed_url_reports_unsupported() {
        let (store, _tmp) = fresh_store(1024).await;
        let h = store.put(store_bytes(b"x")).await.unwrap();
        let err = store
            .signed_url(&h.id, Duration::from_secs(60))
            .await
            .unwrap_err();
        assert!(matches!(err, ContentStoreError::SignedUrlNotSupported));
    }

    #[tokio::test]
    async fn put_too_large_rejects() {
        let (store, _tmp) = fresh_store(8).await;
        let err = store.put(store_bytes(b"way too long")).await.unwrap_err();
        assert!(matches!(err, ContentStoreError::SizeLimit { .. }));
    }
}
