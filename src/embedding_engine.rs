//! Provider-agnostic embedding execution engine.
//!
//! Drives the per-call flow for embedding bindings:
//!
//! 1. Normalise the operator-supplied `args` — accepts `{input: "..."}`
//!    (single string), `{input: ["...","..."]}` (batch), or any
//!    arg field whose name matches `prompt.image_inputs` style; for
//!    embeddings the convention is fixed at `input`.
//! 2. Chunk inputs at the smaller of operator's `max_batch_size`
//!    and the provider's [`crate::EmbeddingProviderAdapter::max_batch_size`].
//! 3. Issue chunks in parallel with retry-on-transient.
//! 4. Stitch the responses back into a single
//!    `{embeddings, dimensions, usage}` JSON object.
//!
//! No agentic loop, no streaming, no schema validation past the
//! shape — embeddings are pure functions.

use std::sync::Arc;
use std::time::Duration;

use futures::stream::{FuturesOrdered, StreamExt};
use mcpg_plugin_protocol::{BackendError, BackendHost, BackendInvocationContext};
use serde_json::{Value, json};
use tracing::{debug, instrument, warn};

use crate::cache::CacheKey;
use crate::embedding::{
    EmbeddingProviderAdapter, NormalizedEmbeddingRequest, NormalizedEmbeddingResponse,
};
use crate::embedding_config::EmbeddingExecutionSpec;
use crate::error::ProviderError;

pub struct EmbeddingEngine {
    pub backend_name: String,
    pub adapter: Arc<dyn EmbeddingProviderAdapter>,
    pub spec: EmbeddingExecutionSpec,
    /// Host capability — used for [`crate::cache::ResponseCache`]
    /// access via the [`BackendHost::cache_get`] / `cache_put`
    /// surface. Pass [`mcpg_plugin_protocol::noop_backend_host`] when
    /// running without a gateway-side cache.
    pub host: Arc<dyn BackendHost>,
}

impl std::fmt::Debug for EmbeddingEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddingEngine")
            .field("backend_name", &self.backend_name)
            .field("provider", &self.adapter.label())
            .field("model", &self.spec.model)
            .finish()
    }
}

impl EmbeddingEngine {
    /// Resolved batch size: smaller of operator override and
    /// provider hard cap. Always >= 1.
    fn effective_batch_size(&self) -> usize {
        let provider_cap = self.adapter.max_batch_size().max(1);
        match self.spec.max_batch_size {
            Some(n) => n.max(1).min(provider_cap),
            None => provider_cap,
        }
    }

    /// Run one embedding request.
    ///
    /// `args` shape:
    /// ```json
    /// { "input": "single text" }
    /// // or
    /// { "input": ["a", "b", "c"] }
    /// ```
    ///
    /// Returns:
    /// ```json
    /// {
    ///   "embeddings": [[…], [...], …],
    ///   "dimensions": 1536,
    ///   "usage": { "input_tokens": 42 }
    /// }
    /// ```
    ///
    /// `embeddings.len()` always matches the input array length —
    /// for scalar `input`, it's a 1-element outer array.
    #[instrument(skip(self, args), fields(binding = %self.backend_name, provider = %self.adapter.label(), model = %self.spec.model))]
    pub async fn execute(&self, args: &Value) -> Result<Value, BackendError> {
        let inputs = parse_inputs(args).map_err(|message| BackendError::InvalidSpec { message })?;
        if inputs.is_empty() {
            return Err(BackendError::InvalidSpec {
                message: "embedding `input` array must not be empty".into(),
            });
        }

        // Response-cache lookup. Embeddings are pure functions; a
        // hit returns the same JSON shape the upstream call would
        // have produced.
        let host_ctx =
            BackendInvocationContext::root("embedding-call", None, self.backend_name.clone());
        let cache_key = if self.spec.cache.enabled {
            let key = build_embedding_cache_key(self, &inputs);
            match self.host.cache_get(&host_ctx, key.as_str()).await {
                Ok(Some(bytes)) => {
                    metrics::counter!(
                        "mcpg_embedding_cache_hits_total",
                        "binding" => self.backend_name.clone(),
                        "provider" => self.adapter.label().to_string(),
                        "model" => self.spec.model.clone(),
                    )
                    .increment(1);
                    let value: Value =
                        serde_json::from_slice(&bytes).map_err(|e| BackendError::Transport {
                            message: format!("decode cached embedding response: {e}"),
                        })?;
                    return Ok(value);
                }
                Ok(None) => {
                    metrics::counter!(
                        "mcpg_embedding_cache_misses_total",
                        "binding" => self.backend_name.clone(),
                        "provider" => self.adapter.label().to_string(),
                        "model" => self.spec.model.clone(),
                    )
                    .increment(1);
                }
                Err(e) => {
                    // Cache failures should not break the call —
                    // log and fall through to the upstream call.
                    debug!(?e, "embedding cache lookup failed; falling through");
                }
            }
            Some(key)
        } else {
            None
        };

        let batch_size = self.effective_batch_size();
        let chunks: Vec<Vec<String>> = inputs.chunks(batch_size).map(|c| c.to_vec()).collect();
        let total_chunks = chunks.len();
        debug!(
            total_inputs = inputs.len(),
            total_chunks, batch_size, "embedding fan-out"
        );

        let timeout = self.spec.timeout();
        let started = std::time::Instant::now();

        let mut futures = FuturesOrdered::new();
        for chunk in chunks {
            let req = NormalizedEmbeddingRequest {
                model: self.spec.model.clone(),
                inputs: chunk,
                dimensions: self.spec.dimensions,
            };
            let adapter = self.adapter.clone();
            let spec = self.spec.clone();
            futures.push_back(async move {
                call_with_retry(adapter.as_ref(), &req, timeout, &spec).await
            });
        }

        let mut all_vectors: Vec<Vec<f32>> = Vec::with_capacity(inputs.len());
        let mut dimensions: u32 = 0;
        let mut usage_total: u32 = 0;
        let mut usage_seen = false;

        while let Some(res) = futures.next().await {
            let resp = res.map_err(|e| binding_error_from_provider(&self.backend_name, e))?;
            if dimensions == 0 {
                dimensions = resp.dimensions;
            } else if resp.dimensions != dimensions {
                return Err(BackendError::Transport {
                    message: format!(
                        "provider returned inconsistent dimensions across batches ({} then {})",
                        dimensions, resp.dimensions
                    ),
                });
            }
            if let Some(u) = resp.usage {
                usage_total = usage_total.saturating_add(u.input_tokens);
                usage_seen = true;
            }
            all_vectors.extend(resp.embeddings);
        }

        if all_vectors.len() != inputs.len() {
            return Err(BackendError::Transport {
                message: format!(
                    "provider returned {} vectors for {} inputs",
                    all_vectors.len(),
                    inputs.len()
                ),
            });
        }

        let elapsed = started.elapsed();
        metrics::histogram!(
            "mcpg_embedding_call_seconds",
            "binding" => self.backend_name.clone(),
            "provider" => self.adapter.label().to_string(),
            "model" => self.spec.model.clone(),
        )
        .record(elapsed.as_secs_f64());
        metrics::counter!(
            "mcpg_embedding_inputs_total",
            "binding" => self.backend_name.clone(),
            "provider" => self.adapter.label().to_string(),
            "model" => self.spec.model.clone(),
        )
        .increment(inputs.len() as u64);

        let mut out = json!({
            "embeddings": all_vectors,
            "dimensions": dimensions,
        });
        if usage_seen {
            out["usage"] = json!({"input_tokens": usage_total});
        }

        // Cost emission. compute_embedding_cost_usd
        // returns None for models not in the rate card; we
        // quietly drop the metric rather than emit a misleading
        // zero.
        if usage_seen
            && let Some(cost) = crate::cost::compute_embedding_cost_usd(
                crate::cost::bundled_rate_card(),
                self.adapter.label(),
                &self.spec.model,
                usage_total,
            )
        {
            metrics::counter!(
                "mcpg_embedding_cost_usd_total",
                "binding" => self.backend_name.clone(),
                "provider" => self.adapter.label().to_string(),
                "model" => self.spec.model.clone(),
            )
            .increment((cost * 1_000_000.0) as u64);
            metrics::histogram!(
                "mcpg_embedding_call_cost_usd",
                "binding" => self.backend_name.clone(),
                "provider" => self.adapter.label().to_string(),
                "model" => self.spec.model.clone(),
            )
            .record(cost);
        }

        // Response-cache write. Skip on hit-path early return
        // above — that branch returned before reaching here.
        if let Some(key) = cache_key {
            let payload = match serde_json::to_vec(&out) {
                Ok(b) => b,
                Err(e) => {
                    debug!(?e, "skip cache write: response not serializable");
                    return Ok(out);
                }
            };
            let ttl = Duration::from_secs(self.spec.cache.ttl_seconds);
            if let Err(e) = self
                .host
                .cache_put(&host_ctx, key.as_str().to_owned(), payload.into(), ttl)
                .await
            {
                debug!(?e, "embedding cache write failed; ignoring");
            }
        }

        Ok(out)
    }
}

fn build_embedding_cache_key(engine: &EmbeddingEngine, inputs: &[String]) -> CacheKey {
    // Stitch all inputs into a single newline-delimited blob — keys
    // never collide across different `[input]` shapes because the
    // joiner is a literal control character that cannot appear in a
    // single input. Adapters never see this string; only used for
    // hashing.
    let joined = inputs.join("\u{0001}");
    CacheKey::for_embedding(
        &engine.backend_name,
        engine.adapter.label(),
        &engine.spec.model,
        &joined,
        engine.spec.dimensions,
    )
}

/// Pull the `input` field out of `args` and normalise to a `Vec<String>`.
fn parse_inputs(args: &Value) -> Result<Vec<String>, String> {
    let v = args
        .get("input")
        .ok_or_else(|| "embedding args must contain `input`".to_owned())?;
    match v {
        Value::String(s) => Ok(vec![s.clone()]),
        Value::Array(arr) => {
            let mut out = Vec::with_capacity(arr.len());
            for (i, item) in arr.iter().enumerate() {
                match item.as_str() {
                    Some(s) => out.push(s.to_owned()),
                    None => {
                        return Err(format!("embedding `input[{i}]` must be a string"));
                    }
                }
            }
            Ok(out)
        }
        _ => Err("embedding `input` must be a string or array of strings".into()),
    }
}

async fn call_with_retry(
    adapter: &dyn EmbeddingProviderAdapter,
    request: &NormalizedEmbeddingRequest,
    timeout: Duration,
    spec: &EmbeddingExecutionSpec,
) -> Result<NormalizedEmbeddingResponse, ProviderError> {
    let mut attempt: u32 = 0;
    loop {
        attempt += 1;
        let result = adapter.embed(request, timeout).await;
        match result {
            Ok(resp) => return Ok(resp),
            Err(err) if attempt >= spec.retry.max_attempts || !err.is_retryable() => {
                if !err.is_retryable() {
                    debug!(?err, "embedding error not retryable");
                } else {
                    warn!(
                        ?err,
                        attempt,
                        max = spec.retry.max_attempts,
                        "embedding retries exhausted"
                    );
                }
                return Err(err);
            }
            Err(err) => {
                let backoff_ms = std::cmp::min(
                    spec.retry
                        .initial_backoff_ms
                        .saturating_mul(1 << (attempt - 1)),
                    spec.retry.max_backoff_ms,
                );
                debug!(?err, attempt, backoff_ms, "embedding retry");
                tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
            }
        }
    }
}

fn binding_error_from_provider(binding: &str, err: ProviderError) -> BackendError {
    match err {
        ProviderError::Network { message } => BackendError::Transport {
            message: format!("embedding {binding}: network: {message}"),
        },
        ProviderError::Server { message } => BackendError::Transport {
            message: format!("embedding {binding}: server: {message}"),
        },
        ProviderError::RateLimited { message } => BackendError::Transport {
            message: format!("embedding {binding}: rate-limited: {message}"),
        },
        ProviderError::AuthFailed { message } => BackendError::InvalidSpec {
            message: format!("embedding {binding}: auth: {message}"),
        },
        ProviderError::BadRequest { message } => BackendError::InvalidSpec {
            message: format!("embedding {binding}: bad request: {message}"),
        },
        ProviderError::ContextLimit { message } => BackendError::InvalidSpec {
            message: format!("embedding {binding}: context limit: {message}"),
        },
        ProviderError::Malformed { message } => BackendError::Transport {
            message: format!("embedding {binding}: malformed: {message}"),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Test stub: returns one zero vector per input, of declared
    /// dimensionality. Counts calls (via shared Arc<AtomicUsize>)
    /// so tests can assert batching while keeping the adapter
    /// behind `Arc<dyn EmbeddingProviderAdapter>`.
    struct StubAdapter {
        max_batch: usize,
        dim: u32,
        call_count: Arc<AtomicUsize>,
    }

    impl std::fmt::Debug for StubAdapter {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("StubAdapter").finish()
        }
    }

    #[async_trait]
    impl EmbeddingProviderAdapter for StubAdapter {
        fn label(&self) -> &'static str {
            "stub"
        }
        fn max_batch_size(&self) -> usize {
            self.max_batch
        }
        async fn embed(
            &self,
            request: &NormalizedEmbeddingRequest,
            _timeout: Duration,
        ) -> Result<NormalizedEmbeddingResponse, ProviderError> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            assert!(
                request.inputs.len() <= self.max_batch,
                "engine sent batch larger than provider cap"
            );
            let dim = self.dim as usize;
            let embeddings = request.inputs.iter().map(|_| vec![0.0_f32; dim]).collect();
            Ok(NormalizedEmbeddingResponse {
                embeddings,
                dimensions: self.dim,
                usage: Some(crate::embedding::EmbeddingTokenUsage {
                    input_tokens: request.inputs.len() as u32 * 5,
                }),
            })
        }
    }

    fn engine(
        max_batch: usize,
        dim: u32,
        max_batch_size: Option<usize>,
    ) -> (EmbeddingEngine, Arc<AtomicUsize>) {
        let call_count = Arc::new(AtomicUsize::new(0));
        let engine = EmbeddingEngine {
            backend_name: "test".into(),
            adapter: Arc::new(StubAdapter {
                max_batch,
                dim,
                call_count: call_count.clone(),
            }),
            spec: EmbeddingExecutionSpec {
                model: "test-model".into(),
                dimensions: None,
                timeout_ms: 1000,
                connect_timeout_ms: 1000,
                max_batch_size,
                retry: Default::default(),
                cache: Default::default(),
            },
            host: mcpg_plugin_protocol::noop_backend_host(),
        };
        (engine, call_count)
    }

    #[tokio::test]
    async fn scalar_input_produces_length_1_embeddings_array() {
        let (e, _calls) = engine(100, 4, None);
        let r = e.execute(&json!({"input": "hello"})).await.unwrap();
        let arr = r["embeddings"].as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0].as_array().unwrap().len(), 4);
        assert_eq!(r["dimensions"], 4);
        assert_eq!(r["usage"]["input_tokens"], 5);
    }

    #[tokio::test]
    async fn array_input_returns_same_length() {
        let (e, _calls) = engine(100, 4, None);
        let r = e.execute(&json!({"input": ["a", "b", "c"]})).await.unwrap();
        let arr = r["embeddings"].as_array().unwrap();
        assert_eq!(arr.len(), 3);
        assert_eq!(r["usage"]["input_tokens"], 15);
    }

    #[tokio::test]
    async fn batch_chunks_at_provider_cap() {
        let (e, calls) = engine(2, 4, None); // provider cap 2
        let r = e
            .execute(&json!({"input": ["a", "b", "c", "d", "e"]}))
            .await
            .unwrap();
        let arr = r["embeddings"].as_array().unwrap();
        assert_eq!(arr.len(), 5);
        // Stub recorded 3 calls: [a,b], [c,d], [e]
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn operator_max_batch_overrides_when_smaller_than_provider() {
        let (e, calls) = engine(100, 4, Some(2)); // operator cap 2 < provider cap 100
        let r = e
            .execute(&json!({"input": ["a", "b", "c", "d"]}))
            .await
            .unwrap();
        assert_eq!(r["embeddings"].as_array().unwrap().len(), 4);
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn missing_input_field_errors() {
        let (e, _calls) = engine(100, 4, None);
        let err = e.execute(&json!({})).await.unwrap_err();
        assert!(matches!(err, BackendError::InvalidSpec { .. }));
    }

    #[tokio::test]
    async fn empty_input_array_errors() {
        let (e, _calls) = engine(100, 4, None);
        let err = e.execute(&json!({"input": []})).await.unwrap_err();
        assert!(matches!(err, BackendError::InvalidSpec { .. }));
    }

    #[tokio::test]
    async fn non_string_array_element_errors() {
        let (e, _calls) = engine(100, 4, None);
        let err = e.execute(&json!({"input": ["ok", 7]})).await.unwrap_err();
        assert!(matches!(err, BackendError::InvalidSpec { .. }));
    }
}
