//! Operator-facing config primitives for embedding bindings.
//!
//! Each per-provider crate's spec carries provider-specific knobs
//! (`api_key`, `base_url` overrides) and flattens
//! [`EmbeddingExecutionSpec`] for the common surface — same
//! flatten-passthrough pattern as `ChatExecutionSpec`.

use serde::{Deserialize, Serialize};

use crate::error::ConfigError;

/// Provider-agnostic execution config for embedding bindings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingExecutionSpec {
    pub model: String,

    /// Reduce vectors to this many dimensions. Only honoured by
    /// providers that support it (OpenAI 3-series, Voyage). Adapters
    /// that don't support it ignore the field.
    #[serde(default)]
    pub dimensions: Option<u32>,

    /// Per-call timeout. Embedding endpoints are typically faster
    /// than chat — default is intentionally tight.
    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: u64,

    /// Connect timeout for the underlying HTTP client.
    #[serde(default = "default_connect_timeout_ms")]
    pub connect_timeout_ms: u64,

    /// Operator-specified per-call max batch size. Capped by the
    /// provider's own hard limit
    /// ([`crate::EmbeddingProviderAdapter::max_batch_size`]).
    /// Larger inputs split into multiple parallel calls.
    #[serde(default)]
    pub max_batch_size: Option<usize>,

    /// Retry policy for transient upstream errors. The defaults
    /// match the chat-side spec (3 attempts, 200 ms initial backoff
    /// with exponential ramp).
    #[serde(default)]
    pub retry: EmbeddingRetrySpec,

    /// Response cache. Embeddings are pure functions —
    /// `text → vector` is deterministic — so the default TTL is one
    /// day. Caching is still off-by-default (operators must
    /// `enabled: true`); the engine never silently dedups otherwise.
    #[serde(default = "default_embedding_cache_spec")]
    pub cache: crate::chat_config::CacheSpec,
}

fn default_embedding_cache_spec() -> crate::chat_config::CacheSpec {
    crate::chat_config::CacheSpec {
        enabled: false,
        ttl_seconds: 86_400,
    }
}

impl Default for EmbeddingExecutionSpec {
    fn default() -> Self {
        Self {
            model: String::new(),
            dimensions: None,
            timeout_ms: default_timeout_ms(),
            connect_timeout_ms: default_connect_timeout_ms(),
            max_batch_size: None,
            retry: EmbeddingRetrySpec::default(),
            cache: default_embedding_cache_spec(),
        }
    }
}

fn default_timeout_ms() -> u64 {
    10_000
}

fn default_connect_timeout_ms() -> u64 {
    5_000
}

/// Subset of the chat-side `RetrySpec` relevant to embeddings.
/// Embeddings have no agentic loop, no tool-result iteration —
/// just provider-call retry on rate-limit / 5xx.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingRetrySpec {
    #[serde(default = "default_max_attempts")]
    pub max_attempts: u32,
    #[serde(default = "default_initial_backoff_ms")]
    pub initial_backoff_ms: u64,
    #[serde(default = "default_max_backoff_ms")]
    pub max_backoff_ms: u64,
}

impl Default for EmbeddingRetrySpec {
    fn default() -> Self {
        Self {
            max_attempts: default_max_attempts(),
            initial_backoff_ms: default_initial_backoff_ms(),
            max_backoff_ms: default_max_backoff_ms(),
        }
    }
}

fn default_max_attempts() -> u32 {
    3
}

fn default_initial_backoff_ms() -> u64 {
    200
}

fn default_max_backoff_ms() -> u64 {
    2_000
}

impl EmbeddingExecutionSpec {
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.model.trim().is_empty() {
            return Err(ConfigError::InvalidSpec("model must not be empty".into()));
        }
        if self.timeout_ms == 0 {
            return Err(ConfigError::InvalidSpec("timeout_ms must be > 0".into()));
        }
        if self.connect_timeout_ms == 0 {
            return Err(ConfigError::InvalidSpec(
                "connect_timeout_ms must be > 0".into(),
            ));
        }
        if let Some(b) = self.max_batch_size
            && b == 0
        {
            return Err(ConfigError::InvalidSpec(
                "max_batch_size must be > 0 when set".into(),
            ));
        }
        if self.retry.max_attempts == 0 {
            return Err(ConfigError::InvalidSpec(
                "retry.max_attempts must be > 0".into(),
            ));
        }
        Ok(())
    }

    pub fn timeout(&self) -> std::time::Duration {
        std::time::Duration::from_millis(self.timeout_ms)
    }

    pub fn connect_timeout(&self) -> std::time::Duration {
        std::time::Duration::from_millis(self.connect_timeout_ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn default_values_pass_validation() {
        let s = EmbeddingExecutionSpec {
            model: "text-embedding-3-small".into(),
            ..Default::default()
        };
        s.validate().unwrap();
        assert_eq!(s.timeout_ms, 10_000);
        assert_eq!(s.retry.max_attempts, 3);
    }

    #[test]
    fn json_round_trip() {
        let v = json!({
            "model": "text-embedding-3-large",
            "dimensions": 1536,
            "timeout_ms": 8000,
            "connect_timeout_ms": 4000,
            "max_batch_size": 512,
            "retry": {"max_attempts": 5}
        });
        let s: EmbeddingExecutionSpec = serde_json::from_value(v).unwrap();
        s.validate().unwrap();
        assert_eq!(s.dimensions, Some(1536));
        assert_eq!(s.max_batch_size, Some(512));
        assert_eq!(s.retry.max_attempts, 5);
    }

    #[test]
    fn empty_model_rejected() {
        let s = EmbeddingExecutionSpec {
            model: "  ".into(),
            ..Default::default()
        };
        assert!(s.validate().is_err());
    }

    #[test]
    fn zero_max_batch_size_rejected() {
        let s = EmbeddingExecutionSpec {
            model: "x".into(),
            max_batch_size: Some(0),
            ..Default::default()
        };
        assert!(s.validate().is_err());
    }
}
