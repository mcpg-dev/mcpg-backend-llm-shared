//! Operator-facing config for image-generation bindings.
//!
//! Each per-provider crate's spec carries provider-specific knobs
//! (`api_key`, `base_url` overrides) and flattens
//! [`ImageExecutionSpec`] for the common surface — same pattern as
//! `ChatExecutionSpec` / `EmbeddingExecutionSpec`.

use serde::{Deserialize, Serialize};

use crate::error::ConfigError;

/// Provider-agnostic execution config for image-generation bindings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageExecutionSpec {
    pub model: String,

    /// Per-call timeout. Image generation is slower than chat /
    /// embedding — default 60 s.
    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: u64,

    #[serde(default = "default_connect_timeout_ms")]
    pub connect_timeout_ms: u64,

    /// Defaults applied when the per-call args don't specify the
    /// corresponding field. Operators encode their preferred shape
    /// once; users supply only what they want to vary.
    #[serde(default)]
    pub defaults: ImageDefaults,

    #[serde(default)]
    pub retry: super::embedding_config::EmbeddingRetrySpec,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ImageDefaults {
    #[serde(default)]
    pub size: Option<String>,
    #[serde(default)]
    pub quality: Option<String>,
    #[serde(default)]
    pub style: Option<String>,
    /// Default `n` images per call. Capped at 1 by adapters that
    /// don't support multiple (DALL-E 3); clamped at the provider
    /// boundary.
    #[serde(default)]
    pub n: Option<u32>,
    /// Default negative-prompt — supported by Stability core/sd3
    /// and Imagen 3. Adapters whose models don't accept it drop
    /// the field.
    #[serde(default)]
    pub negative_prompt: Option<String>,
    /// Default wire-format hint — `png` | `jpeg` | `webp`.
    /// Stability uses this for the `output_format` form field;
    /// OpenAI passes it through for gpt-image-1; DALL-E and
    /// Imagen ignore it.
    #[serde(default)]
    pub output_format: Option<String>,
}

fn default_timeout_ms() -> u64 {
    60_000
}

fn default_connect_timeout_ms() -> u64 {
    5_000
}

impl Default for ImageExecutionSpec {
    fn default() -> Self {
        Self {
            model: String::new(),
            timeout_ms: default_timeout_ms(),
            connect_timeout_ms: default_connect_timeout_ms(),
            defaults: ImageDefaults::default(),
            retry: Default::default(),
        }
    }
}

impl ImageExecutionSpec {
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
        if let Some(n) = self.defaults.n
            && n == 0
        {
            return Err(ConfigError::InvalidSpec(
                "defaults.n must be > 0 when set".into(),
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
    fn default_validates_with_model_set() {
        let s = ImageExecutionSpec {
            model: "dall-e-3".into(),
            ..Default::default()
        };
        s.validate().unwrap();
    }

    #[test]
    fn json_round_trip_with_defaults() {
        let v = json!({
            "model": "dall-e-3",
            "defaults": {
                "size": "1792x1024",
                "quality": "hd",
                "style": "vivid",
                "n": 1,
                "negative_prompt": "blurry, low quality",
                "output_format": "webp"
            }
        });
        let s: ImageExecutionSpec = serde_json::from_value(v).unwrap();
        s.validate().unwrap();
        assert_eq!(s.defaults.size.as_deref(), Some("1792x1024"));
        assert_eq!(s.defaults.quality.as_deref(), Some("hd"));
        assert_eq!(s.defaults.n, Some(1));
        assert_eq!(
            s.defaults.negative_prompt.as_deref(),
            Some("blurry, low quality")
        );
        assert_eq!(s.defaults.output_format.as_deref(), Some("webp"));
    }

    #[test]
    fn empty_model_rejected() {
        let s = ImageExecutionSpec {
            model: "  ".into(),
            ..Default::default()
        };
        assert!(s.validate().is_err());
    }

    #[test]
    fn zero_default_n_rejected() {
        let s = ImageExecutionSpec {
            model: "x".into(),
            defaults: ImageDefaults {
                n: Some(0),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(s.validate().is_err());
    }
}
