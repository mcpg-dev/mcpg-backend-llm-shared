//! Operator-facing config for TTS + STT bindings.

use serde::{Deserialize, Serialize};

use crate::embedding_config::EmbeddingRetrySpec;
use crate::error::ConfigError;
use crate::normalized::AudioFormat;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtsExecutionSpec {
    pub model: String,

    /// Default voice. Per-call args may override.
    pub voice: String,

    /// Default output format.
    #[serde(default = "default_audio_format")]
    pub format: AudioFormat,

    /// Default speed multiplier (1.0 = normal).
    #[serde(default)]
    pub speed: Option<f32>,

    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: u64,
    #[serde(default = "default_connect_timeout_ms")]
    pub connect_timeout_ms: u64,

    #[serde(default)]
    pub retry: EmbeddingRetrySpec,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SttExecutionSpec {
    pub model: String,

    /// Default ISO-639-1 hint passed to the recogniser. Per-call
    /// args may override.
    #[serde(default)]
    pub language: Option<String>,

    /// Per-call inline byte cap when the input is `mcpg-resource://`
    /// or URL — the engine's `multimodal::prefetch` resolver uses
    /// this to refuse oversized inputs early. `None` = use
    /// [`crate::multimodal::DEFAULT_MAX_INLINE_BYTES`].
    #[serde(default)]
    pub max_input_bytes: Option<usize>,

    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: u64,
    #[serde(default = "default_connect_timeout_ms")]
    pub connect_timeout_ms: u64,

    #[serde(default)]
    pub retry: EmbeddingRetrySpec,
}

fn default_timeout_ms() -> u64 {
    60_000
}

fn default_connect_timeout_ms() -> u64 {
    5_000
}

fn default_audio_format() -> AudioFormat {
    AudioFormat::Mp3
}

impl Default for TtsExecutionSpec {
    fn default() -> Self {
        Self {
            model: String::new(),
            voice: String::new(),
            format: default_audio_format(),
            speed: None,
            timeout_ms: default_timeout_ms(),
            connect_timeout_ms: default_connect_timeout_ms(),
            retry: Default::default(),
        }
    }
}

impl Default for SttExecutionSpec {
    fn default() -> Self {
        Self {
            model: String::new(),
            language: None,
            max_input_bytes: None,
            timeout_ms: default_timeout_ms(),
            connect_timeout_ms: default_connect_timeout_ms(),
            retry: Default::default(),
        }
    }
}

impl TtsExecutionSpec {
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.model.trim().is_empty() {
            return Err(ConfigError::InvalidSpec("model must not be empty".into()));
        }
        if self.voice.trim().is_empty() {
            return Err(ConfigError::InvalidSpec("voice must not be empty".into()));
        }
        if let Some(s) = self.speed
            && (!(0.25..=4.0).contains(&s))
        {
            return Err(ConfigError::InvalidSpec(
                "speed must be between 0.25 and 4.0".into(),
            ));
        }
        if self.timeout_ms == 0 {
            return Err(ConfigError::InvalidSpec("timeout_ms must be > 0".into()));
        }
        if self.connect_timeout_ms == 0 {
            return Err(ConfigError::InvalidSpec(
                "connect_timeout_ms must be > 0".into(),
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

impl SttExecutionSpec {
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

    #[test]
    fn tts_default_with_required_fields_validates() {
        let s = TtsExecutionSpec {
            model: "tts-1".into(),
            voice: "alloy".into(),
            ..Default::default()
        };
        s.validate().unwrap();
    }

    #[test]
    fn tts_speed_must_be_in_range() {
        let mut s = TtsExecutionSpec {
            model: "tts-1".into(),
            voice: "alloy".into(),
            ..Default::default()
        };
        s.speed = Some(0.0);
        assert!(s.validate().is_err());
        s.speed = Some(5.0);
        assert!(s.validate().is_err());
        s.speed = Some(2.0);
        s.validate().unwrap();
    }

    #[test]
    fn stt_default_validates_with_model() {
        let s = SttExecutionSpec {
            model: "whisper-1".into(),
            ..Default::default()
        };
        s.validate().unwrap();
    }

    #[test]
    fn empty_model_rejected() {
        let s = TtsExecutionSpec::default();
        assert!(s.validate().is_err());
        let s = SttExecutionSpec::default();
        assert!(s.validate().is_err());
    }
}
