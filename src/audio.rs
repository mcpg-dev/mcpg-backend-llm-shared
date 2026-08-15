//! # TTS / STT — provider-agnostic shape
//!
//! Two parallel surfaces:
//!
//! - **TTS** (`text → audio bytes`): output bytes land in the
//!   gateway's `ContentStore`; the binding response carries
//!   `mcpg-resource://<id>` URIs.
//! - **STT** (`audio bytes → text`): input audio resolved via
//!   `multimodal::resolve_audio_source` (URL / base64 /
//!   `mcpg-resource://`). Output is a structured `{text, language?,
//!   duration_seconds?}` JSON blob.
//!
//! Both share the same retry shape as embeddings (no agentic loop,
//! no streaming).

use async_trait::async_trait;
use std::time::Duration;

use crate::error::ProviderError;
use crate::normalized::AudioFormat;

// ---------------------------------------------------------------------------
// TTS
// ---------------------------------------------------------------------------

/// Inputs to one TTS call.
#[derive(Debug, Clone)]
pub struct NormalizedTtsRequest {
    pub model: String,
    pub text: String,
    /// Voice identifier (`alloy` / `echo` / `nova` / etc. on
    /// OpenAI). Provider-specific values pass through verbatim.
    pub voice: String,
    /// Output format. The provider returns bytes encoded as this
    /// format; the resulting MIME type is derived via
    /// [`AudioFormat::mime_type`].
    pub format: AudioFormat,
    /// Speed multiplier (`1.0` = normal). OpenAI accepts 0.25–4.0;
    /// adapters validate the range.
    pub speed: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct NormalizedTtsResponse {
    /// Encoded audio bytes (mp3, wav, etc. — caller infers format
    /// from the request).
    pub bytes: bytes::Bytes,
    /// MIME type the provider returned. Adapters derive this from
    /// `request.format` when the upstream doesn't carry an explicit
    /// content-type header.
    pub mime_type: String,
}

#[async_trait]
pub trait TtsProviderAdapter: Send + Sync + std::fmt::Debug {
    fn label(&self) -> &'static str;

    async fn synthesize(
        &self,
        request: &NormalizedTtsRequest,
        timeout: Duration,
    ) -> Result<NormalizedTtsResponse, ProviderError>;
}

// ---------------------------------------------------------------------------
// STT
// ---------------------------------------------------------------------------

/// Inputs to one STT call. Audio bytes are resolved upstream by the
/// engine using `multimodal::resolve_audio_source`; adapters always
/// receive concrete `Bytes`.
#[derive(Debug, Clone)]
pub struct NormalizedSttRequest {
    pub model: String,
    pub bytes: bytes::Bytes,
    /// MIME type of the audio bytes. Adapters that need a
    /// filename-extension hint derive it from the MIME (audio/mpeg
    /// → `audio.mp3`).
    pub mime_type: String,
    /// ISO-639-1 language hint for the recogniser. `None` lets the
    /// provider auto-detect.
    pub language: Option<String>,
    /// Optional prompt to bias the recogniser (OpenAI Whisper
    /// supports this; Gemini ignores it).
    pub prompt: Option<String>,
}

#[derive(Debug, Clone)]
pub struct NormalizedSttResponse {
    pub text: String,
    /// Detected language code (`en` / `de` / etc.). `None` from
    /// providers that don't surface it.
    pub language: Option<String>,
    /// Audio duration in seconds, when reported. OpenAI Whisper
    /// returns this in verbose response shapes; basic shapes don't.
    pub duration_seconds: Option<f64>,
}

#[async_trait]
pub trait SttProviderAdapter: Send + Sync + std::fmt::Debug {
    fn label(&self) -> &'static str;

    async fn transcribe(
        &self,
        request: &NormalizedSttRequest,
        timeout: Duration,
    ) -> Result<NormalizedSttResponse, ProviderError>;
}
