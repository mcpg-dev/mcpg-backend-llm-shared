//! # Image generation adapter — provider-agnostic shape
//!
//! Generated images always land
//! in the gateway's `ContentStore` — the binding response carries
//! `mcpg-resource://<id>` URIs rather than inline base64 blobs, so
//! clients fetch large images on demand via standard MCP
//! `resources/read`.
//!
//! ## Surface
//!
//! - [`NormalizedImageRequest`] / [`NormalizedImageResponse`]: the
//!   provider-agnostic request and response shapes.
//! - [`GeneratedImage`]: one image in a multi-image response.
//! - [`ImageProviderAdapter`]: the per-provider HTTP wire-format
//!   contract; OpenAI / Azure / Gemini implement it.

use async_trait::async_trait;
use std::time::Duration;

use crate::error::ProviderError;

/// Request a generation of `n` images. Adapters that don't support
/// `n > 1` (DALL-E 3 — `n=1` only) MUST surface that as a
/// [`ProviderError::BadRequest`] at the request boundary; the
/// engine doesn't try to fan out and chain calls. Operators wanting
/// multiple from a `n=1`-only model issue multiple binding calls.
#[derive(Debug, Clone)]
pub struct NormalizedImageRequest {
    pub model: String,
    pub prompt: String,
    pub n: u32,
    /// Size string in WxH form (`"1024x1024"` / `"1792x1024"` /
    /// `"1024x1792"` for DALL-E 3). Provider-specific validation
    /// happens at adapter entry.
    pub size: Option<String>,
    /// `standard` | `hd` (DALL-E 3) — provider-specific values pass
    /// through verbatim.
    pub quality: Option<String>,
    /// `natural` | `vivid` (DALL-E 3) — provider-specific values
    /// pass through verbatim.
    pub style: Option<String>,
    /// Provider-supplied seed where deterministic output is
    /// supported (Imagen). Models that ignore seed quietly drop
    /// the field.
    pub seed: Option<i64>,
    /// Stuff the model should *not* generate. Adapters that
    /// support it surface it directly (Stability `negative_prompt`,
    /// Imagen `negativePrompt`); adapters whose models don't
    /// support negative-prompting drop the field.
    pub negative_prompt: Option<String>,
    /// Wire-format hint — `png` | `jpeg` | `webp` for Stability;
    /// `png` | `jpeg` | `webp` for OpenAI gpt-image-1. Adapters
    /// for models that don't accept it drop the field; the
    /// `defaults.output_format` slot on `ImageExecutionSpec` is
    /// the typical operator-facing source.
    pub output_format: Option<String>,
}

#[derive(Debug, Clone)]
pub struct NormalizedImageResponse {
    pub images: Vec<GeneratedImage>,
}

#[derive(Debug, Clone)]
pub struct GeneratedImage {
    /// Raw image bytes — the engine pushes these into the
    /// `ContentStore` and embeds the resulting URI in the binding
    /// response.
    pub bytes: bytes::Bytes,
    /// MIME type the provider returned. PNG is the common case;
    /// some providers (Imagen) return JPEG.
    pub mime_type: String,
    /// DALL-E 3 returns the prompt it actually used — useful for
    /// audit + UI surfacing. `None` from providers that don't
    /// surface this.
    pub revised_prompt: Option<String>,
}

#[async_trait]
pub trait ImageProviderAdapter: Send + Sync + std::fmt::Debug {
    /// Short stable identifier used in metrics labels and audit
    /// records. e.g. `openai`, `azure-openai`, `gemini`.
    fn label(&self) -> &'static str;

    /// Issue one generation request. Adapters return raw bytes for
    /// each image — the engine handles ContentStore put and URI
    /// embedding. Auth + status mapping mirrors chat/embedding.
    async fn generate(
        &self,
        request: &NormalizedImageRequest,
        timeout: Duration,
    ) -> Result<NormalizedImageResponse, ProviderError>;
}
