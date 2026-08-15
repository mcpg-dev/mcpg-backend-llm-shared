//! # Embedding adapter — provider-agnostic shape
//!
//! Mirrors the chat-side surface ([`crate::adapter::ChatProviderAdapter`])
//! for embedding endpoints. Each provider that supports embeddings
//! ships a concrete adapter (in the same crate as its chat plugin)
//! plus a thin `BackendPlugin` impl that drives the
//! [`EmbeddingEngine`].
//!
//! Embeddings are pure functions — same model + same input ⇒ same
//! output bytes. That makes them ideal cache fodder; the
//! response-cache layer keys on
//! `(model, normalized_input)` and dedupes calls automatically.

use async_trait::async_trait;
use std::time::Duration;

use crate::error::ProviderError;

/// Inputs to one embedding call. Adapters always send an array
/// upstream — scalar input is sent as a length-1 array. The engine
/// handles input fan-out + reassembly.
#[derive(Debug, Clone)]
pub struct NormalizedEmbeddingRequest {
    pub model: String,
    /// Texts to embed. Sent verbatim to the provider; the engine
    /// shoulders any pre-tokenisation / preprocessing the operator
    /// asked for.
    pub inputs: Vec<String>,
    /// Reduce vector to this many dimensions when the provider
    /// supports it (OpenAI 3-series, Voyage). `None` = native
    /// dimensionality. Provider-side enforcement; adapters that
    /// don't honour it pass through unchanged.
    pub dimensions: Option<u32>,
}

/// Provider response. Vector ordering matches the request inputs;
/// `usage` is best-effort and may be `None` on providers that don't
/// surface token counts on embedding calls (Voyage notably).
#[derive(Debug, Clone)]
pub struct NormalizedEmbeddingResponse {
    pub embeddings: Vec<Vec<f32>>,
    pub dimensions: u32,
    pub usage: Option<EmbeddingTokenUsage>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct EmbeddingTokenUsage {
    pub input_tokens: u32,
}

/// Provider-specific HTTP wire-format encoder/decoder. Mirror of
/// [`crate::adapter::ChatProviderAdapter`] for embeddings.
#[async_trait]
pub trait EmbeddingProviderAdapter: Send + Sync + std::fmt::Debug {
    /// Short stable identifier used in metrics labels and audit
    /// records. e.g. `openai`, `azure-openai`, `gemini`,
    /// `openai-compatible`, `voyage`.
    fn label(&self) -> &'static str;

    /// Provider-side hard cap on inputs per request. The
    /// [`EmbeddingEngine`] respects this when chunking large
    /// operator-supplied batches.
    fn max_batch_size(&self) -> usize;

    /// Issue one embedding request. Adapters MUST split larger
    /// batches at `max_batch_size` themselves OR rely on the engine
    /// (the engine does the split — this method always sees a batch
    /// the adapter can send in one upstream call).
    async fn embed(
        &self,
        request: &NormalizedEmbeddingRequest,
        timeout: Duration,
    ) -> Result<NormalizedEmbeddingResponse, ProviderError>;
}
