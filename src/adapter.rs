//! Provider adapter trait — `ChatProviderAdapter`.
//!
//! Each per-provider plugin crate ships a struct implementing this
//! trait. The shared [`crate::engine::ChatEngine`] holds an
//! `Arc<dyn ChatProviderAdapter>` and drives the agentic loop
//! provider-agnostically. Adapters concern themselves only with:
//!
//! - Encoding [`NormalizedChatRequest`] to the provider's wire format.
//! - Dispatching the HTTP call.
//! - Decoding the response into [`NormalizedChatResponse`].
//! - For streaming: parsing the upstream's SSE/chunked frames into
//!   [`NormalizedStreamEvent`]s on a channel.
//!
//! Adapters do **not** retry, do **not** validate operator output
//! schemas, do **not** dispatch child tool calls. Those are the
//! engine's job.

use async_trait::async_trait;
use std::time::Duration;

use crate::error::ProviderError;
use crate::normalized::{NormalizedChatRequest, NormalizedChatResponse};
use crate::streaming::{NormalizedStreamEvent, StreamEventReceiver};

/// One adapter per provider. Object-safe so the engine can hold
/// `Arc<dyn ChatProviderAdapter>` without a generic bubble.
#[async_trait]
pub trait ChatProviderAdapter: Send + Sync {
    /// Provider label for metrics / audit (`"openai"`, `"anthropic"`,
    /// `"gemini"`, `"azure-openai"`, `"openai-compatible"`, …).
    fn label(&self) -> &'static str;

    /// Translate `request` to the provider's wire format, dispatch
    /// over HTTP, and translate the response back. Returns the
    /// canonical shape regardless of provider.
    async fn chat_completion(
        &self,
        request: &NormalizedChatRequest,
        timeout: Duration,
    ) -> Result<NormalizedChatResponse, ProviderError>;

    /// Streaming variant of [`chat_completion`].
    ///
    /// Returns a [`StreamEventReceiver`] the engine drains for
    /// [`NormalizedStreamEvent`]s. The adapter spawns a background
    /// task that pulls upstream chunks (provider-specific SSE / NDJSON
    /// shape), normalizes them, and forwards via the receiver's
    /// channel. The channel closes when the upstream stream ends or a
    /// `Finish` event has been emitted.
    ///
    /// **Default implementation** wraps the non-streaming
    /// `chat_completion` and emits a synthetic event sequence
    /// (`TextDelta` for any content text, `ToolCallReady` per
    /// tool_call, then `Finish`). This makes adapters that haven't
    /// adopted streaming work end-to-end — clients see the same final
    /// content, just without true incremental tokens. Adapters
    /// override to enable real per-token streaming.
    async fn stream_chat_completion(
        &self,
        request: &NormalizedChatRequest,
        timeout: Duration,
    ) -> Result<StreamEventReceiver, ProviderError> {
        let response = self.chat_completion(request, timeout).await?;
        let (tx, rx) = tokio::sync::mpsc::channel(8);
        if !response.content.is_empty() {
            let _ = tx
                .send(Ok(NormalizedStreamEvent::TextDelta(
                    response.content.clone(),
                )))
                .await;
        }
        for tc in response.tool_calls {
            let _ = tx.send(Ok(NormalizedStreamEvent::ToolCallReady(tc))).await;
        }
        let _ = tx
            .send(Ok(NormalizedStreamEvent::Finish {
                reason: response.finish_reason,
                usage: response.usage,
            }))
            .await;
        Ok(rx)
    }
}
