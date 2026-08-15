//! Provider-agnostic streaming primitives.
//!
//! Each [`crate::adapter::ChatProviderAdapter`] that supports
//! streaming returns a [`StreamEventReceiver`] from
//! `stream_chat_completion`. The engine drains it, mapping events
//! to the engine's loop steps and emitting
//! [`mcpg_plugin_protocol::BackendChunk`]s on its own outbound
//! channel.
//!
//! Adapters that haven't implemented native streaming inherit the
//! default trait impl that synthesizes events from the non-streaming
//! `chat_completion` (see [`crate::adapter::ChatProviderAdapter`]).

use crate::error::ProviderError;
use crate::normalized::{FinishReason, TokenUsage, ToolCall};

/// One unit of normalized streaming output. Provider adapters emit
/// these as upstream tokens / SSE events arrive; the engine
/// consumes them and emits [`mcpg_plugin_protocol::BackendChunk`]s.
#[derive(Debug, Clone)]
pub enum NormalizedStreamEvent {
    /// Incremental text content. Concatenated across the stream gives
    /// the iteration's full assistant content.
    TextDelta(String),
    /// A tool call has fully accumulated. Adapters buffer fragmented
    /// arguments and emit this once the call is complete (the
    /// upstream's signal varies — for OpenAI the next tool_call's
    /// `index` flipping or the finish_reason arriving).
    ToolCallReady(ToolCall),
    /// Iteration finished. `usage` carries the per-iteration token
    /// counts. `reason` is the upstream's terminal classification.
    Finish {
        reason: FinishReason,
        usage: TokenUsage,
    },
}

/// Channel receiver for normalized stream events. The underlying
/// channel is closed when the adapter's stream task exits — use
/// [`tokio_stream::wrappers::ReceiverStream`] in callers that want
/// `Stream` semantics. We keep the type as a plain `Receiver`
/// so the engine can `recv()` on it without the wrapper crate's
/// surface bleeding into the trait.
pub type StreamEventReceiver =
    tokio::sync::mpsc::Receiver<Result<NormalizedStreamEvent, ProviderError>>;
