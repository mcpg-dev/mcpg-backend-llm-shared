//! # mcpg-backend-llm-shared
//!
//! Provider-agnostic core for MCPG's per-provider LLM binding plugins.
//! See `Cargo.toml` for the full list of crates that depend on this
//! one.
//!
//! ## What's here
//!
//! - **Canonical types** ([`normalized`]): `NormalizedChatRequest`,
//!   `NormalizedChatResponse`, `Message`, `ToolCall`, `ToolDef`,
//!   `TokenUsage`, `FinishReason`. The shared currency between the
//!   engine and any provider adapter.
//!
//! - **Streaming primitives** ([`streaming`]): `NormalizedStreamEvent`
//!   and the `StreamEventReceiver` channel alias adapters return from
//!   `stream_chat_completion`.
//!
//! - **Adapter trait** ([`adapter::ChatProviderAdapter`]): the
//!   contract per-provider plugin crates implement.
//!
//! - **Engine** ([`engine::ChatEngine`]): drives the agentic loop,
//!   templating, retry, schema validation, child-tool dispatch.
//!
//! - **Shared config primitives** ([`chat_config`]): provider-
//!   agnostic execution config (`ChatExecutionSpec`, `SamplingSpec`,
//!   `ResponseFormatSpec`, `ToolsSpec`, `RetrySpec`, `GuardrailsSpec`,
//!   `PromptSpec`, `StreamingSpec`).
//!
//! - **Helpers**: prompt templating ([`template`]), SSE event-
//!   boundary parsing ([`sse`]), API-key resolution ([`secret`]),
//!   error taxonomy ([`error`]).
//!
//! ## What's not here
//!
//! Provider-specific HTTP wire format (encoding, decoding, error
//! status mapping) lives in the per-provider plugin crates
//! (`mcpg-plugin-backend-llm-{openai,anthropic,gemini,compat,voyage}`).
//! The gateway integration code (`BackendTypeConfig` variants, route
//! discriminators, dispatcher wiring) lives in `apps/gateway`.

pub mod adapter;
pub mod audio;
pub mod audio_config;
pub mod audio_engine;
pub mod budget;
pub mod cache;
pub mod chat_config;
pub mod content_store;
pub mod content_store_fs;
pub mod cost;
pub mod embedding;
pub mod embedding_config;
pub mod embedding_engine;
pub mod engine;
pub mod image;
pub mod image_config;
pub mod image_engine;
pub mod multimodal;
pub mod secret;
pub mod sse;
pub mod streaming;
pub mod template;

// The pure-types leaf. Re-exported as `error` / `normalized` modules so
// `mcpg_backend_llm_shared::{error, normalized}::…` paths, and this
// crate's own `crate::error` / `crate::normalized` references, keep
// resolving unchanged.
pub use mcpg_backend_llm_types::{error, normalized};

// Re-exports — the public API third-party plugin authors build against.

pub use adapter::ChatProviderAdapter;
pub use audio::{
    NormalizedSttRequest, NormalizedSttResponse, NormalizedTtsRequest, NormalizedTtsResponse,
    SttProviderAdapter, TtsProviderAdapter,
};
pub use audio_config::{SttExecutionSpec, TtsExecutionSpec};
pub use audio_engine::{SttEngine, TtsEngine};
pub use cache::{CacheKey, CacheStats, LruResponseCache, ResponseCache};
pub use chat_config::{
    CacheSpec, ChatExecutionSpec, GuardrailsSpec, IterationExhaustedPolicy, PromptSpec,
    ResponseFormatMode, ResponseFormatSpec, RetryReason, RetrySpec, SamplingSpec,
    SchemaMismatchPolicy, StreamingSpec, ToolChoice, ToolsSpec,
};
pub use content_store::{
    ContentStore, ContentStoreError, ContentStorePlugin, ContentStoreStats, ContentToStore,
    InProcessContentStore, ResourceContent, ResourceHandle,
};
pub use content_store_fs::FileSystemContentStore;
pub use cost::{
    ChatModelEntry, EmbeddingModelEntry, RateCard, RateCardError, bundled_rate_card,
    compute_chat_cost_usd, compute_embedding_cost_usd,
};
pub use embedding::{
    EmbeddingProviderAdapter, EmbeddingTokenUsage, NormalizedEmbeddingRequest,
    NormalizedEmbeddingResponse,
};
pub use embedding_config::{EmbeddingExecutionSpec, EmbeddingRetrySpec};
pub use embedding_engine::EmbeddingEngine;
pub use engine::{ChatEngine, ChildToolMeta, build_child_tool_defs, compile_validator};
pub use error::{ConfigError, ProviderError};
pub use image::{
    GeneratedImage, ImageProviderAdapter, NormalizedImageRequest, NormalizedImageResponse,
};
pub use image_config::{ImageDefaults, ImageExecutionSpec};
pub use image_engine::ImageEngine;
pub use multimodal::{
    DEFAULT_MAX_INLINE_BYTES, MultimodalError, prefetch_message_resources,
    prefetch_messages_resources,
};
pub use normalized::{
    AudioContent, AudioFormat, AudioSource, ContentPart, FileContent, FileSource, FinishReason,
    ImageContent, ImageDetail, ImageSource, Message, MessageContent, NormalizedChatRequest,
    NormalizedChatResponse, Role, TokenUsage, ToolCall, ToolChoiceWire, ToolDef,
};
pub use secret::{ApiKeyRef, resolve_api_key};
pub use streaming::{NormalizedStreamEvent, StreamEventReceiver};
pub use template::{TemplateContext, TemplateMeta, Templates};
