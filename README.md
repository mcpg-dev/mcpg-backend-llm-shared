# mcpg-backend-llm-shared

> Provider-agnostic core for MCPG's LLM binding plugins: canonical types, the chat/embedding/image/speech engines, and the shared operator-facing config specs.

This crate is the part of an MCPG LLM plugin that has nothing to do with any
particular vendor. It owns the canonical request and response shapes every
adapter converts to and from, the engines that drive an actual call — prompt
templating, the agentic tool-calling loop, JSON Schema validation, retry,
streaming, response caching, cost accounting and budget enforcement — and the
config structs that give every provider the same operator-facing YAML. It is
deliberately *not* a plugin: there is no `plugin.yaml`, no `mcpg_plugin_register`
symbol, and nothing here speaks a provider's HTTP wire format. Encoding,
decoding and provider-specific error mapping live in the per-provider crates
that depend on this one.

## What's here

- **Canonical types** (the `normalized` module, re-exported from
  `mcpg-backend-llm-types`): `NormalizedChatRequest`, `NormalizedChatResponse`,
  `Message`, `MessageContent`, `ContentPart`, `ImageContent`, `AudioContent`,
  `FileContent`, `Role`, `ToolCall`, `ToolDef`, `ToolChoiceWire`, `TokenUsage`,
  `FinishReason`.
- **Adapter traits** — the contract a provider crate implements:
  `ChatProviderAdapter`, `EmbeddingProviderAdapter`, `ImageProviderAdapter`,
  `TtsProviderAdapter`, `SttProviderAdapter`. Adapters encode, dispatch and
  decode; they never retry, validate operator schemas or dispatch child tools.
- **Engines** — `ChatEngine` (agentic loop, templating, validation, retry,
  cache, budget), `EmbeddingEngine` (batch splitting and stitching),
  `ImageEngine`, `TtsEngine`, `SttEngine` — plus `build_child_tool_defs`,
  `ChildToolMeta` and `compile_validator`.
- **Operator-facing config specs** — `ChatExecutionSpec` and the pieces it
  flattens (`PromptSpec`, `SamplingSpec`, `ResponseFormatSpec`, `ToolsSpec`,
  `RetrySpec`, `GuardrailsSpec`, `StreamingSpec`, `CacheSpec`), plus
  `EmbeddingExecutionSpec`, `EmbeddingRetrySpec`, `ImageExecutionSpec`,
  `ImageDefaults`, `TtsExecutionSpec` and `SttExecutionSpec`. Each carries its
  own `validate()`; a provider spec embeds one with `#[serde(flatten)]` and
  layers its own invariants on top, which is why the YAML for every MCPG chat
  binding looks the same.
- **Streaming** — `NormalizedStreamEvent` and the `StreamEventReceiver` channel
  alias an adapter returns from `stream_chat_completion`, plus SSE
  event-boundary parsing in the `sse` module.
- **Content store** — `ContentStore`, `ContentStorePlugin`, `ContentToStore`,
  `ResourceHandle`, `ResourceContent`, `ContentStoreStats`, with two
  implementations: `InProcessContentStore` and `FileSystemContentStore`.
  Generated image and audio bytes go in here and come back as
  `mcpg-resource://<id>` URIs.
- **Response cache** — the `ResponseCache` trait, `CacheKey`, `CacheStats` and
  the `LruResponseCache` implementation.
- **Cost and budget** — `RateCard`, `ChatModelEntry`, `EmbeddingModelEntry`,
  `bundled_rate_card`, `compute_chat_cost_usd`, `compute_embedding_cost_usd`,
  and the process-wide per-binding daily-USD ledger in the `budget` module.
  Pricing comes from the `models.toml` vendored in this crate; a model the card
  does not list is not priced, so cost counters stay silent rather than
  reporting a misleading zero.
- **Multimodal resolution** — `prefetch_message_resources`,
  `prefetch_messages_resources`, `MultimodalError` and
  `DEFAULT_MAX_INLINE_BYTES` (20 MiB), which turn `mcpg-resource://` URIs into
  inline base64 before an adapter ever sees them.
- **Templating** — `Templates`, `TemplateContext`, `TemplateMeta`. A locked-down
  MiniJinja environment: templates see only `input.*` and `meta.*`, undefined
  variables are strict errors, and there is no filesystem loader, no env-var
  lookup and no `debug` filter, so a template cannot exfiltrate host state.
- **Secrets and errors** — `ApiKeyRef` (a redacting wrapper whose `Debug`
  renders `***`), `resolve_api_key`, `ConfigError`, `ProviderError`.

Rust edition 2024, Apache-2.0. No cargo features — the whole surface is always
compiled.

## Used by

- The first-party LLM binding plugins under `libs/plugins/backend/llms/`
  (`openai`, `anthropic`, `gemini`, `compat`, `stability`), which supply only
  the wire-format adapter for their provider.
- The content-store plugins `libs/plugins/storage/builtin` and
  `libs/plugins/storage/s3`, which implement `ContentStorePlugin`.
- The MCPG gateway itself, for `ContentStore`, `ResponseCache` and `CacheKey` on
  its backend-host surface.
- Third-party crates adding a provider MCPG does not ship: implement one adapter
  trait and you inherit the engine, the config schema and the observability.

## Usage

```toml
[dependencies]
mcpg-backend-llm-shared = "<version>"
async-trait = "0.1"
```

A minimal chat adapter. The engine supplies everything else, including a default
`stream_chat_completion` that synthesises a `TextDelta` / `ToolCallReady` /
`Finish` sequence from the non-streaming call, so an adapter works end-to-end
before it adopts real incremental streaming:

```rust
use std::time::Duration;

use async_trait::async_trait;
use mcpg_backend_llm_shared::{
    ChatProviderAdapter, FinishReason, NormalizedChatRequest, NormalizedChatResponse,
    ProviderError, TokenUsage,
};

pub struct EchoAdapter;

#[async_trait]
impl ChatProviderAdapter for EchoAdapter {
    fn label(&self) -> &'static str {
        "echo"
    }

    async fn chat_completion(
        &self,
        request: &NormalizedChatRequest,
        _timeout: Duration,
    ) -> Result<NormalizedChatResponse, ProviderError> {
        Ok(NormalizedChatResponse {
            content: format!("model={} messages={}", request.model, request.messages.len()),
            tool_calls: Vec::new(),
            finish_reason: FinishReason::Stop,
            usage: TokenUsage::default(),
        })
    }
}
```

Embed `ChatExecutionSpec` to inherit the shared operator-facing schema, then add
only what your provider needs on top:

```rust
use mcpg_backend_llm_shared::{ApiKeyRef, ChatExecutionSpec, ConfigError};
use serde::Deserialize;

#[derive(Deserialize)]
pub struct EchoChatSpec {
    #[serde(default)]
    pub base_url: Option<String>,
    pub api_key: ApiKeyRef,
    #[serde(flatten)]
    pub chat: ChatExecutionSpec,
}

impl EchoChatSpec {
    pub fn validate(&self) -> Result<(), ConfigError> {
        // Shared invariants first; layer provider-specific checks after.
        self.chat.validate()
    }
}
```

## Build / test

```bash
cargo build -p mcpg-backend-llm-shared
cargo test  -p mcpg-backend-llm-shared
```

## See also

- What a plugin is, and the ABI it loads across: <https://mcpg.dev/docs/plugins/plugins-and-protocol>
- Writing a plugin: <https://mcpg.dev/docs/plugins/plugin-authoring>
- The pure-types leaf this crate re-exports: `libs/plugins/backend/llms/types`
- Reference adapter implementations: `libs/plugins/backend/llms/openai`, `libs/plugins/backend/llms/anthropic`
