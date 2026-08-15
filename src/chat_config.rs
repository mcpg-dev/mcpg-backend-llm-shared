//! Provider-agnostic chat-binding config primitives.
//!
//! Every per-provider plugin's spec embeds [`ChatExecutionSpec`] via
//! `#[serde(flatten)]` and adds only the provider-specific knobs on
//! top (api_key, base_url, optional connect timeout). This keeps the
//! operator-facing YAML shape identical across providers for every
//! field that's semantically the same — sampling, response_format,
//! tools, retry, guardrails, prompt — so switching providers means
//! changing the binding `type` and `model`, not relearning the
//! config schema.
//!
//! All structs are `Serialize + Deserialize + Clone`. None encode a
//! provider identity.

use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::error::ConfigError;

/// The provider-agnostic execution config every chat plugin shares.
///
/// Per-provider config wraps this via `#[serde(flatten)]`:
///
/// ```ignore
/// #[derive(Deserialize)]
/// struct OpenAiChatSpec {
///     #[serde(default)]
///     base_url: Option<String>,
///     api_key: ApiKeyRef,
///     #[serde(flatten)]
///     chat: ChatExecutionSpec,
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatExecutionSpec {
    pub model: String,

    /// Total per-iteration wall-clock budget upstream (milliseconds).
    /// Includes retries within one iteration; the agentic loop's overall
    /// budget is `max_iterations * timeout_ms`.
    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: u64,

    /// TCP connect timeout. Separate from `timeout_ms` so a slow
    /// upstream that has *connected* doesn't get prematurely killed.
    #[serde(default = "default_connect_timeout_ms")]
    pub connect_timeout_ms: u64,

    pub prompt: PromptSpec,

    #[serde(default)]
    pub sampling: SamplingSpec,

    #[serde(default)]
    pub response_format: ResponseFormatSpec,

    #[serde(default)]
    pub tools: ToolsSpec,

    #[serde(default)]
    pub streaming: StreamingSpec,

    #[serde(default)]
    pub retry: RetrySpec,

    #[serde(default)]
    pub guardrails: GuardrailsSpec,

    /// Response cache. Off by default; operators opt in per binding.
    /// Cache is sound for `temperature: 0` + no tools + no streaming.
    /// The validator does NOT enforce determinism — that's the
    /// operator's responsibility — but does refuse `enabled: true`
    /// when the binding has `tools.allowed` non-empty (cache + agentic
    /// loop is meaningless: child tools can write external state).
    #[serde(default)]
    pub cache: CacheSpec,

    /// Per-binding budget caps. Both fields default to `0`
    /// (uncapped). When a cap is exceeded, the engine refuses the
    /// call with a `BackendError::Transport` whose message starts
    /// with `budget refused:` so the gateway logs surface a clean
    /// classification. See `crate::budget` for the daily ledger.
    #[serde(default)]
    pub budget: BudgetSpec,
}

impl ChatExecutionSpec {
    pub fn timeout(&self) -> Duration {
        Duration::from_millis(self.timeout_ms)
    }

    pub fn connect_timeout(&self) -> Duration {
        Duration::from_millis(self.connect_timeout_ms)
    }

    /// Apply provider-agnostic invariants. Per-provider specs add
    /// their own `validate()` calling this and layering provider-
    /// specific checks (e.g., Azure requires base_url).
    pub fn validate(&self) -> Result<(), ConfigError> {
        // Mismatch policy / mode legality.
        if matches!(
            self.response_format.on_mismatch,
            SchemaMismatchPolicy::ReturnRaw
        ) && !matches!(self.response_format.mode, ResponseFormatMode::Text)
        {
            return Err(ConfigError::InvalidSpec(
                "response_format.on_mismatch=return_raw is only allowed with mode=text".into(),
            ));
        }

        // Iteration bounds.
        let max_iter = self.tools.resolved_max_iterations();
        if max_iter == 0 {
            return Err(ConfigError::InvalidSpec(
                "tools.max_iterations must be >= 1".into(),
            ));
        }
        if max_iter > 50 {
            return Err(ConfigError::InvalidSpec(format!(
                "tools.max_iterations={max_iter} exceeds safety cap of 50"
            )));
        }

        // Timeouts.
        if self.timeout_ms == 0 {
            return Err(ConfigError::InvalidSpec("timeout_ms must be > 0".into()));
        }
        if self.connect_timeout_ms == 0 {
            return Err(ConfigError::InvalidSpec(
                "connect_timeout_ms must be > 0".into(),
            ));
        }

        // Retry consistency.
        if self.retry.max_attempts == 0 {
            return Err(ConfigError::InvalidSpec(
                "retry.max_attempts must be >= 1".into(),
            ));
        }
        if self.retry.initial_backoff_ms > self.retry.max_backoff_ms {
            return Err(ConfigError::InvalidSpec(
                "retry.initial_backoff_ms must be <= retry.max_backoff_ms".into(),
            ));
        }

        // Cache + tools is meaningless — child tools have side
        // effects and the cache key wouldn't capture child-call
        // results. Refuse loud rather than silently caching wrong
        // outputs.
        if self.cache.enabled && !self.tools.allowed.is_empty() {
            return Err(ConfigError::InvalidSpec(
                "cache.enabled=true is incompatible with non-empty tools.allowed".into(),
            ));
        }

        // Empty allowed-tool names are nonsense.
        if self.tools.allowed.iter().any(|t| t.trim().is_empty()) {
            return Err(ConfigError::InvalidSpec(
                "tools.allowed entries must be non-empty binding names".into(),
            ));
        }

        // Prompts must be non-empty after trimming.
        if self.prompt.system.trim().is_empty() {
            return Err(ConfigError::InvalidSpec(
                "prompt.system must be non-empty".into(),
            ));
        }
        if self.prompt.user.trim().is_empty() {
            return Err(ConfigError::InvalidSpec(
                "prompt.user must be non-empty".into(),
            ));
        }

        if self.budget.usd_daily_cap.is_sign_negative() || self.budget.usd_daily_cap.is_nan() {
            return Err(ConfigError::InvalidSpec(
                "budget.usd_daily_cap must be a non-negative finite number".into(),
            ));
        }

        Ok(())
    }
}

/// Per-binding response-cache configuration. Caching is opt-in:
/// chat with `temperature: 0` and no tools is a candidate; chat with
/// non-zero temperature or active tool calls is not (output is
/// non-deterministic). Embeddings turn it on by default in their
/// spec (see [`crate::EmbeddingExecutionSpec`]).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheSpec {
    /// `true` to look up + write entries on every call.
    #[serde(default)]
    pub enabled: bool,

    /// Per-entry TTL. `0` = no expiry (still bounded by the gateway's
    /// configured byte cap and LRU). Default 1 hour for chat; the
    /// embedding spec overrides to 1 day.
    #[serde(default = "default_cache_ttl_ms")]
    pub ttl_seconds: u64,
}

impl Default for CacheSpec {
    fn default() -> Self {
        Self {
            enabled: false,
            ttl_seconds: default_cache_ttl_ms(),
        }
    }
}

fn default_cache_ttl_ms() -> u64 {
    3600000
}

/// Per-binding budget caps. Both fields default to `0`
/// (uncapped). The engine evaluates them in [`crate::engine`]:
///
/// - `tokens_per_call_cap` is checked between agentic-loop
///   iterations against the running sum of `input_tokens +
///   output_tokens`. The first iteration is never refused (we
///   need at least one upstream call to even know the cap is
///   approaching); subsequent iterations are gated.
///
/// - `usd_daily_cap` is checked at the *start* of each
///   `execute()` call against a per-binding running daily total
///   maintained in [`crate::budget`]. Resets at UTC midnight.
///   Models not in the rate card cannot accumulate cost — calls
///   on those models still pass through with a logged warning.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BudgetSpec {
    /// Maximum total tokens (input + output, summed across all
    /// agentic-loop iterations) for a single binding call.
    /// `0` = uncapped (default).
    #[serde(default)]
    pub tokens_per_call_cap: u64,

    /// Maximum aggregate USD spend for this binding across all
    /// callers within one UTC day. `0` = uncapped (default).
    #[serde(default)]
    pub usd_daily_cap: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PromptSpec {
    pub system: String,
    pub user: String,
    /// Names of input arg fields that carry image content. The
    /// engine pulls each value (string URI / `mcpg-resource://` /
    /// base64 / explicit `{source, detail}` object) out of
    /// `args` and appends it to the rendered user text as
    /// [`crate::normalized::ContentPart::Image`]. Multi-image
    /// support: a value that is a JSON array fans out to multiple
    /// parts. Unset = text-only.
    #[serde(default)]
    pub image_inputs: Vec<String>,
    /// Names of input arg fields that carry audio content. Same
    /// resolution semantics as `image_inputs`. The arg may be a
    /// string (URL / `mcpg-resource://` / base64) or an object
    /// `{source, format?}`. The default audio format when the
    /// argument doesn't specify one is [`AudioFormat::Mp3`].
    #[serde(default)]
    pub audio_inputs: Vec<String>,
    /// Names of input arg fields that carry document content.
    /// String values are interpreted as URL / resource URI / base64;
    /// object values may set `mime_type` and `filename`.
    #[serde(default)]
    pub file_inputs: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SamplingSpec {
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub max_completion_tokens: Option<u32>,
    #[serde(default)]
    pub seed: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseFormatSpec {
    #[serde(default)]
    pub mode: ResponseFormatMode,
    /// Provider-side strict mode where supported. Binding-side
    /// validation runs regardless.
    #[serde(default = "default_true")]
    pub strict: bool,
    /// Pointer into the binding's schemas. `None` means the binding's
    /// own `output_schema` (the common case).
    #[serde(default)]
    pub schema_ref: Option<String>,
    /// What to do when the response fails binding-side validation.
    #[serde(default)]
    pub on_mismatch: SchemaMismatchPolicy,
}

impl Default for ResponseFormatSpec {
    fn default() -> Self {
        Self {
            mode: ResponseFormatMode::default(),
            strict: true,
            schema_ref: None,
            on_mismatch: SchemaMismatchPolicy::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseFormatMode {
    /// Structured output validated against the binding's `output_schema`.
    /// Provider-side enforcement uses the provider's native mechanism
    /// (OpenAI `json_schema`, Gemini `responseSchema`, Anthropic
    /// forced-tool, etc.).
    #[default]
    JsonSchema,
    /// Free-form text. The binding wraps the response as
    /// `{ "text": "..." }` so the gateway's JSON contract is preserved.
    /// Schema validation is skipped.
    Text,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SchemaMismatchPolicy {
    #[default]
    Error,
    RetryOnce,
    /// Allowed only with `mode: text`. Forbidden under `json_schema`.
    ReturnRaw,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolsSpec {
    /// Names of other bindings registered in this gateway that the LLM
    /// may invoke during its reasoning loop. Empty list = single-shot
    /// mode (no tool calls).
    #[serde(default)]
    pub allowed: Vec<String>,
    /// Maximum LLM round-trips. With `allowed` empty, defaults to 1
    /// (single-shot). With `allowed` non-empty, the resolved default is
    /// 5 — see [`ToolsSpec::resolved_max_iterations`].
    #[serde(default)]
    pub max_iterations: Option<u32>,
    #[serde(default)]
    pub on_iteration_exhausted: IterationExhaustedPolicy,
    /// Truncate each child tool's result before appending to the
    /// conversation. Defaults to 16 KiB.
    #[serde(default = "default_tool_result_max_bytes")]
    pub tool_result_max_bytes: usize,
    /// Provider-level tool-choice hint. `Auto` (default) lets the model
    /// decide; `Required` forces at least one tool call before
    /// terminal; `None` makes tools visible but unselectable.
    #[serde(default)]
    pub tool_choice: ToolChoice,
}

impl ToolsSpec {
    pub fn resolved_max_iterations(&self) -> u32 {
        let default = if self.allowed.is_empty() { 1 } else { 5 };
        self.max_iterations.unwrap_or(default)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IterationExhaustedPolicy {
    #[default]
    Error,
    ReturnPartial,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoice {
    #[default]
    Auto,
    Required,
    None,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingSpec {
    #[serde(default = "default_true")]
    pub enabled: bool,
}

impl Default for StreamingSpec {
    fn default() -> Self {
        Self { enabled: true }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrySpec {
    #[serde(default = "default_retry_max_attempts")]
    pub max_attempts: u32,
    #[serde(default = "default_retry_initial_backoff_ms")]
    pub initial_backoff_ms: u64,
    #[serde(default = "default_retry_max_backoff_ms")]
    pub max_backoff_ms: u64,
    #[serde(default = "default_retry_on")]
    pub retry_on: Vec<RetryReason>,
}

impl Default for RetrySpec {
    fn default() -> Self {
        Self {
            max_attempts: default_retry_max_attempts(),
            initial_backoff_ms: default_retry_initial_backoff_ms(),
            max_backoff_ms: default_retry_max_backoff_ms(),
            retry_on: default_retry_on(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetryReason {
    RateLimited,
    Server,
    Network,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GuardrailsSpec {
    /// Refuse the call before reaching the provider when the rendered
    /// prompt + tool definitions exceed this token-count estimate.
    /// `None` means no pre-flight check.
    #[serde(default)]
    pub max_input_tokens: Option<u32>,
    /// Cap the per-iteration `max_completion_tokens` regardless of what
    /// the operator's [`SamplingSpec`] requests. `None` defers to
    /// sampling.
    #[serde(default)]
    pub max_output_tokens_per_iteration: Option<u32>,
}

// ---------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------

pub(crate) fn default_timeout_ms() -> u64 {
    60_000
}
pub(crate) fn default_connect_timeout_ms() -> u64 {
    5_000
}
pub(crate) fn default_true() -> bool {
    true
}
pub(crate) fn default_tool_result_max_bytes() -> usize {
    16 * 1024
}
pub(crate) fn default_retry_max_attempts() -> u32 {
    3
}
pub(crate) fn default_retry_initial_backoff_ms() -> u64 {
    500
}
pub(crate) fn default_retry_max_backoff_ms() -> u64 {
    8_000
}
pub(crate) fn default_retry_on() -> Vec<RetryReason> {
    vec![
        RetryReason::RateLimited,
        RetryReason::Server,
        RetryReason::Network,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal() -> ChatExecutionSpec {
        ChatExecutionSpec {
            model: "test-model".into(),
            timeout_ms: 30_000,
            connect_timeout_ms: 5_000,
            prompt: PromptSpec {
                system: "you are helpful".into(),
                user: "{{ input.text }}".into(),
                ..Default::default()
            },
            sampling: SamplingSpec::default(),
            response_format: ResponseFormatSpec::default(),
            tools: ToolsSpec::default(),
            streaming: StreamingSpec::default(),
            retry: RetrySpec::default(),
            guardrails: GuardrailsSpec::default(),
            cache: CacheSpec::default(),
            budget: BudgetSpec::default(),
        }
    }

    #[test]
    fn minimal_validates() {
        minimal().validate().unwrap();
    }

    #[test]
    fn return_raw_requires_text_mode() {
        let mut s = minimal();
        s.response_format.on_mismatch = SchemaMismatchPolicy::ReturnRaw;
        s.response_format.mode = ResponseFormatMode::JsonSchema;
        let err = s.validate().unwrap_err();
        assert!(format!("{err}").contains("return_raw"));

        s.response_format.mode = ResponseFormatMode::Text;
        s.validate().unwrap();
    }

    #[test]
    fn tools_resolved_max_iterations_defaults_correctly() {
        let mut s = minimal();
        assert_eq!(s.tools.resolved_max_iterations(), 1);
        s.tools.allowed = vec!["foo".into()];
        assert_eq!(s.tools.resolved_max_iterations(), 5);
        s.tools.max_iterations = Some(8);
        assert_eq!(s.tools.resolved_max_iterations(), 8);
    }

    #[test]
    fn budget_defaults_validate_uncapped() {
        let s = minimal();
        assert_eq!(s.budget.tokens_per_call_cap, 0);
        assert_eq!(s.budget.usd_daily_cap, 0.0);
        s.validate().unwrap();
    }

    #[test]
    fn budget_positive_caps_validate() {
        let mut s = minimal();
        s.budget.tokens_per_call_cap = 50_000;
        s.budget.usd_daily_cap = 25.0;
        s.validate().unwrap();
    }

    #[test]
    fn budget_negative_usd_cap_rejected() {
        let mut s = minimal();
        s.budget.usd_daily_cap = -1.0;
        let err = s.validate().unwrap_err();
        assert!(format!("{err}").contains("usd_daily_cap"));
    }

    #[test]
    fn budget_nan_usd_cap_rejected() {
        let mut s = minimal();
        s.budget.usd_daily_cap = f64::NAN;
        assert!(s.validate().is_err());
    }

    #[test]
    fn budget_serde_roundtrip_preserves_fields() {
        let mut s = minimal();
        s.budget.tokens_per_call_cap = 12_345;
        s.budget.usd_daily_cap = 9.99;
        let json = serde_json::to_string(&s).unwrap();
        let back: ChatExecutionSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(back.budget.tokens_per_call_cap, 12_345);
        assert!((back.budget.usd_daily_cap - 9.99).abs() < 1e-9);
    }

    #[test]
    fn budget_omitted_field_uses_default() {
        let json = serde_json::json!({
            "model": "test-model",
            "timeout_ms": 30_000,
            "connect_timeout_ms": 5_000,
            "prompt": { "system": "s", "user": "u" }
            // no `budget` key — should fall back to default
        });
        let s: ChatExecutionSpec = serde_json::from_value(json).unwrap();
        assert_eq!(s.budget.tokens_per_call_cap, 0);
        assert_eq!(s.budget.usd_daily_cap, 0.0);
    }

    #[test]
    fn empty_allowed_tool_name_rejected() {
        let mut s = minimal();
        s.tools.allowed = vec!["foo".into(), "  ".into()];
        assert!(s.validate().is_err());
    }

    #[test]
    fn zero_timeouts_rejected() {
        let mut s = minimal();
        s.timeout_ms = 0;
        assert!(s.validate().is_err());

        let mut s = minimal();
        s.connect_timeout_ms = 0;
        assert!(s.validate().is_err());
    }

    #[test]
    fn iteration_cap_safety_limit() {
        let mut s = minimal();
        s.tools.allowed = vec!["t".into()];
        s.tools.max_iterations = Some(100);
        assert!(s.validate().is_err());
    }

    #[test]
    fn empty_prompts_rejected() {
        let mut s = minimal();
        s.prompt.system = "   ".into();
        assert!(s.validate().is_err());

        let mut s = minimal();
        s.prompt.user = "".into();
        assert!(s.validate().is_err());
    }
}
