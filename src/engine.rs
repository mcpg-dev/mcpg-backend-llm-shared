//! Provider-agnostic chat execution engine.
//!
//! Drives the agentic loop: prompt rendering, retry-with-backoff,
//! upstream call, child-tool dispatch via `BackendHost`, JSON Schema
//! validation. One [`ChatEngine`] per registered binding profile,
//! held by the per-provider plugin crate's `BackendPlugin` impl.
//!
//! The engine is generic over a [`ChatProviderAdapter`]; provider-
//! specific wire encoding lives in the per-provider plugin crate.

use std::sync::Arc;
use std::time::Duration;

use jsonschema::Validator;
use mcpg_plugin_protocol::{
    BackendChunk, BackendChunkStream, BackendError, BackendHost, BackendInvocationContext,
    BackendResponse,
};
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tracing::{debug, instrument, warn};

use crate::adapter::ChatProviderAdapter;
use crate::cache::CacheKey;
use crate::chat_config::{
    ChatExecutionSpec, IterationExhaustedPolicy, PromptSpec, ResponseFormatMode, RetryReason,
    RetrySpec, SchemaMismatchPolicy, ToolChoice, ToolsSpec,
};
use crate::error::{ConfigError, ProviderError};
use crate::multimodal::{DEFAULT_MAX_INLINE_BYTES, prefetch_messages_resources};
use crate::normalized::{
    AudioContent, AudioFormat, AudioSource, ContentPart, FileContent, FileSource, FinishReason,
    ImageContent, ImageDetail, ImageSource, Message, NormalizedChatRequest, NormalizedChatResponse,
    ToolCall, ToolChoiceWire, ToolDef,
};
use crate::streaming::NormalizedStreamEvent;
use crate::template::{TemplateContext, TemplateMeta, Templates};

pub struct ChatEngine {
    pub backend_name: String,
    pub adapter: Arc<dyn ChatProviderAdapter>,
    pub templates: Templates,
    /// Validator for the operator's `output_schema`. `None` = the
    /// binding doesn't validate (mode: `Text`, no schema).
    pub validator: Option<Arc<Validator>>,
    /// The raw output schema as supplied by the operator. Stored
    /// alongside the [`Validator`] so we can both check responses
    /// and re-serialize the schema for the provider's
    /// `response_format.json_schema`.
    pub raw_output_schema: Option<Value>,
    pub spec: ChatExecutionSpec,
    /// Per-call host capability (for child-tool dispatch). The plugin
    /// stores it at register-profile time and the engine uses it inside
    /// `execute`.
    pub host: Arc<dyn BackendHost>,
    /// Definitions for child tools (resolved at register time so we
    /// don't repeat the lookup per call). Empty when single-shot.
    pub child_tool_defs: Vec<ToolDef>,
    /// Compiled schemas for child-tool input validation, indexed by
    /// tool name. We refuse a tool_call whose args don't validate
    /// against the child binding's input schema *before* dispatching,
    /// so the model can correct itself rather than the child binding
    /// having to surface a low-level error.
    pub child_tool_validators: Vec<(String, Arc<Validator>)>,
}

impl ChatEngine {
    /// Run one tool call.
    ///
    /// `args` are the arguments the gateway forwarded; `request_id`
    /// and `session_id` are the parent call's identifiers.
    #[instrument(skip(self, args), fields(binding = %self.backend_name, provider = %self.adapter.label(), model = %self.spec.model))]
    pub async fn execute(
        &self,
        args: &Value,
        request_id: &str,
        session_id: Option<&str>,
    ) -> Result<Value, BackendError> {
        let now = chrono::Utc::now().to_rfc3339();
        let ctx = TemplateContext {
            input: args,
            meta: TemplateMeta {
                backend_name: &self.backend_name,
                request_id,
                session_id,
                timestamp_iso8601: now,
            },
        };
        let system =
            self.templates
                .render("system", &ctx)
                .map_err(|e| BackendError::Transport {
                    message: format!("template (system): {e}"),
                })?;
        let user = self
            .templates
            .render("user", &ctx)
            .map_err(|e| BackendError::Transport {
                message: format!("template (user): {e}"),
            })?;

        let user_message = build_user_message(user.clone(), args, &self.spec.prompt);
        let host_ctx = BackendInvocationContext::root(
            request_id,
            session_id.map(|s| s.to_owned()),
            self.backend_name.clone(),
        );

        // Per-binding USD daily-cap check.
        // Runs BEFORE any upstream call so we don't pay for one
        // model invocation we'd then refuse. Models not in the
        // rate card cannot accumulate cost; the cap is effectively
        // off for those bindings (see `crate::budget` rationale).
        if self.spec.budget.usd_daily_cap > 0.0 {
            let used = crate::budget::current_daily_usd(&self.backend_name);
            if used >= self.spec.budget.usd_daily_cap {
                metrics::counter!(
                    "mcpg_llm_budget_refusals_total",
                    "binding" => self.backend_name.clone(),
                    "reason" => "usd_daily",
                )
                .increment(1);
                return Err(BackendError::Transport {
                    message: format!(
                        "budget refused: usd_daily_cap of ${:.4} reached for binding `{}` \
                         (current spend: ${used:.4})",
                        self.spec.budget.usd_daily_cap, self.backend_name,
                    ),
                });
            }
        }

        // Response-cache lookup. The validator forbids
        // `cache.enabled + tools.allowed`, so a hit returns the
        // single-shot response straight to the caller.
        let cache_key = if self.spec.cache.enabled {
            let key = build_chat_cache_key(self, &system, &user);
            match self.host.cache_get(&host_ctx, key.as_str()).await {
                Ok(Some(bytes)) => {
                    metrics::counter!(
                        "mcpg_llm_cache_hits_total",
                        "binding" => self.backend_name.clone(),
                        "provider" => self.adapter.label().to_string(),
                        "model" => self.spec.model.clone(),
                    )
                    .increment(1);
                    let value: Value =
                        serde_json::from_slice(&bytes).map_err(|e| BackendError::Transport {
                            message: format!("decode cached chat response: {e}"),
                        })?;
                    return Ok(value);
                }
                Ok(None) => {
                    metrics::counter!(
                        "mcpg_llm_cache_misses_total",
                        "binding" => self.backend_name.clone(),
                        "provider" => self.adapter.label().to_string(),
                        "model" => self.spec.model.clone(),
                    )
                    .increment(1);
                }
                Err(e) => {
                    debug!(?e, "chat cache lookup failed; falling through");
                }
            }
            Some(key)
        } else {
            None
        };

        let mut messages: Vec<Message> = prefetch_messages_resources(
            vec![Message::system(system), user_message],
            self.host.as_ref(),
            &host_ctx,
            DEFAULT_MAX_INLINE_BYTES,
        )
        .await
        .map_err(|e| BackendError::Transport {
            message: format!("multimodal prefetch: {e}"),
        })?;

        let max_iterations = self.spec.tools.resolved_max_iterations();
        let mut last_response: Option<NormalizedChatResponse> = None;
        let mut total_input_tokens: u64 = 0;
        let mut total_output_tokens: u64 = 0;
        let mut total_cached_input_tokens: u64 = 0;

        for iter in 0..max_iterations {
            let req = self.build_request(&messages);
            let resp = self
                .call_with_retry(&req)
                .await
                .map_err(|e| binding_error_from_provider(&self.backend_name, e))?;

            total_input_tokens += resp.usage.input_tokens as u64;
            total_output_tokens += resp.usage.output_tokens as u64;
            total_cached_input_tokens += resp.usage.cached_input_tokens as u64;

            metrics::counter!(
                "mcpg_llm_tokens_total",
                "binding" => self.backend_name.clone(),
                "provider" => self.adapter.label().to_string(),
                "model" => self.spec.model.clone(),
                "direction" => "input"
            )
            .increment(resp.usage.input_tokens as u64);
            metrics::counter!(
                "mcpg_llm_tokens_total",
                "binding" => self.backend_name.clone(),
                "provider" => self.adapter.label().to_string(),
                "model" => self.spec.model.clone(),
                "direction" => "output"
            )
            .increment(resp.usage.output_tokens as u64);

            // Per-call token-cap check. Applied
            // after the running totals are updated so the FIRST
            // iteration is always allowed (we need at least one
            // upstream call before we can know whether the cap is
            // reached). Subsequent iterations are gated.
            if self.spec.budget.tokens_per_call_cap > 0
                && total_input_tokens + total_output_tokens >= self.spec.budget.tokens_per_call_cap
            {
                metrics::counter!(
                    "mcpg_llm_budget_refusals_total",
                    "binding" => self.backend_name.clone(),
                    "reason" => "tokens_per_call",
                )
                .increment(1);
                return Err(BackendError::Transport {
                    message: format!(
                        "budget refused: tokens_per_call_cap of {} reached for binding `{}` \
                         after {iter} iteration(s) (input={total_input_tokens}, \
                         output={total_output_tokens})",
                        self.spec.budget.tokens_per_call_cap, self.backend_name,
                    ),
                });
            }

            // Terminal: no tool calls (or tools disabled).
            if resp.tool_calls.is_empty() {
                last_response = Some(resp);
                break;
            }

            // The model wants to call tools. If the operator hasn't
            // allowed any, that's a configuration drift — the model
            // shouldn't have seen tools at all. Treat as Transport.
            if self.child_tool_defs.is_empty() {
                return Err(BackendError::Transport {
                    message: format!(
                        "model emitted {} tool_calls but no child tools are configured",
                        resp.tool_calls.len()
                    ),
                });
            }

            // Append the assistant turn first (provider conventions
            // expect the assistant's tool_calls before tool messages).
            // Preserve any text the model emitted alongside its
            // tool_calls — Claude in particular routinely narrates
            // ("let me check the database…") before invoking tools,
            // and dropping the narration breaks the second-iteration
            // request structure.
            let calls_for_loop = resp.tool_calls.clone();
            messages.push(Message::assistant_text_and_tool_calls(
                resp.content.clone(),
                resp.tool_calls.clone(),
            ));

            let mut dispatched_count = 0u32;
            for call in calls_for_loop {
                let tool_msg = self
                    .dispatch_child_tool(&call, request_id, session_id, iter)
                    .await;
                messages.push(Message::tool_result(call.id.clone(), tool_msg));
                dispatched_count += 1;
            }

            metrics::counter!(
                "mcpg_llm_iterations",
                "binding" => self.backend_name.clone(),
                "provider" => self.adapter.label().to_string(),
                "model" => self.spec.model.clone(),
            )
            .increment(1);

            debug!(
                iter,
                tool_calls = dispatched_count,
                "engine iteration completed with tool calls"
            );

            last_response = Some(resp);
        }

        let final_resp = match last_response {
            Some(r) => r,
            None => {
                return Err(BackendError::Transport {
                    message: "engine produced no response (max_iterations=0?)".into(),
                });
            }
        };

        if final_resp.finish_reason == FinishReason::ToolCalls {
            metrics::counter!(
                "mcpg_llm_iteration_exhausted_total",
                "binding" => self.backend_name.clone(),
                "provider" => self.adapter.label().to_string(),
                "model" => self.spec.model.clone(),
            )
            .increment(1);
            match self.spec.tools.on_iteration_exhausted {
                IterationExhaustedPolicy::Error => {
                    return Err(BackendError::Transport {
                        message: format!(
                            "iteration cap {max_iterations} reached without terminal response"
                        ),
                    });
                }
                IterationExhaustedPolicy::ReturnPartial => {
                    // Return whatever last content was, raw. Often empty.
                }
            }
        }

        let validated = self
            .validate_final_content(&final_resp, &mut messages)
            .await?;

        metrics::histogram!(
            "mcpg_llm_call_tokens_input",
            "binding" => self.backend_name.clone(),
            "provider" => self.adapter.label().to_string(),
            "model" => self.spec.model.clone(),
        )
        .record(total_input_tokens as f64);
        metrics::histogram!(
            "mcpg_llm_call_tokens_output",
            "binding" => self.backend_name.clone(),
            "provider" => self.adapter.label().to_string(),
            "model" => self.spec.model.clone(),
        )
        .record(total_output_tokens as f64);

        // Cost emission. `compute_chat_cost_usd` returns
        // None for models not in the rate card — quietly drop the
        // metric rather than emit a misleading zero. Sum across all
        // iterations of the agentic loop so multi-step calls report
        // their full cost.
        let aggregate_usage = crate::normalized::TokenUsage {
            input_tokens: total_input_tokens.min(u32::MAX as u64) as u32,
            output_tokens: total_output_tokens.min(u32::MAX as u64) as u32,
            cached_input_tokens: total_cached_input_tokens.min(u32::MAX as u64) as u32,
        };
        if let Some(cost) = crate::cost::compute_chat_cost_usd(
            crate::cost::bundled_rate_card(),
            self.adapter.label(),
            &self.spec.model,
            &aggregate_usage,
        ) {
            metrics::counter!(
                "mcpg_llm_cost_usd_total",
                "binding" => self.backend_name.clone(),
                "provider" => self.adapter.label().to_string(),
                "model" => self.spec.model.clone(),
            )
            .increment((cost * 1_000_000.0) as u64);
            metrics::histogram!(
                "mcpg_llm_call_cost_usd",
                "binding" => self.backend_name.clone(),
                "provider" => self.adapter.label().to_string(),
                "model" => self.spec.model.clone(),
            )
            .record(cost);
            // Update the per-binding daily ledger so the
            // usd_daily_cap check on the next call sees the
            // accumulated spend. We only
            // record after a successful call — a refused call or
            // upstream failure doesn't count against the budget.
            crate::budget::record_cost(&self.backend_name, cost);
        }

        // Response-cache write. Skipped on the hit-path early
        // return above. We deliberately cache after validation so a
        // schema-failed response doesn't poison the cache.
        if let Some(key) = cache_key {
            match serde_json::to_vec(&validated) {
                Ok(payload) => {
                    let ttl = std::time::Duration::from_secs(self.spec.cache.ttl_seconds);
                    if let Err(e) = self
                        .host
                        .cache_put(&host_ctx, key.as_str().to_owned(), payload.into(), ttl)
                        .await
                    {
                        debug!(?e, "chat cache write failed; ignoring");
                    }
                }
                Err(e) => {
                    debug!(?e, "skip cache write: response not serializable");
                }
            }
        }

        Ok(validated)
    }

    /// Dispatch one child tool call back through the gateway.
    /// Returns the *string* to insert as the tool-result message —
    /// truncated to the operator's `tool_result_max_bytes` and with a
    /// suffix marker if truncated.
    async fn dispatch_child_tool(
        &self,
        call: &ToolCall,
        request_id: &str,
        session_id: Option<&str>,
        depth: u32,
    ) -> String {
        if !self.spec.tools.allowed.iter().any(|t| t == &call.name) {
            metrics::counter!(
                "mcpg_llm_tool_calls_total",
                "binding" => self.backend_name.clone(),
                "child_tool" => call.name.clone(),
                "status" => "denied_by_allowlist"
            )
            .increment(1);
            return json!({
                "error": format!("tool '{}' is not in this binding's allowed list", call.name)
            })
            .to_string();
        }

        if let Some((_, validator)) = self
            .child_tool_validators
            .iter()
            .find(|(n, _)| n == &call.name)
            && let Err(error) = validator.validate(&call.arguments)
        {
            metrics::counter!(
                "mcpg_llm_tool_calls_total",
                "binding" => self.backend_name.clone(),
                "child_tool" => call.name.clone(),
                "status" => "invalid_args"
            )
            .increment(1);
            return json!({
                "error": "tool arguments did not match the child binding's input_schema",
                "details": error.to_string()
            })
            .to_string();
        }

        let ctx = BackendInvocationContext {
            parent_request_id: request_id.to_owned(),
            session_id: session_id.map(|s| s.to_owned()),
            initiating_backend: self.backend_name.clone(),
            depth,
            // LLM agentic-loop child-tool calls do not carry the parent
            // caller identity; cred:// identity threading is unsupported
            // on this path.
            identity: None,
        };

        let result = self
            .host
            .invoke_tool(&ctx, &call.name, &call.arguments)
            .await;

        let (status_label, body) = match result {
            Ok(v) => ("ok", v.to_string()),
            Err(e) => (
                "error",
                json!({
                    "error": e.to_string()
                })
                .to_string(),
            ),
        };

        metrics::counter!(
            "mcpg_llm_tool_calls_total",
            "binding" => self.backend_name.clone(),
            "child_tool" => call.name.clone(),
            "status" => status_label
        )
        .increment(1);

        truncate_tool_result(body, self.spec.tools.tool_result_max_bytes)
    }

    fn build_request(&self, messages: &[Message]) -> NormalizedChatRequest {
        let response_schema = match self.spec.response_format.mode {
            ResponseFormatMode::JsonSchema => self.raw_output_schema.clone(),
            ResponseFormatMode::Text => None,
        };

        let tool_choice = match (self.child_tool_defs.is_empty(), self.spec.tools.tool_choice) {
            (true, _) => ToolChoiceWire::None,
            (false, ToolChoice::Auto) => ToolChoiceWire::Auto,
            (false, ToolChoice::Required) => ToolChoiceWire::Required,
            (false, ToolChoice::None) => ToolChoiceWire::None,
        };

        let max_tokens = self
            .spec
            .guardrails
            .max_output_tokens_per_iteration
            .or(self.spec.sampling.max_completion_tokens);

        NormalizedChatRequest {
            model: self.spec.model.clone(),
            messages: messages.to_vec(),
            response_schema,
            strict_response: self.spec.response_format.strict,
            tools: self.child_tool_defs.clone(),
            tool_choice,
            temperature: self.spec.sampling.temperature,
            top_p: self.spec.sampling.top_p,
            max_completion_tokens: max_tokens,
            seed: self.spec.sampling.seed,
        }
    }

    async fn validate_final_content(
        &self,
        response: &NormalizedChatResponse,
        messages: &mut Vec<Message>,
    ) -> Result<Value, BackendError> {
        if matches!(self.spec.response_format.mode, ResponseFormatMode::Text) {
            return Ok(json!({"text": response.content}));
        }

        let parsed_first = match parse_response_json(&response.content) {
            Ok(v) => v,
            Err(e) => {
                if matches!(
                    self.spec.response_format.on_mismatch,
                    SchemaMismatchPolicy::ReturnRaw
                ) {
                    return Ok(json!({"text": response.content, "raw": true}));
                }
                if matches!(
                    self.spec.response_format.on_mismatch,
                    SchemaMismatchPolicy::RetryOnce
                ) {
                    return self
                        .retry_with_corrective_message(
                            messages,
                            &format!("response was not valid JSON: {e}"),
                        )
                        .await;
                }
                metrics::counter!(
                    "mcpg_llm_schema_validation_errors_total",
                    "binding" => self.backend_name.clone(),
                    "mode" => "binding",
                )
                .increment(1);
                return Err(BackendError::Transport {
                    message: format!("schema validation failed: response was not valid JSON: {e}"),
                });
            }
        };

        if let Some(validator) = &self.validator
            && let Err(error) = validator.validate(&parsed_first)
        {
            metrics::counter!(
                "mcpg_llm_schema_validation_errors_total",
                "binding" => self.backend_name.clone(),
                "mode" => "binding",
            )
            .increment(1);
            if matches!(
                self.spec.response_format.on_mismatch,
                SchemaMismatchPolicy::RetryOnce
            ) {
                return self
                    .retry_with_corrective_message(messages, &format!("schema mismatch: {error}"))
                    .await;
            }
            return Err(BackendError::Transport {
                message: format!("schema validation failed: {error}"),
            });
        }

        Ok(parsed_first)
    }

    /// One corrective retry: append the schema-error as a system note
    /// and re-invoke the model with the same conversation. Counts as
    /// one extra iteration but is bounded — a second mismatch is fatal.
    async fn retry_with_corrective_message(
        &self,
        messages: &mut Vec<Message>,
        reason: &str,
    ) -> Result<Value, BackendError> {
        messages.push(Message::system(format!(
            "Your previous response did not match the required schema: {reason}. \
             Respond again, conforming exactly to the schema."
        )));
        let req = self.build_request(messages);
        let resp = self
            .call_with_retry(&req)
            .await
            .map_err(|e| binding_error_from_provider(&self.backend_name, e))?;

        let parsed: Value = parse_response_json(&resp.content).map_err(|e| {
            metrics::counter!(
                "mcpg_llm_schema_validation_errors_total",
                "binding" => self.backend_name.clone(),
                "mode" => "binding",
            )
            .increment(1);
            BackendError::Transport {
                message: format!("schema validation failed after retry: {e}"),
            }
        })?;

        if let Some(validator) = &self.validator
            && let Err(error) = validator.validate(&parsed)
        {
            metrics::counter!(
                "mcpg_llm_schema_validation_errors_total",
                "binding" => self.backend_name.clone(),
                "mode" => "binding",
            )
            .increment(1);
            return Err(BackendError::Transport {
                message: format!("schema validation failed after retry: {error}"),
            });
        }
        Ok(parsed)
    }

    /// Provider call with retry policy applied.
    async fn call_with_retry(
        &self,
        req: &NormalizedChatRequest,
    ) -> Result<NormalizedChatResponse, ProviderError> {
        let RetrySpec {
            max_attempts,
            initial_backoff_ms,
            max_backoff_ms,
            retry_on,
        } = &self.spec.retry;

        let mut attempt: u32 = 0;
        loop {
            attempt += 1;
            let started = std::time::Instant::now();
            let result = self.adapter.chat_completion(req, self.spec.timeout()).await;
            let elapsed = started.elapsed();

            metrics::histogram!(
                "mcpg_llm_call_duration_seconds",
                "binding" => self.backend_name.clone(),
                "provider" => self.adapter.label().to_string(),
                "model" => self.spec.model.clone(),
            )
            .record(elapsed.as_secs_f64());

            match result {
                Ok(resp) => return Ok(resp),
                Err(err) => {
                    let category = err.category();
                    metrics::counter!(
                        "mcpg_llm_calls_total",
                        "binding" => self.backend_name.clone(),
                        "provider" => self.adapter.label().to_string(),
                        "model" => self.spec.model.clone(),
                        "status" => category,
                    )
                    .increment(1);

                    let retryable = err.is_retryable() && retry_reason_allowed(&err, retry_on);
                    if !retryable || attempt >= *max_attempts {
                        return Err(err);
                    }

                    let delay_ms = compute_backoff(*initial_backoff_ms, *max_backoff_ms, attempt);
                    warn!(attempt, delay_ms, error = %err, "retrying provider call");
                    metrics::counter!(
                        "mcpg_llm_retries_total",
                        "binding" => self.backend_name.clone(),
                        "reason" => category,
                    )
                    .increment(1);
                    tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Streaming variant
// ---------------------------------------------------------------------------

impl ChatEngine {
    /// Streaming counterpart of [`execute`]. Returns a
    /// [`BackendChunkStream`] the per-provider plugin forwards to the
    /// gateway transport.
    ///
    /// The stream emits:
    /// - `TextDelta` for each upstream text delta as it arrives.
    /// - `ToolCall` + `ToolResult` per child invocation in the agentic loop.
    /// - `IterationBoundary` between iterations.
    /// - `Usage` once per iteration when reported.
    /// - Terminal `Done(BackendResponse)` carrying the validated structured
    ///   response — same payload `execute` would have returned. Stream
    ///   ends after this chunk.
    ///
    /// Errors mid-stream are emitted as `Err(BackendError)` items, after
    /// which the stream ends.
    pub fn execute_streaming(
        self: std::sync::Arc<Self>,
        args: Value,
        request_id: String,
        session_id: Option<String>,
    ) -> BackendChunkStream {
        let (tx, rx) = mpsc::channel::<Result<BackendChunk, BackendError>>(64);
        let engine = self.clone();
        tokio::spawn(async move {
            engine.run_streaming(args, request_id, session_id, tx).await;
        });
        Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx))
    }

    async fn run_streaming(
        self: std::sync::Arc<Self>,
        args: Value,
        request_id: String,
        session_id: Option<String>,
        tx: mpsc::Sender<Result<BackendChunk, BackendError>>,
    ) {
        let now = chrono::Utc::now().to_rfc3339();
        let ctx = TemplateContext {
            input: &args,
            meta: TemplateMeta {
                backend_name: &self.backend_name,
                request_id: &request_id,
                session_id: session_id.as_deref(),
                timestamp_iso8601: now,
            },
        };
        let system = match self.templates.render("system", &ctx) {
            Ok(s) => s,
            Err(e) => {
                let _ = tx
                    .send(Err(BackendError::Transport {
                        message: format!("template (system): {e}"),
                    }))
                    .await;
                return;
            }
        };
        let user = match self.templates.render("user", &ctx) {
            Ok(s) => s,
            Err(e) => {
                let _ = tx
                    .send(Err(BackendError::Transport {
                        message: format!("template (user): {e}"),
                    }))
                    .await;
                return;
            }
        };

        let user_message = build_user_message(user, &args, &self.spec.prompt);
        let host_ctx = BackendInvocationContext::root(
            request_id.clone(),
            session_id.clone(),
            self.backend_name.clone(),
        );
        let mut messages = match prefetch_messages_resources(
            vec![Message::system(system), user_message],
            self.host.as_ref(),
            &host_ctx,
            DEFAULT_MAX_INLINE_BYTES,
        )
        .await
        {
            Ok(m) => m,
            Err(e) => {
                let _ = tx
                    .send(Err(BackendError::Transport {
                        message: format!("multimodal prefetch: {e}"),
                    }))
                    .await;
                return;
            }
        };

        let max_iterations = self.spec.tools.resolved_max_iterations();
        let mut accumulated_content = String::new();
        let mut last_finish: Option<FinishReason> = None;
        let mut last_tool_calls: Vec<ToolCall> = Vec::new();
        let mut total_input: u64 = 0;
        let mut total_output: u64 = 0;

        for iter in 0..max_iterations {
            if iter > 0
                && tx
                    .send(Ok(BackendChunk::IterationBoundary { iteration: iter }))
                    .await
                    .is_err()
            {
                return;
            }

            let req = self.build_request(&messages);
            let mut event_rx = match self
                .adapter
                .stream_chat_completion(&req, self.spec.timeout())
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    let _ = tx
                        .send(Err(binding_error_from_provider(&self.backend_name, e)))
                        .await;
                    return;
                }
            };

            accumulated_content.clear();
            last_tool_calls.clear();
            last_finish = None;

            while let Some(event) = event_rx.recv().await {
                match event {
                    Ok(NormalizedStreamEvent::TextDelta(delta)) => {
                        accumulated_content.push_str(&delta);
                        if tx
                            .send(Ok(BackendChunk::TextDelta { delta }))
                            .await
                            .is_err()
                        {
                            return;
                        }
                    }
                    Ok(NormalizedStreamEvent::ToolCallReady(call)) => {
                        last_tool_calls.push(call);
                    }
                    Ok(NormalizedStreamEvent::Finish { reason, usage }) => {
                        total_input += usage.input_tokens as u64;
                        total_output += usage.output_tokens as u64;
                        if usage.input_tokens > 0 || usage.output_tokens > 0 {
                            let _ = tx
                                .send(Ok(BackendChunk::Usage {
                                    input_tokens: usage.input_tokens,
                                    output_tokens: usage.output_tokens,
                                    cached_input_tokens: usage.cached_input_tokens,
                                }))
                                .await;
                        }
                        last_finish = Some(reason);
                        break;
                    }
                    Err(e) => {
                        let _ = tx
                            .send(Err(binding_error_from_provider(&self.backend_name, e)))
                            .await;
                        return;
                    }
                }
            }

            metrics::counter!(
                "mcpg_llm_tokens_total",
                "binding" => self.backend_name.clone(),
                "provider" => self.adapter.label().to_string(),
                "model" => self.spec.model.clone(),
                "direction" => "input"
            )
            .increment(total_input);
            metrics::counter!(
                "mcpg_llm_tokens_total",
                "binding" => self.backend_name.clone(),
                "provider" => self.adapter.label().to_string(),
                "model" => self.spec.model.clone(),
                "direction" => "output"
            )
            .increment(total_output);

            if last_tool_calls.is_empty() {
                break;
            }

            if self.child_tool_defs.is_empty() {
                let _ = tx
                    .send(Err(BackendError::Transport {
                        message: format!(
                            "model emitted {} tool_calls but no child tools are configured",
                            last_tool_calls.len()
                        ),
                    }))
                    .await;
                return;
            }

            messages.push(Message::assistant_text_and_tool_calls(
                accumulated_content.clone(),
                last_tool_calls.clone(),
            ));

            let calls_to_dispatch = std::mem::take(&mut last_tool_calls);
            for call in calls_to_dispatch {
                if tx
                    .send(Ok(BackendChunk::ToolCall {
                        id: call.id.clone(),
                        name: call.name.clone(),
                        arguments: call.arguments.clone(),
                    }))
                    .await
                    .is_err()
                {
                    return;
                }

                let tool_msg = self
                    .dispatch_child_tool(&call, &request_id, session_id.as_deref(), iter)
                    .await;

                let parsed_result: Value =
                    serde_json::from_str(&tool_msg).unwrap_or(Value::String(tool_msg.clone()));
                if tx
                    .send(Ok(BackendChunk::ToolResult {
                        id: call.id.clone(),
                        result: parsed_result,
                    }))
                    .await
                    .is_err()
                {
                    return;
                }
                messages.push(Message::tool_result(call.id, tool_msg));
            }
        }

        if matches!(last_finish, Some(FinishReason::ToolCalls)) && !last_tool_calls.is_empty() {
            match self.spec.tools.on_iteration_exhausted {
                IterationExhaustedPolicy::Error => {
                    let _ = tx
                        .send(Err(BackendError::Transport {
                            message: format!(
                                "iteration cap {max_iterations} reached without terminal response"
                            ),
                        }))
                        .await;
                    return;
                }
                IterationExhaustedPolicy::ReturnPartial => {}
            }
        }

        let final_value = match self.validate_accumulated(&accumulated_content).await {
            Ok(v) => v,
            Err(e) => {
                let _ = tx.send(Err(e)).await;
                return;
            }
        };
        let payload = match serde_json::to_vec(&final_value) {
            Ok(p) => p,
            Err(e) => {
                let _ = tx
                    .send(Err(BackendError::Transport {
                        message: format!("serialize response: {e}"),
                    }))
                    .await;
                return;
            }
        };
        let _ = tx
            .send(Ok(BackendChunk::Done(BackendResponse {
                payload,
                truncated: false,
            })))
            .await;
    }

    /// Standalone validator the streaming path uses to apply schema
    /// checks to the accumulated text content. The non-streaming path
    /// uses `validate_final_content` which has corrective-retry support
    /// — the streaming path skips it (would require re-streaming with a
    /// new turn — ergonomically tricky in a chunk pipeline) and falls
    /// back to error-on-mismatch even when `on_mismatch: retry_once` is
    /// configured.
    async fn validate_accumulated(&self, content: &str) -> Result<Value, BackendError> {
        if matches!(self.spec.response_format.mode, ResponseFormatMode::Text) {
            return Ok(json!({"text": content}));
        }

        let parsed: Value =
            serde_json::from_str(content.trim()).map_err(|e| BackendError::Transport {
                message: format!("schema validation failed: response was not valid JSON: {e}"),
            })?;

        if let Some(validator) = &self.validator
            && let Err(error) = validator.validate(&parsed)
        {
            metrics::counter!(
                "mcpg_llm_schema_validation_errors_total",
                "binding" => self.backend_name.clone(),
                "mode" => "binding",
            )
            .increment(1);
            return Err(BackendError::Transport {
                message: format!("schema validation failed: {error}"),
            });
        }
        Ok(parsed)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn truncate_tool_result(mut body: String, max_bytes: usize) -> String {
    if body.len() <= max_bytes {
        return body;
    }
    body.truncate(max_bytes);
    body.push_str(&format!("\n[truncated; original_size_bytes>{max_bytes}]"));
    body
}

fn parse_response_json(content: &str) -> Result<Value, String> {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return Err("response content is empty".into());
    }
    serde_json::from_str(trimmed).map_err(|e| format!("{e}"))
}

fn retry_reason_allowed(err: &ProviderError, allowed: &[RetryReason]) -> bool {
    let category = match err {
        ProviderError::RateLimited { .. } => RetryReason::RateLimited,
        ProviderError::Server { .. } => RetryReason::Server,
        ProviderError::Network { .. } => RetryReason::Network,
        _ => return false,
    };
    allowed.contains(&category)
}

fn compute_backoff(initial_ms: u64, max_ms: u64, attempt: u32) -> u64 {
    let exp = 2u64.pow(attempt.saturating_sub(1).min(20));
    initial_ms.saturating_mul(exp).min(max_ms)
}

fn binding_error_from_provider(_binding_name: &str, err: ProviderError) -> BackendError {
    match err {
        ProviderError::AuthFailed { message } => BackendError::Transport {
            message: format!("provider auth failed: {message}"),
        },
        ProviderError::BadRequest { message } => BackendError::Transport {
            message: format!("provider rejected request: {message}"),
        },
        ProviderError::ContextLimit { message } => BackendError::Transport {
            message: format!("context limit: {message}"),
        },
        ProviderError::RateLimited { message } => BackendError::Transport {
            message: format!("rate limited (retries exhausted): {message}"),
        },
        ProviderError::Server { message } => BackendError::Transport {
            message: format!("provider server error: {message}"),
        },
        ProviderError::Network { message } => BackendError::Transport {
            message: format!("network error: {message}"),
        },
        ProviderError::Malformed { message } => BackendError::Transport {
            message: format!("provider returned malformed response: {message}"),
        },
    }
}

/// Compile a JSON Schema once; engine reuses the validator across calls.
pub fn compile_validator(schema: &Value) -> Result<Arc<Validator>, ConfigError> {
    let v = jsonschema::validator_for(schema)
        .map_err(|e| ConfigError::Schema(format!("output_schema is not valid JSON Schema: {e}")))?;
    Ok(Arc::new(v))
}

/// Construct the user-side `Message` from the rendered text plus any
/// declared multimodal inputs. Pulls each named arg out of `args`,
/// parses it into `ContentPart`s, and returns either:
///
/// - `Message::user(text)` when no multimodal inputs are declared
///   or when none of them resolved to non-empty content.
/// - `Message::user_parts(parts)` where `parts` starts with the
///   rendered text (when non-empty) and is followed by the parsed
///   media in declaration order: image → audio → file. Multi-image
///   args (JSON arrays) fan out element-by-element.
///
/// Args may be:
/// - **string**: a `mcpg-resource://...` URI, an `https?://...` URL,
///   a `data:<mime>;base64,...` data-URL, or raw base64 (interpreted
///   as `image/png` / format-default for audio / `mime_type`-set for
///   file). The string form covers the common case where the operator
///   exposes the binding via a simple `{ type: string, contentEncoding:
///   base64 }` input schema.
/// - **object**: `{url, detail?}` / `{base64, mime_type?}` /
///   `{resource}` / `{source: ..., detail?}` / etc. The full
///   `ContentPart::Image / Audio / File` shape may also be supplied
///   verbatim — the engine deserialises it via `serde_json::from_value`
///   for advanced operators that need to set every field.
/// - **array**: each element parsed independently, fanning into
///   multiple parts.
///
/// Unparseable values are silently dropped — the operator's input
/// schema is the right place to enforce shape; the engine treats
/// already-validated args as best-effort inputs.
fn build_user_message(text: String, args: &Value, prompt: &PromptSpec) -> Message {
    let mut parts: Vec<ContentPart> = Vec::new();
    if !text.is_empty() {
        parts.push(ContentPart::Text(text.clone()));
    }
    for name in &prompt.image_inputs {
        if let Some(v) = args.get(name) {
            for img in parse_images(v) {
                parts.push(ContentPart::Image(img));
            }
        }
    }
    for name in &prompt.audio_inputs {
        if let Some(v) = args.get(name) {
            for au in parse_audios(v) {
                parts.push(ContentPart::Audio(au));
            }
        }
    }
    for name in &prompt.file_inputs {
        if let Some(v) = args.get(name) {
            for f in parse_files(v) {
                parts.push(ContentPart::File(f));
            }
        }
    }

    // Promote to a multimodal user message only if there's actually
    // non-text content — pure text stays on the lean `Text` path so
    // adapter encoders skip the parts pipeline entirely.
    let has_media = parts.iter().any(|p| !matches!(p, ContentPart::Text(_)));
    if has_media {
        Message::user_parts(parts)
    } else {
        Message::user(text)
    }
}

/// Compose a cache key for a chat call. Hashes the rendered system +
/// user text along with binding identity, model, sampling, and the
/// tool-allowlist signature. The validator already refuses
/// `cache.enabled + tools.allowed`, so the tools_signature is
/// always empty here, but we include the field for future-proofing
/// (in case a deterministic-tool path is added later).
fn build_chat_cache_key(
    engine: &ChatEngine,
    rendered_system: &str,
    rendered_user: &str,
) -> CacheKey {
    let mut allowed: Vec<String> = engine.spec.tools.allowed.clone();
    allowed.sort();
    let tools_signature = allowed.join(",");
    CacheKey::for_chat(
        &engine.backend_name,
        engine.adapter.label(),
        &engine.spec.model,
        rendered_system,
        rendered_user,
        &tools_signature,
        &engine.spec.sampling,
    )
}

fn parse_images(value: &Value) -> Vec<ImageContent> {
    match value {
        Value::Array(arr) => arr.iter().flat_map(parse_images).collect(),
        Value::String(s) => vec![ImageContent {
            source: parse_image_string_source(s),
            detail: None,
        }],
        Value::Object(obj) => {
            // `{source: ..., detail?}` shape
            if let Some(src) = obj.get("source")
                && let Ok(source) = serde_json::from_value::<ImageSourceWire>(src.clone())
            {
                return vec![ImageContent {
                    source: source.into(),
                    detail: parse_image_detail(obj.get("detail")),
                }];
            }
            // Shorthand: top-level `url` / `resource` / `data` keys.
            if let Some(url) = obj.get("url").and_then(|v| v.as_str()) {
                return vec![ImageContent {
                    source: ImageSource::Url(url.to_owned()),
                    detail: parse_image_detail(obj.get("detail")),
                }];
            }
            if let Some(uri) = obj.get("resource").and_then(|v| v.as_str()) {
                return vec![ImageContent {
                    source: ImageSource::McpResource(uri.to_owned()),
                    detail: parse_image_detail(obj.get("detail")),
                }];
            }
            if let Some(data) = obj.get("data").and_then(|v| v.as_str()) {
                let mime = obj
                    .get("mime_type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("image/png")
                    .to_owned();
                return vec![ImageContent {
                    source: ImageSource::Base64 {
                        mime_type: mime,
                        data: data.to_owned(),
                    },
                    detail: parse_image_detail(obj.get("detail")),
                }];
            }
            Vec::new()
        }
        _ => Vec::new(),
    }
}

#[derive(serde::Deserialize)]
#[serde(untagged)]
enum ImageSourceWire {
    Resource {
        resource: String,
    },
    Url {
        url: String,
    },
    Base64 {
        mime_type: String,
        data: String,
    },
    /// Catch-all for `{kind: "url|base64|resource", ...}` style.
    Tagged {
        kind: String,
        #[serde(flatten)]
        rest: serde_json::Map<String, Value>,
    },
}

impl From<ImageSourceWire> for ImageSource {
    fn from(w: ImageSourceWire) -> Self {
        match w {
            ImageSourceWire::Resource { resource } => Self::McpResource(resource),
            ImageSourceWire::Url { url } => Self::Url(url),
            ImageSourceWire::Base64 { mime_type, data } => Self::Base64 { mime_type, data },
            ImageSourceWire::Tagged { kind, rest } => match kind.as_str() {
                "resource" => rest
                    .get("resource")
                    .and_then(|v| v.as_str())
                    .map(|s| Self::McpResource(s.to_owned()))
                    .unwrap_or_else(|| Self::Url(String::new())),
                "url" => rest
                    .get("url")
                    .and_then(|v| v.as_str())
                    .map(|s| Self::Url(s.to_owned()))
                    .unwrap_or_else(|| Self::Url(String::new())),
                "base64" => Self::Base64 {
                    mime_type: rest
                        .get("mime_type")
                        .and_then(|v| v.as_str())
                        .unwrap_or("image/png")
                        .to_owned(),
                    data: rest
                        .get("data")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_owned(),
                },
                _ => Self::Url(String::new()),
            },
        }
    }
}

fn parse_image_string_source(s: &str) -> ImageSource {
    if let Some(rest) = s.strip_prefix("mcpg-resource://") {
        ImageSource::McpResource(format!("mcpg-resource://{rest}"))
    } else if let Some(data_part) = s.strip_prefix("data:") {
        // `data:<mime>;base64,<data>`. Forgiving parser: bail to URL
        // when the structure doesn't match.
        if let Some((mime_part, payload)) = data_part.split_once(',') {
            let mime = mime_part.trim_end_matches(";base64").to_owned();
            ImageSource::Base64 {
                mime_type: if mime.is_empty() {
                    "image/png".into()
                } else {
                    mime
                },
                data: payload.to_owned(),
            }
        } else {
            ImageSource::Url(s.to_owned())
        }
    } else if s.starts_with("http://") || s.starts_with("https://") {
        ImageSource::Url(s.to_owned())
    } else {
        // Raw base64 with no scheme — assume image/png. Operators
        // should prefer the data-URL form for clarity.
        ImageSource::Base64 {
            mime_type: "image/png".into(),
            data: s.to_owned(),
        }
    }
}

fn parse_image_detail(v: Option<&Value>) -> Option<ImageDetail> {
    match v.and_then(|v| v.as_str()) {
        Some("auto") => Some(ImageDetail::Auto),
        Some("high") => Some(ImageDetail::High),
        Some("low") => Some(ImageDetail::Low),
        _ => None,
    }
}

fn parse_audios(value: &Value) -> Vec<AudioContent> {
    let parse_format = |v: Option<&Value>| -> AudioFormat {
        match v.and_then(|v| v.as_str()).unwrap_or("mp3") {
            "wav" => AudioFormat::Wav,
            "flac" => AudioFormat::Flac,
            "ogg" => AudioFormat::Ogg,
            "aac" => AudioFormat::Aac,
            "pcm" => AudioFormat::Pcm,
            _ => AudioFormat::Mp3,
        }
    };
    match value {
        Value::Array(arr) => arr.iter().flat_map(parse_audios).collect(),
        Value::String(s) => vec![AudioContent {
            source: parse_audio_string_source(s),
            format: AudioFormat::Mp3,
        }],
        Value::Object(obj) => {
            let format = parse_format(obj.get("format"));
            if let Some(uri) = obj.get("resource").and_then(|v| v.as_str()) {
                return vec![AudioContent {
                    source: AudioSource::McpResource(uri.to_owned()),
                    format,
                }];
            }
            if let Some(url) = obj.get("url").and_then(|v| v.as_str()) {
                return vec![AudioContent {
                    source: AudioSource::Url(url.to_owned()),
                    format,
                }];
            }
            if let Some(data) = obj.get("data").and_then(|v| v.as_str()) {
                return vec![AudioContent {
                    source: AudioSource::Base64 {
                        data: data.to_owned(),
                    },
                    format,
                }];
            }
            Vec::new()
        }
        _ => Vec::new(),
    }
}

fn parse_audio_string_source(s: &str) -> AudioSource {
    if s.starts_with("mcpg-resource://") {
        AudioSource::McpResource(s.to_owned())
    } else if s.starts_with("http://") || s.starts_with("https://") {
        AudioSource::Url(s.to_owned())
    } else {
        AudioSource::Base64 { data: s.to_owned() }
    }
}

fn parse_files(value: &Value) -> Vec<FileContent> {
    match value {
        Value::Array(arr) => arr.iter().flat_map(parse_files).collect(),
        Value::String(s) => vec![FileContent {
            source: parse_file_string_source(s),
            mime_type: String::new(), // Sniffed downstream when blank.
            filename: None,
        }],
        Value::Object(obj) => {
            let mime_type = obj
                .get("mime_type")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_owned();
            let filename = obj
                .get("filename")
                .and_then(|v| v.as_str())
                .map(|s| s.to_owned());
            if let Some(uri) = obj.get("resource").and_then(|v| v.as_str()) {
                return vec![FileContent {
                    source: FileSource::McpResource(uri.to_owned()),
                    mime_type,
                    filename,
                }];
            }
            if let Some(url) = obj.get("url").and_then(|v| v.as_str()) {
                return vec![FileContent {
                    source: FileSource::Url(url.to_owned()),
                    mime_type,
                    filename,
                }];
            }
            if let Some(data) = obj.get("data").and_then(|v| v.as_str()) {
                return vec![FileContent {
                    source: FileSource::Base64 {
                        data: data.to_owned(),
                    },
                    mime_type,
                    filename,
                }];
            }
            Vec::new()
        }
        _ => Vec::new(),
    }
}

fn parse_file_string_source(s: &str) -> FileSource {
    if s.starts_with("mcpg-resource://") {
        FileSource::McpResource(s.to_owned())
    } else if s.starts_with("http://") || s.starts_with("https://") {
        FileSource::Url(s.to_owned())
    } else {
        FileSource::Base64 { data: s.to_owned() }
    }
}

/// Helper bound to [`ToolsSpec`] that resolves child-tool definitions
/// the LLM will see. The `lookup` callback is the gateway-side helper
/// that, given a binding name, returns its `description` and
/// `input_schema` (when known). Unknown bindings yield a permissive
/// `{type: object}` shape — operators can supply per-binding schemas
/// in their own catalogs.
pub fn build_child_tool_defs(
    tools: &ToolsSpec,
    mut lookup: impl FnMut(&str) -> Option<ChildToolMeta>,
) -> Vec<ToolDef> {
    tools
        .allowed
        .iter()
        .map(|name| {
            let meta = lookup(name).unwrap_or_default();
            ToolDef {
                name: name.clone(),
                description: meta.description.unwrap_or_else(|| name.clone()),
                parameters: meta
                    .input_schema
                    .unwrap_or_else(|| json!({"type": "object", "additionalProperties": true})),
            }
        })
        .collect()
}

/// What [`build_child_tool_defs`] needs about each child binding.
#[derive(Debug, Clone, Default)]
pub struct ChildToolMeta {
    pub description: Option<String>,
    pub input_schema: Option<Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_tool_result_under_limit_unchanged() {
        let s = "x".repeat(100);
        let out = truncate_tool_result(s.clone(), 200);
        assert_eq!(out, s);
    }

    #[test]
    fn truncate_tool_result_over_limit_marked() {
        let s = "x".repeat(500);
        let out = truncate_tool_result(s, 100);
        assert!(out.starts_with("xxxx"));
        assert!(out.contains("[truncated"));
        assert!(out.len() < 200);
    }

    #[test]
    fn parse_response_json_handles_whitespace() {
        let v = parse_response_json("  \n {\"x\":1}  \n").unwrap();
        assert_eq!(v, json!({"x":1}));
    }

    #[test]
    fn parse_response_json_rejects_empty() {
        assert!(parse_response_json("").is_err());
        assert!(parse_response_json("   ").is_err());
    }

    #[test]
    fn parse_response_json_rejects_non_json() {
        assert!(parse_response_json("just text").is_err());
    }

    #[test]
    fn compute_backoff_caps_at_max() {
        assert_eq!(compute_backoff(500, 4_000, 1), 500);
        assert_eq!(compute_backoff(500, 4_000, 2), 1_000);
        assert_eq!(compute_backoff(500, 4_000, 4), 4_000);
        assert_eq!(compute_backoff(500, 4_000, 30), 4_000);
    }

    #[test]
    fn retry_reason_allowed_filters_correctly() {
        let allowed = vec![RetryReason::RateLimited, RetryReason::Server];
        assert!(retry_reason_allowed(
            &ProviderError::RateLimited {
                message: "x".into()
            },
            &allowed
        ));
        assert!(retry_reason_allowed(
            &ProviderError::Server {
                message: "x".into()
            },
            &allowed
        ));
        assert!(!retry_reason_allowed(
            &ProviderError::Network {
                message: "x".into()
            },
            &allowed
        ));
        assert!(!retry_reason_allowed(
            &ProviderError::AuthFailed {
                message: "x".into()
            },
            &allowed
        ));
    }

    #[test]
    fn build_child_tool_defs_respects_lookup_and_falls_back() {
        let tools = ToolsSpec {
            allowed: vec!["known".into(), "unknown".into()],
            ..Default::default()
        };
        let defs = build_child_tool_defs(&tools, |name| {
            if name == "known" {
                Some(ChildToolMeta {
                    description: Some("a known tool".into()),
                    input_schema: Some(
                        json!({"type": "object", "properties": {"k": {"type": "string"}}}),
                    ),
                })
            } else {
                None
            }
        });
        assert_eq!(defs.len(), 2);
        assert_eq!(defs[0].name, "known");
        assert_eq!(defs[0].description, "a known tool");
        assert_eq!(defs[1].name, "unknown");
        assert_eq!(defs[1].description, "unknown");
        assert_eq!(defs[1].parameters["type"], json!("object"));
    }

    #[test]
    fn binding_error_from_provider_preserves_category_in_message() {
        let e = binding_error_from_provider(
            "b",
            ProviderError::RateLimited {
                message: "slow".into(),
            },
        );
        if let BackendError::Transport { message } = e {
            assert!(message.contains("rate limited"));
        } else {
            panic!("expected Transport");
        }
    }

    // ----- build_user_message + parsers -----

    fn prompt(image: &[&str], audio: &[&str], file: &[&str]) -> PromptSpec {
        PromptSpec {
            system: "s".into(),
            user: "{{ input.text }}".into(),
            image_inputs: image.iter().map(|s| s.to_string()).collect(),
            audio_inputs: audio.iter().map(|s| s.to_string()).collect(),
            file_inputs: file.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn build_user_message_text_only_when_no_inputs_declared() {
        let m = build_user_message(
            "hello".into(),
            &json!({"image": "ignored"}),
            &prompt(&[], &[], &[]),
        );
        assert!(
            matches!(m.content, crate::normalized::MessageContent::Text(ref s) if s == "hello")
        );
    }

    #[test]
    fn build_user_message_promotes_to_parts_with_image_string() {
        let m = build_user_message(
            "describe".into(),
            &json!({"image": "https://ex.com/a.png"}),
            &prompt(&["image"], &[], &[]),
        );
        match &m.content {
            crate::normalized::MessageContent::Parts(parts) => {
                assert!(matches!(&parts[0], ContentPart::Text(s) if s == "describe"));
                assert!(
                    matches!(&parts[1], ContentPart::Image(img) if matches!(&img.source, ImageSource::Url(u) if u == "https://ex.com/a.png"))
                );
            }
            _ => panic!("expected Parts"),
        }
    }

    #[test]
    fn build_user_message_recognises_data_url() {
        let m = build_user_message(
            "x".into(),
            &json!({"image": "data:image/jpeg;base64,abc=="}),
            &prompt(&["image"], &[], &[]),
        );
        if let crate::normalized::MessageContent::Parts(parts) = &m.content {
            assert!(
                matches!(&parts[1], ContentPart::Image(img) if matches!(&img.source, ImageSource::Base64{mime_type, data} if mime_type == "image/jpeg" && data == "abc=="))
            );
        } else {
            panic!("expected Parts");
        }
    }

    #[test]
    fn build_user_message_recognises_mcpg_resource_string() {
        let m = build_user_message(
            "x".into(),
            &json!({"image": "mcpg-resource://hash:abc"}),
            &prompt(&["image"], &[], &[]),
        );
        if let crate::normalized::MessageContent::Parts(parts) = &m.content {
            assert!(
                matches!(&parts[1], ContentPart::Image(img) if matches!(&img.source, ImageSource::McpResource(s) if s == "mcpg-resource://hash:abc"))
            );
        } else {
            panic!("expected Parts");
        }
    }

    #[test]
    fn build_user_message_object_form_with_url_and_detail() {
        let m = build_user_message(
            "x".into(),
            &json!({"image": {"url": "https://ex.com/a.png", "detail": "low"}}),
            &prompt(&["image"], &[], &[]),
        );
        if let crate::normalized::MessageContent::Parts(parts) = &m.content {
            assert!(
                matches!(&parts[1], ContentPart::Image(img) if img.detail == Some(ImageDetail::Low))
            );
        } else {
            panic!("expected Parts");
        }
    }

    #[test]
    fn build_user_message_array_fans_out() {
        let m = build_user_message(
            "x".into(),
            &json!({
                "image": [
                    "https://ex.com/a.png",
                    "https://ex.com/b.png",
                ]
            }),
            &prompt(&["image"], &[], &[]),
        );
        if let crate::normalized::MessageContent::Parts(parts) = &m.content {
            // 1 text + 2 images
            assert_eq!(parts.len(), 3);
            assert!(matches!(&parts[1], ContentPart::Image(_)));
            assert!(matches!(&parts[2], ContentPart::Image(_)));
        } else {
            panic!("expected Parts");
        }
    }

    #[test]
    fn build_user_message_audio_with_format_object() {
        let m = build_user_message(
            "transcribe".into(),
            &json!({"audio": {"data": "QUJD", "format": "wav"}}),
            &prompt(&[], &["audio"], &[]),
        );
        if let crate::normalized::MessageContent::Parts(parts) = &m.content {
            match &parts[1] {
                ContentPart::Audio(au) => {
                    assert_eq!(au.format, AudioFormat::Wav);
                    assert!(matches!(&au.source, AudioSource::Base64{data} if data == "QUJD"));
                }
                _ => panic!("expected Audio"),
            }
        } else {
            panic!("expected Parts");
        }
    }

    #[test]
    fn build_user_message_file_string_is_base64_default() {
        let m = build_user_message(
            "summarize".into(),
            &json!({"doc": "JVBERi0="}),
            &prompt(&[], &[], &["doc"]),
        );
        if let crate::normalized::MessageContent::Parts(parts) = &m.content {
            match &parts[1] {
                ContentPart::File(f) => {
                    assert!(matches!(&f.source, FileSource::Base64 { .. }));
                }
                _ => panic!("expected File"),
            }
        } else {
            panic!("expected Parts");
        }
    }

    #[test]
    fn build_user_message_file_object_with_filename() {
        let m = build_user_message(
            "x".into(),
            &json!({"doc": {
                "url": "https://ex.com/x.pdf",
                "mime_type": "application/pdf",
                "filename": "report.pdf"
            }}),
            &prompt(&[], &[], &["doc"]),
        );
        if let crate::normalized::MessageContent::Parts(parts) = &m.content {
            match &parts[1] {
                ContentPart::File(f) => {
                    assert_eq!(f.filename.as_deref(), Some("report.pdf"));
                    assert_eq!(f.mime_type, "application/pdf");
                    assert!(matches!(&f.source, FileSource::Url(u) if u == "https://ex.com/x.pdf"));
                }
                _ => panic!("expected File"),
            }
        } else {
            panic!("expected Parts");
        }
    }

    #[test]
    fn build_user_message_missing_arg_produces_text_only() {
        // Operator declared `image_inputs: ["image"]` but the call
        // didn't supply one. Engine falls back to the text-only
        // shape so the model still gets the prompt.
        let m = build_user_message(
            "no image given".into(),
            &json!({"unrelated": "x"}),
            &prompt(&["image"], &[], &[]),
        );
        assert!(matches!(
            m.content,
            crate::normalized::MessageContent::Text(_)
        ));
    }
}
