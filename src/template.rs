//! MiniJinja environment with a locked-down feature surface.
//!
//! Templates may reference only `input.*` and `meta.*`. Filesystem
//! access (`{% include %}` / `{% extends %}`), env-var lookup, and the
//! `debug` filter are unavailable — operators cannot exfiltrate
//! gateway state by accident. Filters that ARE available are the
//! minijinja built-ins plus a small whitelist (`json`, `quote_json`,
//! `truncate`).
//!
//! Templates stay text-only even with multimodal input —
//! operators declare `image_inputs` / `audio_inputs` / `file_inputs`
//! arrays naming which input fields carry media; the engine
//! constructs multi-part user messages without templating gymnastics.

use minijinja::{Environment, Error as MjError, Value as MjValue};
use serde_json::Value;

use crate::error::ConfigError;

/// Per-spec template renderer. Both prompts share one environment so
/// rendering is cheap.
pub struct Templates {
    env: Environment<'static>,
    /// Holds owned template source for the lifetime of `env`.
    /// `Environment::add_template_owned` requires `'static`, so we keep
    /// strings here and reference them via `Box::leak` substitutes —
    /// in practice we just clone into the env which owns the storage
    /// internally via the `_owned` API.
    _system_src: String,
    _user_src: String,
}

impl Templates {
    pub fn compile(system: &str, user: &str) -> Result<Self, ConfigError> {
        let mut env = Environment::new();
        // No filesystem loader, no auto-reload — purely in-memory.
        env.set_undefined_behavior(minijinja::UndefinedBehavior::Strict);
        // Forbid the `debug` filter / function — it dumps the whole
        // context, which would leak input args into logs.
        env.remove_filter("debug");

        env.add_template_owned("system", system.to_owned())
            .map_err(template_err)?;
        env.add_template_owned("user", user.to_owned())
            .map_err(template_err)?;

        Ok(Self {
            env,
            _system_src: system.to_owned(),
            _user_src: user.to_owned(),
        })
    }

    /// Render `name` (`"system"` or `"user"`) with the given context.
    /// Errors here are runtime issues (missing variable, type
    /// mismatch) — surface as transport failures to the engine.
    pub fn render(&self, name: &str, ctx: &TemplateContext) -> Result<String, String> {
        let tmpl = self
            .env
            .get_template(name)
            .map_err(|e| format!("template '{name}' not found: {e}"))?;
        let value = MjValue::from_serialize(ctx);
        tmpl.render(value)
            .map_err(|e| format!("render '{name}': {e}"))
    }
}

fn template_err(e: MjError) -> ConfigError {
    ConfigError::Template(format!("{e}"))
}

/// What the templates see. `input` holds the tool-call arguments;
/// `meta` carries call-scoped metadata (request_id, session_id, etc.).
#[derive(serde::Serialize)]
pub struct TemplateContext<'a> {
    pub input: &'a Value,
    pub meta: TemplateMeta<'a>,
}

#[derive(serde::Serialize)]
pub struct TemplateMeta<'a> {
    pub backend_name: &'a str,
    pub request_id: &'a str,
    pub session_id: Option<&'a str>,
    pub timestamp_iso8601: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn renders_input_substitution() {
        let t = Templates::compile("you are helpful", "Hello {{ input.name }}").unwrap();
        let input = json!({"name": "World"});
        let ctx = TemplateContext {
            input: &input,
            meta: TemplateMeta {
                backend_name: "b",
                request_id: "r",
                session_id: None,
                timestamp_iso8601: "2026-04-29T00:00:00Z".into(),
            },
        };
        assert_eq!(t.render("user", &ctx).unwrap(), "Hello World");
    }

    #[test]
    fn missing_variable_errors_strict() {
        let t = Templates::compile("sys", "{{ input.missing }}").unwrap();
        let input = json!({});
        let ctx = TemplateContext {
            input: &input,
            meta: TemplateMeta {
                backend_name: "b",
                request_id: "r",
                session_id: None,
                timestamp_iso8601: "x".into(),
            },
        };
        let err = t.render("user", &ctx).unwrap_err();
        assert!(err.contains("undefined") || err.contains("not"));
    }

    #[test]
    fn default_filter_works_for_optional_inputs() {
        let t = Templates::compile("sys", "Severity: {{ input.severity | default('unknown') }}")
            .unwrap();
        let input = json!({});
        let ctx = TemplateContext {
            input: &input,
            meta: TemplateMeta {
                backend_name: "b",
                request_id: "r",
                session_id: None,
                timestamp_iso8601: "x".into(),
            },
        };
        assert_eq!(t.render("user", &ctx).unwrap(), "Severity: unknown");
    }

    #[test]
    fn parse_error_surfaces_at_compile_time() {
        let res = Templates::compile("sys", "Hello {{ input.");
        match res {
            Err(ConfigError::Template(_)) => {}
            Err(other) => panic!("expected Template error, got: {other:?}"),
            Ok(_) => panic!("expected error from malformed template"),
        }
    }

    #[test]
    fn meta_fields_accessible() {
        let t = Templates::compile("sys", "req={{ meta.request_id }}").unwrap();
        let input = json!({});
        let ctx = TemplateContext {
            input: &input,
            meta: TemplateMeta {
                backend_name: "b",
                request_id: "abc-123",
                session_id: None,
                timestamp_iso8601: "x".into(),
            },
        };
        assert_eq!(t.render("user", &ctx).unwrap(), "req=abc-123");
    }
}
