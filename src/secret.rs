//! API-key resolution shared across every per-provider plugin.
//!
//! [`ApiKeyRef`] is the operator-facing config type for the provider
//! API key. Operators supply it as a plain string — typically a
//! `${env.X}` or `cred://…` reference that the gateway substitutes to
//! the literal value at config load, before the plugin ever sees it.
//! [`resolve_api_key`] returns that already-resolved value at
//! `register_profile` time.
//!
//! Plugins that need the resolved key as `Arc<str>` (the typical
//! case — the adapter holds it for the binding's lifetime) call
//! `Arc::from(resolve_api_key(&spec)?)`.

use mcpg_plugin_protocol::BackendError;
use mcpg_sensitive::Sensitive;
use serde::{Deserialize, Serialize};

/// The provider API key. Deserializes from a plain config string;
/// the gateway substitutes `${env.X}` / `cred://…` references to the
/// literal value at config load, so the plugin reads it directly.
///
/// Wraps the value in [`mcpg_sensitive::Sensitive`] so `Debug`
/// renders as `***` instead of leaking the key.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(transparent)]
pub struct ApiKeyRef(Sensitive<String>);

impl ApiKeyRef {
    /// Construct from an already-resolved key value.
    #[must_use]
    pub fn new(value: impl Into<String>) -> Self {
        Self(Sensitive::new(value.into()))
    }

    /// Borrow the resolved key string.
    #[must_use]
    pub fn expose(&self) -> &str {
        self.0.expose()
    }
}

/// Resolve a credential to a concrete string.
///
/// The value arrives already resolved by the gateway's secret layer.
/// An empty key is rejected with [`BackendError::InvalidSpec`] so a
/// missing `${env.X}` / `cred://…` substitution fails loudly rather
/// than producing unauthenticated requests.
pub fn resolve_api_key(key: &ApiKeyRef) -> Result<String, BackendError> {
    let value = key.expose();
    if value.is_empty() {
        return Err(BackendError::InvalidSpec {
            message: "api_key resolved to an empty value; check the \
                      ${env.X} / cred:// reference is set"
                .to_owned(),
        });
    }
    Ok(value.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_passes_through() {
        let key = ApiKeyRef::new("sk-test");
        assert_eq!(resolve_api_key(&key).unwrap(), "sk-test");
    }

    #[test]
    fn empty_value_errors() {
        let key = ApiKeyRef::new("");
        assert!(resolve_api_key(&key).is_err());
    }

    #[test]
    fn deserializes_from_plain_string() {
        let key: ApiKeyRef = serde_json::from_str("\"sk-prod\"").unwrap();
        assert_eq!(resolve_api_key(&key).unwrap(), "sk-prod");
    }

    #[test]
    fn json_round_trip() {
        let key = ApiKeyRef::new("z");
        let s = serde_json::to_string(&key).unwrap();
        let back: ApiKeyRef = serde_json::from_str(&s).unwrap();
        assert_eq!(key, back);
    }

    #[test]
    fn debug_redacts_the_value() {
        let v = ApiKeyRef::new("sk-prod-deadbeef");
        let rendered = format!("{v:?}");
        assert!(rendered.contains("***"), "{rendered}");
        assert!(!rendered.contains("sk-prod"), "leak: {rendered}");
    }
}
