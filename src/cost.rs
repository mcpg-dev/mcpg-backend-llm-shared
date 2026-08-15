//! # Cost rate cards
//!
//! The engine emits per-call cost metrics. Pricing is read from a
//! TOML file vendored in this crate (`models.toml`); operators with
//! custom rate cards (resold pricing, negotiated discounts, in-house
//! models) point at an override file via
//! `plugins.cost.rate_card_path`.
//!
//! ## What's here
//!
//! - [`RateCard`]: parsed rate card.
//! - [`bundled_rate_card`]: the shared crate's vendored copy. Cheap
//!   — the file is `include_str!`'d at compile time, parsed once
//!   per process via `OnceLock`.
//! - [`compute_chat_cost_usd`] / [`compute_embedding_cost_usd`]:
//!   arithmetic helpers. Return `None` when the model isn't in the
//!   card so the caller can quietly drop the metric rather than
//!   emit a misleading zero.

use serde::Deserialize;
use std::sync::OnceLock;

use crate::normalized::TokenUsage;

/// Parsed model row in the rate-card TOML.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatModelEntry {
    pub provider: String,
    pub model: String,
    pub input_per_million_usd: f64,
    pub output_per_million_usd: f64,
    /// OpenAI / Anthropic surface prompt-cache discounts; absent on
    /// providers that don't.
    #[serde(default)]
    pub cached_input_per_million_usd: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingModelEntry {
    pub provider: String,
    pub model: String,
    pub per_million_usd: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RateCard {
    pub schema_version: u32,
    pub last_updated: String,
    #[serde(default, rename = "model")]
    pub chat: Vec<ChatModelEntry>,
    #[serde(default, rename = "embedding_model")]
    pub embedding: Vec<EmbeddingModelEntry>,
}

impl RateCard {
    /// Parse a TOML rate card.
    pub fn parse(toml_str: &str) -> Result<Self, RateCardError> {
        toml::from_str(toml_str).map_err(|e| RateCardError::Parse {
            message: e.to_string(),
        })
    }

    pub fn find_chat(&self, provider: &str, model: &str) -> Option<&ChatModelEntry> {
        // Provider-label match is case-insensitive (cheap; saves
        // operators tripping on `OpenAI` vs `openai`); model match
        // is exact since model names contain dates.
        self.chat
            .iter()
            .find(|e| e.provider.eq_ignore_ascii_case(provider) && e.model == model)
    }

    pub fn find_embedding(&self, provider: &str, model: &str) -> Option<&EmbeddingModelEntry> {
        self.embedding
            .iter()
            .find(|e| e.provider.eq_ignore_ascii_case(provider) && e.model == model)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RateCardError {
    #[error("rate card parse error: {message}")]
    Parse { message: String },
}

/// The crate-vendored rate card, parsed lazily and cached.
pub fn bundled_rate_card() -> &'static RateCard {
    static CARD: OnceLock<RateCard> = OnceLock::new();
    CARD.get_or_init(|| {
        let raw = include_str!("../models.toml");
        RateCard::parse(raw).expect("vendored models.toml is well-formed (compile-time invariant)")
    })
}

/// Compute USD cost for one chat call. Returns `None` when the
/// `(provider, model)` pair isn't in the card — the caller is
/// expected to silently skip emitting cost metrics rather than
/// emit a misleading zero.
pub fn compute_chat_cost_usd(
    rate_card: &RateCard,
    provider: &str,
    model: &str,
    usage: &TokenUsage,
) -> Option<f64> {
    let entry = rate_card.find_chat(provider, model)?;
    let input_uncached = usage.input_tokens.saturating_sub(usage.cached_input_tokens);
    let cached_per_m = entry
        .cached_input_per_million_usd
        .unwrap_or(entry.input_per_million_usd);
    let input = (input_uncached as f64) / 1_000_000.0 * entry.input_per_million_usd;
    let cached = (usage.cached_input_tokens as f64) / 1_000_000.0 * cached_per_m;
    let output = (usage.output_tokens as f64) / 1_000_000.0 * entry.output_per_million_usd;
    Some(input + cached + output)
}

/// Compute USD cost for one embedding call. `total_input_tokens`
/// matches the `usage.input_tokens` summed across batches by the
/// engine.
pub fn compute_embedding_cost_usd(
    rate_card: &RateCard,
    provider: &str,
    model: &str,
    total_input_tokens: u32,
) -> Option<f64> {
    let entry = rate_card.find_embedding(provider, model)?;
    Some((total_input_tokens as f64) / 1_000_000.0 * entry.per_million_usd)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn usage(input: u32, output: u32, cached: u32) -> TokenUsage {
        TokenUsage {
            input_tokens: input,
            output_tokens: output,
            cached_input_tokens: cached,
        }
    }

    #[test]
    fn bundled_rate_card_parses() {
        let card = bundled_rate_card();
        assert_eq!(card.schema_version, 1);
        assert!(!card.chat.is_empty());
        assert!(!card.embedding.is_empty());
    }

    #[test]
    fn find_chat_known_model() {
        let card = bundled_rate_card();
        let entry = card.find_chat("openai", "gpt-4o-mini").unwrap();
        assert_eq!(entry.input_per_million_usd, 0.15);
        assert_eq!(entry.output_per_million_usd, 0.60);
    }

    #[test]
    fn find_chat_provider_match_is_case_insensitive() {
        let card = bundled_rate_card();
        assert!(card.find_chat("OpenAI", "gpt-4o-mini").is_some());
        assert!(card.find_chat("OPENAI", "gpt-4o-mini").is_some());
    }

    #[test]
    fn find_chat_model_match_is_case_sensitive() {
        let card = bundled_rate_card();
        // Exact match: "gpt-4o-mini" hits.
        assert!(card.find_chat("openai", "gpt-4o-mini").is_some());
        // "GPT-4o-Mini" does not — model strings carry dates and
        // operator-facing names that must match exactly.
        assert!(card.find_chat("openai", "GPT-4o-Mini").is_none());
    }

    #[test]
    fn find_embedding_known_model() {
        let card = bundled_rate_card();
        let entry = card
            .find_embedding("openai", "text-embedding-3-small")
            .unwrap();
        assert_eq!(entry.per_million_usd, 0.02);
    }

    #[test]
    fn unknown_chat_model_returns_none() {
        let card = bundled_rate_card();
        assert!(compute_chat_cost_usd(card, "openai", "nope-9000", &usage(1, 1, 0)).is_none());
    }

    #[test]
    fn compute_chat_cost_basic() {
        let card = bundled_rate_card();
        // gpt-4o-mini: 0.15 input / 0.60 output. 1M in + 0.5M out
        // → 0.15 + 0.30 = 0.45 USD.
        let cost =
            compute_chat_cost_usd(card, "openai", "gpt-4o-mini", &usage(1_000_000, 500_000, 0))
                .unwrap();
        assert!((cost - 0.45).abs() < 1e-9);
    }

    #[test]
    fn compute_chat_cost_with_cached_discount() {
        let card = bundled_rate_card();
        // gpt-4o: 2.50 input / 10.00 output / 1.25 cached.
        // 1M total input of which 600k cached, 100k output:
        //   uncached = (1_000_000 - 600_000) * 2.50 / 1M = 1.0
        //   cached   = 600_000 * 1.25 / 1M = 0.75
        //   output   = 100_000 * 10.00 / 1M = 1.0
        //   total    = 2.75
        let cost = compute_chat_cost_usd(
            card,
            "openai",
            "gpt-4o",
            &usage(1_000_000, 100_000, 600_000),
        )
        .unwrap();
        assert!((cost - 2.75).abs() < 1e-9);
    }

    #[test]
    fn compute_chat_cost_falls_back_when_no_cached_rate() {
        let card = bundled_rate_card();
        // gpt-4o-mini has no `cached_input_per_million_usd`; the
        // formula falls back to the regular input rate. Tokens
        // accounted as cached still bill at the input rate, so:
        //   1M input total, 200k cached → 800k uncached + 200k cached
        //   = 1.0M @ 0.15 = 0.15
        let cost =
            compute_chat_cost_usd(card, "openai", "gpt-4o-mini", &usage(1_000_000, 0, 200_000))
                .unwrap();
        assert!((cost - 0.15).abs() < 1e-9);
    }

    #[test]
    fn compute_embedding_cost_basic() {
        let card = bundled_rate_card();
        // text-embedding-3-small: 0.02 / 1M.
        // 5M tokens → 5 * 0.02 = 0.10 USD.
        let cost = compute_embedding_cost_usd(card, "openai", "text-embedding-3-small", 5_000_000)
            .unwrap();
        assert!((cost - 0.10).abs() < 1e-9);
    }

    #[test]
    fn compute_embedding_cost_unknown_returns_none() {
        let card = bundled_rate_card();
        assert!(compute_embedding_cost_usd(card, "openai", "nope-embed", 1_000_000).is_none());
    }

    #[test]
    fn parse_rejects_malformed_toml() {
        let bad = "schema_version = \"not a number\"";
        assert!(RateCard::parse(bad).is_err());
    }
}
