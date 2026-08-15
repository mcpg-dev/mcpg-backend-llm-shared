//! Per-binding daily USD spend ledger.
//!
//! Tracks aggregate cost per binding within one UTC day for
//! `BudgetSpec::usd_daily_cap` enforcement. Process-wide; resets
//! at UTC midnight. Single-instance only — multi-instance prod
//! deployments that need shared budget state should look at the
//! cluster_backend-backed cache pattern (see
//! `mcpg_plugin_host::credential_cache_clustered` for the
//! template). For v1.0 manual testing, the in-process ledger is
//! sufficient: it stops a runaway agentic loop on one node from
//! billing $100; cluster-wide quota is a separate epic.
//!
//! ## Why a global static
//!
//! The engine is constructed per `(plugin, backend_name)` at
//! `register_profile` time. Multiple `ChatEngine` instances may
//! exist for one binding name across plugin reloads, but the
//! daily ledger should persist for the calendar day regardless.
//! A `LazyLock<Mutex<HashMap<...>>>` keyed on binding name gives
//! that property without any per-engine state to thread through
//! constructors.
//!
//! ## What gets counted
//!
//! Only successful calls whose cost the rate card resolves.
//! A binding using a model the rate card doesn't know cannot
//! accumulate; the engine logs once per call and skips the
//! ledger update. This is by design: a budget cap that
//! *silently* fails to count would be worse than no cap.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

use chrono::{DateTime, NaiveDate, Utc};

/// Ledger entry for one binding.
struct DailyEntry {
    /// UTC date the running total applies to. When the next
    /// `record` call happens on a different day, the running
    /// total resets.
    day: NaiveDate,
    /// Aggregate USD spent today. Stored as f64 — at typical LLM
    /// per-call costs (sub-cent to a few dollars) accumulating
    /// over a day stays comfortably inside f64 precision; we're
    /// not engineering a financial general ledger.
    usd: f64,
}

static LEDGER: LazyLock<Mutex<HashMap<String, DailyEntry>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Look up the binding's running daily total. Returns `0.0` for
/// any binding never recorded today (or recorded only on a
/// previous calendar day — stale rows are pruned on next write).
#[must_use]
pub fn current_daily_usd(backend_name: &str) -> f64 {
    current_daily_usd_at(backend_name, Utc::now())
}

/// Same as [`current_daily_usd`] but with an injectable clock
/// for tests. Production callers use the no-arg variant.
#[must_use]
pub fn current_daily_usd_at(backend_name: &str, now: DateTime<Utc>) -> f64 {
    let today = now.date_naive();
    let guard = LEDGER.lock().unwrap_or_else(|e| e.into_inner());
    match guard.get(backend_name) {
        Some(entry) if entry.day == today => entry.usd,
        _ => 0.0,
    }
}

/// Add `cost_usd` to the binding's daily total. Resets the
/// running total when the calendar day has rolled over.
pub fn record_cost(backend_name: &str, cost_usd: f64) {
    record_cost_at(backend_name, cost_usd, Utc::now());
}

/// Same as [`record_cost`] but with an injectable clock for tests.
pub fn record_cost_at(backend_name: &str, cost_usd: f64, now: DateTime<Utc>) {
    if !cost_usd.is_finite() || cost_usd <= 0.0 {
        return;
    }
    let today = now.date_naive();
    let mut guard = LEDGER.lock().unwrap_or_else(|e| e.into_inner());
    let entry = guard.entry(backend_name.to_owned()).or_insert(DailyEntry {
        day: today,
        usd: 0.0,
    });
    if entry.day != today {
        entry.day = today;
        entry.usd = 0.0;
    }
    entry.usd += cost_usd;
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn day(y: i32, m: u32, d: u32, h: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(y, m, d, h, 0, 0).unwrap()
    }

    fn fresh_name(suffix: &str) -> String {
        format!(
            "test-{suffix}-{}",
            std::time::UNIX_EPOCH.elapsed().unwrap().as_nanos()
        )
    }

    #[test]
    fn unknown_binding_reports_zero() {
        let name = fresh_name("unknown");
        assert_eq!(current_daily_usd_at(&name, day(2026, 4, 30, 10)), 0.0);
    }

    #[test]
    fn record_then_read_same_day() {
        let name = fresh_name("same-day");
        let t = day(2026, 4, 30, 12);
        record_cost_at(&name, 0.50, t);
        record_cost_at(&name, 0.25, t);
        let got = current_daily_usd_at(&name, t);
        assert!((got - 0.75).abs() < 1e-9, "got {got}");
    }

    #[test]
    fn day_rollover_resets() {
        let name = fresh_name("rollover");
        record_cost_at(&name, 1.0, day(2026, 4, 30, 23));
        record_cost_at(&name, 2.0, day(2026, 5, 1, 0));
        let got = current_daily_usd_at(&name, day(2026, 5, 1, 0));
        assert!((got - 2.0).abs() < 1e-9);
    }

    #[test]
    fn rejects_non_finite_or_zero() {
        let name = fresh_name("nan");
        let t = day(2026, 4, 30, 12);
        record_cost_at(&name, f64::NAN, t);
        record_cost_at(&name, f64::INFINITY, t);
        record_cost_at(&name, 0.0, t);
        record_cost_at(&name, -5.0, t);
        assert_eq!(current_daily_usd_at(&name, t), 0.0);
    }

    #[test]
    fn previous_day_treated_as_zero_without_recording() {
        let name = fresh_name("stale");
        record_cost_at(&name, 0.1, day(2026, 4, 1, 12));
        // Reading on a much later day, without recording, returns 0.
        let got = current_daily_usd_at(&name, day(2026, 4, 30, 12));
        assert_eq!(got, 0.0);
    }
}
