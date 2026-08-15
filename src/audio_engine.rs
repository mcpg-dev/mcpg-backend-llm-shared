//! Provider-agnostic TTS / STT engines.
//!
//! Both engines mirror the embedding engine's retry shape — no
//! agentic loop, no streaming, just a single upstream call with
//! exponential backoff on retryable errors. The TTS engine pushes
//! synthesized bytes into the gateway's `ContentStore` and returns
//! a `mcpg-resource://<id>` URI; the STT engine resolves the input
//! audio source (URL / base64 / `mcpg-resource://`) before the
//! adapter sees it.

use std::sync::Arc;
use std::time::Duration;

use mcpg_plugin_protocol::{BackendError, BackendHost, BackendHostError, BackendInvocationContext};
use serde_json::{Value, json};
use tracing::{debug, instrument, warn};

use crate::audio::{
    NormalizedSttRequest, NormalizedSttResponse, NormalizedTtsRequest, NormalizedTtsResponse,
    SttProviderAdapter, TtsProviderAdapter,
};
use crate::audio_config::{SttExecutionSpec, TtsExecutionSpec};
use crate::embedding_config::EmbeddingRetrySpec;
use crate::error::ProviderError;
use crate::normalized::AudioFormat;

// ---------------------------------------------------------------------------
// TTS engine
// ---------------------------------------------------------------------------

pub struct TtsEngine {
    pub backend_name: String,
    pub adapter: Arc<dyn TtsProviderAdapter>,
    pub spec: TtsExecutionSpec,
    pub host: Arc<dyn BackendHost>,
}

impl std::fmt::Debug for TtsEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TtsEngine")
            .field("backend_name", &self.backend_name)
            .field("provider", &self.adapter.label())
            .field("model", &self.spec.model)
            .finish()
    }
}

impl TtsEngine {
    /// Run one TTS call.
    ///
    /// `args`:
    /// ```json
    /// {
    ///   "text": "Hello world",
    ///   "voice": "alloy",      // optional, falls back to spec.voice
    ///   "format": "mp3",       // optional, falls back to spec.format
    ///   "speed": 1.0           // optional, falls back to spec.speed
    /// }
    /// ```
    ///
    /// Returns: `{audio_uri, mime_type, format}`.
    #[instrument(skip(self, args), fields(binding = %self.backend_name, provider = %self.adapter.label(), model = %self.spec.model))]
    pub async fn execute(
        &self,
        args: &Value,
        request_id: &str,
        session_id: Option<&str>,
    ) -> Result<Value, BackendError> {
        let text =
            args.get("text")
                .and_then(|v| v.as_str())
                .ok_or_else(|| BackendError::InvalidSpec {
                    message: "tts args must include `text` (string)".into(),
                })?;
        if text.trim().is_empty() {
            return Err(BackendError::InvalidSpec {
                message: "`text` must not be empty".into(),
            });
        }

        let voice = args
            .get("voice")
            .and_then(|v| v.as_str())
            .map(|s| s.to_owned())
            .unwrap_or_else(|| self.spec.voice.clone());
        if voice.trim().is_empty() {
            return Err(BackendError::InvalidSpec {
                message: "`voice` must not be empty (set in spec or per-call args)".into(),
            });
        }

        let format = args
            .get("format")
            .and_then(|v| v.as_str())
            .and_then(parse_audio_format)
            .unwrap_or(self.spec.format);
        let speed = args
            .get("speed")
            .and_then(|v| v.as_f64())
            .map(|f| f as f32)
            .or(self.spec.speed);

        let request = NormalizedTtsRequest {
            model: self.spec.model.clone(),
            text: text.to_owned(),
            voice,
            format,
            speed,
        };

        let started = std::time::Instant::now();
        let resp = call_with_retry_tts(
            self.adapter.as_ref(),
            &request,
            self.spec.timeout(),
            &self.spec.retry,
        )
        .await
        .map_err(|e| binding_error_from_provider("tts", &self.backend_name, e))?;
        let elapsed = started.elapsed();

        metrics::histogram!(
            "mcpg_tts_call_seconds",
            "binding" => self.backend_name.clone(),
            "provider" => self.adapter.label().to_string(),
            "model" => self.spec.model.clone(),
        )
        .record(elapsed.as_secs_f64());
        metrics::counter!(
            "mcpg_tts_calls_total",
            "binding" => self.backend_name.clone(),
            "provider" => self.adapter.label().to_string(),
            "model" => self.spec.model.clone(),
            "status" => "ok",
        )
        .increment(1);

        let host_ctx = BackendInvocationContext::root(
            request_id,
            session_id.map(|s| s.to_owned()),
            self.backend_name.clone(),
        );
        let resource = self
            .host
            .store_content(&host_ctx, resp.bytes, resp.mime_type.clone(), None)
            .await
            .map_err(|e| BackendError::Transport {
                message: format!("store synthesized audio: {e}"),
            })?;

        Ok(json!({
            "audio_uri": resource.uri,
            "mime_type": resp.mime_type,
            "format": audio_format_label(format),
        }))
    }
}

// ---------------------------------------------------------------------------
// STT engine
// ---------------------------------------------------------------------------

pub struct SttEngine {
    pub backend_name: String,
    pub adapter: Arc<dyn SttProviderAdapter>,
    pub spec: SttExecutionSpec,
    pub host: Arc<dyn BackendHost>,
}

impl std::fmt::Debug for SttEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SttEngine")
            .field("backend_name", &self.backend_name)
            .field("provider", &self.adapter.label())
            .field("model", &self.spec.model)
            .finish()
    }
}

impl SttEngine {
    /// Run one transcription call.
    ///
    /// `args`:
    /// ```json
    /// {
    ///   "audio": "mcpg-resource://hash:abc",  // string OR object form
    ///   "language": "en",                     // optional
    ///   "prompt": "Custom vocab: …"           // optional
    /// }
    /// ```
    ///
    /// `audio` accepts the same shapes as multimodal audio inputs:
    /// `mcpg-resource://`, `https://…`, raw base64, or
    /// `{source: ...}` / `{url|resource|data, format?}` objects.
    ///
    /// Returns: `{text, language?, duration_seconds?}`.
    #[instrument(skip(self, args), fields(binding = %self.backend_name, provider = %self.adapter.label(), model = %self.spec.model))]
    pub async fn execute(
        &self,
        args: &Value,
        request_id: &str,
        session_id: Option<&str>,
    ) -> Result<Value, BackendError> {
        let audio_arg = args.get("audio").ok_or_else(|| BackendError::InvalidSpec {
            message: "stt args must include `audio`".into(),
        })?;

        let host_ctx = BackendInvocationContext::root(
            request_id,
            session_id.map(|s| s.to_owned()),
            self.backend_name.clone(),
        );

        let (bytes, mime_type) = self
            .resolve_audio_input(audio_arg, &host_ctx)
            .await
            .map_err(|e| match e {
                ResolveAudioError::Missing => BackendError::InvalidSpec {
                    message: "could not resolve `audio` input".into(),
                },
                ResolveAudioError::SizeLimit { actual, limit } => BackendError::InvalidSpec {
                    message: format!(
                        "audio input too large: {actual} bytes exceeds limit of {limit}"
                    ),
                },
                ResolveAudioError::Unsupported(msg) => BackendError::InvalidSpec { message: msg },
                ResolveAudioError::Host(e) => BackendError::Transport {
                    message: format!("resolve audio input: {e}"),
                },
            })?;

        let language = args
            .get("language")
            .and_then(|v| v.as_str())
            .map(|s| s.to_owned())
            .or_else(|| self.spec.language.clone());
        let prompt = args
            .get("prompt")
            .and_then(|v| v.as_str())
            .map(|s| s.to_owned());

        let request = NormalizedSttRequest {
            model: self.spec.model.clone(),
            bytes,
            mime_type,
            language,
            prompt,
        };

        let started = std::time::Instant::now();
        let resp = call_with_retry_stt(
            self.adapter.as_ref(),
            &request,
            self.spec.timeout(),
            &self.spec.retry,
        )
        .await
        .map_err(|e| binding_error_from_provider("stt", &self.backend_name, e))?;
        let elapsed = started.elapsed();

        metrics::histogram!(
            "mcpg_stt_call_seconds",
            "binding" => self.backend_name.clone(),
            "provider" => self.adapter.label().to_string(),
            "model" => self.spec.model.clone(),
        )
        .record(elapsed.as_secs_f64());
        metrics::counter!(
            "mcpg_stt_calls_total",
            "binding" => self.backend_name.clone(),
            "provider" => self.adapter.label().to_string(),
            "model" => self.spec.model.clone(),
            "status" => "ok",
        )
        .increment(1);

        let mut out = json!({"text": resp.text});
        if let Some(lang) = resp.language {
            out["language"] = json!(lang);
        }
        if let Some(d) = resp.duration_seconds {
            out["duration_seconds"] = json!(d);
        }
        Ok(out)
    }

    async fn resolve_audio_input(
        &self,
        arg: &Value,
        ctx: &BackendInvocationContext,
    ) -> Result<(bytes::Bytes, String), ResolveAudioError> {
        let max_bytes = self
            .spec
            .max_input_bytes
            .unwrap_or(crate::multimodal::DEFAULT_MAX_INLINE_BYTES);
        // Accept either a string or an object with `url|resource|data`
        // keys. This is the same shape `parse_audio_string_source`
        // / `parse_audios` accept on the chat path; we duplicate
        // the resolution here because STT bindings don't go through
        // the chat engine.
        let (kind, value) = classify_audio_arg(arg).ok_or(ResolveAudioError::Missing)?;
        let mime_default_for_format = |fmt: &str| -> String {
            parse_audio_format(fmt)
                .map(|f| f.mime_type().to_owned())
                .unwrap_or_else(|| "audio/mpeg".to_owned())
        };
        match kind {
            AudioKind::Resource(uri) => match self.host.fetch_content(ctx, &uri).await {
                Ok(Some(b)) => {
                    if max_bytes != 0 && b.len() > max_bytes {
                        return Err(ResolveAudioError::SizeLimit {
                            actual: b.len(),
                            limit: max_bytes,
                        });
                    }
                    let mime = sniff_audio_mime(&b)
                        .or_else(|| {
                            value
                                .get("format")
                                .and_then(|v| v.as_str())
                                .map(mime_default_for_format)
                        })
                        .unwrap_or_else(|| "audio/mpeg".to_owned());
                    Ok((b, mime))
                }
                Ok(None) => Err(ResolveAudioError::Host(format!(
                    "resource not found: {uri}"
                ))),
                Err(e) => Err(ResolveAudioError::Host(e.to_string())),
            },
            AudioKind::Url => {
                // STT inputs need bytes for the multipart upload —
                // adapters don't accept URLs upstream. We fail loud
                // rather than silently fetching from arbitrary URLs.
                Err(ResolveAudioError::Unsupported(
                    "audio URL inputs are not supported by stt; pass mcpg-resource:// or base64"
                        .into(),
                ))
            }
            AudioKind::Base64(b64) => {
                use base64::Engine;
                let raw = base64::engine::general_purpose::STANDARD
                    .decode(&b64)
                    .map_err(|e| ResolveAudioError::Host(format!("decode base64: {e}")))?;
                if max_bytes != 0 && raw.len() > max_bytes {
                    return Err(ResolveAudioError::SizeLimit {
                        actual: raw.len(),
                        limit: max_bytes,
                    });
                }
                let bytes = bytes::Bytes::from(raw);
                let mime = sniff_audio_mime(&bytes)
                    .or_else(|| {
                        value
                            .get("format")
                            .and_then(|v| v.as_str())
                            .map(mime_default_for_format)
                    })
                    .unwrap_or_else(|| "audio/mpeg".to_owned());
                Ok((bytes, mime))
            }
        }
    }
}

#[derive(Debug)]
enum ResolveAudioError {
    Missing,
    SizeLimit {
        actual: usize,
        limit: usize,
    },
    /// Operator-facing error — the input shape is unsupported by
    /// this binding (e.g. URL inputs to STT). Translated to
    /// `BackendError::InvalidSpec` upstream.
    Unsupported(String),
    /// Host-side error during fetch — translated to
    /// `BackendError::Transport` upstream.
    Host(String),
}

impl From<BackendHostError> for ResolveAudioError {
    fn from(e: BackendHostError) -> Self {
        Self::Host(e.to_string())
    }
}

#[derive(Debug)]
enum AudioKind {
    Resource(String),
    /// URL inputs are accepted by the classifier but rejected at
    /// resolution time — STT bindings need bytes for the multipart
    /// upload. The variant carries no payload because the URL
    /// value never crosses the resolver boundary.
    Url,
    Base64(String),
}

fn classify_audio_arg(arg: &Value) -> Option<(AudioKind, &Value)> {
    match arg {
        Value::String(s) => Some((classify_audio_string(s), arg)),
        Value::Object(obj) => {
            if let Some(uri) = obj.get("resource").and_then(|v| v.as_str()) {
                Some((AudioKind::Resource(uri.to_owned()), arg))
            } else if obj.get("url").and_then(|v| v.as_str()).is_some() {
                Some((AudioKind::Url, arg))
            } else if let Some(data) = obj.get("data").and_then(|v| v.as_str()) {
                Some((AudioKind::Base64(data.to_owned()), arg))
            } else {
                None
            }
        }
        _ => None,
    }
}

fn classify_audio_string(s: &str) -> AudioKind {
    if s.starts_with("mcpg-resource://") {
        AudioKind::Resource(s.to_owned())
    } else if s.starts_with("http://") || s.starts_with("https://") {
        AudioKind::Url
    } else {
        AudioKind::Base64(s.to_owned())
    }
}

fn parse_audio_format(s: &str) -> Option<AudioFormat> {
    match s {
        "mp3" => Some(AudioFormat::Mp3),
        "wav" => Some(AudioFormat::Wav),
        "flac" => Some(AudioFormat::Flac),
        "ogg" | "opus" => Some(AudioFormat::Ogg),
        "aac" => Some(AudioFormat::Aac),
        "pcm" => Some(AudioFormat::Pcm),
        _ => None,
    }
}

fn audio_format_label(f: AudioFormat) -> &'static str {
    match f {
        AudioFormat::Mp3 => "mp3",
        AudioFormat::Wav => "wav",
        AudioFormat::Flac => "flac",
        AudioFormat::Ogg => "ogg",
        AudioFormat::Aac => "aac",
        AudioFormat::Pcm => "pcm",
    }
}

fn sniff_audio_mime(bytes: &[u8]) -> Option<String> {
    match bytes {
        [b'O', b'g', b'g', b'S', ..] => Some("audio/ogg".into()),
        [b'f', b'L', b'a', b'C', ..] => Some("audio/flac".into()),
        [b'I', b'D', b'3', ..] | [0xFF, 0xFB, ..] => Some("audio/mpeg".into()),
        [
            b'R',
            b'I',
            b'F',
            b'F',
            _,
            _,
            _,
            _,
            b'W',
            b'A',
            b'V',
            b'E',
            ..,
        ] => Some("audio/wav".into()),
        _ => None,
    }
}

async fn call_with_retry_tts(
    adapter: &dyn TtsProviderAdapter,
    request: &NormalizedTtsRequest,
    timeout: Duration,
    retry: &EmbeddingRetrySpec,
) -> Result<NormalizedTtsResponse, ProviderError> {
    let mut attempt: u32 = 0;
    loop {
        attempt += 1;
        match adapter.synthesize(request, timeout).await {
            Ok(resp) => return Ok(resp),
            Err(err) if attempt >= retry.max_attempts || !err.is_retryable() => {
                if !err.is_retryable() {
                    debug!(?err, "tts error not retryable");
                } else {
                    warn!(
                        ?err,
                        attempt,
                        max = retry.max_attempts,
                        "tts retries exhausted"
                    );
                }
                return Err(err);
            }
            Err(err) => {
                let backoff_ms = std::cmp::min(
                    retry.initial_backoff_ms.saturating_mul(1 << (attempt - 1)),
                    retry.max_backoff_ms,
                );
                debug!(?err, attempt, backoff_ms, "tts retry");
                tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
            }
        }
    }
}

async fn call_with_retry_stt(
    adapter: &dyn SttProviderAdapter,
    request: &NormalizedSttRequest,
    timeout: Duration,
    retry: &EmbeddingRetrySpec,
) -> Result<NormalizedSttResponse, ProviderError> {
    let mut attempt: u32 = 0;
    loop {
        attempt += 1;
        match adapter.transcribe(request, timeout).await {
            Ok(resp) => return Ok(resp),
            Err(err) if attempt >= retry.max_attempts || !err.is_retryable() => {
                if !err.is_retryable() {
                    debug!(?err, "stt error not retryable");
                } else {
                    warn!(
                        ?err,
                        attempt,
                        max = retry.max_attempts,
                        "stt retries exhausted"
                    );
                }
                return Err(err);
            }
            Err(err) => {
                let backoff_ms = std::cmp::min(
                    retry.initial_backoff_ms.saturating_mul(1 << (attempt - 1)),
                    retry.max_backoff_ms,
                );
                debug!(?err, attempt, backoff_ms, "stt retry");
                tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
            }
        }
    }
}

fn binding_error_from_provider(kind: &str, binding: &str, err: ProviderError) -> BackendError {
    match err {
        ProviderError::Network { message } => BackendError::Transport {
            message: format!("{kind} {binding}: network: {message}"),
        },
        ProviderError::Server { message } => BackendError::Transport {
            message: format!("{kind} {binding}: server: {message}"),
        },
        ProviderError::RateLimited { message } => BackendError::Transport {
            message: format!("{kind} {binding}: rate-limited: {message}"),
        },
        ProviderError::AuthFailed { message } => BackendError::InvalidSpec {
            message: format!("{kind} {binding}: auth: {message}"),
        },
        ProviderError::BadRequest { message } => BackendError::InvalidSpec {
            message: format!("{kind} {binding}: bad request: {message}"),
        },
        ProviderError::ContextLimit { message } => BackendError::InvalidSpec {
            message: format!("{kind} {binding}: context limit: {message}"),
        },
        ProviderError::Malformed { message } => BackendError::Transport {
            message: format!("{kind} {binding}: malformed: {message}"),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use mcpg_plugin_protocol::BackendResource;
    use std::collections::HashMap;
    use std::sync::Mutex;

    struct StubTts {
        bytes: Vec<u8>,
        mime: &'static str,
    }
    impl std::fmt::Debug for StubTts {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("StubTts").finish()
        }
    }
    #[async_trait]
    impl TtsProviderAdapter for StubTts {
        fn label(&self) -> &'static str {
            "stub"
        }
        async fn synthesize(
            &self,
            _r: &NormalizedTtsRequest,
            _t: Duration,
        ) -> Result<NormalizedTtsResponse, ProviderError> {
            Ok(NormalizedTtsResponse {
                bytes: bytes::Bytes::copy_from_slice(&self.bytes),
                mime_type: self.mime.to_owned(),
            })
        }
    }

    struct StubStt {
        text: String,
        language: Option<String>,
    }
    impl std::fmt::Debug for StubStt {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("StubStt").finish()
        }
    }
    #[async_trait]
    impl SttProviderAdapter for StubStt {
        fn label(&self) -> &'static str {
            "stub"
        }
        async fn transcribe(
            &self,
            _r: &NormalizedSttRequest,
            _t: Duration,
        ) -> Result<NormalizedSttResponse, ProviderError> {
            Ok(NormalizedSttResponse {
                text: self.text.clone(),
                language: self.language.clone(),
                duration_seconds: Some(2.5),
            })
        }
    }

    struct ContentHost {
        store: Mutex<HashMap<String, bytes::Bytes>>,
    }
    impl ContentHost {
        fn new(initial: &[(&str, &[u8])]) -> Self {
            let mut store = HashMap::new();
            for (k, v) in initial {
                store.insert((*k).to_owned(), bytes::Bytes::copy_from_slice(v));
            }
            Self {
                store: Mutex::new(store),
            }
        }
    }
    #[async_trait]
    impl BackendHost for ContentHost {
        async fn invoke_tool(
            &self,
            _ctx: &BackendInvocationContext,
            _t: &str,
            _a: &Value,
        ) -> Result<Value, BackendHostError> {
            unreachable!()
        }
        async fn store_content(
            &self,
            _ctx: &BackendInvocationContext,
            bytes: bytes::Bytes,
            mime_type: String,
            _ttl: Option<Duration>,
        ) -> Result<BackendResource, BackendHostError> {
            let id = format!("hash:{}", self.store.lock().unwrap().len());
            self.store.lock().unwrap().insert(id.clone(), bytes.clone());
            Ok(BackendResource {
                id: id.clone(),
                uri: format!("mcpg-resource://{id}"),
                size_bytes: bytes.len(),
                mime_type,
                content_hash: format!("blake3:{id}"),
                expires_at_unix: None,
            })
        }
        async fn fetch_content(
            &self,
            _ctx: &BackendInvocationContext,
            uri: &str,
        ) -> Result<Option<bytes::Bytes>, BackendHostError> {
            let id = uri.strip_prefix("mcpg-resource://").unwrap_or(uri);
            Ok(self.store.lock().unwrap().get(id).cloned())
        }
    }

    #[tokio::test]
    async fn tts_returns_audio_uri() {
        let host = Arc::new(ContentHost::new(&[]));
        let engine = TtsEngine {
            backend_name: "t".into(),
            adapter: Arc::new(StubTts {
                bytes: vec![0xFF, 0xFB, 0x01, 0x02],
                mime: "audio/mpeg",
            }),
            spec: TtsExecutionSpec {
                model: "tts-1".into(),
                voice: "alloy".into(),
                ..Default::default()
            },
            host: host.clone() as Arc<dyn BackendHost>,
        };
        let r = engine
            .execute(&json!({"text": "hello"}), "r", None)
            .await
            .unwrap();
        assert!(
            r["audio_uri"]
                .as_str()
                .unwrap()
                .starts_with("mcpg-resource://")
        );
        assert_eq!(r["mime_type"], "audio/mpeg");
        assert_eq!(r["format"], "mp3");
    }

    #[tokio::test]
    async fn tts_missing_text_errors() {
        let host = Arc::new(ContentHost::new(&[]));
        let engine = TtsEngine {
            backend_name: "t".into(),
            adapter: Arc::new(StubTts {
                bytes: vec![],
                mime: "audio/mpeg",
            }),
            spec: TtsExecutionSpec {
                model: "x".into(),
                voice: "v".into(),
                ..Default::default()
            },
            host: host.clone() as Arc<dyn BackendHost>,
        };
        let err = engine.execute(&json!({}), "r", None).await.unwrap_err();
        assert!(matches!(err, BackendError::InvalidSpec { .. }));
    }

    #[tokio::test]
    async fn tts_per_call_voice_overrides_spec() {
        let host = Arc::new(ContentHost::new(&[]));
        let engine = TtsEngine {
            backend_name: "t".into(),
            adapter: Arc::new(StubTts {
                bytes: vec![0xFF, 0xFB],
                mime: "audio/mpeg",
            }),
            spec: TtsExecutionSpec {
                model: "tts-1".into(),
                voice: "alloy".into(),
                ..Default::default()
            },
            host: host.clone() as Arc<dyn BackendHost>,
        };
        // No assertion on adapter receiving the override since
        // StubTts ignores; this just confirms no error path.
        engine
            .execute(&json!({"text": "x", "voice": "nova"}), "r", None)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn stt_resolves_resource_input_and_returns_text() {
        let host = Arc::new(ContentHost::new(&[(
            "hash:audio1",
            // OggS magic + filler so the sniff classifies as audio/ogg.
            b"OggSffillerbyteshere",
        )]));
        let engine = SttEngine {
            backend_name: "s".into(),
            adapter: Arc::new(StubStt {
                text: "hello".into(),
                language: Some("en".into()),
            }),
            spec: SttExecutionSpec {
                model: "whisper-1".into(),
                ..Default::default()
            },
            host: host.clone() as Arc<dyn BackendHost>,
        };
        let r = engine
            .execute(&json!({"audio": "mcpg-resource://hash:audio1"}), "r", None)
            .await
            .unwrap();
        assert_eq!(r["text"], "hello");
        assert_eq!(r["language"], "en");
        assert_eq!(r["duration_seconds"], 2.5);
    }

    #[tokio::test]
    async fn stt_rejects_url_input() {
        let host = Arc::new(ContentHost::new(&[]));
        let engine = SttEngine {
            backend_name: "s".into(),
            adapter: Arc::new(StubStt {
                text: "x".into(),
                language: None,
            }),
            spec: SttExecutionSpec {
                model: "whisper-1".into(),
                ..Default::default()
            },
            host: host.clone() as Arc<dyn BackendHost>,
        };
        let err = engine
            .execute(&json!({"audio": "https://ex.com/a.mp3"}), "r", None)
            .await
            .unwrap_err();
        assert!(matches!(err, BackendError::InvalidSpec { .. }));
    }

    #[tokio::test]
    async fn stt_oversize_input_errors() {
        let host = Arc::new(ContentHost::new(&[("hash:big", &[0u8; 100][..])]));
        let engine = SttEngine {
            backend_name: "s".into(),
            adapter: Arc::new(StubStt {
                text: "x".into(),
                language: None,
            }),
            spec: SttExecutionSpec {
                model: "whisper-1".into(),
                max_input_bytes: Some(50),
                ..Default::default()
            },
            host: host.clone() as Arc<dyn BackendHost>,
        };
        let err = engine
            .execute(&json!({"audio": "mcpg-resource://hash:big"}), "r", None)
            .await
            .unwrap_err();
        assert!(matches!(err, BackendError::InvalidSpec { .. }));
    }

    #[tokio::test]
    async fn stt_missing_audio_errors() {
        let host = Arc::new(ContentHost::new(&[]));
        let engine = SttEngine {
            backend_name: "s".into(),
            adapter: Arc::new(StubStt {
                text: "x".into(),
                language: None,
            }),
            spec: SttExecutionSpec {
                model: "whisper-1".into(),
                ..Default::default()
            },
            host: host.clone() as Arc<dyn BackendHost>,
        };
        let err = engine.execute(&json!({}), "r", None).await.unwrap_err();
        assert!(matches!(err, BackendError::InvalidSpec { .. }));
    }
}
