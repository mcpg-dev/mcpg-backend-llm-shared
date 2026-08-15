//! # Multimodal helpers
//!
//! Pre-resolves `mcpg-resource://` references in chat messages
//! before they reach the adapter's encoding path. The adapters'
//! `encode_messages` are intentionally synchronous — pulling bytes
//! out of the [`mcpg_plugin_protocol::BackendHost`] async store would
//! force every encoder to become async and to know about hosts. The
//! engine instead walks the message list once before each provider
//! call and inlines any resource URIs as `Base64` sources, so the
//! adapter only ever sees `Url(_)` or `Base64 { … }` variants.

use base64::Engine;
use mcpg_plugin_protocol::{BackendHost, BackendHostError, BackendInvocationContext};

use crate::normalized::{
    AudioSource, ContentPart, FileSource, ImageSource, Message, MessageContent,
};

/// Errors raised while resolving multimodal sources. The engine
/// translates these into [`crate::error::ProviderError::BadRequest`]
/// — they all surface to the operator as a single "preflight failed"
/// classification, but each carries a distinct message so logs +
/// audit can pinpoint the failure mode.
#[derive(Debug, thiserror::Error)]
pub enum MultimodalError {
    /// `mcpg-resource://<id>` did not resolve. May be expired,
    /// session-ACL-refused, or never stored.
    #[error("resource not found: {uri}")]
    ResourceNotFound { uri: String },

    /// The `BackendHost::fetch_content` returned an error.
    #[error("resource lookup failed: {uri} — {cause}")]
    ResourceLookupFailed { uri: String, cause: String },

    /// MIME-type sniffing returned nothing usable. Some providers
    /// reject unspecified MIME on inline blobs; we surface this as
    /// a clear preflight failure rather than letting the adapter's
    /// upstream fail with a less obvious message.
    #[error("could not infer mime type for resource: {uri}")]
    MissingMimeType { uri: String },

    /// Resolved content exceeds the configured per-call inline cap.
    /// Operator either bumps the cap or switches to a flow that
    /// keeps the resource as an `mcpg-resource://` URL the model
    /// fetches itself.
    #[error("resolved content too large: {actual_bytes} bytes exceeds limit of {limit_bytes}")]
    SizeLimit {
        limit_bytes: usize,
        actual_bytes: usize,
    },
}

/// Default per-call inline cap (20 MiB). Matches the OpenAI /
/// Anthropic ceilings; Gemini's lower 9.5 MiB cap is enforced
/// upstream by the API itself when payloads exceed it.
pub const DEFAULT_MAX_INLINE_BYTES: usize = 20 * 1024 * 1024;

/// Walk the message and replace every `ContentPart` whose source is
/// `McpResource(_)` with the equivalent `Base64 { mime_type, data }`
/// variant. Other source variants pass through unchanged. Returns
/// the rewritten message; the original is consumed.
///
/// `max_inline_bytes = 0` disables the per-call size cap (useful for
/// tests). Production callers should pass [`DEFAULT_MAX_INLINE_BYTES`]
/// or operator-overridden value.
pub async fn prefetch_message_resources(
    mut message: Message,
    host: &dyn BackendHost,
    ctx: &BackendInvocationContext,
    max_inline_bytes: usize,
) -> Result<Message, MultimodalError> {
    if let MessageContent::Parts(parts) = &mut message.content {
        for part in parts.iter_mut() {
            match part {
                ContentPart::Image(img) => {
                    if let ImageSource::McpResource(uri) = &img.source {
                        let (data, mime_type) =
                            fetch_inline(uri, host, ctx, max_inline_bytes, true).await?;
                        img.source = ImageSource::Base64 { mime_type, data };
                    }
                }
                ContentPart::Audio(au) => {
                    if let AudioSource::McpResource(uri) = &au.source {
                        let (data, _mime) =
                            fetch_inline(uri, host, ctx, max_inline_bytes, false).await?;
                        // We discard the resolved MIME for audio
                        // because `AudioContent` carries `format`
                        // explicitly — adapters use that, not the
                        // store's recorded MIME.
                        au.source = AudioSource::Base64 { data };
                    }
                }
                ContentPart::File(f) => {
                    if let FileSource::McpResource(uri) = &f.source {
                        let (data, mime_type) =
                            fetch_inline(uri, host, ctx, max_inline_bytes, false).await?;
                        if f.mime_type.is_empty() {
                            f.mime_type = mime_type;
                        }
                        f.source = FileSource::Base64 { data };
                    }
                }
                ContentPart::Text(_) => {}
            }
        }
    }
    Ok(message)
}

/// Same as [`prefetch_message_resources`] but operates on a slice of
/// messages — convenience used by the engine when constructing the
/// full `NormalizedChatRequest`. Each message is rewritten in place;
/// errors short-circuit and bubble up the index of the failed
/// message in their `Display` impl via the underlying URI.
pub async fn prefetch_messages_resources(
    messages: Vec<Message>,
    host: &dyn BackendHost,
    ctx: &BackendInvocationContext,
    max_inline_bytes: usize,
) -> Result<Vec<Message>, MultimodalError> {
    let mut out = Vec::with_capacity(messages.len());
    for m in messages {
        out.push(prefetch_message_resources(m, host, ctx, max_inline_bytes).await?);
    }
    Ok(out)
}

/// Fetch a single `mcpg-resource://` URI through the host and return
/// `(base64-encoded data, sniffed mime_type)`.
///
/// `require_mime` is set when the caller (image content) cannot
/// downstream tolerate an empty MIME — every provider rejects an
/// image payload without an explicit MIME. For audio/file it's
/// false because audio carries format independently and file
/// callers pre-populate `mime_type` themselves.
async fn fetch_inline(
    uri: &str,
    host: &dyn BackendHost,
    ctx: &BackendInvocationContext,
    max_inline_bytes: usize,
    require_mime: bool,
) -> Result<(String, String), MultimodalError> {
    let bytes = match host.fetch_content(ctx, uri).await {
        Ok(Some(b)) => b,
        Ok(None) => {
            return Err(MultimodalError::ResourceNotFound {
                uri: uri.to_owned(),
            });
        }
        Err(BackendHostError::NotFound { .. }) => {
            return Err(MultimodalError::ResourceNotFound {
                uri: uri.to_owned(),
            });
        }
        Err(e) => {
            return Err(MultimodalError::ResourceLookupFailed {
                uri: uri.to_owned(),
                cause: e.to_string(),
            });
        }
    };
    if max_inline_bytes != 0 && bytes.len() > max_inline_bytes {
        return Err(MultimodalError::SizeLimit {
            limit_bytes: max_inline_bytes,
            actual_bytes: bytes.len(),
        });
    }
    let mime = sniff_mime(&bytes);
    if require_mime && mime.is_empty() {
        return Err(MultimodalError::MissingMimeType {
            uri: uri.to_owned(),
        });
    }
    let encoded = base64::engine::general_purpose::STANDARD.encode(&bytes);
    Ok((encoded, mime))
}

/// Cheap magic-byte MIME sniff. Covers the formats every supported
/// provider accepts inline. Returns an empty string when nothing
/// matches — caller decides whether to treat that as an error.
fn sniff_mime(bytes: &[u8]) -> String {
    match bytes {
        // PNG: 89 50 4E 47 0D 0A 1A 0A
        [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, ..] => "image/png".into(),
        // JPEG: FF D8 FF
        [0xFF, 0xD8, 0xFF, ..] => "image/jpeg".into(),
        // GIF: GIF8
        [b'G', b'I', b'F', b'8', ..] => "image/gif".into(),
        // WebP: RIFF....WEBP — match the RIFF prefix; adapters
        // accept any RIFF-coded image as image/webp.
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
            b'E',
            b'B',
            b'P',
            ..,
        ] => "image/webp".into(),
        // PDF: %PDF
        [b'%', b'P', b'D', b'F', ..] => "application/pdf".into(),
        // OGG: OggS
        [b'O', b'g', b'g', b'S', ..] => "audio/ogg".into(),
        // FLAC: fLaC
        [b'f', b'L', b'a', b'C', ..] => "audio/flac".into(),
        // MP3 / ID3: matches `ID3` tag-header or 0xFF 0xFB MP3
        // frame-sync.
        [b'I', b'D', b'3', ..] | [0xFF, 0xFB, ..] => "audio/mpeg".into(),
        // WAV: RIFF....WAVE
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
        ] => "audio/wav".into(),
        _ => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalized::{
        AudioContent, AudioFormat, FileContent, ImageContent, ImageDetail, ToolCall,
    };
    use bytes::Bytes;
    use mcpg_plugin_protocol::async_trait;
    use std::collections::HashMap;
    use std::sync::Mutex;

    /// Test stub: maps URIs to fixed payloads. Returns
    /// `BackendHostError::NotFound` for unknown URIs.
    struct StubHost {
        store: Mutex<HashMap<String, Bytes>>,
    }

    impl StubHost {
        fn new(entries: &[(&str, &[u8])]) -> Self {
            let mut store = HashMap::new();
            for (k, v) in entries {
                store.insert((*k).to_owned(), Bytes::copy_from_slice(v));
            }
            Self {
                store: Mutex::new(store),
            }
        }
    }

    #[async_trait]
    impl BackendHost for StubHost {
        async fn invoke_tool(
            &self,
            _ctx: &BackendInvocationContext,
            _tool: &str,
            _args: &serde_json::Value,
        ) -> Result<serde_json::Value, BackendHostError> {
            unreachable!("stub host: invoke_tool not used")
        }
        async fn fetch_content(
            &self,
            _ctx: &BackendInvocationContext,
            uri: &str,
        ) -> Result<Option<bytes::Bytes>, BackendHostError> {
            Ok(self.store.lock().unwrap().get(uri).cloned())
        }
    }

    fn ctx() -> BackendInvocationContext {
        BackendInvocationContext::root("r1", Some("sess".into()), "test")
    }

    fn png_bytes() -> Vec<u8> {
        let mut v = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        v.extend(b"more pixels here");
        v
    }

    #[tokio::test]
    async fn prefetch_replaces_image_resource_with_base64() {
        let host = StubHost::new(&[("mcpg-resource://hash:img1", &png_bytes())]);
        let m = Message::user_parts(vec![ContentPart::Image(ImageContent {
            source: ImageSource::McpResource("mcpg-resource://hash:img1".into()),
            detail: Some(ImageDetail::Auto),
        })]);
        let m = prefetch_message_resources(m, &host, &ctx(), 0)
            .await
            .unwrap();
        match &m.content {
            MessageContent::Parts(parts) => match &parts[0] {
                ContentPart::Image(img) => match &img.source {
                    ImageSource::Base64 { mime_type, data } => {
                        assert_eq!(mime_type, "image/png");
                        // Decoded base64 should round-trip to the
                        // input bytes.
                        let decoded = base64::engine::general_purpose::STANDARD
                            .decode(data)
                            .unwrap();
                        assert_eq!(decoded, png_bytes());
                    }
                    other => panic!("expected Base64 source, got {other:?}"),
                },
                other => panic!("expected Image part, got {other:?}"),
            },
            other => panic!("expected Parts content, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn prefetch_replaces_audio_resource_keeps_format() {
        let host = StubHost::new(&[("mcpg-resource://hash:au", b"OggS_audio_data_here")]);
        let m = Message::user_parts(vec![ContentPart::Audio(AudioContent {
            source: AudioSource::McpResource("mcpg-resource://hash:au".into()),
            format: AudioFormat::Ogg,
        })]);
        let m = prefetch_message_resources(m, &host, &ctx(), 0)
            .await
            .unwrap();
        match &m.content {
            MessageContent::Parts(parts) => match &parts[0] {
                ContentPart::Audio(au) => {
                    assert_eq!(au.format, AudioFormat::Ogg);
                    assert!(matches!(au.source, AudioSource::Base64 { .. }));
                }
                _ => panic!("expected Audio part"),
            },
            _ => panic!("expected Parts content"),
        }
    }

    #[tokio::test]
    async fn prefetch_replaces_file_resource_and_fills_missing_mime() {
        let host = StubHost::new(&[("mcpg-resource://hash:doc", b"%PDF-1.4 ...")]);
        let m = Message::user_parts(vec![ContentPart::File(FileContent {
            source: FileSource::McpResource("mcpg-resource://hash:doc".into()),
            mime_type: String::new(), // unset; sniff fills it.
            filename: None,
        })]);
        let m = prefetch_message_resources(m, &host, &ctx(), 0)
            .await
            .unwrap();
        match &m.content {
            MessageContent::Parts(parts) => match &parts[0] {
                ContentPart::File(f) => {
                    assert_eq!(f.mime_type, "application/pdf");
                    assert!(matches!(f.source, FileSource::Base64 { .. }));
                }
                _ => panic!("expected File part"),
            },
            _ => panic!("expected Parts content"),
        }
    }

    #[tokio::test]
    async fn prefetch_passes_through_url_and_base64_sources() {
        // No mcpg-resource:// → no host calls needed.
        let host = StubHost::new(&[]);
        let parts = vec![
            ContentPart::Text("hello".into()),
            ContentPart::Image(ImageContent {
                source: ImageSource::Url("https://ex.com/a.png".into()),
                detail: None,
            }),
            ContentPart::Image(ImageContent {
                source: ImageSource::Base64 {
                    mime_type: "image/png".into(),
                    data: "zzz".into(),
                },
                detail: None,
            }),
        ];
        let m = Message::user_parts(parts);
        let m = prefetch_message_resources(m, &host, &ctx(), 0)
            .await
            .unwrap();
        match &m.content {
            MessageContent::Parts(parts) => {
                assert!(
                    matches!(&parts[1], ContentPart::Image(img) if matches!(&img.source, ImageSource::Url(u) if u == "https://ex.com/a.png"))
                );
                assert!(
                    matches!(&parts[2], ContentPart::Image(img) if matches!(&img.source, ImageSource::Base64{..}))
                );
            }
            _ => panic!("expected Parts content"),
        }
    }

    #[tokio::test]
    async fn prefetch_text_message_passes_through_unchanged() {
        let host = StubHost::new(&[]);
        let m = Message::system("rules");
        let m = prefetch_message_resources(m, &host, &ctx(), 0)
            .await
            .unwrap();
        assert!(matches!(m.content, MessageContent::Text(ref s) if s == "rules"));
    }

    #[tokio::test]
    async fn prefetch_assistant_text_with_tool_calls_passes_through() {
        let host = StubHost::new(&[]);
        let calls = vec![ToolCall {
            id: "t1".into(),
            name: "fetch".into(),
            arguments: serde_json::json!({}),
        }];
        let m = Message::assistant_text_and_tool_calls("thinking…", calls);
        let m = prefetch_message_resources(m, &host, &ctx(), 0)
            .await
            .unwrap();
        assert!(matches!(m.content, MessageContent::Text(_)));
        assert_eq!(m.tool_calls.len(), 1);
    }

    #[tokio::test]
    async fn unknown_resource_uri_surfaces_not_found() {
        let host = StubHost::new(&[]);
        let m = Message::user_parts(vec![ContentPart::Image(ImageContent {
            source: ImageSource::McpResource("mcpg-resource://hash:nope".into()),
            detail: None,
        })]);
        let err = prefetch_message_resources(m, &host, &ctx(), 0)
            .await
            .unwrap_err();
        assert!(matches!(err, MultimodalError::ResourceNotFound { .. }));
    }

    #[tokio::test]
    async fn oversize_resource_returns_size_limit_error() {
        let big = vec![0u8; 100];
        let host = StubHost::new(&[("mcpg-resource://hash:big", &big)]);
        let m = Message::user_parts(vec![ContentPart::Image(ImageContent {
            source: ImageSource::McpResource("mcpg-resource://hash:big".into()),
            detail: None,
        })]);
        let err = prefetch_message_resources(m, &host, &ctx(), 50)
            .await
            .unwrap_err();
        assert!(matches!(err, MultimodalError::SizeLimit { .. }));
    }

    #[tokio::test]
    async fn missing_mime_for_image_surfaces_error() {
        // Random bytes with no recognisable magic.
        let host = StubHost::new(&[("mcpg-resource://hash:weird", b"random_unknown_bytes")]);
        let m = Message::user_parts(vec![ContentPart::Image(ImageContent {
            source: ImageSource::McpResource("mcpg-resource://hash:weird".into()),
            detail: None,
        })]);
        let err = prefetch_message_resources(m, &host, &ctx(), 0)
            .await
            .unwrap_err();
        assert!(matches!(err, MultimodalError::MissingMimeType { .. }));
    }

    #[test]
    fn sniff_mime_recognises_common_magic() {
        assert_eq!(
            sniff_mime(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]),
            "image/png"
        );
        assert_eq!(sniff_mime(&[0xFF, 0xD8, 0xFF, 0xE0]), "image/jpeg");
        assert_eq!(sniff_mime(b"GIF89a"), "image/gif");
        assert_eq!(sniff_mime(b"%PDF-1.7"), "application/pdf");
        assert_eq!(sniff_mime(b"OggS....."), "audio/ogg");
        assert_eq!(sniff_mime(b"fLaC...."), "audio/flac");
        assert_eq!(sniff_mime(b"unknown"), "");
    }

    #[tokio::test]
    async fn prefetch_messages_walks_each_message_in_sequence() {
        let host = StubHost::new(&[
            ("mcpg-resource://hash:a", b"%PDF-X"),
            ("mcpg-resource://hash:b", &png_bytes()),
        ]);
        let msgs = vec![
            Message::system("hi"),
            Message::user_parts(vec![ContentPart::File(FileContent {
                source: FileSource::McpResource("mcpg-resource://hash:a".into()),
                mime_type: String::new(),
                filename: None,
            })]),
            Message::user_parts(vec![ContentPart::Image(ImageContent {
                source: ImageSource::McpResource("mcpg-resource://hash:b".into()),
                detail: None,
            })]),
        ];
        let out = prefetch_messages_resources(msgs, &host, &ctx(), 0)
            .await
            .unwrap();
        assert_eq!(out.len(), 3);
        assert!(matches!(out[0].content, MessageContent::Text(_)));
        if let MessageContent::Parts(parts) = &out[1].content {
            assert!(
                matches!(&parts[0], ContentPart::File(f) if matches!(&f.source, FileSource::Base64 {..}))
            );
        } else {
            panic!("expected file message rewritten");
        }
        if let MessageContent::Parts(parts) = &out[2].content {
            assert!(
                matches!(&parts[0], ContentPart::Image(img) if matches!(&img.source, ImageSource::Base64 {..}))
            );
        } else {
            panic!("expected image message rewritten");
        }
    }
}
