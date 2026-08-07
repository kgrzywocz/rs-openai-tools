//! Forward-compatibility guarantees for the public enums.
//!
//! OpenAI adds values to these enums continuously - new models, new voices,
//! new lifecycle states, new realtime events. Two separate mechanisms keep the
//! library from breaking when that happens, and they solve different problems:
//!
//! - `#[non_exhaustive]` is a **compile-time / semver** tool: adding a variant
//!   stops being a breaking change for downstream crates.
//! - A catch-all variant (`Other(String)`, `Unknown`) is a **runtime** tool:
//!   an unrecognised value deserializes instead of failing the whole response.
//!
//! `#[non_exhaustive]` does nothing for the runtime case, which is why the
//! response-side enums need both.

use openai_tools::batch::response::BatchStatus;
use openai_tools::common::role::Role;
use openai_tools::fine_tuning::response::FineTuningJobStatus;
use openai_tools::realtime::events::server::{ResponseStatus, ServerEvent};
use openai_tools::realtime::ItemStatus;
use openai_tools::videos::response::VideoStatus;

// ===========================================================================
// Role
// ===========================================================================

/// `developer` is the recommended replacement for `system` on reasoning
/// models, and the Conversations API echoes it back on stored items.
#[test]
fn role_supports_developer() {
    assert_eq!(Role::Developer.as_str(), "developer");
    assert_eq!(serde_json::to_string(&Role::Developer).unwrap(), "\"developer\"");
    assert_eq!(serde_json::from_str::<Role>("\"developer\"").unwrap(), Role::Developer);
    assert_eq!(Role::try_from("developer".to_string()).unwrap(), Role::Developer);
    assert_eq!(format!("{}", Role::Developer), "developer");
}

#[test]
fn role_preserves_unknown_values() {
    let parsed: Role = serde_json::from_str("\"moderator\"").expect("an unknown role must still deserialize");
    assert_eq!(parsed, Role::Other("moderator".to_string()));
    assert_eq!(parsed.as_str(), "moderator");
    // Round-trips back out unchanged.
    assert_eq!(serde_json::to_string(&parsed).unwrap(), "\"moderator\"");
}

/// The existing roles must keep their exact wire representation.
#[test]
fn role_known_values_unchanged() {
    for (role, expected) in
        [(Role::System, "system"), (Role::User, "user"), (Role::Assistant, "assistant"), (Role::Function, "function"), (Role::Tool, "tool")]
    {
        assert_eq!(role.as_str(), expected);
        assert_eq!(format!("{}", role), expected);
        assert_eq!(serde_json::to_string(&role).unwrap(), format!("\"{}\"", expected));
        assert_eq!(serde_json::from_str::<Role>(&format!("\"{}\"", expected)).unwrap(), role);
    }
}

/// `TryFrom` stays strict: it validates caller-supplied input, unlike serde
/// which has to tolerate whatever the API sends.
#[test]
fn role_try_from_rejects_unknown() {
    assert!(Role::try_from("moderator".to_string()).is_err());
}

// ===========================================================================
// Response-side status enums
// ===========================================================================

#[test]
fn batch_status_preserves_unknown_values() {
    let parsed: BatchStatus = serde_json::from_str("\"expired_soon\"").expect("unknown batch status must deserialize");
    assert_eq!(parsed, BatchStatus::Other("expired_soon".to_string()));

    // Known values are unaffected.
    assert_eq!(serde_json::from_str::<BatchStatus>("\"completed\"").unwrap(), BatchStatus::Completed);
}

#[test]
fn fine_tuning_job_status_preserves_unknown_values() {
    let parsed: FineTuningJobStatus = serde_json::from_str("\"paused\"").expect("unknown job status must deserialize");
    assert_eq!(parsed, FineTuningJobStatus::Other("paused".to_string()));

    assert_eq!(serde_json::from_str::<FineTuningJobStatus>("\"succeeded\"").unwrap(), FineTuningJobStatus::Succeeded);
}

#[test]
fn item_status_preserves_unknown_values() {
    let parsed: ItemStatus = serde_json::from_str("\"interrupted\"").expect("unknown item status must deserialize");
    assert_eq!(parsed, ItemStatus::Other("interrupted".to_string()));

    assert_eq!(serde_json::from_str::<ItemStatus>("\"completed\"").unwrap(), ItemStatus::Completed);
}

#[test]
fn response_status_preserves_unknown_values() {
    let parsed: ResponseStatus = serde_json::from_str("\"interrupted\"").expect("unknown response status must deserialize");
    assert_eq!(parsed, ResponseStatus::Other("interrupted".to_string()));

    assert_eq!(serde_json::from_str::<ResponseStatus>("\"failed\"").unwrap(), ResponseStatus::Failed);
}

/// Already covered in 2.0.0; asserted here so the whole family is checked in
/// one place.
#[test]
fn video_status_preserves_unknown_values() {
    let parsed: VideoStatus = serde_json::from_str("\"cancelled\"").expect("unknown video status must deserialize");
    assert_eq!(parsed, VideoStatus::Other("cancelled".to_string()));
}

// ===========================================================================
// Realtime server events
// ===========================================================================

/// A realtime session must not tear down because OpenAI shipped a new event
/// type. Unrecognised events surface as `Unknown` rather than an error.
#[test]
fn server_event_tolerates_unknown_event_types() {
    let json = r#"{"type":"response.brand_new_thing","event_id":"evt_1","payload":{"a":1}}"#;
    let parsed: ServerEvent = serde_json::from_str(json).expect("an unknown server event must deserialize");
    assert!(matches!(parsed, ServerEvent::Unknown));
}

/// The `other` fallback must not swallow event types the library does know
/// about.
#[test]
fn server_event_known_types_still_parse() {
    let json = r#"{"type":"conversation.item.deleted","event_id":"evt_1","item_id":"item_1"}"#;
    let parsed: ServerEvent = serde_json::from_str(json).expect("a known server event must deserialize");

    assert!(!matches!(parsed, ServerEvent::Unknown), "a known event must not fall through to Unknown");
    assert_eq!(parsed.event_id(), Some("evt_1"));
}

/// `event_id()` has no value to return for an unrecognised event.
#[test]
fn unknown_server_event_has_no_event_id() {
    let json = r#"{"type":"response.brand_new_thing","event_id":"evt_1"}"#;
    let parsed: ServerEvent = serde_json::from_str(json).unwrap();
    assert_eq!(parsed.event_id(), None);
}

// ===========================================================================
// #[non_exhaustive] audit
// ===========================================================================

/// `#[non_exhaustive]` only has an effect on *downstream* crates, so it cannot
/// be observed from a test. This audit reads the source instead, which also
/// stops a future model enum from being added without the attribute - the
/// exact mistake that forced 2.0.0 to be a major release.
#[test]
fn designated_enums_are_non_exhaustive() {
    /// (file relative to openai-tools/src, enum name)
    const REQUIRED: &[(&str, &str)] = &[
        // --- Model identifiers ---
        ("common/models.rs", "ChatModel"),
        ("common/models.rs", "EmbeddingModel"),
        ("common/models.rs", "RealtimeModel"),
        ("common/models.rs", "FineTuningModel"),
        ("images/request.rs", "ImageModel"),
        ("videos/request.rs", "VideoModel"),
        ("audio/request.rs", "TtsModel"),
        ("audio/request.rs", "SttModel"),
        ("moderations/request.rs", "ModerationModel"),
        ("realtime/audio.rs", "TranscriptionModel"),
        // --- Request-side value enums ---
        ("audio/request.rs", "Voice"),
        ("audio/request.rs", "AudioFormat"),
        ("audio/request.rs", "TranscriptionFormat"),
        ("audio/request.rs", "TimestampGranularity"),
        ("realtime/audio.rs", "Voice"),
        ("realtime/audio.rs", "AudioFormat"),
        ("realtime/audio.rs", "NoiseReductionType"),
        ("realtime/session.rs", "Modality"),
        ("realtime/vad.rs", "Eagerness"),
        ("images/request.rs", "ImageSize"),
        ("images/request.rs", "ImageQuality"),
        ("videos/request.rs", "VideoSize"),
        ("videos/request.rs", "VideoSeconds"),
        ("videos/request.rs", "VideoVariant"),
        ("responses/request.rs", "ReasoningEffort"),
        ("responses/request.rs", "ReasoningSummary"),
        ("responses/request.rs", "TextVerbosity"),
        ("responses/request.rs", "Include"),
        ("conversations/request.rs", "ConversationInclude"),
        ("files/request.rs", "FilePurpose"),
        ("batch/request.rs", "BatchEndpoint"),
        ("batch/request.rs", "CompletionWindow"),
        // --- Response-side enums (these also carry a catch-all variant) ---
        ("common/role.rs", "Role"),
        ("batch/response.rs", "BatchStatus"),
        ("fine_tuning/response.rs", "FineTuningJobStatus"),
        ("realtime/conversation.rs", "ItemStatus"),
        ("realtime/events/server.rs", "ResponseStatus"),
        ("realtime/events/server.rs", "ServerEvent"),
        ("videos/response.rs", "VideoStatus"),
    ];

    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut missing = Vec::new();

    for (file, name) in REQUIRED {
        let src = std::fs::read_to_string(root.join(file)).unwrap_or_else(|e| panic!("cannot read {file}: {e}"));

        let decl = format!("pub enum {name} {{");
        let at = src.find(&decl).unwrap_or_else(|| panic!("{name} not found in {file}"));

        // Walk back over the attribute block immediately above the decl.
        let preceding = &src[..at];
        if !preceding.trim_end().ends_with(']') && !preceding.contains("#[non_exhaustive]") {
            missing.push(format!("{file}::{name}"));
            continue;
        }
        let window_start = preceding.len().saturating_sub(400);
        if !preceding[window_start..].contains("#[non_exhaustive]") {
            missing.push(format!("{file}::{name}"));
        }
    }

    assert!(missing.is_empty(), "these public enums must be marked #[non_exhaustive]:\n  {}", missing.join("\n  "));
}
