//! Client and server events for the OpenAI Live API.

use serde::{Deserialize, Serialize};

use crate::common::models::LiveModel;

use super::session::{LiveAudioConfig, LiveSessionConfig, LiveSessionUpdate};

/// Events sent from the client to a Live WebSocket.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum ClientEvent {
    #[serde(rename = "session.start")]
    SessionStart {
        #[serde(skip_serializing_if = "Option::is_none")]
        event_id: Option<String>,
        session: LiveSessionConfig,
    },
    #[serde(rename = "session.update")]
    SessionUpdate {
        #[serde(skip_serializing_if = "Option::is_none")]
        event_id: Option<String>,
        session: LiveSessionUpdate,
    },
    #[serde(rename = "session.input_audio.append")]
    InputAudioAppend {
        #[serde(skip_serializing_if = "Option::is_none")]
        event_id: Option<String>,
        audio: String,
    },
    #[serde(rename = "session.input_audio.mute")]
    InputAudioMute {
        #[serde(skip_serializing_if = "Option::is_none")]
        event_id: Option<String>,
    },
    #[serde(rename = "session.input_audio.unmute")]
    InputAudioUnmute {
        #[serde(skip_serializing_if = "Option::is_none")]
        event_id: Option<String>,
    },
    #[serde(rename = "session.instructions.append")]
    InstructionsAppend {
        #[serde(skip_serializing_if = "Option::is_none")]
        event_id: Option<String>,
        content: String,
        delegation_id: Option<String>,
    },
    #[serde(rename = "session.thinking.append")]
    ThinkingAppend {
        #[serde(skip_serializing_if = "Option::is_none")]
        event_id: Option<String>,
        content: String,
        delegation_id: Option<String>,
    },
    #[serde(rename = "session.commentary.append")]
    CommentaryAppend {
        #[serde(skip_serializing_if = "Option::is_none")]
        event_id: Option<String>,
        content: String,
        delegation_id: Option<String>,
    },
    #[serde(rename = "response.item.create")]
    ResponseItemCreate {
        #[serde(skip_serializing_if = "Option::is_none")]
        event_id: Option<String>,
        item: serde_json::Value,
    },
    #[serde(rename = "response.create")]
    ResponseCreate {
        #[serde(skip_serializing_if = "Option::is_none")]
        event_id: Option<String>,
    },
    #[serde(rename = "session.close")]
    SessionClose {
        #[serde(skip_serializing_if = "Option::is_none")]
        event_id: Option<String>,
    },
}

/// Events received from a Live WebSocket.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
#[non_exhaustive]
pub enum ServerEvent {
    #[serde(rename = "session.started")]
    SessionStarted(SessionStartedEvent),
    #[serde(rename = "session.updated")]
    SessionUpdated(SessionUpdatedEvent),
    #[serde(rename = "session.input_audio.muted")]
    InputAudioMuted(AcknowledgementEvent),
    #[serde(rename = "session.input_audio.unmuted")]
    InputAudioUnmuted(AcknowledgementEvent),
    #[serde(rename = "session.instructions.appended")]
    InstructionsAppended(TimelineEvent),
    #[serde(rename = "session.thinking.appended")]
    ThinkingAppended(TimelineEvent),
    #[serde(rename = "session.commentary.appended")]
    CommentaryAppended(TimelineEvent),
    #[serde(rename = "session.output_audio.delta")]
    OutputAudioDelta(OutputAudioDeltaEvent),
    #[serde(rename = "session.input_transcript.delta")]
    InputTranscriptDelta(TranscriptDeltaEvent),
    #[serde(rename = "session.output_transcript.delta")]
    OutputTranscriptDelta(TranscriptDeltaEvent),
    #[serde(rename = "session.delegation.created")]
    DelegationCreated(DelegationCreatedEvent),
    #[serde(rename = "response.event")]
    ResponseEvent(ResponseEvent),
    #[serde(rename = "session.usage.updated")]
    SessionUsageUpdated(SessionUsageUpdatedEvent),
    #[serde(rename = "session.closed")]
    SessionClosed(SessionClosedEvent),
    #[serde(rename = "error")]
    Error(ErrorEvent),
    #[serde(rename = "info")]
    Info(InfoEvent),
    /// An event added after this library version.
    #[serde(other)]
    Unknown,
}

/// Resolved Live session state.
#[derive(Debug, Clone, Deserialize)]
pub struct SessionResource {
    pub id: String,
    pub expires_at: u64,
    pub model: LiveModel,
    pub status: String,
    pub audio: Option<LiveAudioConfig>,
    pub client: Option<serde_json::Value>,
    pub delegation: Option<serde_json::Value>,
    #[serde(default)]
    pub input: Vec<serde_json::Value>,
    pub instructions: Option<String>,
    pub store: Option<bool>,
}

/// Successful response to `session.start`.
#[derive(Debug, Clone, Deserialize)]
pub struct SessionStartedEvent {
    pub event_id: Option<String>,
    pub client_event_id: Option<String>,
    pub session: SessionResource,
}

/// Successful response to `session.update`.
#[derive(Debug, Clone, Deserialize)]
pub struct SessionUpdatedEvent {
    pub event_id: Option<String>,
    pub client_event_id: Option<String>,
    pub session: SessionResource,
}

/// Acknowledgement for an event without a payload.
#[derive(Debug, Clone, Deserialize)]
pub struct AcknowledgementEvent {
    pub event_id: Option<String>,
    pub client_event_id: Option<String>,
}

/// Acknowledgement positioned on the Live session timeline.
#[derive(Debug, Clone, Deserialize)]
pub struct TimelineEvent {
    pub event_id: Option<String>,
    pub client_event_id: Option<String>,
    pub start_ms: Option<u64>,
    pub end_ms: Option<u64>,
}

/// Generated output audio bytes, encoded as base64.
#[derive(Debug, Clone, Deserialize)]
pub struct OutputAudioDeltaEvent {
    pub event_id: Option<String>,
    pub client_event_id: Option<String>,
    pub delta: String,
    pub start_ms: Option<u64>,
    pub end_ms: Option<u64>,
}

/// Incremental input or output speech transcript.
#[derive(Debug, Clone, Deserialize)]
pub struct TranscriptDeltaEvent {
    pub event_id: Option<String>,
    pub client_event_id: Option<String>,
    pub delta: String,
    pub start_ms: Option<u64>,
    pub end_ms: Option<u64>,
}

/// A delegation created by the Live model.
#[derive(Debug, Clone, Deserialize)]
pub struct DelegationCreatedEvent {
    pub event_id: Option<String>,
    pub client_event_id: Option<String>,
    pub offset_ms: Option<u64>,
    pub delegation: DelegationResource,
}

/// Delegation metadata.
#[derive(Debug, Clone, Deserialize)]
pub struct DelegationResource {
    pub id: String,
    #[serde(rename = "type")]
    pub type_name: String,
    pub target: DelegationTarget,
    pub response_id: Option<String>,
}

/// The service handling delegated work.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum DelegationTarget {
    Client,
    Responses,
    /// A delegation target added after this library version.
    #[serde(untagged)]
    Other(String),
}

/// Envelope for a Responses API streaming event.
#[derive(Debug, Clone, Deserialize)]
pub struct ResponseEvent {
    pub event_id: Option<String>,
    pub client_event_id: Option<String>,
    pub delegation_id: Option<String>,
    pub event: serde_json::Value,
}

/// Cumulative Live audio usage.
///
/// `seconds` is cumulative; do not add values from successive events.
#[derive(Debug, Clone, Deserialize)]
pub struct SessionUsage {
    pub seconds: f64,
}

/// Current Live context-window utilization.
#[derive(Debug, Clone, Deserialize)]
pub struct ContextWindowUsage {
    pub usage_ratio: f64,
}

/// Updated cumulative usage and optional context-window utilization.
#[derive(Debug, Clone, Deserialize)]
pub struct SessionUsageUpdatedEvent {
    pub event_id: Option<String>,
    pub client_event_id: Option<String>,
    pub usage: SessionUsage,
    pub context_window: Option<ContextWindowUsage>,
}

/// Terminal Live session event.
#[derive(Debug, Clone, Deserialize)]
pub struct SessionClosedEvent {
    pub event_id: Option<String>,
    pub client_event_id: Option<String>,
    pub reason: CloseReason,
    pub session: SessionResource,
    pub usage: SessionUsage,
}

/// Why a Live session ended.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum CloseReason {
    CloseRequested,
    Expired,
    Content,
    RemoteHangup,
    ConnectionLost,
    /// A close reason added after this library version.
    #[serde(untagged)]
    Other(String),
}

/// Live API error payload.
#[derive(Debug, Clone, Deserialize)]
pub struct LiveError {
    #[serde(rename = "type")]
    pub type_name: String,
    pub code: String,
    pub message: String,
    pub param: Option<String>,
    pub client_event_id: Option<String>,
}

/// Error event sent by the Live service.
#[derive(Debug, Clone, Deserialize)]
pub struct ErrorEvent {
    pub event_id: Option<String>,
    pub client_event_id: Option<String>,
    pub error: LiveError,
}

/// Informational event sent by the Live service.
#[derive(Debug, Clone, Deserialize)]
pub struct InfoEvent {
    pub event_id: Option<String>,
    pub client_event_id: Option<String>,
    pub code: String,
    pub message: String,
}
