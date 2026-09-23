//! OpenAI Live API support over primary and fork WebSockets.
//!
//! GPT Live advances only while input audio is streaming in real time. Keep
//! sending audio frames continuously—even silence—while appending commentary
//! or instructions. For 24 kHz PCM16, 100 ms of silence is 4,800 zero bytes.
//!
//! ```rust,no_run
//! use std::time::Duration;
//! use openai_tools::live::LiveClient;
//!
//! # async fn run() -> Result<(), openai_tools::common::errors::OpenAIToolError> {
//! let session = LiveClient::new().connect().await?;
//! let (sender, mut receiver) = session.split();
//! let audio_sender = sender.clone();
//! let audio_task = tokio::spawn(async move {
//!     let silence = [0_u8; 4_800];
//!     loop {
//!         audio_sender.append_audio_bytes(&silence).await?;
//!         tokio::time::sleep(Duration::from_millis(100)).await;
//!     }
//!     #[allow(unreachable_code)]
//!     Ok::<(), openai_tools::common::errors::OpenAIToolError>(())
//! });
//!
//! sender.append_commentary("Hello from GPT Live.", None).await?;
//! let _event = receiver.recv().await?;
//! audio_task.abort();
//! receiver.close().await?;
//! # Ok(())
//! # }
//! ```
//!
//! «未対応»: WebRTC session creation, sideband, SIP, `transport.*` events, and recording downloads.
//! Azure authentication is also unsupported.

mod client;
pub mod events;
mod session;

pub use client::{LiveClient, LiveReceiver, LiveSender, LiveSession};
pub use events::*;
pub use session::*;

#[cfg(test)]
mod tests {
    use serde_json::{json, Value};

    use crate::common::models::LiveModel;

    use super::{
        ClientEvent, CloseReason, Delegation, DelegationTarget, LiveAudioEncoding, LiveAudioFormat, LiveSessionConfig, LiveVoice,
        ResponsesDelegationConfig, ServerEvent,
    };

    fn assert_json(event: ClientEvent, expected: Value) {
        assert_eq!(serde_json::to_value(event).unwrap(), expected);
    }

    #[test]
    fn every_client_event_has_the_documented_wire_shape() {
        let session = LiveSessionConfig::default();

        assert_json(ClientEvent::SessionStart { event_id: None, session: session.clone() }, json!({"type": "session.start", "session": {}}));
        assert_json(
            ClientEvent::SessionUpdate { event_id: None, session: super::LiveSessionUpdate { delegation: Delegation::Client } },
            json!({"type": "session.update", "session": {"delegation": {"type": "client"}}}),
        );
        assert_json(
            ClientEvent::InputAudioAppend { event_id: None, audio: "AA==".to_string() },
            json!({"type": "session.input_audio.append", "audio": "AA=="}),
        );
        assert_json(ClientEvent::InputAudioMute { event_id: None }, json!({"type": "session.input_audio.mute"}));
        assert_json(ClientEvent::InputAudioUnmute { event_id: None }, json!({"type": "session.input_audio.unmute"}));

        for (event, event_type) in [
            (ClientEvent::InstructionsAppend { event_id: None, content: "context".to_string(), delegation_id: None }, "session.instructions.append"),
            (ClientEvent::ThinkingAppend { event_id: None, content: "context".to_string(), delegation_id: None }, "session.thinking.append"),
            (ClientEvent::CommentaryAppend { event_id: None, content: "context".to_string(), delegation_id: None }, "session.commentary.append"),
        ] {
            assert_json(event, json!({"type": event_type, "content": "context", "delegation_id": null}));
        }

        assert_json(
            ClientEvent::ResponseItemCreate { event_id: None, item: json!({"type": "message"}) },
            json!({"type": "response.item.create", "item": {"type": "message"}}),
        );
        assert_json(ClientEvent::ResponseCreate { event_id: None }, json!({"type": "response.create"}));
        assert_json(ClientEvent::SessionClose { event_id: None }, json!({"type": "session.close"}));
    }

    #[test]
    fn client_event_serializes_present_event_id() {
        assert_json(ClientEvent::ResponseCreate { event_id: Some("evt_1".to_string()) }, json!({"type": "response.create", "event_id": "evt_1"}));
    }

    #[test]
    fn live_model_uses_gpt_live_1_and_preserves_custom_ids() {
        assert_eq!(LiveModel::default(), LiveModel::GptLive1);
        assert_eq!(LiveModel::GptLive1.as_str(), "gpt-live-1");
        assert_eq!(LiveModel::from("gpt-live-1"), LiveModel::GptLive1);
        assert_eq!(serde_json::to_string(&LiveModel::GptLive1).unwrap(), "\"gpt-live-1\"");

        let custom = LiveModel::from("gpt-live-next");
        assert_eq!(custom, LiveModel::Custom("gpt-live-next".to_string()));
        assert_eq!(serde_json::from_str::<LiveModel>("\"gpt-live-next\"").unwrap(), custom);
    }

    #[test]
    fn measured_session_started_deserializes() {
        let json = r#"{"event_id":"event_123","type":"session.started","session":{"id":"live_u7_123","expires_at":1790170679,"model":"gpt-live-1","instructions":"Hello","audio":{"output":{"voice":"marin"},"format":{"type":"audio/pcm","rate":24000}},"delegation":{"type":"client"},"status":"active","input":[]}}"#;
        let event: ServerEvent = serde_json::from_str(json).unwrap();

        match event {
            ServerEvent::SessionStarted(event) => {
                assert_eq!(event.client_event_id, None);
                assert_eq!(event.session.id, "live_u7_123");
                assert_eq!(event.session.status, "active");
                assert_eq!(event.session.audio.unwrap().output.unwrap().voice, Some(LiveVoice::Marin));
            }
            other => panic!("unexpected event: {other:?}"),
        }
    }

    #[test]
    fn measured_usage_and_error_events_deserialize() {
        let usage: ServerEvent = serde_json::from_str(
            r#"{"type":"session.usage.updated","usage":{"seconds":14.0},"context_window":{"usage_ratio":0.0094921875},"event_id":"event_123"}"#,
        )
        .unwrap();
        match usage {
            ServerEvent::SessionUsageUpdated(event) => {
                assert_eq!(event.usage.seconds, 14.0);
                assert_eq!(event.context_window.unwrap().usage_ratio, 0.0094921875);
            }
            other => panic!("unexpected event: {other:?}"),
        }

        let first_usage: ServerEvent =
            serde_json::from_str(r#"{"type":"session.usage.updated","usage":{"seconds":1.0},"event_id":"event_124"}"#).unwrap();
        assert!(matches!(first_usage, ServerEvent::SessionUsageUpdated(event) if event.context_window.is_none()));

        let error: ServerEvent = serde_json::from_str(
            r#"{"type":"error","event_id":"event_125","error":{"type":"server_error","code":"context_injection_incomplete","message":"incomplete","param":null,"client_event_id":"evt_c1"}}"#,
        )
        .unwrap();
        assert!(
            matches!(error, ServerEvent::Error(event) if event.error.param.is_none() && event.error.client_event_id.as_deref() == Some("evt_c1"))
        );
    }

    #[test]
    fn primary_output_audio_delta_accepts_only_type_and_delta() {
        let event: ServerEvent = serde_json::from_str(r#"{"type":"session.output_audio.delta","delta":"AA=="}"#).unwrap();
        assert!(
            matches!(event, ServerEvent::OutputAudioDelta(event) if event.event_id.is_none() && event.start_ms.is_none() && event.end_ms.is_none())
        );
    }

    #[test]
    fn unknown_server_event_type_is_tolerated() {
        let event: ServerEvent = serde_json::from_str(r#"{"type":"session.something.new","event_id":"evt_1"}"#).unwrap();
        assert!(matches!(event, ServerEvent::Unknown));
    }

    #[test]
    fn live_voice_built_in_and_custom_round_trip() {
        let built_ins = [
            (LiveVoice::Alloy, "alloy"),
            (LiveVoice::Ash, "ash"),
            (LiveVoice::Ballad, "ballad"),
            (LiveVoice::Beacon, "beacon"),
            (LiveVoice::Bossa, "bossa"),
            (LiveVoice::Cedar, "cedar"),
            (LiveVoice::Cinder, "cinder"),
            (LiveVoice::Coral, "coral"),
            (LiveVoice::Delta, "delta"),
            (LiveVoice::Echo, "echo"),
            (LiveVoice::Gleam, "gleam"),
            (LiveVoice::Marin, "marin"),
            (LiveVoice::Meridian, "meridian"),
            (LiveVoice::Quartz, "quartz"),
            (LiveVoice::Ripple, "ripple"),
            (LiveVoice::Sage, "sage"),
            (LiveVoice::Shimmer, "shimmer"),
            (LiveVoice::Stone, "stone"),
            (LiveVoice::Tempo, "tempo"),
            (LiveVoice::Verse, "verse"),
            (LiveVoice::Vesper, "vesper"),
            (LiveVoice::Willow, "willow"),
        ];
        for (voice, name) in built_ins {
            let json = serde_json::to_string(&voice).unwrap();
            assert_eq!(json, format!("\"{name}\""));
            assert_eq!(serde_json::from_str::<LiveVoice>(&json).unwrap(), voice);
        }

        let voice = LiveVoice::Custom { id: "voice_custom_1".to_string() };
        assert_eq!(serde_json::to_value(&voice).unwrap(), json!({"id": "voice_custom_1"}));
        assert_eq!(serde_json::from_value::<LiveVoice>(json!({"id": "voice_custom_1"})).unwrap(), voice);
    }

    #[test]
    fn live_audio_formats_round_trip() {
        for format in [LiveAudioFormat::pcm24k(), LiveAudioFormat::pcm16k(), LiveAudioFormat::pcmu(), LiveAudioFormat::pcma()] {
            let json = serde_json::to_value(&format).unwrap();
            assert_eq!(serde_json::from_value::<LiveAudioFormat>(json).unwrap(), format);
        }

        assert_eq!(serde_json::to_value(LiveAudioFormat::pcm24k()).unwrap(), json!({"type": "audio/pcm", "rate": 24000}));
        let other: LiveAudioEncoding = serde_json::from_str("\"audio/flac\"").unwrap();
        assert_eq!(other, LiveAudioEncoding::Other("audio/flac".to_string()));
    }

    #[test]
    fn close_reason_and_delegation_target_preserve_unknown_values() {
        let reason: CloseReason = serde_json::from_str("\"maintenance\"").unwrap();
        assert_eq!(reason, CloseReason::Other("maintenance".to_string()));
        assert_eq!(serde_json::to_string(&reason).unwrap(), "\"maintenance\"");

        let target: DelegationTarget = serde_json::from_str("\"agent\"").unwrap();
        assert_eq!(target, DelegationTarget::Other("agent".to_string()));
        assert_eq!(serde_json::to_string(&target).unwrap(), "\"agent\"");
    }

    #[test]
    fn responses_delegation_has_nested_responses_shape() {
        let delegation = Delegation::Responses(ResponsesDelegationConfig {
            model: "gpt-5.6-terra".to_string(),
            instructions: Some("Use the tools".to_string()),
            max_output_tokens: Some(1024),
            parallel_tool_calls: Some(true),
            reasoning: None,
            service_tier: Some(json!("priority")),
            text: Some(super::ResponsesTextConfig { verbosity: Some("low".to_string()) }),
            tool_choice: Some(json!("auto")),
            tools: Some(json!([{"type": "web_search"}])),
        });

        assert_eq!(
            serde_json::to_value(delegation).unwrap(),
            json!({
                "type": "responses",
                "responses": {
                    "model": "gpt-5.6-terra",
                    "instructions": "Use the tools",
                    "max_output_tokens": 1024,
                    "parallel_tool_calls": true,
                    "service_tier": "priority",
                    "text": {"verbosity": "low"},
                    "tool_choice": "auto",
                    "tools": [{"type": "web_search"}]
                }
            })
        );
    }
}
