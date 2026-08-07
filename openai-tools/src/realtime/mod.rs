//! # OpenAI Realtime API
//!
//! This module provides a type-safe interface for OpenAI's Realtime API, enabling
//! real-time audio and text communication with OpenAI models through WebSocket connections.
//!
//! ## Features
//!
//! - **WebSocket Connection**: Real-time bidirectional communication
//! - **Audio Streaming**: Send and receive audio in PCM16, G.711 formats
//! - **Text Messages**: Real-time text-based conversations
//! - **Voice Activity Detection (VAD)**: Server-side and semantic turn detection
//! - **Function Calling**: Integrate tools and functions into conversations
//! - **Transcription**: Automatic speech-to-text via Whisper
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use openai_tools::realtime::{RealtimeClient, Modality, Voice};
//! use openai_tools::realtime::vad::ServerVadConfig;
//! use openai_tools::realtime::events::server::ServerEvent;
//! use openai_tools::common::models::RealtimeModel;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut client = RealtimeClient::new();
//!     client
//!         .model(RealtimeModel::GptRealtime_2025_08_28)
//!         .modalities(vec![Modality::Text, Modality::Audio])
//!         .voice(Voice::Alloy)
//!         .instructions("You are a helpful voice assistant.");
//!
//!     let mut session = client.connect().await?;
//!
//!     // Send a text message
//!     session.send_text("Hello!").await?;
//!     session.create_response(None).await?;
//!
//!     // Process events
//!     while let Some(event) = session.recv().await? {
//!         match event {
//!             ServerEvent::ResponseTextDelta(e) => {
//!                 print!("{}", e.delta);
//!             }
//!             ServerEvent::ResponseDone(_) => break,
//!             _ => {}
//!         }
//!     }
//!
//!     session.close().await?;
//!     Ok(())
//! }
//! ```
//!
//! ## Audio Streaming Example
//!
//! ```rust,no_run
//! use openai_tools::realtime::{RealtimeClient, Modality, Voice, AudioFormat};
//! use openai_tools::realtime::vad::ServerVadConfig;
//! use openai_tools::realtime::events::server::ServerEvent;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut client = RealtimeClient::new();
//!     client
//!         .modalities(vec![Modality::Text, Modality::Audio])
//!         .voice(Voice::Alloy)
//!         .input_audio_format(AudioFormat::Pcm16)
//!         .output_audio_format(AudioFormat::Pcm16)
//!         .server_vad(ServerVadConfig {
//!             threshold: Some(0.5),
//!             silence_duration_ms: Some(500),
//!             ..Default::default()
//!         });
//!
//!     let mut session = client.connect().await?;
//!
//!     // Append audio data (PCM16, 24kHz, mono)
//!     let audio_bytes: Vec<u8> = vec![]; // Your audio data
//!     session.append_audio_bytes(&audio_bytes).await?;
//!
//!     // With server VAD enabled, audio is automatically committed
//!     // and responses are generated when speech ends
//!
//!     // Process response events
//!     while let Some(event) = session.recv().await? {
//!         match event {
//!             ServerEvent::ResponseAudioDelta(e) => {
//!                 // Play audio: base64::decode(&e.delta)
//!             }
//!             ServerEvent::ResponseAudioTranscriptDelta(e) => {
//!                 print!("{}", e.delta);
//!             }
//!             ServerEvent::ResponseDone(_) => break,
//!             _ => {}
//!         }
//!     }
//!
//!     session.close().await?;
//!     Ok(())
//! }
//! ```
//!
//! ## Function Calling Example
//!
//! ```rust,no_run
//! use openai_tools::realtime::{RealtimeClient, Modality};
//! use openai_tools::realtime::events::server::ServerEvent;
//! use openai_tools::common::{tool::Tool, parameters::ParameterProperty};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut client = RealtimeClient::new();
//!
//!     let weather_tool = Tool::function(
//!         "get_weather",
//!         "Get weather for a location",
//!         vec![("location", ParameterProperty::from_string("City name"))],
//!         false,
//!     );
//!
//!     client
//!         .modalities(vec![Modality::Text])
//!         .tools(vec![weather_tool]);
//!
//!     let mut session = client.connect().await?;
//!
//!     session.send_text("What's the weather in Tokyo?").await?;
//!     session.create_response(None).await?;
//!
//!     while let Some(event) = session.recv().await? {
//!         match event {
//!             ServerEvent::ResponseFunctionCallArgumentsDone(e) => {
//!                 // Execute function and submit result
//!                 let result = r#"{"temp": "22C", "condition": "sunny"}"#;
//!                 session.submit_function_output(&e.call_id, result).await?;
//!                 session.create_response(None).await?;
//!             }
//!             ServerEvent::ResponseDone(_) => break,
//!             _ => {}
//!         }
//!     }
//!
//!     session.close().await?;
//!     Ok(())
//! }
//! ```

mod audio;
mod client;
mod conversation;
pub mod events;
mod session;
mod stream;
pub mod vad;

// Re-export main types
pub use audio::{AudioFormat, InputAudioTranscription, TranscriptionModel, Voice};
pub use client::{RealtimeClient, RealtimeSession};
pub use conversation::{ContentPart, ConversationItem, FunctionCallItem, FunctionCallOutputItem, ItemStatus, MessageItem, MessageRole};
pub use events::client::ClientEvent;
pub use events::server::ServerEvent;
pub use session::{MaxTokens, Modality, NamedFunction, NamedToolChoice, RealtimeTool, SessionConfig, SimpleToolChoice, ToolChoice};
pub use stream::EventHandler;
pub use vad::{Eagerness, SemanticVadConfig, ServerVadConfig, TurnDetection};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{parameters::ParameterProperty, tool::Tool};

    #[test]
    fn test_session_config_serialization() {
        let config = SessionConfig {
            modalities: Some(vec![Modality::Text, Modality::Audio]),
            voice: Some(Voice::Alloy),
            temperature: Some(0.8),
            ..Default::default()
        };
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("\"modalities\""));
        assert!(json.contains("\"text\""));
        assert!(json.contains("\"audio\""));
        assert!(json.contains("\"voice\":\"alloy\""));
    }

    #[test]
    fn test_audio_format_serialization() {
        let format = AudioFormat::Pcm16;
        let json = serde_json::to_string(&format).unwrap();
        assert_eq!(json, "\"pcm16\"");

        let format = AudioFormat::G711Ulaw;
        let json = serde_json::to_string(&format).unwrap();
        assert_eq!(json, "\"g711_ulaw\"");
    }

    #[test]
    fn test_voice_serialization() {
        let voice = Voice::Alloy;
        let json = serde_json::to_string(&voice).unwrap();
        assert_eq!(json, "\"alloy\"");

        let voice = Voice::Shimmer;
        let json = serde_json::to_string(&voice).unwrap();
        assert_eq!(json, "\"shimmer\"");
    }

    #[test]
    fn test_vad_config_serialization() {
        let vad = TurnDetection::ServerVad(ServerVadConfig { threshold: Some(0.5), silence_duration_ms: Some(500), ..Default::default() });
        let json = serde_json::to_string(&vad).unwrap();
        assert!(json.contains("\"type\":\"server_vad\""));
        assert!(json.contains("\"threshold\":0.5"));
    }

    #[test]
    fn test_semantic_vad_config_serialization() {
        let vad =
            TurnDetection::SemanticVad(SemanticVadConfig { eagerness: Some(Eagerness::High), create_response: Some(true), ..Default::default() });
        let json = serde_json::to_string(&vad).unwrap();
        assert!(json.contains("\"type\":\"semantic_vad\""));
        assert!(json.contains("\"eagerness\":\"high\""));
    }

    #[test]
    fn test_tool_integration() {
        // Test with RealtimeTool directly
        let tool = RealtimeTool::function("get_weather", "Get weather for location", vec![("location", ParameterProperty::from_string("City name"))]);

        let config = SessionConfig { tools: Some(vec![tool]), ..Default::default() };

        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("get_weather"));
        // Verify flattened format (no "function" wrapper)
        assert!(json.contains("\"name\":\"get_weather\""));
        assert!(!json.contains("\"function\":{"));
    }

    #[test]
    fn test_tool_conversion_from_chat_tool() {
        // Test conversion from common Tool to RealtimeTool
        let chat_tool =
            Tool::function("get_weather", "Get weather for location", vec![("location", ParameterProperty::from_string("City name"))], false);

        let realtime_tool = RealtimeTool::from(chat_tool);
        assert_eq!(realtime_tool.name, "get_weather");
        assert_eq!(realtime_tool.type_name, "function");
        assert!(realtime_tool.description.is_some());

        let json = serde_json::to_string(&realtime_tool).unwrap();
        // Verify flattened format
        assert!(json.contains("\"name\":\"get_weather\""));
        assert!(!json.contains("\"function\":{"));
    }

    #[test]
    fn test_conversation_item_serialization() {
        let item = ConversationItem::Message(MessageItem {
            id: Some("msg_123".to_string()),
            role: MessageRole::User,
            content: vec![ContentPart::InputText { text: "Hello".to_string() }],
        });
        let json = serde_json::to_string(&item).unwrap();
        assert!(json.contains("\"type\":\"message\""));
        assert!(json.contains("\"role\":\"user\""));
    }

    #[test]
    fn test_modality_serialization() {
        let modalities = vec![Modality::Text, Modality::Audio];
        let json = serde_json::to_string(&modalities).unwrap();
        assert_eq!(json, "[\"text\",\"audio\"]");
    }

    #[test]
    fn test_max_tokens_serialization() {
        let count = MaxTokens::Count(1000);
        let json = serde_json::to_string(&count).unwrap();
        assert_eq!(json, "1000");

        let infinite = MaxTokens::Infinite;
        let json = serde_json::to_string(&infinite).unwrap();
        assert_eq!(json, "\"inf\"");
    }

    #[test]
    fn test_tool_choice_serialization() {
        // Simple choices
        let auto = ToolChoice::auto();
        let json = serde_json::to_string(&auto).unwrap();
        assert_eq!(json, "\"auto\"");

        let none = ToolChoice::none();
        let json = serde_json::to_string(&none).unwrap();
        assert_eq!(json, "\"none\"");

        let required = ToolChoice::required();
        let json = serde_json::to_string(&required).unwrap();
        assert_eq!(json, "\"required\"");

        // Named function choice
        let func = ToolChoice::function("get_weather");
        let json = serde_json::to_string(&func).unwrap();
        assert!(json.contains("\"type\":\"function\""));
        assert!(json.contains("\"name\":\"get_weather\""));
    }

    #[test]
    fn test_client_event_session_update_serialization() {
        use events::client::ClientEvent;

        let event = ClientEvent::SessionUpdate {
            event_id: Some("evt_123".to_string()),
            session: SessionConfig { modalities: Some(vec![Modality::Text]), ..Default::default() },
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"type\":\"session.update\""));
        assert!(json.contains("\"event_id\":\"evt_123\""));
    }

    #[test]
    fn test_client_event_input_audio_buffer_append() {
        use events::client::ClientEvent;

        let event = ClientEvent::InputAudioBufferAppend { event_id: None, audio: "SGVsbG8gV29ybGQ=".to_string() };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"type\":\"input_audio_buffer.append\""));
        assert!(json.contains("\"audio\":\"SGVsbG8gV29ybGQ=\""));
    }

    #[test]
    fn test_server_event_deserialization() {
        use events::server::ServerEvent;

        let json = r#"{"type":"error","event_id":"evt_123","error":{"type":"invalid_request_error","code":"invalid_value","message":"Invalid model","param":"model"}}"#;
        let event: ServerEvent = serde_json::from_str(json).unwrap();
        match event {
            ServerEvent::Error(e) => {
                assert_eq!(e.event_id, "evt_123");
                assert_eq!(e.error.code, Some("invalid_value".to_string()));
            }
            _ => panic!("Expected Error event"),
        }
    }

    #[test]
    fn test_response_text_delta_deserialization() {
        use events::server::ServerEvent;

        let json = r#"{"type":"response.text.delta","event_id":"evt_456","response_id":"resp_123","item_id":"item_123","output_index":0,"content_index":0,"delta":"Hello"}"#;
        let event: ServerEvent = serde_json::from_str(json).unwrap();
        match event {
            ServerEvent::ResponseTextDelta(e) => {
                assert_eq!(e.delta, "Hello");
                assert_eq!(e.response_id, "resp_123");
            }
            _ => panic!("Expected ResponseTextDelta event"),
        }
    }
}
