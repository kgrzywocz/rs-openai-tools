//! Realtime API client implementation.

use base64::prelude::*;
use futures_util::{SinkExt, StreamExt};
use tokio::net::TcpStream;
use tokio_tungstenite::{
    connect_async_with_config,
    tungstenite::{client::IntoClientRequest, Message as WsMessage},
    MaybeTlsStream, WebSocketStream,
};

use crate::common::auth::AuthProvider;
use crate::common::errors::{OpenAIToolError, Result};
use crate::common::models::RealtimeModel;
use crate::common::tool::Tool;

use super::audio::{AudioFormat, InputAudioTranscription, TranscriptionModel, Voice};
use super::conversation::{ConversationItem, FunctionCallOutputItem, MessageItem};
use super::events::client::ClientEvent;
use super::events::server::ServerEvent;
use super::session::{Modality, RealtimeTool, ResponseCreateConfig, SessionConfig};
use super::vad::{SemanticVadConfig, ServerVadConfig, TurnDetection};

/// The Realtime API WebSocket endpoint path.
const REALTIME_PATH: &str = "realtime";

/// Builder for creating Realtime API connections.
///
/// # Example
///
/// ```rust,no_run
/// use openai_tools::realtime::{RealtimeClient, Modality, Voice};
/// use openai_tools::common::models::RealtimeModel;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let mut client = RealtimeClient::new();
///     client
///         .model(RealtimeModel::GptRealtime_2025_08_28)
///         .modalities(vec![Modality::Text, Modality::Audio])
///         .voice(Voice::Alloy)
///         .instructions("You are a helpful assistant.");
///
///     let mut session = client.connect().await?;
///     // Use session...
///     session.close().await?;
///     Ok(())
/// }
/// ```
#[derive(Debug, Clone)]
pub struct RealtimeClient {
    /// Authentication provider (OpenAI or Azure)
    auth: AuthProvider,
    model: RealtimeModel,
    session_config: SessionConfig,
}

impl RealtimeClient {
    /// Create a new RealtimeClient for OpenAI API.
    ///
    /// Loads the API key from the `OPENAI_API_KEY` environment variable.
    pub fn new() -> Self {
        let auth = AuthProvider::openai_from_env().expect("OPENAI_API_KEY must be set");
        Self { auth, model: RealtimeModel::default(), session_config: SessionConfig::default() }
    }

    /// Create a new RealtimeClient with a custom authentication provider.
    pub fn with_auth(auth: AuthProvider) -> Self {
        Self { auth, model: RealtimeModel::default(), session_config: SessionConfig::default() }
    }

    /// Create a new RealtimeClient for Azure OpenAI API.
    pub fn azure() -> Result<Self> {
        let auth = AuthProvider::azure_from_env()?;
        Ok(Self { auth, model: RealtimeModel::default(), session_config: SessionConfig::default() })
    }

    /// Create a new RealtimeClient by auto-detecting the provider.
    pub fn detect_provider() -> Result<Self> {
        let auth = AuthProvider::from_env()?;
        Ok(Self { auth, model: RealtimeModel::default(), session_config: SessionConfig::default() })
    }

    /// Creates a new RealtimeClient with URL-based provider detection.
    pub fn with_url<S: Into<String>>(base_url: S, api_key: S) -> Self {
        let auth = AuthProvider::from_url_with_key(base_url, api_key);
        Self { auth, model: RealtimeModel::default(), session_config: SessionConfig::default() }
    }

    /// Creates a new RealtimeClient from URL using environment variables.
    pub fn from_url<S: Into<String>>(url: S) -> Result<Self> {
        let auth = AuthProvider::from_url(url)?;
        Ok(Self { auth, model: RealtimeModel::default(), session_config: SessionConfig::default() })
    }

    /// Create a new RealtimeClient with an explicit API key.
    #[deprecated(since = "0.3.0", note = "Use `with_auth(AuthProvider::OpenAI(...))` instead")]
    pub fn with_api_key(api_key: impl Into<String>) -> Self {
        let auth = AuthProvider::OpenAI(crate::common::auth::OpenAIAuth::new(api_key));
        Self { auth, model: RealtimeModel::default(), session_config: SessionConfig::default() }
    }

    /// Returns the authentication provider.
    pub fn auth(&self) -> &AuthProvider {
        &self.auth
    }

    /// Set the model for the Realtime API.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::realtime::RealtimeClient;
    /// use openai_tools::common::models::RealtimeModel;
    ///
    /// let mut client = RealtimeClient::new();
    /// client.model(RealtimeModel::GptRealtime_2025_08_28);
    /// ```
    pub fn model(&mut self, model: RealtimeModel) -> &mut Self {
        self.model = model;
        self
    }

    /// Set the model using a string ID (for backward compatibility).
    ///
    /// Prefer using [`model`](Self::model) with `RealtimeModel` enum for type safety.
    #[deprecated(since = "0.2.0", note = "Use `model(RealtimeModel)` instead for type safety")]
    pub fn model_id(&mut self, model_id: impl Into<String>) -> &mut Self {
        self.model = RealtimeModel::from(model_id.into().as_str());
        self
    }

    /// Set the supported modalities.
    pub fn modalities(&mut self, modalities: Vec<Modality>) -> &mut Self {
        self.session_config.modalities = Some(modalities);
        self
    }

    /// Set the system instructions.
    pub fn instructions(&mut self, instructions: impl Into<String>) -> &mut Self {
        self.session_config.instructions = Some(instructions.into());
        self
    }

    /// Set the voice for audio output.
    pub fn voice(&mut self, voice: Voice) -> &mut Self {
        self.session_config.voice = Some(voice);
        self
    }

    /// Set the input audio format.
    pub fn input_audio_format(&mut self, format: AudioFormat) -> &mut Self {
        self.session_config.input_audio_format = Some(format);
        self
    }

    /// Set the output audio format.
    pub fn output_audio_format(&mut self, format: AudioFormat) -> &mut Self {
        self.session_config.output_audio_format = Some(format);
        self
    }

    /// Enable input audio transcription.
    pub fn enable_transcription(&mut self, model: TranscriptionModel) -> &mut Self {
        self.session_config.input_audio_transcription = Some(InputAudioTranscription::new(model));
        self
    }

    /// Set input audio transcription configuration.
    pub fn transcription(&mut self, config: InputAudioTranscription) -> &mut Self {
        self.session_config.input_audio_transcription = Some(config);
        self
    }

    /// Set Server VAD turn detection.
    pub fn server_vad(&mut self, config: ServerVadConfig) -> &mut Self {
        self.session_config.turn_detection = Some(TurnDetection::ServerVad(config));
        self
    }

    /// Set Semantic VAD turn detection.
    pub fn semantic_vad(&mut self, config: SemanticVadConfig) -> &mut Self {
        self.session_config.turn_detection = Some(TurnDetection::SemanticVad(config));
        self
    }

    /// Disable turn detection (manual mode).
    pub fn disable_turn_detection(&mut self) -> &mut Self {
        self.session_config.turn_detection = None;
        self
    }

    /// Set available tools for function calling.
    ///
    /// Accepts `Tool` from the common module and converts to `RealtimeTool`.
    pub fn tools(&mut self, tools: Vec<Tool>) -> &mut Self {
        self.session_config.tools = Some(tools.into_iter().map(RealtimeTool::from).collect());
        self
    }

    /// Set available realtime tools directly.
    pub fn realtime_tools(&mut self, tools: Vec<RealtimeTool>) -> &mut Self {
        self.session_config.tools = Some(tools);
        self
    }

    /// Set the sampling temperature.
    pub fn temperature(&mut self, temp: f32) -> &mut Self {
        self.session_config.temperature = Some(temp);
        self
    }

    /// Connect to the Realtime API.
    ///
    /// Returns a `RealtimeSession` for sending and receiving events.
    pub async fn connect(&self) -> Result<RealtimeSession> {
        // Get the WebSocket URL based on auth provider
        let url = self.ws_endpoint();

        // Build WebSocket request with headers
        let mut request = url.into_client_request().map_err(|e| OpenAIToolError::Error(format!("Failed to build request: {}", e)))?;

        let headers = request.headers_mut();

        // Apply auth headers based on provider
        match &self.auth {
            AuthProvider::OpenAI(auth) => {
                headers.insert(
                    "Authorization",
                    format!("Bearer {}", auth.api_key()).parse().map_err(|e| OpenAIToolError::Error(format!("Invalid header value: {}", e)))?,
                );
            }
            AuthProvider::Azure(auth) => {
                headers.insert("api-key", auth.api_key().parse().map_err(|e| OpenAIToolError::Error(format!("Invalid header value: {}", e)))?);
            }
        }
        headers.insert("OpenAI-Beta", "realtime=v1".parse().map_err(|e| OpenAIToolError::Error(format!("Invalid header value: {}", e)))?);

        let (ws_stream, _response) = connect_async_with_config(request, None, false)
            .await
            .map_err(|e| OpenAIToolError::Error(format!("WebSocket connection failed: {}", e)))?;

        let mut session = RealtimeSession::new(ws_stream);

        // Wait for session.created event
        session.wait_for_session_created().await?;

        // Send initial session.update if we have configuration
        if self.session_config.modalities.is_some()
            || self.session_config.instructions.is_some()
            || self.session_config.voice.is_some()
            || self.session_config.tools.is_some()
            || self.session_config.turn_detection.is_some()
        {
            session.update_session(self.session_config.clone()).await?;
        }

        Ok(session)
    }

    /// Get the WebSocket endpoint URL based on auth provider.
    fn ws_endpoint(&self) -> String {
        match &self.auth {
            AuthProvider::OpenAI(_) => {
                format!("wss://api.openai.com/v1/{}?model={}", REALTIME_PATH, self.model.as_str())
            }
            AuthProvider::Azure(auth) => {
                // Azure WebSocket endpoint: convert https to wss and use the base_url directly
                // User should provide a WebSocket-compatible URL like:
                // "wss://my-resource.openai.azure.com/openai/realtime?api-version=2024-10-01-preview&deployment=my-deployment"
                // or if they provide https, we convert it to wss
                let base = auth.base_url();
                if base.starts_with("https://") {
                    base.replacen("https://", "wss://", 1)
                } else if base.starts_with("http://") {
                    base.replacen("http://", "ws://", 1)
                } else {
                    base.to_string()
                }
            }
        }
    }
}

impl Default for RealtimeClient {
    fn default() -> Self {
        Self::new()
    }
}

/// An active Realtime API session.
///
/// Provides methods for sending events and receiving responses.
pub struct RealtimeSession {
    ws_stream: WebSocketStream<MaybeTlsStream<TcpStream>>,
}

impl RealtimeSession {
    /// Create a new session from a WebSocket stream.
    fn new(ws_stream: WebSocketStream<MaybeTlsStream<TcpStream>>) -> Self {
        Self { ws_stream }
    }

    /// Send a client event to the server.
    pub async fn send(&mut self, event: ClientEvent) -> Result<()> {
        let json = serde_json::to_string(&event)?;
        self.ws_stream.send(WsMessage::Text(json.into())).await.map_err(|e| OpenAIToolError::Error(format!("Failed to send event: {}", e)))?;
        Ok(())
    }

    /// Receive the next server event.
    ///
    /// Returns `None` if the connection is closed.
    pub async fn recv(&mut self) -> Result<Option<ServerEvent>> {
        loop {
            match self.ws_stream.next().await {
                Some(Ok(WsMessage::Text(text))) => {
                    let event: ServerEvent = serde_json::from_str(&text)?;
                    return Ok(Some(event));
                }
                Some(Ok(WsMessage::Close(_))) => {
                    return Ok(None);
                }
                Some(Ok(WsMessage::Ping(data))) => {
                    self.ws_stream.send(WsMessage::Pong(data)).await.map_err(|e| OpenAIToolError::Error(format!("Failed to send pong: {}", e)))?;
                    continue;
                }
                Some(Ok(_)) => continue, // Ignore other message types
                Some(Err(e)) => {
                    return Err(OpenAIToolError::Error(format!("WebSocket error: {}", e)));
                }
                None => {
                    return Ok(None);
                }
            }
        }
    }

    /// Wait for the session.created event.
    async fn wait_for_session_created(&mut self) -> Result<()> {
        match self.recv().await? {
            Some(ServerEvent::SessionCreated(_)) => Ok(()),
            Some(ServerEvent::Error(e)) => Err(OpenAIToolError::Error(format!("Session creation failed: {}", e.error.message))),
            Some(event) => {
                Err(OpenAIToolError::Error(format!("Unexpected event while waiting for session.created: {:?}", std::mem::discriminant(&event))))
            }
            None => Err(OpenAIToolError::Error("Connection closed before session.created".to_string())),
        }
    }

    /// Update the session configuration.
    pub async fn update_session(&mut self, config: SessionConfig) -> Result<()> {
        self.send(ClientEvent::SessionUpdate { event_id: None, session: config }).await
    }

    /// Append base64-encoded audio to the input buffer.
    pub async fn append_audio(&mut self, audio_base64: &str) -> Result<()> {
        self.send(ClientEvent::InputAudioBufferAppend { event_id: None, audio: audio_base64.to_string() }).await
    }

    /// Append raw audio bytes to the input buffer.
    ///
    /// The bytes will be base64 encoded automatically.
    pub async fn append_audio_bytes(&mut self, audio: &[u8]) -> Result<()> {
        let encoded = BASE64_STANDARD.encode(audio);
        self.append_audio(&encoded).await
    }

    /// Commit the input audio buffer.
    ///
    /// Creates a user message item from the buffered audio.
    pub async fn commit_audio(&mut self) -> Result<()> {
        self.send(ClientEvent::InputAudioBufferCommit { event_id: None }).await
    }

    /// Clear the input audio buffer.
    pub async fn clear_audio(&mut self) -> Result<()> {
        self.send(ClientEvent::InputAudioBufferClear { event_id: None }).await
    }

    /// Create a conversation item.
    pub async fn create_item(&mut self, item: ConversationItem) -> Result<()> {
        self.send(ClientEvent::ConversationItemCreate { event_id: None, previous_item_id: None, item }).await
    }

    /// Send a text message.
    pub async fn send_text(&mut self, text: &str) -> Result<()> {
        let item = ConversationItem::Message(MessageItem::user_text(text));
        self.create_item(item).await
    }

    /// Create a response from the model.
    pub async fn create_response(&mut self, config: Option<ResponseCreateConfig>) -> Result<()> {
        self.send(ClientEvent::ResponseCreate { event_id: None, response: config }).await
    }

    /// Cancel the current response.
    pub async fn cancel_response(&mut self) -> Result<()> {
        self.send(ClientEvent::ResponseCancel { event_id: None }).await
    }

    /// Submit the output of a function call.
    pub async fn submit_function_output(&mut self, call_id: &str, output: &str) -> Result<()> {
        let item = ConversationItem::FunctionCallOutput(FunctionCallOutputItem::new(call_id, output));
        self.create_item(item).await
    }

    /// Delete a conversation item.
    pub async fn delete_item(&mut self, item_id: &str) -> Result<()> {
        self.send(ClientEvent::ConversationItemDelete { event_id: None, item_id: item_id.to_string() }).await
    }

    /// Truncate a conversation item's audio.
    pub async fn truncate_item(&mut self, item_id: &str, content_index: u32, audio_end_ms: u32) -> Result<()> {
        self.send(ClientEvent::ConversationItemTruncate { event_id: None, item_id: item_id.to_string(), content_index, audio_end_ms }).await
    }

    /// Close the session.
    pub async fn close(&mut self) -> Result<()> {
        self.ws_stream.close(None).await.map_err(|e| OpenAIToolError::Error(format!("Failed to close session: {}", e)))
    }
}
