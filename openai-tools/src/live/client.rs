//! WebSocket client and active-session types for the OpenAI Live API.

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use base64::prelude::*;
use futures_util::{stream::SplitSink, stream::SplitStream, SinkExt, StreamExt};
use tokio::{net::TcpStream, sync::Mutex};
use tokio_tungstenite::{
    connect_async_with_config,
    tungstenite::{client::IntoClientRequest, Message as WsMessage},
    MaybeTlsStream, WebSocketStream,
};

use crate::common::{
    auth::{AuthProvider, OpenAIAuth},
    errors::{OpenAIToolError, Result},
    models::LiveModel,
};

use super::{
    events::{ClientEvent, ServerEvent, SessionClosedEvent, SessionResource},
    session::{
        Delegation, LiveAudioConfig, LiveAudioEncoding, LiveAudioFormat, LiveAudioOutputConfig, LiveSessionConfig, LiveSessionUpdate, LiveVoice,
    },
};

const LIVE_SESSIONS_URL: &str = "wss://api.openai.com/v1/live/sessions";

type LiveWebSocket = WebSocketStream<MaybeTlsStream<TcpStream>>;
type LiveSink = SplitSink<LiveWebSocket, WsMessage>;
type LiveStream = SplitStream<LiveWebSocket>;

/// Builder for a primary or forked Live WebSocket session.
#[derive(Debug, Clone)]
pub struct LiveClient {
    auth: AuthProvider,
    session_config: LiveSessionConfig,
}

impl LiveClient {
    /// Creates a client using `OPENAI_API_KEY`.
    pub fn new() -> Self {
        let auth = AuthProvider::openai_from_env().expect("OPENAI_API_KEY must be set");
        Self::with_auth(auth)
    }

    /// Creates a client using an explicit OpenAI API key.
    pub fn with_api_key(api_key: impl Into<String>) -> Self {
        Self::with_auth(AuthProvider::OpenAI(OpenAIAuth::new(api_key)))
    }

    /// Creates a client with an authentication provider.
    ///
    /// The Live API currently supports only [`AuthProvider::OpenAI`]. Passing
    /// Azure authentication is allowed here so that configuration can be
    /// shared, but [`connect`](Self::connect) and [`fork`](Self::fork) reject it.
    pub fn with_auth(auth: AuthProvider) -> Self {
        Self { auth, session_config: LiveSessionConfig::default() }
    }

    /// Selects the frontend Live model.
    pub fn model(&mut self, model: LiveModel) -> &mut Self {
        self.session_config.model = Some(model);
        self
    }

    /// Sets immutable frontend instructions.
    pub fn instructions(&mut self, instructions: impl Into<String>) -> &mut Self {
        self.session_config.instructions = Some(instructions.into());
        self
    }

    /// Selects the output voice.
    pub fn voice(&mut self, voice: LiveVoice) -> &mut Self {
        self.audio_mut().output.get_or_insert_with(LiveAudioOutputConfig::default).voice = Some(voice);
        self
    }

    /// Selects the common input/output audio format.
    pub fn audio_format(&mut self, format: LiveAudioFormat) -> &mut Self {
        self.audio_mut().format = Some(format);
        self
    }

    /// Selects the delegation strategy.
    pub fn delegation(&mut self, delegation: Delegation) -> &mut Self {
        self.session_config.delegation = Some(delegation);
        self
    }

    /// Enables or disables storage for later forks.
    pub fn store(&mut self, store: bool) -> &mut Self {
        self.session_config.store = Some(store);
        self
    }

    /// Supplies initial Responses-style input items.
    pub fn input(&mut self, input: Vec<serde_json::Value>) -> &mut Self {
        self.session_config.input = Some(input);
        self
    }

    /// Starts a new primary Live WebSocket session.
    pub async fn connect(&self) -> Result<LiveSession> {
        let mut config = self.session_config.clone();
        config.model.get_or_insert_default();
        self.connect_to(LIVE_SESSIONS_URL.to_string(), config).await
    }

    /// Forks a stored Live session, applying only explicitly configured fields.
    pub async fn fork(&self, session_id: &str) -> Result<LiveSession> {
        let url = format!("{LIVE_SESSIONS_URL}/{session_id}/fork");
        self.connect_to(url, self.session_config.clone()).await
    }

    fn audio_mut(&mut self) -> &mut LiveAudioConfig {
        self.session_config.audio.get_or_insert_with(LiveAudioConfig::default)
    }

    async fn connect_to(&self, url: String, config: LiveSessionConfig) -> Result<LiveSession> {
        let api_key = match &self.auth {
            AuthProvider::OpenAI(auth) => auth.api_key(),
            AuthProvider::Azure(_) => {
                return Err(OpenAIToolError::Error("Azure authentication is not supported by the OpenAI Live API".to_string()));
            }
        };

        let mut request =
            url.into_client_request().map_err(|error| OpenAIToolError::Error(format!("Failed to build Live WebSocket request: {error}")))?;
        request.headers_mut().insert(
            "Authorization",
            format!("Bearer {api_key}").parse().map_err(|error| OpenAIToolError::Error(format!("Invalid Authorization header: {error}")))?,
        );

        let (socket, _) = connect_async_with_config(request, None, false)
            .await
            .map_err(|error| OpenAIToolError::WebSocketError(format!("Live connection failed: {error}")))?;
        let (sink, stream) = socket.split();
        let sender = LiveSender::new(sink);
        let mut receiver = LiveReceiver::new(sender.clone(), stream);

        sender.send(ClientEvent::SessionStart { event_id: None, session: config }).await?;

        let started = loop {
            match receiver.recv().await? {
                Some(ServerEvent::SessionStarted(event)) => break event.session,
                Some(ServerEvent::Error(event)) => {
                    return Err(OpenAIToolError::Error(format!("Live session start failed: {} - {}", event.error.code, event.error.message)));
                }
                Some(_) => continue,
                None => {
                    return Err(OpenAIToolError::Error("Live WebSocket closed before session.started".to_string()));
                }
            }
        };

        let pcm16 =
            started.audio.as_ref().and_then(|audio| audio.format.as_ref()).is_none_or(|format| matches!(format.encoding, LiveAudioEncoding::Pcm));
        sender.pcm16.store(pcm16, Ordering::Relaxed);

        Ok(LiveSession { started, sender, receiver })
    }
}

impl Default for LiveClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Sending half of a split Live session.
#[derive(Clone)]
pub struct LiveSender {
    sink: Arc<Mutex<LiveSink>>,
    pcm16: Arc<AtomicBool>,
}

impl LiveSender {
    fn new(sink: LiveSink) -> Self {
        Self { sink: Arc::new(Mutex::new(sink)), pcm16: Arc::new(AtomicBool::new(true)) }
    }

    /// Sends a raw typed client event.
    pub async fn send(&self, event: ClientEvent) -> Result<()> {
        let json = serde_json::to_string(&event)?;
        self.send_message(WsMessage::Text(json.into())).await
    }

    /// Appends base64-encoded audio input.
    pub async fn append_audio(&self, base64: &str) -> Result<()> {
        self.send(ClientEvent::InputAudioAppend { event_id: None, audio: base64.to_string() }).await
    }

    /// Base64-encodes and appends raw audio input.
    ///
    /// PCM input must contain complete signed 16-bit samples, so odd byte
    /// lengths are rejected before anything is sent.
    pub async fn append_audio_bytes(&self, audio: &[u8]) -> Result<()> {
        if self.pcm16.load(Ordering::Relaxed) && !audio.len().is_multiple_of(2) {
            return Err(OpenAIToolError::Error("PCM audio must contain an even number of bytes (16-bit sample boundary)".to_string()));
        }
        self.append_audio(&BASE64_STANDARD.encode(audio)).await
    }

    /// Mutes Live input audio.
    pub async fn mute(&self) -> Result<()> {
        self.send(ClientEvent::InputAudioMute { event_id: None }).await
    }

    /// Unmutes Live input audio.
    pub async fn unmute(&self) -> Result<()> {
        self.send(ClientEvent::InputAudioUnmute { event_id: None }).await
    }

    /// Appends system-level instructions.
    pub async fn append_instructions(&self, content: impl Into<String>, delegation_id: Option<&str>) -> Result<()> {
        self.send(ClientEvent::InstructionsAppend { event_id: None, content: content.into(), delegation_id: delegation_id.map(str::to_string) }).await
    }

    /// Appends silent context for model reasoning.
    pub async fn append_thinking(&self, content: impl Into<String>, delegation_id: Option<&str>) -> Result<()> {
        self.send(ClientEvent::ThinkingAppend { event_id: None, content: content.into(), delegation_id: delegation_id.map(str::to_string) }).await
    }

    /// Appends content that the model should speak.
    pub async fn append_commentary(&self, content: impl Into<String>, delegation_id: Option<&str>) -> Result<()> {
        self.send(ClientEvent::CommentaryAppend { event_id: None, content: content.into(), delegation_id: delegation_id.map(str::to_string) }).await
    }

    /// Adds an input item to a Responses delegation.
    pub async fn create_response_item(&self, item: serde_json::Value) -> Result<()> {
        self.send(ClientEvent::ResponseItemCreate { event_id: None, item }).await
    }

    /// Submits a function-call result to a Responses delegation.
    pub async fn submit_function_output(&self, call_id: impl Into<String>, output: impl Into<String>) -> Result<()> {
        self.create_response_item(serde_json::json!({
            "type": "function_call_output",
            "call_id": call_id.into(),
            "output": output.into(),
        }))
        .await
    }

    /// Requests that a Responses delegation continue generating.
    pub async fn create_response(&self) -> Result<()> {
        self.send(ClientEvent::ResponseCreate { event_id: None }).await
    }

    /// Updates settings within the current delegation type.
    pub async fn update_delegation(&self, delegation: Delegation) -> Result<()> {
        self.send(ClientEvent::SessionUpdate { event_id: None, session: LiveSessionUpdate { delegation } }).await
    }

    async fn send_message(&self, message: WsMessage) -> Result<()> {
        self.sink.lock().await.send(message).await.map_err(|error| OpenAIToolError::WebSocketError(format!("Failed to send Live event: {error}")))
    }

    async fn close_transport(&self) -> Result<()> {
        self.sink.lock().await.close().await.map_err(|error| OpenAIToolError::WebSocketError(format!("Failed to close Live WebSocket: {error}")))
    }
}

/// Receiving half of a split Live session.
pub struct LiveReceiver {
    sender: LiveSender,
    stream: LiveStream,
}

impl LiveReceiver {
    fn new(sender: LiveSender, stream: LiveStream) -> Self {
        Self { sender, stream }
    }

    /// Receives the next typed server event.
    ///
    /// Returns `None` after a WebSocket Close frame or end of stream. Ping
    /// frames are answered with Pong and are not surfaced to the caller.
    pub async fn recv(&mut self) -> Result<Option<ServerEvent>> {
        loop {
            match self.stream.next().await {
                Some(Ok(WsMessage::Text(text))) => return Ok(Some(serde_json::from_str(&text)?)),
                Some(Ok(WsMessage::Close(_))) | None => return Ok(None),
                Some(Ok(WsMessage::Ping(payload))) => {
                    self.sender.send_message(WsMessage::Pong(payload)).await?;
                }
                Some(Ok(_)) => {}
                Some(Err(error)) => {
                    return Err(OpenAIToolError::WebSocketError(format!("Live WebSocket receive failed: {error}")));
                }
            }
        }
    }

    /// Gracefully closes the Live session and then the WebSocket.
    pub async fn close(mut self) -> Result<Option<SessionClosedEvent>> {
        if let Err(error) = self.sender.send(ClientEvent::SessionClose { event_id: None }).await {
            let _ = self.sender.close_transport().await;
            return Err(error);
        }

        let closed = loop {
            match self.recv().await {
                Ok(Some(ServerEvent::SessionClosed(event))) => break Ok(Some(event)),
                Ok(Some(_)) => continue,
                Ok(None) => break Ok(None),
                Err(error) => break Err(error),
            }
        };

        let transport_close = self.sender.close_transport().await;
        match closed {
            Ok(event) => {
                transport_close?;
                Ok(event)
            }
            Err(error) => {
                let _ = transport_close;
                Err(error)
            }
        }
    }
}

/// An active primary or forked Live WebSocket session.
///
/// The model advances only while input audio is streaming in real time. Keep
/// sending audio frames (silence is acceptable) even when injecting commentary
/// or instructions; otherwise the injection can remain incomplete until close.
pub struct LiveSession {
    started: SessionResource,
    sender: LiveSender,
    receiver: LiveReceiver,
}

impl LiveSession {
    /// Returns the opaque Live session identifier.
    pub fn session_id(&self) -> &str {
        &self.started.id
    }

    /// Returns the full resolved session from `session.started`.
    pub fn started(&self) -> &SessionResource {
        &self.started
    }

    /// Splits the session for concurrent sending and receiving.
    pub fn split(self) -> (LiveSender, LiveReceiver) {
        (self.sender, self.receiver)
    }

    /// Sends a raw typed client event.
    pub async fn send(&self, event: ClientEvent) -> Result<()> {
        self.sender.send(event).await
    }

    /// Receives the next typed server event.
    pub async fn recv(&mut self) -> Result<Option<ServerEvent>> {
        self.receiver.recv().await
    }

    /// Appends base64-encoded audio input.
    pub async fn append_audio(&self, base64: &str) -> Result<()> {
        self.sender.append_audio(base64).await
    }

    /// Base64-encodes and appends raw audio input.
    pub async fn append_audio_bytes(&self, audio: &[u8]) -> Result<()> {
        self.sender.append_audio_bytes(audio).await
    }

    /// Mutes input audio.
    pub async fn mute(&self) -> Result<()> {
        self.sender.mute().await
    }

    /// Unmutes input audio.
    pub async fn unmute(&self) -> Result<()> {
        self.sender.unmute().await
    }

    /// Appends system-level instructions.
    pub async fn append_instructions(&self, content: impl Into<String>, delegation_id: Option<&str>) -> Result<()> {
        self.sender.append_instructions(content, delegation_id).await
    }

    /// Appends silent reasoning context.
    pub async fn append_thinking(&self, content: impl Into<String>, delegation_id: Option<&str>) -> Result<()> {
        self.sender.append_thinking(content, delegation_id).await
    }

    /// Appends content that the model should speak.
    pub async fn append_commentary(&self, content: impl Into<String>, delegation_id: Option<&str>) -> Result<()> {
        self.sender.append_commentary(content, delegation_id).await
    }

    /// Adds an input item to a Responses delegation.
    pub async fn create_response_item(&self, item: serde_json::Value) -> Result<()> {
        self.sender.create_response_item(item).await
    }

    /// Submits a function-call result to a Responses delegation.
    pub async fn submit_function_output(&self, call_id: impl Into<String>, output: impl Into<String>) -> Result<()> {
        self.sender.submit_function_output(call_id, output).await
    }

    /// Requests that a Responses delegation continue generating.
    pub async fn create_response(&self) -> Result<()> {
        self.sender.create_response().await
    }

    /// Updates settings within the current delegation type.
    pub async fn update_delegation(&self, delegation: Delegation) -> Result<()> {
        self.sender.update_delegation(delegation).await
    }

    /// Sends `session.close`, waits for `session.closed`, and closes the socket.
    pub async fn close(self) -> Result<Option<SessionClosedEvent>> {
        self.receiver.close().await
    }
}
