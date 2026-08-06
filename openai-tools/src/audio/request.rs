//! OpenAI Audio API Request Module
//!
//! This module provides the functionality to interact with the OpenAI Audio API.
//! It supports text-to-speech (TTS), transcription, and translation.
//!
//! # Key Features
//!
//! - **Text-to-Speech**: Convert text to natural-sounding audio
//! - **Transcription**: Convert audio to text (speech-to-text)
//! - **Translation**: Translate audio to English text
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use openai_tools::audio::request::{Audio, TtsOptions, Voice};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let audio = Audio::new()?;
//!
//!     // Generate speech from text
//!     let options = TtsOptions::default();
//!     let audio_bytes = audio.text_to_speech("Hello, world!", options).await?;
//!     std::fs::write("output.mp3", audio_bytes)?;
//!
//!     Ok(())
//! }
//! ```

use crate::audio::response::TranscriptionResponse;
use crate::common::auth::AuthProvider;
use crate::common::client::create_http_client;
use crate::common::errors::{ErrorResponse, OpenAIToolError, Result};
use request::multipart::{Form, Part};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Duration;

/// Default API path for Audio
const AUDIO_PATH: &str = "audio";

/// Text-to-speech models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TtsModel {
    /// Standard quality TTS model
    #[serde(rename = "tts-1")]
    #[default]
    Tts1,
    /// High definition TTS model
    #[serde(rename = "tts-1-hd")]
    Tts1Hd,
    /// GPT-4o Mini TTS model
    #[serde(rename = "gpt-4o-mini-tts")]
    Gpt4oMiniTts,
    /// tts-1-1106 - dated snapshot of the standard TTS model
    #[serde(rename = "tts-1-1106")]
    Tts1_1106,
    /// tts-1-hd-1106 - dated snapshot of the HD TTS model
    #[serde(rename = "tts-1-hd-1106")]
    Tts1Hd1106,
}

impl TtsModel {
    /// Returns the model identifier string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Tts1 => "tts-1",
            Self::Tts1Hd => "tts-1-hd",
            Self::Gpt4oMiniTts => "gpt-4o-mini-tts",
            Self::Tts1_1106 => "tts-1-1106",
            Self::Tts1Hd1106 => "tts-1-hd-1106",
        }
    }

    /// Checks if this model supports the `instructions` parameter.
    ///
    /// Only `gpt-4o-mini-tts` supports the instructions parameter for
    /// controlling voice characteristics like tone, emotion, and pacing.
    ///
    /// # Example
    ///
    /// ```rust
    /// use openai_tools::audio::request::TtsModel;
    ///
    /// assert!(TtsModel::Gpt4oMiniTts.supports_instructions());
    /// assert!(!TtsModel::Tts1.supports_instructions());
    /// assert!(!TtsModel::Tts1Hd.supports_instructions());
    /// ```
    pub fn supports_instructions(&self) -> bool {
        matches!(self, Self::Gpt4oMiniTts)
    }
}

impl std::fmt::Display for TtsModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Voice options for text-to-speech.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Voice {
    /// Alloy voice
    #[default]
    Alloy,
    /// Ash voice
    Ash,
    /// Ballad voice
    Ballad,
    /// Cedar voice (recommended for quality)
    Cedar,
    /// Coral voice
    Coral,
    /// Echo voice
    Echo,
    /// Fable voice
    Fable,
    /// Marin voice (recommended for quality)
    Marin,
    /// Nova voice
    Nova,
    /// Onyx voice
    Onyx,
    /// Sage voice
    Sage,
    /// Shimmer voice
    Shimmer,
    /// Verse voice
    Verse,
}

impl Voice {
    /// Returns the voice identifier string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Alloy => "alloy",
            Self::Ash => "ash",
            Self::Ballad => "ballad",
            Self::Cedar => "cedar",
            Self::Coral => "coral",
            Self::Echo => "echo",
            Self::Fable => "fable",
            Self::Marin => "marin",
            Self::Nova => "nova",
            Self::Onyx => "onyx",
            Self::Sage => "sage",
            Self::Shimmer => "shimmer",
            Self::Verse => "verse",
        }
    }
}

impl std::fmt::Display for Voice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Audio output formats for TTS.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum AudioFormat {
    /// MP3 format (default)
    #[default]
    Mp3,
    /// Opus format
    Opus,
    /// AAC format
    Aac,
    /// FLAC format
    Flac,
    /// WAV format
    Wav,
    /// PCM format
    Pcm,
}

impl AudioFormat {
    /// Returns the format string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Mp3 => "mp3",
            Self::Opus => "opus",
            Self::Aac => "aac",
            Self::Flac => "flac",
            Self::Wav => "wav",
            Self::Pcm => "pcm",
        }
    }

    /// Returns the file extension for this format.
    pub fn file_extension(&self) -> &'static str {
        self.as_str()
    }
}

impl std::fmt::Display for AudioFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Speech-to-text models for transcription and translation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SttModel {
    /// Whisper v1 model
    #[serde(rename = "whisper-1")]
    #[default]
    Whisper1,
    /// GPT-4o Transcribe model
    #[serde(rename = "gpt-4o-transcribe")]
    Gpt4oTranscribe,
    /// GPT-4o Mini Transcribe - cheaper GPT-4o Transcribe
    #[serde(rename = "gpt-4o-mini-transcribe")]
    Gpt4oMiniTranscribe,
    /// GPT-4o Transcribe Diarize - transcription with speaker diarization
    #[serde(rename = "gpt-4o-transcribe-diarize")]
    Gpt4oTranscribeDiarize,
    /// GPT Transcribe - high-accuracy speech-to-text for file and realtime input
    #[serde(rename = "gpt-transcribe")]
    GptTranscribe,
    /// GPT Live Transcribe - low-latency streaming transcript deltas
    ///
    /// Realtime transcription sessions only; not available on
    /// `/v1/audio/transcriptions`.
    #[serde(rename = "gpt-live-transcribe")]
    GptLiveTranscribe,
    /// GPT Realtime Whisper - Whisper for realtime transcription sessions
    ///
    /// Realtime transcription sessions only; not available on
    /// `/v1/audio/transcriptions`.
    #[serde(rename = "gpt-realtime-whisper")]
    GptRealtimeWhisper,
}

impl SttModel {
    /// Returns the model identifier string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Whisper1 => "whisper-1",
            Self::Gpt4oTranscribe => "gpt-4o-transcribe",
            Self::Gpt4oMiniTranscribe => "gpt-4o-mini-transcribe",
            Self::Gpt4oTranscribeDiarize => "gpt-4o-transcribe-diarize",
            Self::GptTranscribe => "gpt-transcribe",
            Self::GptLiveTranscribe => "gpt-live-transcribe",
            Self::GptRealtimeWhisper => "gpt-realtime-whisper",
        }
    }

    /// Returns `true` if the model can transcribe uploaded files via
    /// `/v1/audio/transcriptions`.
    ///
    /// [`GptLiveTranscribe`](Self::GptLiveTranscribe) and
    /// [`GptRealtimeWhisper`](Self::GptRealtimeWhisper) are exposed only on
    /// `/v1/realtime/transcription_sessions`, so passing them to
    /// [`Audio::transcribe`] would be rejected by the API.
    pub fn supports_file_transcription(&self) -> bool {
        !matches!(self, Self::GptLiveTranscribe | Self::GptRealtimeWhisper)
    }
}

impl std::fmt::Display for SttModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Transcription response formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum TranscriptionFormat {
    /// JSON format
    #[default]
    Json,
    /// Plain text format
    Text,
    /// SRT subtitle format
    Srt,
    /// Verbose JSON with timestamps
    VerboseJson,
    /// VTT subtitle format
    Vtt,
}

impl TranscriptionFormat {
    /// Returns the format string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Json => "json",
            Self::Text => "text",
            Self::Srt => "srt",
            Self::VerboseJson => "verbose_json",
            Self::Vtt => "vtt",
        }
    }
}

/// Timestamp granularity options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TimestampGranularity {
    /// Word-level timestamps
    Word,
    /// Segment-level timestamps
    Segment,
}

impl TimestampGranularity {
    /// Returns the granularity string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Word => "word",
            Self::Segment => "segment",
        }
    }
}

/// Options for text-to-speech generation.
#[derive(Debug, Clone, Default)]
pub struct TtsOptions {
    /// The model to use (defaults to tts-1)
    pub model: TtsModel,
    /// The voice to use (defaults to alloy)
    pub voice: Voice,
    /// The output audio format (defaults to mp3)
    pub response_format: AudioFormat,
    /// Speech speed (0.25 to 4.0, defaults to 1.0)
    pub speed: Option<f32>,
    /// Instructions for controlling voice characteristics.
    ///
    /// Only supported by `gpt-4o-mini-tts` model.
    /// Use natural language to control tone, emotion, and pacing.
    ///
    /// # Examples
    ///
    /// - `"Speak in a cheerful and positive tone."`
    /// - `"Use a calm and soothing voice."`
    /// - `"Speak with enthusiasm and energy."`
    ///
    /// If set with an unsupported model (`tts-1` or `tts-1-hd`),
    /// this parameter will be ignored and a warning will be logged.
    pub instructions: Option<String>,
}

/// Options for audio transcription.
#[derive(Debug, Clone, Default)]
pub struct TranscribeOptions {
    /// The model to use (defaults to whisper-1)
    pub model: Option<SttModel>,
    /// The language of the input audio (ISO-639-1 code)
    pub language: Option<String>,
    /// Optional prompt to guide the model's style
    pub prompt: Option<String>,
    /// Response format (defaults to json)
    pub response_format: Option<TranscriptionFormat>,
    /// Temperature for sampling (0.0 to 1.0)
    pub temperature: Option<f32>,
    /// Timestamp granularities to include
    pub timestamp_granularities: Option<Vec<TimestampGranularity>>,
}

/// Options for audio translation.
#[derive(Debug, Clone, Default)]
pub struct TranslateOptions {
    /// The model to use (only whisper-1 is supported)
    pub model: Option<SttModel>,
    /// Optional prompt to guide the model's style
    pub prompt: Option<String>,
    /// Response format (defaults to json)
    pub response_format: Option<TranscriptionFormat>,
    /// Temperature for sampling (0.0 to 1.0)
    pub temperature: Option<f32>,
}

/// Request payload for TTS.
#[derive(Debug, Clone, Serialize)]
struct TtsRequest {
    model: String,
    input: String,
    voice: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    speed: Option<f32>,
    /// Instructions for voice control (only for gpt-4o-mini-tts).
    #[serde(skip_serializing_if = "Option::is_none")]
    instructions: Option<String>,
}

/// Client for interacting with the OpenAI Audio API.
///
/// This struct provides methods for text-to-speech, transcription, and translation.
/// Use [`Audio::new()`] to create a new instance.
///
/// # Example
///
/// ```rust,no_run
/// use openai_tools::audio::request::{Audio, TtsOptions, Voice, AudioFormat};
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let audio = Audio::new()?;
///
///     let options = TtsOptions {
///         voice: Voice::Nova,
///         response_format: AudioFormat::Mp3,
///         ..Default::default()
///     };
///
///     let bytes = audio.text_to_speech("Welcome to our app!", options).await?;
///     std::fs::write("welcome.mp3", bytes)?;
///
///     Ok(())
/// }
/// ```
pub struct Audio {
    /// Authentication provider (OpenAI or Azure)
    auth: AuthProvider,
    /// Optional request timeout duration
    timeout: Option<Duration>,
}

impl Audio {
    /// Creates a new Audio client for OpenAI API.
    ///
    /// Initializes the client by loading the OpenAI API key from
    /// the environment variable `OPENAI_API_KEY`. Supports `.env` file loading
    /// via dotenvy.
    ///
    /// # Returns
    ///
    /// * `Ok(Audio)` - A new Audio client ready for use
    /// * `Err(OpenAIToolError)` - If the API key is not found in the environment
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::audio::request::Audio;
    ///
    /// let audio = Audio::new().expect("API key should be set");
    /// ```
    pub fn new() -> Result<Self> {
        let auth = AuthProvider::openai_from_env()?;
        Ok(Self { auth, timeout: None })
    }

    /// Creates a new Audio client with a custom authentication provider
    pub fn with_auth(auth: AuthProvider) -> Self {
        Self { auth, timeout: None }
    }

    /// Creates a new Audio client for Azure OpenAI API
    pub fn azure() -> Result<Self> {
        let auth = AuthProvider::azure_from_env()?;
        Ok(Self { auth, timeout: None })
    }

    /// Creates a new Audio client by auto-detecting the provider
    pub fn detect_provider() -> Result<Self> {
        let auth = AuthProvider::from_env()?;
        Ok(Self { auth, timeout: None })
    }

    /// Creates a new Audio client with URL-based provider detection
    pub fn with_url<S: Into<String>>(base_url: S, api_key: S) -> Self {
        let auth = AuthProvider::from_url_with_key(base_url, api_key);
        Self { auth, timeout: None }
    }

    /// Creates a new Audio client from URL using environment variables
    pub fn from_url<S: Into<String>>(url: S) -> Result<Self> {
        let auth = AuthProvider::from_url(url)?;
        Ok(Self { auth, timeout: None })
    }

    /// Returns the authentication provider
    pub fn auth(&self) -> &AuthProvider {
        &self.auth
    }

    /// Sets the request timeout duration.
    ///
    /// # Arguments
    ///
    /// * `timeout` - The maximum time to wait for a response
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    pub fn timeout(&mut self, timeout: Duration) -> &mut Self {
        self.timeout = Some(timeout);
        self
    }

    /// Creates the HTTP client with default headers.
    fn create_client(&self) -> Result<(request::Client, request::header::HeaderMap)> {
        let client = create_http_client(self.timeout)?;
        let mut headers = request::header::HeaderMap::new();
        self.auth.apply_headers(&mut headers)?;
        headers.insert("User-Agent", request::header::HeaderValue::from_static("openai-tools-rust"));
        Ok((client, headers))
    }

    /// Converts text to speech.
    ///
    /// Returns audio bytes in the specified format.
    ///
    /// # Arguments
    ///
    /// * `text` - The text to convert to speech (max 4096 characters)
    /// * `options` - TTS options (model, voice, format, speed)
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<u8>)` - The audio data as bytes
    /// * `Err(OpenAIToolError)` - If the request fails
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::audio::request::{Audio, TtsOptions, TtsModel, Voice};
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let audio = Audio::new()?;
    ///
    ///     let options = TtsOptions {
    ///         model: TtsModel::Tts1Hd,
    ///         voice: Voice::Shimmer,
    ///         speed: Some(1.2),
    ///         ..Default::default()
    ///     };
    ///
    ///     let bytes = audio.text_to_speech("Hello, this is a test.", options).await?;
    ///     std::fs::write("speech.mp3", bytes)?;
    ///
    ///     Ok(())
    /// }
    /// ```
    pub async fn text_to_speech(&self, text: &str, options: TtsOptions) -> Result<Vec<u8>> {
        let (client, mut headers) = self.create_client()?;
        headers.insert("Content-Type", request::header::HeaderValue::from_static("application/json"));

        // Check if instructions parameter is supported by the model
        let instructions = if options.instructions.is_some() {
            if options.model.supports_instructions() {
                options.instructions
            } else {
                tracing::warn!("Model '{}' does not support instructions parameter. Ignoring instructions.", options.model);
                None
            }
        } else {
            None
        };

        let request_body = TtsRequest {
            model: options.model.as_str().to_string(),
            input: text.to_string(),
            voice: options.voice.as_str().to_string(),
            response_format: Some(options.response_format.as_str().to_string()),
            speed: options.speed,
            instructions,
        };

        let body = serde_json::to_string(&request_body).map_err(OpenAIToolError::SerdeJsonError)?;

        let url = format!("{}/speech", self.auth.endpoint(AUDIO_PATH));

        let response = client.post(&url).headers(headers).body(body).send().await.map_err(OpenAIToolError::RequestError)?;

        let bytes = response.bytes().await.map_err(OpenAIToolError::RequestError)?;

        Ok(bytes.to_vec())
    }

    /// Transcribes audio from a file path.
    ///
    /// # Arguments
    ///
    /// * `audio_path` - Path to the audio file
    /// * `options` - Transcription options
    ///
    /// # Returns
    ///
    /// * `Ok(TranscriptionResponse)` - The transcription result
    /// * `Err(OpenAIToolError)` - If the request fails
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::audio::request::{Audio, TranscribeOptions};
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let audio = Audio::new()?;
    ///
    ///     let options = TranscribeOptions {
    ///         language: Some("en".to_string()),
    ///         ..Default::default()
    ///     };
    ///
    ///     let response = audio.transcribe("audio.mp3", options).await?;
    ///     println!("Transcription: {}", response.text);
    ///
    ///     Ok(())
    /// }
    /// ```
    pub async fn transcribe(&self, audio_path: &str, options: TranscribeOptions) -> Result<TranscriptionResponse> {
        let audio_content = tokio::fs::read(audio_path).await.map_err(|e| OpenAIToolError::Error(format!("Failed to read audio file: {}", e)))?;

        let filename = Path::new(audio_path).file_name().and_then(|n| n.to_str()).unwrap_or("audio.mp3").to_string();

        self.transcribe_bytes(&audio_content, &filename, options).await
    }

    /// Transcribes audio from bytes.
    ///
    /// # Arguments
    ///
    /// * `audio_data` - The audio data as bytes
    /// * `filename` - The filename with extension (e.g., "audio.mp3")
    /// * `options` - Transcription options
    ///
    /// # Returns
    ///
    /// * `Ok(TranscriptionResponse)` - The transcription result
    /// * `Err(OpenAIToolError)` - If the request fails
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::audio::request::{Audio, TranscribeOptions, SttModel};
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let audio = Audio::new()?;
    ///
    ///     let audio_data = std::fs::read("recording.mp3")?;
    ///     let options = TranscribeOptions {
    ///         model: Some(SttModel::Whisper1),
    ///         ..Default::default()
    ///     };
    ///
    ///     let response = audio.transcribe_bytes(&audio_data, "recording.mp3", options).await?;
    ///     println!("Transcription: {}", response.text);
    ///
    ///     Ok(())
    /// }
    /// ```
    pub async fn transcribe_bytes(&self, audio_data: &[u8], filename: &str, options: TranscribeOptions) -> Result<TranscriptionResponse> {
        let (client, headers) = self.create_client()?;

        let audio_part = Part::bytes(audio_data.to_vec())
            .file_name(filename.to_string())
            .mime_str("audio/mpeg")
            .map_err(|e| OpenAIToolError::Error(format!("Failed to set MIME type: {}", e)))?;

        let mut form = Form::new().part("file", audio_part);

        // Add model
        let model = options.model.unwrap_or_default();
        form = form.text("model", model.as_str().to_string());

        // Add optional parameters
        if let Some(language) = options.language {
            form = form.text("language", language);
        }
        if let Some(prompt) = options.prompt {
            form = form.text("prompt", prompt);
        }
        if let Some(response_format) = options.response_format {
            form = form.text("response_format", response_format.as_str().to_string());
        }
        if let Some(temperature) = options.temperature {
            form = form.text("temperature", temperature.to_string());
        }
        if let Some(granularities) = options.timestamp_granularities {
            for g in granularities {
                form = form.text("timestamp_granularities[]", g.as_str().to_string());
            }
        }

        let url = format!("{}/transcriptions", self.auth.endpoint(AUDIO_PATH));

        let response = client.post(&url).headers(headers).multipart(form).send().await.map_err(OpenAIToolError::RequestError)?;

        let status = response.status();
        let content = response.text().await.map_err(OpenAIToolError::RequestError)?;

        if cfg!(test) {
            tracing::info!("Response content: {}", content);
        }

        if !status.is_success() {
            if let Ok(error_resp) = serde_json::from_str::<ErrorResponse>(&content) {
                return Err(OpenAIToolError::Error(error_resp.error.message.unwrap_or_default()));
            }
            return Err(OpenAIToolError::Error(format!("API error ({}): {}", status, content)));
        }

        serde_json::from_str::<TranscriptionResponse>(&content).map_err(OpenAIToolError::SerdeJsonError)
    }

    /// Translates audio to English text.
    ///
    /// Only supports translation to English using the whisper-1 model.
    ///
    /// # Arguments
    ///
    /// * `audio_path` - Path to the audio file
    /// * `options` - Translation options
    ///
    /// # Returns
    ///
    /// * `Ok(TranscriptionResponse)` - The translation result
    /// * `Err(OpenAIToolError)` - If the request fails
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::audio::request::{Audio, TranslateOptions};
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let audio = Audio::new()?;
    ///
    ///     let options = TranslateOptions::default();
    ///     let response = audio.translate("french_audio.mp3", options).await?;
    ///     println!("English translation: {}", response.text);
    ///
    ///     Ok(())
    /// }
    /// ```
    pub async fn translate(&self, audio_path: &str, options: TranslateOptions) -> Result<TranscriptionResponse> {
        let audio_content = tokio::fs::read(audio_path).await.map_err(|e| OpenAIToolError::Error(format!("Failed to read audio file: {}", e)))?;

        let filename = Path::new(audio_path).file_name().and_then(|n| n.to_str()).unwrap_or("audio.mp3").to_string();

        self.translate_bytes(&audio_content, &filename, options).await
    }

    /// Translates audio from bytes to English text.
    ///
    /// # Arguments
    ///
    /// * `audio_data` - The audio data as bytes
    /// * `filename` - The filename with extension (e.g., "audio.mp3")
    /// * `options` - Translation options
    ///
    /// # Returns
    ///
    /// * `Ok(TranscriptionResponse)` - The translation result
    /// * `Err(OpenAIToolError)` - If the request fails
    pub async fn translate_bytes(&self, audio_data: &[u8], filename: &str, options: TranslateOptions) -> Result<TranscriptionResponse> {
        let (client, headers) = self.create_client()?;

        let audio_part = Part::bytes(audio_data.to_vec())
            .file_name(filename.to_string())
            .mime_str("audio/mpeg")
            .map_err(|e| OpenAIToolError::Error(format!("Failed to set MIME type: {}", e)))?;

        let mut form = Form::new().part("file", audio_part);

        // Add model (whisper-1 is the only supported model for translation)
        let model = options.model.unwrap_or(SttModel::Whisper1);
        form = form.text("model", model.as_str().to_string());

        // Add optional parameters
        if let Some(prompt) = options.prompt {
            form = form.text("prompt", prompt);
        }
        if let Some(response_format) = options.response_format {
            form = form.text("response_format", response_format.as_str().to_string());
        }
        if let Some(temperature) = options.temperature {
            form = form.text("temperature", temperature.to_string());
        }

        let url = format!("{}/translations", self.auth.endpoint(AUDIO_PATH));

        let response = client.post(&url).headers(headers).multipart(form).send().await.map_err(OpenAIToolError::RequestError)?;

        let status = response.status();
        let content = response.text().await.map_err(OpenAIToolError::RequestError)?;

        if cfg!(test) {
            tracing::info!("Response content: {}", content);
        }

        if !status.is_success() {
            if let Ok(error_resp) = serde_json::from_str::<ErrorResponse>(&content) {
                return Err(OpenAIToolError::Error(error_resp.error.message.unwrap_or_default()));
            }
            return Err(OpenAIToolError::Error(format!("API error ({}): {}", status, content)));
        }

        serde_json::from_str::<TranscriptionResponse>(&content).map_err(OpenAIToolError::SerdeJsonError)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // TtsModel Tests
    // =========================================================================

    #[test]
    fn test_tts_model_as_str() {
        assert_eq!(TtsModel::Tts1.as_str(), "tts-1");
        assert_eq!(TtsModel::Tts1Hd.as_str(), "tts-1-hd");
        assert_eq!(TtsModel::Gpt4oMiniTts.as_str(), "gpt-4o-mini-tts");
    }

    #[test]
    fn test_tts_model_supports_instructions() {
        // Only gpt-4o-mini-tts supports instructions
        assert!(TtsModel::Gpt4oMiniTts.supports_instructions());
        assert!(!TtsModel::Tts1.supports_instructions());
        assert!(!TtsModel::Tts1Hd.supports_instructions());
    }

    #[test]
    fn test_tts_model_default() {
        let model = TtsModel::default();
        assert_eq!(model, TtsModel::Tts1);
    }

    #[test]
    fn test_tts_model_display() {
        assert_eq!(format!("{}", TtsModel::Gpt4oMiniTts), "gpt-4o-mini-tts");
    }

    // =========================================================================
    // Voice Tests
    // =========================================================================

    #[test]
    fn test_voice_as_str_all_voices() {
        assert_eq!(Voice::Alloy.as_str(), "alloy");
        assert_eq!(Voice::Ash.as_str(), "ash");
        assert_eq!(Voice::Ballad.as_str(), "ballad");
        assert_eq!(Voice::Cedar.as_str(), "cedar");
        assert_eq!(Voice::Coral.as_str(), "coral");
        assert_eq!(Voice::Echo.as_str(), "echo");
        assert_eq!(Voice::Fable.as_str(), "fable");
        assert_eq!(Voice::Marin.as_str(), "marin");
        assert_eq!(Voice::Nova.as_str(), "nova");
        assert_eq!(Voice::Onyx.as_str(), "onyx");
        assert_eq!(Voice::Sage.as_str(), "sage");
        assert_eq!(Voice::Shimmer.as_str(), "shimmer");
        assert_eq!(Voice::Verse.as_str(), "verse");
    }

    #[test]
    fn test_voice_new_voices() {
        // Test the newly added voices
        assert_eq!(Voice::Ballad.as_str(), "ballad");
        assert_eq!(Voice::Cedar.as_str(), "cedar");
        assert_eq!(Voice::Marin.as_str(), "marin");
        assert_eq!(Voice::Verse.as_str(), "verse");
    }

    #[test]
    fn test_voice_default() {
        let voice = Voice::default();
        assert_eq!(voice, Voice::Alloy);
    }

    #[test]
    fn test_voice_serialization() {
        let voice = Voice::Coral;
        let json = serde_json::to_string(&voice).unwrap();
        assert_eq!(json, "\"coral\"");

        // Test new voices
        let ballad = Voice::Ballad;
        let json = serde_json::to_string(&ballad).unwrap();
        assert_eq!(json, "\"ballad\"");
    }

    #[test]
    fn test_voice_deserialization() {
        let voice: Voice = serde_json::from_str("\"coral\"").unwrap();
        assert_eq!(voice, Voice::Coral);

        // Test new voices
        let cedar: Voice = serde_json::from_str("\"cedar\"").unwrap();
        assert_eq!(cedar, Voice::Cedar);

        let marin: Voice = serde_json::from_str("\"marin\"").unwrap();
        assert_eq!(marin, Voice::Marin);
    }

    // =========================================================================
    // TtsOptions Tests
    // =========================================================================

    #[test]
    fn test_tts_options_default() {
        let options = TtsOptions::default();
        assert_eq!(options.model, TtsModel::Tts1);
        assert_eq!(options.voice, Voice::Alloy);
        assert_eq!(options.response_format, AudioFormat::Mp3);
        assert!(options.speed.is_none());
        assert!(options.instructions.is_none());
    }

    #[test]
    fn test_tts_options_with_instructions() {
        let options = TtsOptions {
            model: TtsModel::Gpt4oMiniTts,
            voice: Voice::Coral,
            instructions: Some("Speak in a cheerful tone.".to_string()),
            ..Default::default()
        };
        assert_eq!(options.model, TtsModel::Gpt4oMiniTts);
        assert_eq!(options.instructions, Some("Speak in a cheerful tone.".to_string()));
    }

    // =========================================================================
    // TtsRequest Tests
    // =========================================================================

    #[test]
    fn test_tts_request_serialization_with_instructions() {
        let request = TtsRequest {
            model: "gpt-4o-mini-tts".to_string(),
            input: "Hello, world!".to_string(),
            voice: "coral".to_string(),
            response_format: Some("mp3".to_string()),
            speed: None,
            instructions: Some("Speak cheerfully.".to_string()),
        };
        let json = serde_json::to_value(&request).unwrap();

        assert_eq!(json["model"], "gpt-4o-mini-tts");
        assert_eq!(json["input"], "Hello, world!");
        assert_eq!(json["voice"], "coral");
        assert_eq!(json["response_format"], "mp3");
        assert_eq!(json["instructions"], "Speak cheerfully.");
        assert!(json.get("speed").is_none());
    }

    #[test]
    fn test_tts_request_serialization_without_instructions() {
        let request = TtsRequest {
            model: "tts-1".to_string(),
            input: "Hello".to_string(),
            voice: "alloy".to_string(),
            response_format: Some("mp3".to_string()),
            speed: Some(1.0),
            instructions: None,
        };
        let json = serde_json::to_value(&request).unwrap();

        assert_eq!(json["model"], "tts-1");
        assert_eq!(json["speed"], 1.0);
        // instructions should be omitted when None
        assert!(json.get("instructions").is_none());
    }

    #[test]
    fn test_tts_request_skip_serializing_none_fields() {
        let request = TtsRequest {
            model: "tts-1".to_string(),
            input: "Test".to_string(),
            voice: "echo".to_string(),
            response_format: None,
            speed: None,
            instructions: None,
        };
        let json = serde_json::to_value(&request).unwrap();

        // Required fields are present
        assert!(json.get("model").is_some());
        assert!(json.get("input").is_some());
        assert!(json.get("voice").is_some());

        // Optional fields with None are omitted
        assert!(json.get("response_format").is_none());
        assert!(json.get("speed").is_none());
        assert!(json.get("instructions").is_none());
    }

    // =========================================================================
    // AudioFormat Tests
    // =========================================================================

    #[test]
    fn test_audio_format_as_str() {
        assert_eq!(AudioFormat::Mp3.as_str(), "mp3");
        assert_eq!(AudioFormat::Opus.as_str(), "opus");
        assert_eq!(AudioFormat::Aac.as_str(), "aac");
        assert_eq!(AudioFormat::Flac.as_str(), "flac");
        assert_eq!(AudioFormat::Wav.as_str(), "wav");
        assert_eq!(AudioFormat::Pcm.as_str(), "pcm");
    }

    #[test]
    fn test_audio_format_file_extension() {
        assert_eq!(AudioFormat::Mp3.file_extension(), "mp3");
        assert_eq!(AudioFormat::Wav.file_extension(), "wav");
    }

    // =========================================================================
    // SttModel Tests
    // =========================================================================

    #[test]
    fn test_stt_model_as_str() {
        assert_eq!(SttModel::Whisper1.as_str(), "whisper-1");
        assert_eq!(SttModel::Gpt4oTranscribe.as_str(), "gpt-4o-transcribe");
    }

    // =========================================================================
    // TranscriptionFormat Tests
    // =========================================================================

    #[test]
    fn test_transcription_format_as_str() {
        assert_eq!(TranscriptionFormat::Json.as_str(), "json");
        assert_eq!(TranscriptionFormat::Text.as_str(), "text");
        assert_eq!(TranscriptionFormat::Srt.as_str(), "srt");
        assert_eq!(TranscriptionFormat::VerboseJson.as_str(), "verbose_json");
        assert_eq!(TranscriptionFormat::Vtt.as_str(), "vtt");
    }

    // =========================================================================
    // TimestampGranularity Tests
    // =========================================================================

    #[test]
    fn test_timestamp_granularity_as_str() {
        assert_eq!(TimestampGranularity::Word.as_str(), "word");
        assert_eq!(TimestampGranularity::Segment.as_str(), "segment");
    }

    // =========================================================================
    // Speech-to-text models added in 2026.
    //
    // Model IDs verified against the OpenAI API reference
    // (https://developers.openai.com/api/docs/models), August 2026.
    // =========================================================================

    fn new_stt_models() -> Vec<(SttModel, &'static str)> {
        vec![
            (SttModel::GptTranscribe, "gpt-transcribe"),
            (SttModel::GptLiveTranscribe, "gpt-live-transcribe"),
            (SttModel::GptRealtimeWhisper, "gpt-realtime-whisper"),
            (SttModel::Gpt4oMiniTranscribe, "gpt-4o-mini-transcribe"),
        ]
    }

    #[test]
    fn test_new_stt_models_as_str() {
        for (model, expected) in new_stt_models() {
            assert_eq!(model.as_str(), expected, "Wrong model ID for {:?}", model);
        }
    }

    #[test]
    fn test_new_stt_models_serialization() {
        for (model, expected) in new_stt_models() {
            let json = serde_json::to_string(&model).unwrap();
            assert_eq!(json, format!("\"{}\"", expected), "Wrong serialization for {:?}", model);
            let deserialized: SttModel = serde_json::from_str(&json).unwrap();
            assert_eq!(deserialized, model, "Serialization roundtrip failed for {:?}", model);
        }
    }

    /// `gpt-live-transcribe` and `gpt-realtime-whisper` are only exposed on
    /// `v1/realtime/transcription_sessions`, not on `v1/audio/transcriptions`.
    #[test]
    fn test_realtime_only_stt_models_are_flagged() {
        assert!(!SttModel::GptLiveTranscribe.supports_file_transcription());
        assert!(!SttModel::GptRealtimeWhisper.supports_file_transcription());

        assert!(SttModel::GptTranscribe.supports_file_transcription());
        assert!(SttModel::Whisper1.supports_file_transcription());
        assert!(SttModel::Gpt4oTranscribe.supports_file_transcription());
        assert!(SttModel::Gpt4oMiniTranscribe.supports_file_transcription());
    }

    /// Audio models present in the live /v1/models listing but previously
    /// missing from the enums. Verified live against the API (August 2026).
    #[test]
    fn test_previously_missing_tts_models() {
        for (model, expected) in [(TtsModel::Tts1_1106, "tts-1-1106"), (TtsModel::Tts1Hd1106, "tts-1-hd-1106")] {
            assert_eq!(model.as_str(), expected, "Wrong model ID for {:?}", model);
            let json = serde_json::to_string(&model).unwrap();
            assert_eq!(json, format!("\"{}\"", expected), "Wrong serialization for {:?}", model);
        }
    }

    #[test]
    fn test_diarizing_stt_model() {
        assert_eq!(SttModel::Gpt4oTranscribeDiarize.as_str(), "gpt-4o-transcribe-diarize");
        assert!(SttModel::Gpt4oTranscribeDiarize.supports_file_transcription());
    }
}
