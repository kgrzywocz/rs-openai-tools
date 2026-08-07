//! Audio types for the Realtime API.

use serde::{Deserialize, Serialize};

/// Audio formats supported by the Realtime API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum AudioFormat {
    /// PCM 16-bit linear encoding (24kHz, mono)
    #[default]
    Pcm16,
    /// G.711 mu-law encoding
    G711Ulaw,
    /// G.711 A-law encoding
    G711Alaw,
}

/// Voice options for text-to-speech output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum Voice {
    #[default]
    Alloy,
    Ash,
    Ballad,
    Coral,
    Echo,
    Sage,
    Shimmer,
    Verse,
}

/// Transcription model options for input audio.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub enum TranscriptionModel {
    #[serde(rename = "whisper-1")]
    #[default]
    Whisper1,
}

/// Input audio transcription configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InputAudioTranscription {
    /// The transcription model to use.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<TranscriptionModel>,

    /// Language hint for transcription (ISO-639-1 code).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,

    /// Optional prompt to guide transcription.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
}

impl InputAudioTranscription {
    /// Create a new transcription configuration with the specified model.
    pub fn new(model: TranscriptionModel) -> Self {
        Self { model: Some(model), language: None, prompt: None }
    }

    /// Set the language hint.
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Set the transcription prompt.
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = Some(prompt.into());
        self
    }
}

/// Input audio noise reduction configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputAudioNoiseReduction {
    /// Type of noise reduction to apply.
    #[serde(rename = "type")]
    pub noise_type: NoiseReductionType,
}

/// Noise reduction type options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum NoiseReductionType {
    /// Optimized for near-field audio (close microphone).
    NearField,
    /// Optimized for far-field audio (distant microphone).
    FarField,
}
