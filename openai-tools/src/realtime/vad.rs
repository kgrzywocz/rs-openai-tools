//! Voice Activity Detection (VAD) configuration types.

use serde::{Deserialize, Serialize};

/// Turn detection configuration.
///
/// Determines how the server detects when a user has finished speaking.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TurnDetection {
    /// Server-side Voice Activity Detection.
    ServerVad(ServerVadConfig),
    /// Semantic-based turn detection using language understanding.
    SemanticVad(SemanticVadConfig),
}

/// Server VAD configuration.
///
/// Uses audio signal analysis to detect speech boundaries.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ServerVadConfig {
    /// Activation threshold for speech detection (0.0 to 1.0).
    /// Lower values are more sensitive.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub threshold: Option<f32>,

    /// Amount of audio to include before detected speech start (milliseconds).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefix_padding_ms: Option<u32>,

    /// Duration of silence required to detect speech end (milliseconds).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub silence_duration_ms: Option<u32>,

    /// Whether to automatically create a response when speech ends.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub create_response: Option<bool>,

    /// Whether to interrupt an active response when new speech is detected.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub interrupt_response: Option<bool>,
}

impl ServerVadConfig {
    /// Create a new ServerVadConfig with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the activation threshold.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = Some(threshold);
        self
    }

    /// Set the prefix padding duration.
    pub fn with_prefix_padding_ms(mut self, ms: u32) -> Self {
        self.prefix_padding_ms = Some(ms);
        self
    }

    /// Set the silence duration for end detection.
    pub fn with_silence_duration_ms(mut self, ms: u32) -> Self {
        self.silence_duration_ms = Some(ms);
        self
    }

    /// Set whether to automatically create responses.
    pub fn with_create_response(mut self, create: bool) -> Self {
        self.create_response = Some(create);
        self
    }

    /// Set whether to interrupt active responses.
    pub fn with_interrupt_response(mut self, interrupt: bool) -> Self {
        self.interrupt_response = Some(interrupt);
        self
    }
}

/// Semantic VAD configuration.
///
/// Uses language understanding to detect natural turn boundaries.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SemanticVadConfig {
    /// Eagerness level for detecting end of utterance.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eagerness: Option<Eagerness>,

    /// Whether to automatically create a response when turn ends.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub create_response: Option<bool>,

    /// Whether to interrupt an active response when new speech is detected.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub interrupt_response: Option<bool>,
}

impl SemanticVadConfig {
    /// Create a new SemanticVadConfig with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the eagerness level.
    pub fn with_eagerness(mut self, eagerness: Eagerness) -> Self {
        self.eagerness = Some(eagerness);
        self
    }

    /// Set whether to automatically create responses.
    pub fn with_create_response(mut self, create: bool) -> Self {
        self.create_response = Some(create);
        self
    }

    /// Set whether to interrupt active responses.
    pub fn with_interrupt_response(mut self, interrupt: bool) -> Self {
        self.interrupt_response = Some(interrupt);
        self
    }
}

/// Eagerness level for semantic turn detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum Eagerness {
    /// Wait longer for user to finish (more conservative).
    Low,
    /// Balanced detection (default).
    Medium,
    /// Respond more quickly (more aggressive).
    High,
    /// Let the model decide.
    #[default]
    Auto,
}
