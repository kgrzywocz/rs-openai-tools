//! Session configuration types for the OpenAI Live API.

use serde::{de, ser::SerializeStruct, Deserialize, Deserializer, Serialize, Serializer};

use crate::common::models::LiveModel;
use crate::responses::request::{ReasoningEffort, ReasoningSummary};

/// A built-in or custom voice for Live audio output.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum LiveVoice {
    Alloy,
    Ash,
    Ballad,
    Beacon,
    Bossa,
    Cedar,
    Cinder,
    Coral,
    Delta,
    Echo,
    Gleam,
    #[default]
    Marin,
    Meridian,
    Quartz,
    Ripple,
    Sage,
    Shimmer,
    Stone,
    Tempo,
    Verse,
    Vesper,
    Willow,
    /// A project-specific custom voice.
    Custom {
        id: String,
    },
}

impl LiveVoice {
    fn built_in_name(&self) -> Option<&'static str> {
        match self {
            Self::Alloy => Some("alloy"),
            Self::Ash => Some("ash"),
            Self::Ballad => Some("ballad"),
            Self::Beacon => Some("beacon"),
            Self::Bossa => Some("bossa"),
            Self::Cedar => Some("cedar"),
            Self::Cinder => Some("cinder"),
            Self::Coral => Some("coral"),
            Self::Delta => Some("delta"),
            Self::Echo => Some("echo"),
            Self::Gleam => Some("gleam"),
            Self::Marin => Some("marin"),
            Self::Meridian => Some("meridian"),
            Self::Quartz => Some("quartz"),
            Self::Ripple => Some("ripple"),
            Self::Sage => Some("sage"),
            Self::Shimmer => Some("shimmer"),
            Self::Stone => Some("stone"),
            Self::Tempo => Some("tempo"),
            Self::Verse => Some("verse"),
            Self::Vesper => Some("vesper"),
            Self::Willow => Some("willow"),
            Self::Custom { .. } => None,
        }
    }

    fn from_built_in(name: &str) -> Option<Self> {
        Some(match name {
            "alloy" => Self::Alloy,
            "ash" => Self::Ash,
            "ballad" => Self::Ballad,
            "beacon" => Self::Beacon,
            "bossa" => Self::Bossa,
            "cedar" => Self::Cedar,
            "cinder" => Self::Cinder,
            "coral" => Self::Coral,
            "delta" => Self::Delta,
            "echo" => Self::Echo,
            "gleam" => Self::Gleam,
            "marin" => Self::Marin,
            "meridian" => Self::Meridian,
            "quartz" => Self::Quartz,
            "ripple" => Self::Ripple,
            "sage" => Self::Sage,
            "shimmer" => Self::Shimmer,
            "stone" => Self::Stone,
            "tempo" => Self::Tempo,
            "verse" => Self::Verse,
            "vesper" => Self::Vesper,
            "willow" => Self::Willow,
            _ => return None,
        })
    }
}

impl Serialize for LiveVoice {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if let Some(name) = self.built_in_name() {
            serializer.serialize_str(name)
        } else if let Self::Custom { id } = self {
            #[derive(Serialize)]
            struct CustomVoice<'a> {
                id: &'a str,
            }
            CustomVoice { id }.serialize(serializer)
        } else {
            unreachable!("all LiveVoice variants are handled")
        }
    }
}

impl<'de> Deserialize<'de> for LiveVoice {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum VoiceValue {
            BuiltIn(String),
            Custom { id: String },
        }

        match VoiceValue::deserialize(deserializer)? {
            VoiceValue::BuiltIn(name) => Self::from_built_in(&name).ok_or_else(|| de::Error::unknown_variant(&name, &[])),
            VoiceValue::Custom { id } => Ok(Self::Custom { id }),
        }
    }
}

/// Encoding used for Live input and output audio.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum LiveAudioEncoding {
    #[serde(rename = "audio/pcm")]
    Pcm,
    #[serde(rename = "audio/pcmu")]
    Pcmu,
    #[serde(rename = "audio/pcma")]
    Pcma,
    /// An encoding added after this library version.
    #[serde(untagged)]
    Other(String),
}

/// Audio encoding and sample rate shared by Live input and output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LiveAudioFormat {
    /// Audio MIME-style encoding name.
    #[serde(rename = "type")]
    pub encoding: LiveAudioEncoding,
    /// Sample rate in hertz.
    pub rate: u32,
}

impl LiveAudioFormat {
    /// Signed PCM16 little-endian audio at 24 kHz.
    pub const fn pcm24k() -> Self {
        Self { encoding: LiveAudioEncoding::Pcm, rate: 24_000 }
    }

    /// Signed PCM16 little-endian audio at 16 kHz.
    pub const fn pcm16k() -> Self {
        Self { encoding: LiveAudioEncoding::Pcm, rate: 16_000 }
    }

    /// G.711 mu-law audio at 8 kHz.
    pub const fn pcmu() -> Self {
        Self { encoding: LiveAudioEncoding::Pcmu, rate: 8_000 }
    }

    /// G.711 A-law audio at 8 kHz.
    pub const fn pcma() -> Self {
        Self { encoding: LiveAudioEncoding::Pcma, rate: 8_000 }
    }
}

/// Live audio settings.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LiveAudioConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<LiveAudioFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<LiveAudioOutputConfig>,
}

/// Live audio output settings.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LiveAudioOutputConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voice: Option<LiveVoice>,
}

/// Delegation target and configuration.
#[derive(Debug, Clone)]
pub enum Delegation {
    /// The application handles delegated work.
    Client,
    /// The Live service delegates to the Responses API.
    Responses(ResponsesDelegationConfig),
}

impl Delegation {
    /// Creates Responses API delegation settings.
    pub fn responses(config: ResponsesDelegationConfig) -> Self {
        Self::Responses(config)
    }
}

impl Serialize for Delegation {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Client => {
                let mut state = serializer.serialize_struct("Delegation", 1)?;
                state.serialize_field("type", "client")?;
                state.end()
            }
            Self::Responses(config) => {
                let mut state = serializer.serialize_struct("Delegation", 2)?;
                state.serialize_field("type", "responses")?;
                state.serialize_field("responses", config)?;
                state.end()
            }
        }
    }
}

/// Responses API backend configuration used by Live delegation.
#[derive(Debug, Clone, Serialize)]
pub struct ResponsesDelegationConfig {
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ResponsesReasoningConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<ResponsesTextConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<serde_json::Value>,
}

/// Reasoning settings for Responses delegation.
#[derive(Debug, Clone, Default, Serialize)]
pub struct ResponsesReasoningConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<ReasoningEffort>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<ReasoningSummary>,
}

/// Text settings for Responses delegation.
#[derive(Debug, Clone, Default, Serialize)]
pub struct ResponsesTextConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verbosity: Option<String>,
}

/// Configuration carried by `session.start`.
#[derive(Debug, Clone, Default, Serialize)]
pub struct LiveSessionConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<LiveModel>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio: Option<LiveAudioConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delegation: Option<Delegation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<Vec<serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
}

/// Mutable portion of a Live session configuration.
#[derive(Debug, Clone, Serialize)]
pub struct LiveSessionUpdate {
    pub delegation: Delegation,
}
