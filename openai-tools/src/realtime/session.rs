//! Session configuration types for the Realtime API.

use crate::common::parameters::{Name, ParameterProperty, Parameters};
use crate::common::tool::Tool;
use serde::{Deserialize, Serialize};

use super::audio::{AudioFormat, InputAudioNoiseReduction, InputAudioTranscription, Voice};
use super::vad::TurnDetection;

/// Tool definition for the Realtime API.
///
/// The Realtime API uses a flattened tool format, unlike the Chat Completions API
/// which nests the function details under a `function` key.
///
/// # Example
///
/// ```rust
/// use openai_tools::realtime::RealtimeTool;
/// use openai_tools::common::parameters::ParameterProperty;
///
/// let tool = RealtimeTool::function(
///     "get_weather",
///     "Get the current weather for a location",
///     vec![("location", ParameterProperty::from_string("The city name"))],
/// );
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeTool {
    /// The type of tool (always "function" for function calling).
    #[serde(rename = "type")]
    pub type_name: String,

    /// The name of the function.
    pub name: String,

    /// A description of what the function does.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// The parameters the function accepts.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Parameters>,
}

impl RealtimeTool {
    /// Create a new function tool.
    pub fn function<T, U, V>(name: T, description: U, parameters: Vec<(V, ParameterProperty)>) -> Self
    where
        T: Into<String>,
        U: Into<String>,
        V: AsRef<str>,
    {
        let params: Vec<(Name, ParameterProperty)> = parameters.into_iter().map(|(k, v)| (k.as_ref().to_string(), v)).collect();

        Self {
            type_name: "function".to_string(),
            name: name.into(),
            description: Some(description.into()),
            parameters: Some(Parameters::new(params, None)),
        }
    }
}

impl From<Tool> for RealtimeTool {
    /// Convert a Chat API tool to a Realtime API tool.
    fn from(tool: Tool) -> Self {
        if let Some(func) = tool.function {
            Self { type_name: "function".to_string(), name: func.name, description: func.description, parameters: func.parameters }
        } else {
            // Fallback for tools without function definition
            Self { type_name: tool.type_name, name: tool.name.unwrap_or_default(), description: None, parameters: tool.parameters }
        }
    }
}

/// Session modality - what types of input/output are supported.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum Modality {
    /// Text input/output
    Text,
    /// Audio input/output
    Audio,
}

/// Session configuration sent in session.update events.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionConfig {
    /// Supported modalities for this session.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modalities: Option<Vec<Modality>>,

    /// System instructions for the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,

    /// Voice for audio output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voice: Option<Voice>,

    /// Format for input audio.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_audio_format: Option<AudioFormat>,

    /// Format for output audio.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_audio_format: Option<AudioFormat>,

    /// Configuration for input audio transcription.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_audio_transcription: Option<InputAudioTranscription>,

    /// Noise reduction configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_audio_noise_reduction: Option<InputAudioNoiseReduction>,

    /// Turn detection configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turn_detection: Option<TurnDetection>,

    /// Available tools for function calling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<RealtimeTool>>,

    /// How to select tools.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,

    /// Sampling temperature (0.6 to 1.2).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Maximum tokens in a response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_response_output_tokens: Option<MaxTokens>,
}

impl SessionConfig {
    /// Create a new empty session configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the modalities.
    pub fn with_modalities(mut self, modalities: Vec<Modality>) -> Self {
        self.modalities = Some(modalities);
        self
    }

    /// Set the instructions.
    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    /// Set the voice.
    pub fn with_voice(mut self, voice: Voice) -> Self {
        self.voice = Some(voice);
        self
    }

    /// Set the input audio format.
    pub fn with_input_audio_format(mut self, format: AudioFormat) -> Self {
        self.input_audio_format = Some(format);
        self
    }

    /// Set the output audio format.
    pub fn with_output_audio_format(mut self, format: AudioFormat) -> Self {
        self.output_audio_format = Some(format);
        self
    }

    /// Set the transcription configuration.
    pub fn with_transcription(mut self, config: InputAudioTranscription) -> Self {
        self.input_audio_transcription = Some(config);
        self
    }

    /// Set the turn detection configuration.
    pub fn with_turn_detection(mut self, config: TurnDetection) -> Self {
        self.turn_detection = Some(config);
        self
    }

    /// Set the available tools.
    ///
    /// Accepts `Tool` from the common module and converts to `RealtimeTool`.
    pub fn with_tools(mut self, tools: Vec<Tool>) -> Self {
        self.tools = Some(tools.into_iter().map(RealtimeTool::from).collect());
        self
    }

    /// Set the available realtime tools directly.
    pub fn with_realtime_tools(mut self, tools: Vec<RealtimeTool>) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Set the tool choice.
    pub fn with_tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = Some(choice);
        self
    }

    /// Set the temperature.
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    /// Set the maximum response tokens.
    pub fn with_max_tokens(mut self, max: MaxTokens) -> Self {
        self.max_response_output_tokens = Some(max);
        self
    }
}

/// Maximum tokens configuration.
#[derive(Debug, Clone)]
pub enum MaxTokens {
    /// Specific token count limit.
    Count(u32),
    /// No limit (infinite).
    Infinite,
}

impl serde::Serialize for MaxTokens {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            MaxTokens::Count(n) => serializer.serialize_u32(*n),
            MaxTokens::Infinite => serializer.serialize_str("inf"),
        }
    }
}

impl<'de> serde::Deserialize<'de> for MaxTokens {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::{self, Visitor};

        struct MaxTokensVisitor;

        impl<'de> Visitor<'de> for MaxTokensVisitor {
            type Value = MaxTokens;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a positive integer or \"inf\"")
            }

            fn visit_u64<E>(self, value: u64) -> std::result::Result<MaxTokens, E>
            where
                E: de::Error,
            {
                Ok(MaxTokens::Count(value as u32))
            }

            fn visit_str<E>(self, value: &str) -> std::result::Result<MaxTokens, E>
            where
                E: de::Error,
            {
                if value == "inf" {
                    Ok(MaxTokens::Infinite)
                } else {
                    Err(de::Error::custom(format!("unknown value: {}", value)))
                }
            }
        }

        deserializer.deserialize_any(MaxTokensVisitor)
    }
}

impl From<u32> for MaxTokens {
    fn from(count: u32) -> Self {
        Self::Count(count)
    }
}

/// How to select tools for function calling.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    /// Simple string-based choices: "auto", "none", "required"
    Simple(SimpleToolChoice),
    /// Force a specific function by name
    Function(NamedToolChoice),
}

/// Simple tool choice options.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SimpleToolChoice {
    /// Model decides whether to use tools.
    Auto,
    /// Never use tools.
    None,
    /// Must use a tool.
    Required,
}

impl Default for ToolChoice {
    fn default() -> Self {
        Self::Simple(SimpleToolChoice::Auto)
    }
}

impl ToolChoice {
    /// Model decides whether to use tools.
    pub fn auto() -> Self {
        Self::Simple(SimpleToolChoice::Auto)
    }

    /// Never use tools.
    pub fn none() -> Self {
        Self::Simple(SimpleToolChoice::None)
    }

    /// Must use a tool.
    pub fn required() -> Self {
        Self::Simple(SimpleToolChoice::Required)
    }

    /// Force a specific function by name.
    pub fn function(name: impl Into<String>) -> Self {
        Self::Function(NamedToolChoice { type_name: "function".to_string(), function: NamedFunction { name: name.into() } })
    }
}

/// Named tool choice for forcing a specific function.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamedToolChoice {
    #[serde(rename = "type")]
    pub type_name: String,
    pub function: NamedFunction,
}

/// Function name for named tool choice.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamedFunction {
    pub name: String,
}

/// Response creation configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResponseCreateConfig {
    /// Modalities for this response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modalities: Option<Vec<Modality>>,

    /// Instructions for this response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,

    /// Voice for this response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voice: Option<Voice>,

    /// Output audio format.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_audio_format: Option<AudioFormat>,

    /// Tools available for this response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<RealtimeTool>>,

    /// Tool choice for this response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,

    /// Temperature for this response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Maximum output tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<MaxTokens>,

    /// Whether to include in conversation history.
    /// Set to "none" to exclude.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation: Option<String>,

    /// Metadata for this response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

impl ResponseCreateConfig {
    /// Create a new empty response configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the modalities.
    pub fn with_modalities(mut self, modalities: Vec<Modality>) -> Self {
        self.modalities = Some(modalities);
        self
    }

    /// Set the instructions.
    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    /// Set the voice.
    pub fn with_voice(mut self, voice: Voice) -> Self {
        self.voice = Some(voice);
        self
    }

    /// Exclude this response from conversation history.
    pub fn out_of_band(mut self) -> Self {
        self.conversation = Some("none".to_string());
        self
    }
}
