//! OpenAI Model Types
//!
//! This module provides strongly-typed enums for specifying OpenAI models
//! across different APIs. Using enums instead of strings provides:
//!
//! - Compile-time validation of model names
//! - IDE autocompletion support
//! - Prevention of typos in model names
//! - Clear documentation of available models
//!
//! # Model Categories
//!
//! - [`ChatModel`]: Models for Chat Completions and Responses APIs
//! - [`EmbeddingModel`]: Models for text embeddings
//! - [`RealtimeModel`]: Models for real-time audio/text interactions
//! - [`FineTuningModel`]: Base models that can be fine-tuned
//!
//! # Example
//!
//! ```rust,no_run
//! use openai_tools::common::models::{ChatModel, EmbeddingModel};
//! use openai_tools::chat::request::ChatCompletion;
//! use openai_tools::embedding::request::Embedding;
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Using ChatModel enum
//! let mut chat = ChatCompletion::new();
//! chat.model(ChatModel::Gpt4oMini);
//!
//! // Using EmbeddingModel enum
//! let mut embedding = Embedding::new()?;
//! embedding.model(EmbeddingModel::TextEmbedding3Small);
//! # Ok(())
//! # }
//! ```
//!
//! # References
//!
//! - [OpenAI Models Documentation](https://platform.openai.com/docs/models)
//! - [Model Deprecations](https://platform.openai.com/docs/deprecations)

use serde::{Deserialize, Serialize};

// ============================================================================
// Parameter Restriction Types
// ============================================================================

/// Defines how a parameter is restricted for a model.
///
/// This enum is used to specify whether a parameter can accept any value,
/// only a fixed value, or is not supported at all.
#[derive(Debug, Clone, PartialEq)]
pub enum ParameterRestriction {
    /// Parameter accepts any value within its valid range
    Any,
    /// Parameter only supports a specific fixed value
    FixedValue(f64),
    /// Parameter is not supported by this model
    NotSupported,
}

/// Parameter support information for a model.
///
/// This struct provides detailed information about which parameters are
/// supported by a model and any restrictions that apply.
///
/// # Example
///
/// ```rust
/// use openai_tools::common::models::{ChatModel, ParameterRestriction};
///
/// let model = ChatModel::O3Mini;
/// let support = model.parameter_support();
///
/// // Reasoning models only support temperature = 1.0
/// assert_eq!(support.temperature, ParameterRestriction::FixedValue(1.0));
///
/// // Reasoning models don't support logprobs
/// assert!(!support.logprobs);
/// ```
#[derive(Debug, Clone)]
pub struct ParameterSupport {
    /// Temperature parameter restriction (Chat & Responses API)
    pub temperature: ParameterRestriction,
    /// Frequency penalty parameter restriction (Chat API only)
    pub frequency_penalty: ParameterRestriction,
    /// Presence penalty parameter restriction (Chat API only)
    pub presence_penalty: ParameterRestriction,
    /// Whether logprobs parameter is supported (Chat API only)
    pub logprobs: bool,
    /// Whether top_logprobs parameter is supported (Chat & Responses API)
    pub top_logprobs: bool,
    /// Whether logit_bias parameter is supported (Chat API only)
    pub logit_bias: bool,
    /// Whether n > 1 (multiple completions) is supported (Chat API only)
    pub n_multiple: bool,
    /// Top P parameter restriction (Responses API only)
    pub top_p: ParameterRestriction,
    /// Whether reasoning parameter is supported (Responses API only, reasoning models)
    pub reasoning: bool,
}

impl ParameterSupport {
    /// Creates parameter support info for standard (non-reasoning) models.
    ///
    /// Standard models support all parameters with full range.
    pub fn standard_model() -> Self {
        Self {
            temperature: ParameterRestriction::Any,
            frequency_penalty: ParameterRestriction::Any,
            presence_penalty: ParameterRestriction::Any,
            logprobs: true,
            top_logprobs: true,
            logit_bias: true,
            n_multiple: true,
            top_p: ParameterRestriction::Any,
            reasoning: false,
        }
    }

    /// Creates parameter support info for reasoning models (GPT-5, o-series).
    ///
    /// Reasoning models have restricted parameter support:
    /// - temperature: only 1.0
    /// - top_p: only 1.0
    /// - frequency_penalty: only 0
    /// - presence_penalty: only 0
    /// - logprobs, top_logprobs, logit_bias: not supported
    /// - n: only 1
    /// - reasoning: supported
    pub fn reasoning_model() -> Self {
        Self {
            temperature: ParameterRestriction::FixedValue(1.0),
            frequency_penalty: ParameterRestriction::FixedValue(0.0),
            presence_penalty: ParameterRestriction::FixedValue(0.0),
            logprobs: false,
            top_logprobs: false,
            logit_bias: false,
            n_multiple: false,
            top_p: ParameterRestriction::FixedValue(1.0),
            reasoning: true,
        }
    }
}

/// Models available for Chat Completions and Responses APIs.
///
/// This enum covers all models that can be used with the Chat Completions API
/// (`/v1/chat/completions`) and the Responses API (`/v1/responses`).
///
/// # Model Categories
///
/// ## GPT-5 Series (Latest Flagship)
/// - [`Gpt5_2`]: GPT-5.2 Thinking - flagship model for coding and agentic tasks
/// - [`Gpt5_2ChatLatest`]: GPT-5.2 Instant - fast workhorse for everyday work
/// - [`Gpt5_2Pro`]: GPT-5.2 Pro - smartest for difficult questions (Responses API only)
/// - [`Gpt5_1`]: GPT-5.1 - configurable reasoning and non-reasoning
/// - [`Gpt5_1CodexMax`]: GPT-5.1 Codex Max - powers Codex CLI
/// - [`Gpt5Mini`]: GPT-5 Mini - smaller, faster variant
///
/// ## GPT-4.1 Series
/// - [`Gpt4_1`]: 1M context window flagship
/// - [`Gpt4_1Mini`]: Balanced performance and cost
/// - [`Gpt4_1Nano`]: Fastest and most cost-efficient
///
/// ## GPT-4o Series
/// - [`Gpt4o`]: High-intelligence flagship model
/// - [`Gpt4oMini`]: Cost-effective GPT-4o variant
/// - [`Gpt4oAudioPreview`]: Audio-capable GPT-4o
///
/// ## Reasoning Models (o-series)
/// - [`O1`], [`O1Pro`]: Full reasoning models
/// - [`O3`], [`O3Mini`]: Latest reasoning models
/// - [`O4Mini`]: Fast, cost-efficient reasoning
///
/// # Reasoning Model Restrictions
///
/// Reasoning models (GPT-5 series, o1, o3, o4 series) have parameter restrictions:
/// - `temperature`: Only 1.0 supported
/// - `top_p`: Only 1.0 supported
/// - `frequency_penalty`: Only 0 supported
/// - `presence_penalty`: Only 0 supported
///
/// GPT-5 models support `reasoning.effort` parameter:
/// - `none`: No reasoning (GPT-5.1 default)
/// - `minimal`: Very few reasoning tokens
/// - `low`, `medium`, `high`: Increasing reasoning depth
/// - `xhigh`: Maximum reasoning (GPT-5.2 Pro, GPT-5.1 Codex Max)
///
/// # Example
///
/// ```rust
/// use openai_tools::common::models::ChatModel;
///
/// // Check if a model is a reasoning model
/// let model = ChatModel::O3Mini;
/// assert!(model.is_reasoning_model());
///
/// // GPT-5 models are also reasoning models
/// let gpt5 = ChatModel::Gpt5_2;
/// assert!(gpt5.is_reasoning_model());
///
/// // Get the API model ID string
/// assert_eq!(model.as_str(), "o3-mini");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum ChatModel {
    // === GPT-5 Series (Latest Flagship) ===
    /// GPT-5.2 Thinking - Flagship model for coding and agentic tasks
    ///
    /// - Context: 128K tokens (256K with thinking)
    /// - Supports: reasoning.effort (none, minimal, low, medium, high, xhigh)
    /// - Supports: verbosity parameter (low, medium, high)
    #[serde(rename = "gpt-5.2")]
    Gpt5_2,

    /// GPT-5.2 Instant - Fast workhorse for everyday work
    ///
    /// Points to the GPT-5.2 snapshot used in ChatGPT
    #[serde(rename = "gpt-5.2-chat-latest")]
    Gpt5_2ChatLatest,

    /// GPT-5.2 Pro - Smartest for difficult questions
    ///
    /// - Available in Responses API only
    /// - Supports: xhigh reasoning effort
    #[serde(rename = "gpt-5.2-pro")]
    Gpt5_2Pro,

    /// GPT-5.1 - Configurable reasoning and non-reasoning
    ///
    /// - Defaults to no reasoning (effort: none)
    /// - Supports: reasoning.effort (none, low, medium, high)
    #[serde(rename = "gpt-5.1")]
    Gpt5_1,

    /// GPT-5.1 Chat Latest - Chat-optimized GPT-5.1
    #[serde(rename = "gpt-5.1-chat-latest")]
    Gpt5_1ChatLatest,

    /// GPT-5.1 Codex Max - Powers Codex and Codex CLI
    ///
    /// - Available in Responses API only
    /// - Supports: reasoning.effort (none, medium, high, xhigh)
    #[serde(rename = "gpt-5.1-codex-max")]
    Gpt5_1CodexMax,

    /// GPT-5 Mini - Smaller, faster GPT-5 variant
    #[serde(rename = "gpt-5-mini")]
    Gpt5Mini,

    /// GPT-5 Nano - Fastest, most cost-efficient GPT-5 variant
    #[serde(rename = "gpt-5-nano")]
    Gpt5Nano,

    // === GPT-4.1 Series ===
    /// GPT-4.1 - Smartest non-reasoning model with 1M token context
    #[serde(rename = "gpt-4.1")]
    Gpt4_1,

    /// GPT-4.1 Mini - Balanced performance and cost
    #[serde(rename = "gpt-4.1-mini")]
    Gpt4_1Mini,

    /// GPT-4.1 Nano - Fastest and most cost-efficient
    #[serde(rename = "gpt-4.1-nano")]
    Gpt4_1Nano,

    // === GPT-4o Series ===
    /// GPT-4o - High-intelligence flagship model (multimodal)
    #[serde(rename = "gpt-4o")]
    Gpt4o,

    /// GPT-4o Mini - Cost-effective GPT-4o variant
    #[serde(rename = "gpt-4o-mini")]
    #[default]
    Gpt4oMini,

    /// GPT-4o Audio Preview - Audio-capable GPT-4o
    #[serde(rename = "gpt-4o-audio-preview")]
    Gpt4oAudioPreview,

    // === GPT-4 Series ===
    /// GPT-4 Turbo - High capability with faster responses
    #[serde(rename = "gpt-4-turbo")]
    Gpt4Turbo,

    /// GPT-4 - Original GPT-4 model
    #[serde(rename = "gpt-4")]
    Gpt4,

    // === GPT-3.5 Series ===
    /// GPT-3.5 Turbo - Fast and cost-effective
    #[serde(rename = "gpt-3.5-turbo")]
    Gpt3_5Turbo,

    // === Reasoning Models (o-series) ===
    /// O1 - Full reasoning model for complex tasks
    #[serde(rename = "o1")]
    O1,

    /// O1 Pro - O1 with more compute for complex problems
    #[serde(rename = "o1-pro")]
    O1Pro,

    /// O3 - Latest full reasoning model
    #[serde(rename = "o3")]
    O3,

    /// O3 Mini - Smaller, faster reasoning model
    #[serde(rename = "o3-mini")]
    O3Mini,

    /// O4 Mini - Fast, cost-efficient reasoning model
    #[serde(rename = "o4-mini")]
    O4Mini,

    // Gemini
    /// Gemini 3.1 Flash Lite - Fast, cost-effective Gemini variant
    #[serde(rename = "gemini-3.1-flash-lite")]
    Gemini3_1FlashLite,

    // === Custom Model ===
    /// Custom model ID for fine-tuned models or new models not yet in enum
    #[serde(untagged)]
    Custom(String),
}

impl ChatModel {
    /// Returns the model identifier string for API requests.
    ///
    /// # Example
    ///
    /// ```rust
    /// use openai_tools::common::models::ChatModel;
    ///
    /// assert_eq!(ChatModel::Gpt4oMini.as_str(), "gpt-4o-mini");
    /// assert_eq!(ChatModel::O3Mini.as_str(), "o3-mini");
    /// assert_eq!(ChatModel::Gpt5_2.as_str(), "gpt-5.2");
    /// ```
    pub fn as_str(&self) -> &str {
        match self {
            // GPT-5 Series
            Self::Gpt5_2 => "gpt-5.2",
            Self::Gpt5_2ChatLatest => "gpt-5.2-chat-latest",
            Self::Gpt5_2Pro => "gpt-5.2-pro",
            Self::Gpt5_1 => "gpt-5.1",
            Self::Gpt5_1ChatLatest => "gpt-5.1-chat-latest",
            Self::Gpt5_1CodexMax => "gpt-5.1-codex-max",
            Self::Gpt5Mini => "gpt-5-mini",
            Self::Gpt5Nano => "gpt-5-nano",
            // GPT-4.1 Series
            Self::Gpt4_1 => "gpt-4.1",
            Self::Gpt4_1Mini => "gpt-4.1-mini",
            Self::Gpt4_1Nano => "gpt-4.1-nano",
            // GPT-4o Series
            Self::Gpt4o => "gpt-4o",
            Self::Gpt4oMini => "gpt-4o-mini",
            Self::Gpt4oAudioPreview => "gpt-4o-audio-preview",
            // GPT-4 Series
            Self::Gpt4Turbo => "gpt-4-turbo",
            Self::Gpt4 => "gpt-4",
            // GPT-3.5 Series
            Self::Gpt3_5Turbo => "gpt-3.5-turbo",
            // Reasoning Models
            Self::O1 => "o1",
            Self::O1Pro => "o1-pro",
            Self::O3 => "o3",
            Self::O3Mini => "o3-mini",
            Self::O4Mini => "o4-mini",
            // Gemini
            Self::Gemini3_1FlashLite => "gemini-3.1-flash-lite",
            // Custom
            Self::Custom(s) => s.as_str(),
        }
    }

    /// Checks if this is a reasoning model with parameter restrictions.
    ///
    /// Reasoning models (GPT-5 series, o1, o3, o4 series) only support:
    /// - `temperature = 1.0`
    /// - `top_p = 1.0`
    /// - `frequency_penalty = 0`
    /// - `presence_penalty = 0`
    ///
    /// # Example
    ///
    /// ```rust
    /// use openai_tools::common::models::ChatModel;
    ///
    /// assert!(ChatModel::O3Mini.is_reasoning_model());
    /// assert!(ChatModel::Gpt5_2.is_reasoning_model());
    /// assert!(!ChatModel::Gpt4oMini.is_reasoning_model());
    /// assert!(!ChatModel::Gpt4_1.is_reasoning_model());
    /// ```
    pub fn is_reasoning_model(&self) -> bool {
        matches!(
            self,
            // GPT-5 series are reasoning models
            Self::Gpt5_2 | Self::Gpt5_2ChatLatest | Self::Gpt5_2Pro |
            Self::Gpt5_1 | Self::Gpt5_1ChatLatest | Self::Gpt5_1CodexMax |
            Self::Gpt5Mini |
            // O-series reasoning models
            Self::O1 | Self::O1Pro | Self::O3 | Self::O3Mini | Self::O4Mini
        ) || matches!(
            self,
            Self::Custom(s) if s.starts_with("gpt-5") || s.starts_with("o1") || s.starts_with("o3") || s.starts_with("o4")
        )
    }

    /// Returns parameter support information for this model.
    ///
    /// This method provides detailed information about which parameters
    /// are supported by the model and any restrictions that apply.
    ///
    /// # Example
    ///
    /// ```rust
    /// use openai_tools::common::models::{ChatModel, ParameterRestriction};
    ///
    /// // Standard model supports all parameters
    /// let standard = ChatModel::Gpt4oMini;
    /// let support = standard.parameter_support();
    /// assert_eq!(support.temperature, ParameterRestriction::Any);
    /// assert!(support.logprobs);
    ///
    /// // Reasoning model has restrictions
    /// let reasoning = ChatModel::O3Mini;
    /// let support = reasoning.parameter_support();
    /// assert_eq!(support.temperature, ParameterRestriction::FixedValue(1.0));
    /// assert!(!support.logprobs);
    /// assert!(support.reasoning);
    /// ```
    pub fn parameter_support(&self) -> ParameterSupport {
        if self.is_reasoning_model() {
            ParameterSupport::reasoning_model()
        } else {
            ParameterSupport::standard_model()
        }
    }

    /// Creates a custom model from a string.
    ///
    /// Use this for fine-tuned models or new models not yet in the enum.
    ///
    /// # Example
    ///
    /// ```rust
    /// use openai_tools::common::models::ChatModel;
    ///
    /// let model = ChatModel::custom("ft:gpt-4o-mini:my-org::abc123");
    /// assert_eq!(model.as_str(), "ft:gpt-4o-mini:my-org::abc123");
    /// ```
    pub fn custom(model_id: impl Into<String>) -> Self {
        Self::Custom(model_id.into())
    }
}

impl std::fmt::Display for ChatModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl From<&str> for ChatModel {
    fn from(s: &str) -> Self {
        match s {
            // GPT-5 Series
            "gpt-5.2" => Self::Gpt5_2,
            "gpt-5.2-chat-latest" => Self::Gpt5_2ChatLatest,
            "gpt-5.2-pro" => Self::Gpt5_2Pro,
            "gpt-5.1" => Self::Gpt5_1,
            "gpt-5.1-chat-latest" => Self::Gpt5_1ChatLatest,
            "gpt-5.1-codex-max" => Self::Gpt5_1CodexMax,
            "gpt-5-mini" => Self::Gpt5Mini,
            // GPT-4.1 Series
            "gpt-4.1" => Self::Gpt4_1,
            "gpt-4.1-mini" => Self::Gpt4_1Mini,
            "gpt-4.1-nano" => Self::Gpt4_1Nano,
            // GPT-4o Series
            "gpt-4o" => Self::Gpt4o,
            "gpt-4o-mini" => Self::Gpt4oMini,
            "gpt-4o-audio-preview" => Self::Gpt4oAudioPreview,
            // GPT-4 Series
            "gpt-4-turbo" => Self::Gpt4Turbo,
            "gpt-4" => Self::Gpt4,
            // GPT-3.5 Series
            "gpt-3.5-turbo" => Self::Gpt3_5Turbo,
            // Reasoning Models
            "o1" => Self::O1,
            "o1-pro" => Self::O1Pro,
            "o3" => Self::O3,
            "o3-mini" => Self::O3Mini,
            "o4-mini" => Self::O4Mini,
            // Custom
            other => Self::Custom(other.to_string()),
        }
    }
}

impl From<String> for ChatModel {
    fn from(s: String) -> Self {
        Self::from(s.as_str())
    }
}

// ============================================================================
// Embedding Models
// ============================================================================

/// Models available for the Embeddings API.
///
/// This enum covers all models that can be used with the Embeddings API
/// (`/v1/embeddings`) for converting text into vector representations.
///
/// # Available Models
///
/// - [`TextEmbedding3Small`]: Improved, performant model (default)
/// - [`TextEmbedding3Large`]: Most capable model for English and non-English
/// - [`TextEmbeddingAda002`]: Legacy model (not recommended for new projects)
///
/// # Example
///
/// ```rust
/// use openai_tools::common::models::EmbeddingModel;
///
/// let model = EmbeddingModel::TextEmbedding3Small;
/// assert_eq!(model.as_str(), "text-embedding-3-small");
/// assert_eq!(model.dimensions(), 1536);
/// ```
///
/// # Reference
///
/// See [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum EmbeddingModel {
    /// text-embedding-3-small - Improved, more performant embedding model
    ///
    /// - Dimensions: 1536
    /// - Max input: 8191 tokens
    /// - Recommended for most use cases
    #[serde(rename = "text-embedding-3-small")]
    #[default]
    TextEmbedding3Small,

    /// text-embedding-3-large - Most capable embedding model
    ///
    /// - Dimensions: 3072
    /// - Max input: 8191 tokens
    /// - Best for high-accuracy tasks
    #[serde(rename = "text-embedding-3-large")]
    TextEmbedding3Large,

    /// text-embedding-ada-002 - Legacy embedding model
    ///
    /// - Dimensions: 1536
    /// - Max input: 8191 tokens
    /// - Not recommended for new projects
    #[serde(rename = "text-embedding-ada-002")]
    TextEmbeddingAda002,
}

impl EmbeddingModel {
    /// Returns the model identifier string for API requests.
    pub fn as_str(&self) -> &str {
        match self {
            Self::TextEmbedding3Small => "text-embedding-3-small",
            Self::TextEmbedding3Large => "text-embedding-3-large",
            Self::TextEmbeddingAda002 => "text-embedding-ada-002",
        }
    }

    /// Returns the default output dimensions for this model.
    ///
    /// Note: For `text-embedding-3-*` models, you can request fewer dimensions
    /// via the API's `dimensions` parameter. This returns the default/maximum.
    pub fn dimensions(&self) -> usize {
        match self {
            Self::TextEmbedding3Small => 1536,
            Self::TextEmbedding3Large => 3072,
            Self::TextEmbeddingAda002 => 1536,
        }
    }
}

impl std::fmt::Display for EmbeddingModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl From<&str> for EmbeddingModel {
    fn from(s: &str) -> Self {
        match s {
            "text-embedding-3-small" => Self::TextEmbedding3Small,
            "text-embedding-3-large" => Self::TextEmbedding3Large,
            "text-embedding-ada-002" => Self::TextEmbeddingAda002,
            _ => Self::TextEmbedding3Small, // Default fallback
        }
    }
}

// ============================================================================
// Realtime Models
// ============================================================================

/// Models available for the Realtime API.
///
/// This enum covers all models that can be used with the Realtime API
/// for real-time audio and text interactions via WebSocket.
///
/// # Available Models
///
/// - [`GptRealtime_2025_08_28`]: GPT Realtime model (default)
///
/// # Example
///
/// ```rust
/// use openai_tools::common::models::RealtimeModel;
///
/// let model = RealtimeModel::GptRealtime_2025_08_28;
/// assert_eq!(model.as_str(), "gpt-realtime-2025-08-28");
/// ```
///
/// # Reference
///
/// See [OpenAI Realtime API Documentation](https://platform.openai.com/docs/guides/realtime)
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum RealtimeModel {
    /// gpt-realtime-2025-08-28 - GPT Realtime model
    #[serde(rename = "gpt-realtime-2025-08-28")]
    #[default]
    GptRealtime_2025_08_28,

    /// Custom model ID for new models not yet in enum
    #[serde(untagged)]
    Custom(String),
}

impl RealtimeModel {
    /// Returns the model identifier string for API requests.
    pub fn as_str(&self) -> &str {
        match self {
            Self::GptRealtime_2025_08_28 => "gpt-realtime-2025-08-28",
            Self::Custom(s) => s.as_str(),
        }
    }

    /// Creates a custom model from a string.
    pub fn custom(model_id: impl Into<String>) -> Self {
        Self::Custom(model_id.into())
    }
}

impl std::fmt::Display for RealtimeModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl From<&str> for RealtimeModel {
    fn from(s: &str) -> Self {
        match s {
            "gpt-realtime-2025-08-28" => Self::GptRealtime_2025_08_28,
            other => Self::Custom(other.to_string()),
        }
    }
}

// ============================================================================
// Fine-tuning Models
// ============================================================================

/// Base models that can be used for fine-tuning.
///
/// This enum covers all models that can be fine-tuned via the Fine-tuning API
/// (`/v1/fine_tuning/jobs`). Note that fine-tuning requires specific dated
/// model versions.
///
/// # Available Models
///
/// ## GPT-4.1 Series (Latest)
/// - [`Gpt41_2025_04_14`]: GPT-4.1 for fine-tuning
/// - [`Gpt41Mini_2025_04_14`]: GPT-4.1 Mini for fine-tuning
/// - [`Gpt41Nano_2025_04_14`]: GPT-4.1 Nano for fine-tuning
///
/// ## GPT-4o Series
/// - [`Gpt4oMini_2024_07_18`]: GPT-4o Mini for fine-tuning
/// - [`Gpt4o_2024_08_06`]: GPT-4o for fine-tuning
///
/// ## GPT-4 Series
/// - [`Gpt4_0613`]: GPT-4 for fine-tuning
///
/// ## GPT-3.5 Series
/// - [`Gpt35Turbo_0125`]: GPT-3.5 Turbo for fine-tuning
///
/// # Example
///
/// ```rust
/// use openai_tools::common::models::FineTuningModel;
///
/// let model = FineTuningModel::Gpt4oMini_2024_07_18;
/// assert_eq!(model.as_str(), "gpt-4o-mini-2024-07-18");
/// ```
///
/// # Reference
///
/// See [OpenAI Fine-tuning Guide](https://platform.openai.com/docs/guides/fine-tuning)
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum FineTuningModel {
    // === GPT-4.1 Series ===
    /// gpt-4.1-2025-04-14 - GPT-4.1 for fine-tuning
    #[serde(rename = "gpt-4.1-2025-04-14")]
    Gpt41_2025_04_14,

    /// gpt-4.1-mini-2025-04-14 - GPT-4.1 Mini for fine-tuning
    #[serde(rename = "gpt-4.1-mini-2025-04-14")]
    Gpt41Mini_2025_04_14,

    /// gpt-4.1-nano-2025-04-14 - GPT-4.1 Nano for fine-tuning
    #[serde(rename = "gpt-4.1-nano-2025-04-14")]
    Gpt41Nano_2025_04_14,

    // === GPT-4o Series ===
    /// gpt-4o-mini-2024-07-18 - GPT-4o Mini for fine-tuning
    #[serde(rename = "gpt-4o-mini-2024-07-18")]
    #[default]
    Gpt4oMini_2024_07_18,

    /// gpt-4o-2024-08-06 - GPT-4o for fine-tuning
    #[serde(rename = "gpt-4o-2024-08-06")]
    Gpt4o_2024_08_06,

    // === GPT-4 Series ===
    /// gpt-4-0613 - GPT-4 for fine-tuning
    #[serde(rename = "gpt-4-0613")]
    Gpt4_0613,

    // === GPT-3.5 Series ===
    /// gpt-3.5-turbo-0125 - GPT-3.5 Turbo for fine-tuning
    #[serde(rename = "gpt-3.5-turbo-0125")]
    Gpt35Turbo_0125,

    /// gpt-3.5-turbo-1106 - GPT-3.5 Turbo (older version)
    #[serde(rename = "gpt-3.5-turbo-1106")]
    Gpt35Turbo_1106,

    /// gpt-3.5-turbo-0613 - GPT-3.5 Turbo (legacy)
    #[serde(rename = "gpt-3.5-turbo-0613")]
    Gpt35Turbo_0613,
}

impl FineTuningModel {
    /// Returns the model identifier string for API requests.
    pub fn as_str(&self) -> &str {
        match self {
            // GPT-4.1 Series
            Self::Gpt41_2025_04_14 => "gpt-4.1-2025-04-14",
            Self::Gpt41Mini_2025_04_14 => "gpt-4.1-mini-2025-04-14",
            Self::Gpt41Nano_2025_04_14 => "gpt-4.1-nano-2025-04-14",
            // GPT-4o Series
            Self::Gpt4oMini_2024_07_18 => "gpt-4o-mini-2024-07-18",
            Self::Gpt4o_2024_08_06 => "gpt-4o-2024-08-06",
            // GPT-4 Series
            Self::Gpt4_0613 => "gpt-4-0613",
            // GPT-3.5 Series
            Self::Gpt35Turbo_0125 => "gpt-3.5-turbo-0125",
            Self::Gpt35Turbo_1106 => "gpt-3.5-turbo-1106",
            Self::Gpt35Turbo_0613 => "gpt-3.5-turbo-0613",
        }
    }
}

impl std::fmt::Display for FineTuningModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_model_as_str() {
        assert_eq!(ChatModel::Gpt4oMini.as_str(), "gpt-4o-mini");
        assert_eq!(ChatModel::O3Mini.as_str(), "o3-mini");
        assert_eq!(ChatModel::Gpt4_1.as_str(), "gpt-4.1");
        // GPT-5 models
        assert_eq!(ChatModel::Gpt5_2.as_str(), "gpt-5.2");
        assert_eq!(ChatModel::Gpt5_2ChatLatest.as_str(), "gpt-5.2-chat-latest");
        assert_eq!(ChatModel::Gpt5_2Pro.as_str(), "gpt-5.2-pro");
        assert_eq!(ChatModel::Gpt5_1.as_str(), "gpt-5.1");
        assert_eq!(ChatModel::Gpt5_1CodexMax.as_str(), "gpt-5.1-codex-max");
        assert_eq!(ChatModel::Gpt5Mini.as_str(), "gpt-5-mini");
    }

    #[test]
    fn test_chat_model_is_reasoning() {
        // O-series reasoning models
        assert!(ChatModel::O1.is_reasoning_model());
        assert!(ChatModel::O3.is_reasoning_model());
        assert!(ChatModel::O3Mini.is_reasoning_model());
        assert!(ChatModel::O4Mini.is_reasoning_model());
        // GPT-5 series are also reasoning models
        assert!(ChatModel::Gpt5_2.is_reasoning_model());
        assert!(ChatModel::Gpt5_2ChatLatest.is_reasoning_model());
        assert!(ChatModel::Gpt5_2Pro.is_reasoning_model());
        assert!(ChatModel::Gpt5_1.is_reasoning_model());
        assert!(ChatModel::Gpt5_1CodexMax.is_reasoning_model());
        assert!(ChatModel::Gpt5Mini.is_reasoning_model());
        // Non-reasoning models
        assert!(!ChatModel::Gpt4oMini.is_reasoning_model());
        assert!(!ChatModel::Gpt4_1.is_reasoning_model());
    }

    #[test]
    fn test_chat_model_from_str() {
        assert_eq!(ChatModel::from("gpt-4o-mini"), ChatModel::Gpt4oMini);
        assert_eq!(ChatModel::from("o3-mini"), ChatModel::O3Mini);
        // GPT-5 models
        assert_eq!(ChatModel::from("gpt-5.2"), ChatModel::Gpt5_2);
        assert_eq!(ChatModel::from("gpt-5.2-chat-latest"), ChatModel::Gpt5_2ChatLatest);
        assert_eq!(ChatModel::from("gpt-5.2-pro"), ChatModel::Gpt5_2Pro);
        assert_eq!(ChatModel::from("gpt-5.1"), ChatModel::Gpt5_1);
        assert_eq!(ChatModel::from("gpt-5.1-codex-max"), ChatModel::Gpt5_1CodexMax);
        assert_eq!(ChatModel::from("gpt-5-mini"), ChatModel::Gpt5Mini);
        // Unknown models become Custom
        assert!(matches!(ChatModel::from("unknown-model"), ChatModel::Custom(_)));
    }

    #[test]
    fn test_chat_model_custom() {
        let custom = ChatModel::custom("ft:gpt-4o-mini:org::123");
        assert_eq!(custom.as_str(), "ft:gpt-4o-mini:org::123");
    }

    #[test]
    fn test_chat_model_custom_gpt5_is_reasoning() {
        // Custom GPT-5 models should also be detected as reasoning models
        let custom_gpt5 = ChatModel::custom("gpt-5.3-preview");
        assert!(custom_gpt5.is_reasoning_model());
    }

    #[test]
    fn test_embedding_model_dimensions() {
        assert_eq!(EmbeddingModel::TextEmbedding3Small.dimensions(), 1536);
        assert_eq!(EmbeddingModel::TextEmbedding3Large.dimensions(), 3072);
    }

    #[test]
    fn test_realtime_model_as_str() {
        assert_eq!(RealtimeModel::GptRealtime_2025_08_28.as_str(), "gpt-realtime-2025-08-28");
    }

    #[test]
    fn test_fine_tuning_model_as_str() {
        assert_eq!(FineTuningModel::Gpt4oMini_2024_07_18.as_str(), "gpt-4o-mini-2024-07-18");
        assert_eq!(FineTuningModel::Gpt41_2025_04_14.as_str(), "gpt-4.1-2025-04-14");
    }

    #[test]
    fn test_chat_model_serialization() {
        let model = ChatModel::Gpt4oMini;
        let json = serde_json::to_string(&model).unwrap();
        assert_eq!(json, "\"gpt-4o-mini\"");
        // GPT-5 serialization
        let gpt52 = ChatModel::Gpt5_2;
        let json = serde_json::to_string(&gpt52).unwrap();
        assert_eq!(json, "\"gpt-5.2\"");
    }

    #[test]
    fn test_chat_model_deserialization() {
        let model: ChatModel = serde_json::from_str("\"gpt-4o-mini\"").unwrap();
        assert_eq!(model, ChatModel::Gpt4oMini);
        // GPT-5 deserialization
        let gpt52: ChatModel = serde_json::from_str("\"gpt-5.2\"").unwrap();
        assert_eq!(gpt52, ChatModel::Gpt5_2);
    }

    #[test]
    fn test_parameter_support_standard_model() {
        let model = ChatModel::Gpt4oMini;
        let support = model.parameter_support();

        // Standard models support all parameters
        assert_eq!(support.temperature, ParameterRestriction::Any);
        assert_eq!(support.frequency_penalty, ParameterRestriction::Any);
        assert_eq!(support.presence_penalty, ParameterRestriction::Any);
        assert_eq!(support.top_p, ParameterRestriction::Any);
        assert!(support.logprobs);
        assert!(support.top_logprobs);
        assert!(support.logit_bias);
        assert!(support.n_multiple);
        assert!(!support.reasoning); // Standard models don't support reasoning
    }

    #[test]
    fn test_parameter_support_reasoning_model() {
        let model = ChatModel::O3Mini;
        let support = model.parameter_support();

        // Reasoning models have restrictions
        assert_eq!(support.temperature, ParameterRestriction::FixedValue(1.0));
        assert_eq!(support.frequency_penalty, ParameterRestriction::FixedValue(0.0));
        assert_eq!(support.presence_penalty, ParameterRestriction::FixedValue(0.0));
        assert_eq!(support.top_p, ParameterRestriction::FixedValue(1.0));
        assert!(!support.logprobs);
        assert!(!support.top_logprobs);
        assert!(!support.logit_bias);
        assert!(!support.n_multiple);
        assert!(support.reasoning); // Reasoning models support reasoning
    }

    #[test]
    fn test_parameter_support_gpt5_model() {
        // GPT-5 models are also reasoning models
        let model = ChatModel::Gpt5_2;
        let support = model.parameter_support();

        assert_eq!(support.temperature, ParameterRestriction::FixedValue(1.0));
        assert!(!support.logprobs);
        assert!(support.reasoning);
    }

    // =============================================================================
    // Comprehensive Reasoning Model Detection Tests
    // =============================================================================

    #[test]
    fn test_all_o_series_models_are_reasoning() {
        // All defined o-series models should be detected as reasoning models
        let o_series = vec![ChatModel::O1, ChatModel::O1Pro, ChatModel::O3, ChatModel::O3Mini, ChatModel::O4Mini];

        for model in o_series {
            assert!(model.is_reasoning_model(), "Expected {} to be a reasoning model", model.as_str());
        }
    }

    #[test]
    fn test_all_gpt5_models_are_reasoning() {
        // All GPT-5 series models should be detected as reasoning models
        let gpt5_series = vec![
            ChatModel::Gpt5_2,
            ChatModel::Gpt5_2ChatLatest,
            ChatModel::Gpt5_2Pro,
            ChatModel::Gpt5_1,
            ChatModel::Gpt5_1ChatLatest,
            ChatModel::Gpt5_1CodexMax,
            ChatModel::Gpt5Mini,
        ];

        for model in gpt5_series {
            assert!(model.is_reasoning_model(), "Expected {} to be a reasoning model", model.as_str());
        }
    }

    #[test]
    fn test_all_standard_models_are_not_reasoning() {
        // Standard models should NOT be detected as reasoning models
        let standard_models = vec![
            ChatModel::Gpt4oMini,
            ChatModel::Gpt4o,
            ChatModel::Gpt4oAudioPreview,
            ChatModel::Gpt4Turbo,
            ChatModel::Gpt4,
            ChatModel::Gpt3_5Turbo,
            ChatModel::Gpt4_1,
            ChatModel::Gpt4_1Mini,
            ChatModel::Gpt4_1Nano,
        ];

        for model in standard_models {
            assert!(!model.is_reasoning_model(), "Expected {} to NOT be a reasoning model", model.as_str());
        }
    }

    // =============================================================================
    // Custom Model Reasoning Detection Tests
    // =============================================================================

    #[test]
    fn test_custom_o1_models_are_reasoning() {
        let custom_o1_variants = vec!["o1-mini", "o1-preview", "o1-pro-2025", "o1-high"];

        for model_str in custom_o1_variants {
            let model = ChatModel::custom(model_str);
            assert!(model.is_reasoning_model(), "Expected custom model '{}' to be a reasoning model", model_str);
        }
    }

    #[test]
    fn test_custom_o3_models_are_reasoning() {
        let custom_o3_variants = vec!["o3-preview", "o3-high", "o3-2025-01-15"];

        for model_str in custom_o3_variants {
            let model = ChatModel::custom(model_str);
            assert!(model.is_reasoning_model(), "Expected custom model '{}' to be a reasoning model", model_str);
        }
    }

    #[test]
    fn test_custom_o4_models_are_reasoning() {
        let custom_o4_variants = vec!["o4-preview", "o4-mini-2025", "o4-high"];

        for model_str in custom_o4_variants {
            let model = ChatModel::custom(model_str);
            assert!(model.is_reasoning_model(), "Expected custom model '{}' to be a reasoning model", model_str);
        }
    }

    #[test]
    fn test_custom_gpt5_models_are_reasoning() {
        let custom_gpt5_variants = vec!["gpt-5.3", "gpt-5.3-preview", "gpt-5-turbo", "gpt-5.0"];

        for model_str in custom_gpt5_variants {
            let model = ChatModel::custom(model_str);
            assert!(model.is_reasoning_model(), "Expected custom model '{}' to be a reasoning model", model_str);
        }
    }

    #[test]
    fn test_custom_standard_models_are_not_reasoning() {
        let custom_standard_variants = vec![
            "ft:gpt-4o-mini:org::123",
            "gpt-4o-2025-01-15",
            "gpt-4-turbo-preview",
            "gpt-3.5-turbo-instruct",
            "text-davinci-003",
            "claude-3-opus", // Non-OpenAI model
        ];

        for model_str in custom_standard_variants {
            let model = ChatModel::custom(model_str);
            assert!(!model.is_reasoning_model(), "Expected custom model '{}' to NOT be a reasoning model", model_str);
        }
    }

    // =============================================================================
    // Parameter Support Tests for Each Model Generation
    // =============================================================================

    #[test]
    fn test_parameter_support_all_o_series() {
        let o_series = vec![ChatModel::O1, ChatModel::O1Pro, ChatModel::O3, ChatModel::O3Mini, ChatModel::O4Mini];

        for model in o_series {
            let support = model.parameter_support();

            assert_eq!(support.temperature, ParameterRestriction::FixedValue(1.0), "{} should only support temperature=1.0", model.as_str());
            assert_eq!(
                support.frequency_penalty,
                ParameterRestriction::FixedValue(0.0),
                "{} should only support frequency_penalty=0.0",
                model.as_str()
            );
            assert_eq!(
                support.presence_penalty,
                ParameterRestriction::FixedValue(0.0),
                "{} should only support presence_penalty=0.0",
                model.as_str()
            );
            assert_eq!(support.top_p, ParameterRestriction::FixedValue(1.0), "{} should only support top_p=1.0", model.as_str());
            assert!(!support.logprobs, "{} should not support logprobs", model.as_str());
            assert!(!support.top_logprobs, "{} should not support top_logprobs", model.as_str());
            assert!(!support.logit_bias, "{} should not support logit_bias", model.as_str());
            assert!(!support.n_multiple, "{} should only support n=1", model.as_str());
            assert!(support.reasoning, "{} should support reasoning parameter", model.as_str());
        }
    }

    #[test]
    fn test_parameter_support_all_gpt5_series() {
        let gpt5_series = vec![
            ChatModel::Gpt5_2,
            ChatModel::Gpt5_2ChatLatest,
            ChatModel::Gpt5_2Pro,
            ChatModel::Gpt5_1,
            ChatModel::Gpt5_1ChatLatest,
            ChatModel::Gpt5_1CodexMax,
            ChatModel::Gpt5Mini,
        ];

        for model in gpt5_series {
            let support = model.parameter_support();

            assert_eq!(support.temperature, ParameterRestriction::FixedValue(1.0), "{} should only support temperature=1.0", model.as_str());
            assert!(support.reasoning, "{} should support reasoning parameter", model.as_str());
        }
    }

    #[test]
    fn test_parameter_support_all_standard_gpt4_series() {
        let gpt4_series = vec![
            ChatModel::Gpt4oMini,
            ChatModel::Gpt4o,
            ChatModel::Gpt4Turbo,
            ChatModel::Gpt4,
            ChatModel::Gpt4_1,
            ChatModel::Gpt4_1Mini,
            ChatModel::Gpt4_1Nano,
        ];

        for model in gpt4_series {
            let support = model.parameter_support();

            assert_eq!(support.temperature, ParameterRestriction::Any, "{} should support any temperature", model.as_str());
            assert_eq!(support.frequency_penalty, ParameterRestriction::Any, "{} should support any frequency_penalty", model.as_str());
            assert_eq!(support.presence_penalty, ParameterRestriction::Any, "{} should support any presence_penalty", model.as_str());
            assert!(support.logprobs, "{} should support logprobs", model.as_str());
            assert!(support.top_logprobs, "{} should support top_logprobs", model.as_str());
            assert!(support.logit_bias, "{} should support logit_bias", model.as_str());
            assert!(support.n_multiple, "{} should support n > 1", model.as_str());
            assert!(!support.reasoning, "{} should NOT support reasoning parameter", model.as_str());
        }
    }

    // =============================================================================
    // ParameterRestriction Enum Tests
    // =============================================================================

    #[test]
    fn test_parameter_restriction_equality() {
        assert_eq!(ParameterRestriction::Any, ParameterRestriction::Any);
        assert_eq!(ParameterRestriction::NotSupported, ParameterRestriction::NotSupported);
        assert_eq!(ParameterRestriction::FixedValue(1.0), ParameterRestriction::FixedValue(1.0));

        assert_ne!(ParameterRestriction::Any, ParameterRestriction::NotSupported);
        assert_ne!(ParameterRestriction::FixedValue(1.0), ParameterRestriction::FixedValue(0.0));
    }

    #[test]
    fn test_parameter_support_factory_methods() {
        let standard = ParameterSupport::standard_model();
        assert_eq!(standard.temperature, ParameterRestriction::Any);
        assert!(standard.logprobs);
        assert!(!standard.reasoning);

        let reasoning = ParameterSupport::reasoning_model();
        assert_eq!(reasoning.temperature, ParameterRestriction::FixedValue(1.0));
        assert!(!reasoning.logprobs);
        assert!(reasoning.reasoning);
    }

    // =============================================================================
    // Model String Conversion Tests
    // =============================================================================

    #[test]
    fn test_all_gpt5_model_string_roundtrip() {
        let gpt5_models = vec![
            ("gpt-5.2", ChatModel::Gpt5_2),
            ("gpt-5.2-chat-latest", ChatModel::Gpt5_2ChatLatest),
            ("gpt-5.2-pro", ChatModel::Gpt5_2Pro),
            ("gpt-5.1", ChatModel::Gpt5_1),
            ("gpt-5.1-chat-latest", ChatModel::Gpt5_1ChatLatest),
            ("gpt-5.1-codex-max", ChatModel::Gpt5_1CodexMax),
            ("gpt-5-mini", ChatModel::Gpt5Mini),
        ];

        for (model_str, expected_model) in gpt5_models {
            // Test from string
            let parsed = ChatModel::from(model_str);
            assert_eq!(parsed, expected_model, "Failed to parse '{}'", model_str);

            // Test to string
            assert_eq!(expected_model.as_str(), model_str, "Failed to convert {:?} to string", expected_model);

            // Test serialization roundtrip
            let json = serde_json::to_string(&expected_model).unwrap();
            let deserialized: ChatModel = serde_json::from_str(&json).unwrap();
            assert_eq!(deserialized, expected_model, "Serialization roundtrip failed for {}", model_str);
        }
    }

    #[test]
    fn test_all_o_series_model_string_roundtrip() {
        let o_series_models = vec![
            ("o1", ChatModel::O1),
            ("o1-pro", ChatModel::O1Pro),
            ("o3", ChatModel::O3),
            ("o3-mini", ChatModel::O3Mini),
            ("o4-mini", ChatModel::O4Mini),
        ];

        for (model_str, expected_model) in o_series_models {
            let parsed = ChatModel::from(model_str);
            assert_eq!(parsed, expected_model, "Failed to parse '{}'", model_str);
            assert_eq!(expected_model.as_str(), model_str, "Failed to convert {:?} to string", expected_model);
        }
    }

    // =============================================================================
    // Embedding Model Tests
    // =============================================================================

    #[test]
    fn test_embedding_model_string_roundtrip() {
        let embedding_models = vec![
            ("text-embedding-3-small", EmbeddingModel::TextEmbedding3Small),
            ("text-embedding-3-large", EmbeddingModel::TextEmbedding3Large),
            ("text-embedding-ada-002", EmbeddingModel::TextEmbeddingAda002),
        ];

        for (model_str, expected_model) in embedding_models {
            let parsed = EmbeddingModel::from(model_str);
            assert_eq!(parsed, expected_model, "Failed to parse '{}'", model_str);
            assert_eq!(expected_model.as_str(), model_str, "Failed to convert {:?} to string", expected_model);
        }
    }

    #[test]
    fn test_embedding_model_all_dimensions() {
        assert_eq!(EmbeddingModel::TextEmbedding3Small.dimensions(), 1536);
        assert_eq!(EmbeddingModel::TextEmbedding3Large.dimensions(), 3072);
        assert_eq!(EmbeddingModel::TextEmbeddingAda002.dimensions(), 1536);
    }

    // =============================================================================
    // Realtime Model Tests
    // =============================================================================

    #[test]
    fn test_realtime_model_string_roundtrip() {
        let realtime_models = vec![("gpt-realtime-2025-08-28", RealtimeModel::GptRealtime_2025_08_28)];

        for (model_str, expected_model) in realtime_models {
            let parsed = RealtimeModel::from(model_str);
            assert_eq!(parsed, expected_model, "Failed to parse '{}'", model_str);
            assert_eq!(expected_model.as_str(), model_str, "Failed to convert {:?} to string", expected_model);
        }
    }

    #[test]
    fn test_realtime_model_custom() {
        let custom = RealtimeModel::custom("gpt-4o-realtime-2025");
        assert_eq!(custom.as_str(), "gpt-4o-realtime-2025");
        assert!(matches!(custom, RealtimeModel::Custom(_)));
    }

    // =============================================================================
    // Fine-tuning Model Tests
    // =============================================================================

    #[test]
    fn test_fine_tuning_model_as_str_all_variants() {
        let fine_tuning_models = vec![
            ("gpt-4.1-2025-04-14", FineTuningModel::Gpt41_2025_04_14),
            ("gpt-4.1-mini-2025-04-14", FineTuningModel::Gpt41Mini_2025_04_14),
            ("gpt-4.1-nano-2025-04-14", FineTuningModel::Gpt41Nano_2025_04_14),
            ("gpt-4o-mini-2024-07-18", FineTuningModel::Gpt4oMini_2024_07_18),
            ("gpt-4o-2024-08-06", FineTuningModel::Gpt4o_2024_08_06),
            ("gpt-4-0613", FineTuningModel::Gpt4_0613),
            ("gpt-3.5-turbo-0125", FineTuningModel::Gpt35Turbo_0125),
            ("gpt-3.5-turbo-1106", FineTuningModel::Gpt35Turbo_1106),
            ("gpt-3.5-turbo-0613", FineTuningModel::Gpt35Turbo_0613),
        ];

        for (model_str, expected_model) in fine_tuning_models {
            assert_eq!(expected_model.as_str(), model_str, "Failed to convert {:?} to string", expected_model);
        }
    }

    #[test]
    fn test_fine_tuning_model_serialization_roundtrip() {
        let models = vec![FineTuningModel::Gpt41_2025_04_14, FineTuningModel::Gpt4oMini_2024_07_18, FineTuningModel::Gpt35Turbo_0125];

        for model in models {
            let json = serde_json::to_string(&model).unwrap();
            let deserialized: FineTuningModel = serde_json::from_str(&json).unwrap();
            assert_eq!(deserialized, model, "Serialization roundtrip failed for {:?}", model);
        }
    }
}
