//! OpenAI Chat Completions API Request Module
//!
//! This module provides the functionality to build and send requests to the OpenAI Chat Completions API.
//! It offers a builder pattern for constructing requests with various parameters and options,
//! making it easy to interact with OpenAI's conversational AI models.
//!
//! # Key Features
//!
//! - **Builder Pattern**: Fluent API for constructing requests
//! - **Structured Output**: Support for JSON schema-based responses
//! - **Function Calling**: Tool integration for extended model capabilities
//! - **Comprehensive Parameters**: Full support for all OpenAI API parameters
//! - **Error Handling**: Robust error management and validation
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use openai_tools::chat::request::ChatCompletion;
//! use openai_tools::common::message::Message;
//! use openai_tools::common::role::Role;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Initialize the chat completion client
//!     let mut chat = ChatCompletion::new();
//!     
//!     // Create a simple conversation
//!     let messages = vec![
//!         Message::from_string(Role::User, "Hello! How are you?")
//!     ];
//!
//!     // Send the request and get a response
//!     let response = chat
//!         .model_id("gpt-4o-mini")
//!         .messages(messages)
//!         .temperature(0.7)
//!         .chat()
//!         .await?;
//!
//!     println!("AI Response: {}",
//!              response.choices[0].message.content.as_ref().unwrap().text.as_ref().unwrap());
//!     Ok(())
//! }
//! ```
//!
//! # Advanced Usage
//!
//! ## Structured Output with JSON Schema
//!
//! ```rust,no_run
//! use openai_tools::chat::request::ChatCompletion;
//! use openai_tools::common::message::Message;
//! use openai_tools::common::role::Role;
//! use openai_tools::common::structured_output::Schema;
//! use serde::{Deserialize, Serialize};
//!
//! #[derive(Serialize, Deserialize)]
//! struct PersonInfo {
//!     name: String,
//!     age: u32,
//!     occupation: String,
//! }
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut chat = ChatCompletion::new();
//!     
//!     // Define JSON schema for structured output
//!     let mut schema = Schema::chat_json_schema("person_info");
//!     schema.add_property("name", "string", "Person's full name");
//!     schema.add_property("age", "number", "Person's age in years");
//!     schema.add_property("occupation", "string", "Person's job or profession");
//!     
//!     let messages = vec![
//!         Message::from_string(Role::User,
//!             "Extract information about: John Smith, 30 years old, software engineer")
//!     ];
//!
//!     let response = chat
//!         .model_id("gpt-4o-mini")
//!         .messages(messages)
//!         .json_schema(schema)
//!         .chat()
//!         .await?;
//!         
//!     // Parse structured response
//!     let person: PersonInfo = serde_json::from_str(
//!         response.choices[0].message.content.as_ref().unwrap().text.as_ref().unwrap()
//!     )?;
//!     
//!     println!("Extracted: {} (age: {}, job: {})",
//!              person.name, person.age, person.occupation);
//!     Ok(())
//! }
//! ```
//!
//! ## Function Calling with Tools
//!
//! ```rust,no_run
//! use openai_tools::chat::request::ChatCompletion;
//! use openai_tools::common::message::Message;
//! use openai_tools::common::role::Role;
//! use openai_tools::common::tool::Tool;
//! use openai_tools::common::parameters::ParameterProperty;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut chat = ChatCompletion::new();
//!     
//!     // Define a weather checking tool
//!     let weather_tool = Tool::function(
//!         "get_weather",
//!         "Get current weather information for a location",
//!         vec![
//!             ("location", ParameterProperty::from_string("The city and country")),
//!             ("unit", ParameterProperty::from_string("Temperature unit (celsius/fahrenheit)")),
//!         ],
//!         false,
//!     );
//!     
//!     let messages = vec![
//!         Message::from_string(Role::User,
//!             "What's the weather like in Tokyo today?")
//!     ];
//!
//!     let response = chat
//!         .model_id("gpt-4o-mini")
//!         .messages(messages)
//!         .tools(vec![weather_tool])
//!         .temperature(0.1)
//!         .chat()
//!         .await?;
//!         
//!     // Handle tool calls
//!     if let Some(tool_calls) = &response.choices[0].message.tool_calls {
//!         for call in tool_calls {
//!             println!("Tool called: {}", call.function.name);
//!             if let Ok(args) = call.function.arguments_as_map() {
//!                 println!("Arguments: {:?}", args);
//!             }
//!             // Execute the function and continue the conversation...
//!         }
//!     }
//!     Ok(())
//! }
//! ```
//!
//! # Environment Setup
//!
//! Before using this module, ensure you have set up your OpenAI API key:
//!
//! ```bash
//! export OPENAI_API_KEY="your-api-key-here"
//! ```
//!
//! Or create a `.env` file in your project root:
//!
//! ```text
//! OPENAI_API_KEY=your-api-key-here
//! ```
//!
//!
//! # Error Handling
//!
//! All methods return a `Result` type for proper error handling:
//!
//! ```rust,no_run
//! use openai_tools::chat::request::ChatCompletion;
//! use openai_tools::common::errors::OpenAIToolError;
//!
//! #[tokio::main]
//! async fn main() {
//!     let mut chat = ChatCompletion::new();
//!     
//!     match chat.model_id("gpt-4o-mini").chat().await {
//!         Ok(response) => {
//!             if let Some(content) = &response.choices[0].message.content {
//!                 if let Some(text) = &content.text {
//!                     println!("Success: {}", text);
//!                 }
//!             }
//!         }
//!         Err(OpenAIToolError::RequestError(e)) => {
//!             eprintln!("Network error: {}", e);
//!         }
//!         Err(OpenAIToolError::SerdeJsonError(e)) => {
//!             eprintln!("JSON parsing error: {}", e);
//!         }
//!         Err(e) => {
//!             eprintln!("Other error: {}", e);
//!         }
//!     }
//! }
//! ```

use crate::chat::response::Response;
use crate::common::{
    auth::{AuthProvider, OpenAIAuth},
    client::create_http_client,
    errors::{ErrorResponse, OpenAIToolError, Result},
    message::{Content, Message},
    models::{ChatModel, ParameterRestriction},
    structured_output::Schema,
    tool::Tool,
};
use core::str;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Response format structure for OpenAI API requests
///
/// This structure is used for structured output when JSON schema is specified.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct Format {
    #[serde(rename = "type")]
    type_name: String,
    json_schema: Schema,
}

impl Format {
    /// Creates a new Format structure
    ///
    /// # Arguments
    ///
    /// * `type_name` - The type name for the response format
    /// * `json_schema` - The JSON schema definition
    ///
    /// # Returns
    ///
    /// A new Format structure instance
    pub fn new<T: AsRef<str>>(type_name: T, json_schema: Schema) -> Self {
        Self { type_name: type_name.as_ref().to_string(), json_schema }
    }
}

// =============================================================================
// Chat API serialization wrappers
//
// The shared `Content` type uses Responses API format ("input_text", "input_image"),
// but Chat Completions API expects different type names and structure:
//   - "input_text"  → {"type": "text", "text": "..."}
//   - "input_image" → {"type": "image_url", "image_url": {"url": "..."}}
//
// These zero-copy wrappers convert at serialization time without changing
// the public API or affecting the Responses API path.
// =============================================================================

/// Wraps `&Content` to serialize in Chat Completions API format.
struct ChatContentRef<'a>(&'a Content);

impl<'a> Serialize for ChatContentRef<'a> {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;

        match self.0.type_name.as_str() {
            "input_text" => {
                let mut state = serializer.serialize_struct("Content", 2)?;
                state.serialize_field("type", "text")?;
                state.serialize_field("text", &self.0.text)?;
                state.end()
            }
            "input_image" => {
                #[derive(Serialize)]
                struct ImageUrl<'b> {
                    url: &'b str,
                }
                let mut state = serializer.serialize_struct("Content", 2)?;
                state.serialize_field("type", "image_url")?;
                if let Some(ref url) = self.0.image_url {
                    state.serialize_field("image_url", &ImageUrl { url })?;
                }
                state.end()
            }
            other => {
                // Pass through unknown types as-is
                let mut state = serializer.serialize_struct("Content", 3)?;
                state.serialize_field("type", other)?;
                if let Some(ref text) = self.0.text {
                    state.serialize_field("text", text)?;
                }
                if let Some(ref url) = self.0.image_url {
                    state.serialize_field("image_url", url)?;
                }
                state.end()
            }
        }
    }
}

/// Wraps `&Message` to serialize in Chat Completions API format.
///
/// - Single content (`content` field): extracts `.text` as a plain string (existing behavior)
/// - Content list (`content_list` field): wraps each element with `ChatContentRef`
struct ChatMessageRef<'a>(&'a Message);

impl<'a> Serialize for ChatMessageRef<'a> {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;

        let msg = self.0;
        let mut state = serializer.serialize_struct("Message", 3)?;
        state.serialize_field("role", &msg.role)?;

        if let Some(ref content) = msg.content {
            // Single content: serialize as plain text string
            state.serialize_field("content", &content.text)?;
        } else if let Some(ref contents) = msg.content_list {
            // Multi-modal content: wrap each element with ChatContentRef
            let chat_contents: Vec<ChatContentRef<'_>> = contents.iter().map(ChatContentRef).collect();
            state.serialize_field("content", &chat_contents)?;
        }

        if let Some(ref tool_call_id) = msg.tool_call_id {
            state.serialize_field("tool_call_id", tool_call_id)?;
        }
        if let Some(ref tool_calls) = msg.tool_calls {
            state.serialize_field("tool_calls", tool_calls)?;
        }

        state.end()
    }
}

/// Custom serializer for `Vec<Message>` that converts to Chat API format.
fn serialize_chat_messages<S>(messages: &Vec<Message>, serializer: S) -> std::result::Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    use serde::ser::SerializeSeq;
    let mut seq = serializer.serialize_seq(Some(messages.len()))?;
    for msg in messages {
        seq.serialize_element(&ChatMessageRef(msg))?;
    }
    seq.end()
}

/// Request body structure for OpenAI Chat Completions API
///
/// This structure represents the parameters that will be sent in the request body
/// to the OpenAI API. Each field corresponds to the API specification.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub(crate) struct Body {
    pub(crate) model: ChatModel,
    #[serde(serialize_with = "serialize_chat_messages")]
    pub(crate) messages: Vec<Message>,
    /// Whether to store the request and response at OpenAI
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) store: Option<bool>,
    /// Frequency penalty parameter to reduce repetition (-2.0 to 2.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) frequency_penalty: Option<f32>,
    /// Logit bias to adjust the probability of specific tokens
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) logit_bias: Option<HashMap<String, i32>>,
    /// Whether to include probability information for each token
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) logprobs: Option<bool>,
    /// Number of top probabilities to return for each token (0-20)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) top_logprobs: Option<u8>,
    /// Maximum number of tokens to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) max_completion_tokens: Option<u64>,
    /// Number of responses to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) n: Option<u32>,
    /// Available modalities for the response (e.g., text, audio)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) modalities: Option<Vec<String>>,
    /// Presence penalty to encourage new topics (-2.0 to 2.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) presence_penalty: Option<f32>,
    /// Temperature parameter to control response randomness (0.0 to 2.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) temperature: Option<f32>,
    /// Response format specification (e.g., JSON schema)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) response_format: Option<Format>,
    /// Optional tools that can be used by the model
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tools: Option<Vec<Tool>>,
    /// A stable identifier for the end user, used for safety monitoring and abuse detection
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) safety_identifier: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) user: Option<String>,
}

/// OpenAI Chat Completions API client
///
/// This structure manages interactions with the OpenAI Chat Completions API.
/// It handles API key management, request parameter configuration, and API calls.
///
/// # Example
///
/// ```rust
/// use openai_tools::chat::request::ChatCompletion;
/// use openai_tools::common::message::Message;
/// use openai_tools::common::role::Role;
///
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut chat = ChatCompletion::new();
/// let messages = vec![Message::from_string(Role::User, "Hello!")];
///
/// let response = chat
///     .model_id("gpt-4o-mini")
///     .messages(messages)
///     .temperature(1.0)
///     .chat()
///     .await?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// # }
/// ```
/// Default API path for Chat Completions
const CHAT_COMPLETIONS_PATH: &str = "chat/completions";

/// OpenAI Chat Completions API client
///
/// This structure manages interactions with the OpenAI Chat Completions API
/// and Azure OpenAI API. It handles authentication, request parameter
/// configuration, and API calls.
///
/// # Providers
///
/// The client supports two providers:
/// - **OpenAI**: Standard OpenAI API (default)
/// - **Azure**: Azure OpenAI Service
///
/// # Examples
///
/// ## OpenAI (existing behavior - unchanged)
///
/// ```rust,no_run
/// use openai_tools::chat::request::ChatCompletion;
/// use openai_tools::common::message::Message;
/// use openai_tools::common::role::Role;
///
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut chat = ChatCompletion::new();
/// let messages = vec![Message::from_string(Role::User, "Hello!")];
///
/// let response = chat
///     .model_id("gpt-4o-mini")
///     .messages(messages)
///     .chat()
///     .await?;
/// # Ok(())
/// # }
/// ```
///
/// ## Azure OpenAI
///
/// ```rust,no_run
/// use openai_tools::chat::request::ChatCompletion;
/// use openai_tools::common::message::Message;
/// use openai_tools::common::role::Role;
///
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // From environment variables
/// let mut chat = ChatCompletion::azure()?;
///
/// let messages = vec![Message::from_string(Role::User, "Hello!")];
/// let response = chat.messages(messages).chat().await?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct ChatCompletion {
    /// Authentication provider (OpenAI or Azure)
    auth: AuthProvider,
    /// The request body containing all parameters for the API call
    pub(crate) request_body: Body,
    /// Optional request timeout duration
    timeout: Option<Duration>,
}

impl Default for ChatCompletion {
    fn default() -> Self {
        Self::new()
    }
}

impl ChatCompletion {
    /// Creates a new ChatCompletion instance for OpenAI API
    ///
    /// Loads the API key from the `OPENAI_API_KEY` environment variable.
    /// If a `.env` file exists, it will also be loaded.
    ///
    /// # Panics
    ///
    /// Panics if the `OPENAI_API_KEY` environment variable is not set.
    ///
    /// # Returns
    ///
    /// A new ChatCompletion instance configured for OpenAI API
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::chat::request::ChatCompletion;
    ///
    /// let mut chat = ChatCompletion::new();
    /// ```
    pub fn new() -> Self {
        let auth = AuthProvider::openai_from_env().map_err(|e| OpenAIToolError::Error(format!("Failed to load OpenAI auth: {}", e))).unwrap();
        Self { auth, request_body: Body::default(), timeout: None }
    }

    /// Creates a new ChatCompletion instance with a specified model
    ///
    /// This is the recommended constructor as it enables parameter validation
    /// at setter time. When you set parameters like `temperature()`, the model's
    /// parameter support is checked and warnings are logged for unsupported values.
    ///
    /// # Arguments
    ///
    /// * `model` - The model to use for chat completion
    ///
    /// # Panics
    ///
    /// Panics if the `OPENAI_API_KEY` environment variable is not set.
    ///
    /// # Returns
    ///
    /// A new ChatCompletion instance with the specified model
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::chat::request::ChatCompletion;
    /// use openai_tools::common::models::ChatModel;
    ///
    /// // Recommended: specify model at creation time
    /// let mut chat = ChatCompletion::with_model(ChatModel::Gpt4oMini);
    ///
    /// // For reasoning models, unsupported parameters are validated at setter time
    /// let mut reasoning_chat = ChatCompletion::with_model(ChatModel::O3Mini);
    /// reasoning_chat.temperature(0.5); // Warning logged, value ignored
    /// ```
    pub fn with_model(model: ChatModel) -> Self {
        let auth = AuthProvider::openai_from_env().map_err(|e| OpenAIToolError::Error(format!("Failed to load OpenAI auth: {}", e))).unwrap();
        Self { auth, request_body: Body { model, ..Default::default() }, timeout: None }
    }

    /// Creates a new ChatCompletion instance with a custom authentication provider
    ///
    /// Use this to explicitly configure OpenAI or Azure authentication.
    ///
    /// # Arguments
    ///
    /// * `auth` - The authentication provider
    ///
    /// # Returns
    ///
    /// A new ChatCompletion instance with the specified auth provider
    ///
    /// # Example
    ///
    /// ```rust
    /// use openai_tools::chat::request::ChatCompletion;
    /// use openai_tools::common::auth::{AuthProvider, AzureAuth};
    ///
    /// // Explicit Azure configuration with complete base URL
    /// let auth = AuthProvider::Azure(
    ///     AzureAuth::new(
    ///         "api-key",
    ///         "https://my-resource.openai.azure.com/openai/deployments/gpt-4o?api-version=2024-08-01-preview"
    ///     )
    /// );
    /// let mut chat = ChatCompletion::with_auth(auth);
    /// ```
    pub fn with_auth(auth: AuthProvider) -> Self {
        Self { auth, request_body: Body::default(), timeout: None }
    }

    /// Creates a new ChatCompletion instance for Azure OpenAI API
    ///
    /// Loads configuration from Azure-specific environment variables.
    ///
    /// # Returns
    ///
    /// `Result<ChatCompletion>` - Configured for Azure or error if env vars missing
    ///
    /// # Environment Variables
    ///
    /// | Variable | Required | Description |
    /// |----------|----------|-------------|
    /// | `AZURE_OPENAI_API_KEY` | Yes | Azure API key |
    /// | `AZURE_OPENAI_BASE_URL` | Yes | Complete endpoint URL including deployment, API path, and api-version |
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::chat::request::ChatCompletion;
    ///
    /// // With environment variables:
    /// // AZURE_OPENAI_API_KEY=xxx
    /// // AZURE_OPENAI_BASE_URL=https://my-resource.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview
    /// let mut chat = ChatCompletion::azure()?;
    /// # Ok::<(), openai_tools::common::errors::OpenAIToolError>(())
    /// ```
    pub fn azure() -> Result<Self> {
        let auth = AuthProvider::azure_from_env()?;
        Ok(Self { auth, request_body: Body::default(), timeout: None })
    }

    /// Creates a new ChatCompletion instance by auto-detecting the provider
    ///
    /// Tries Azure first (if AZURE_OPENAI_API_KEY is set), then falls back to OpenAI.
    ///
    /// # Returns
    ///
    /// `Result<ChatCompletion>` - Auto-configured client or error
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::chat::request::ChatCompletion;
    ///
    /// // Uses Azure if AZURE_OPENAI_API_KEY is set, otherwise OpenAI
    /// let mut chat = ChatCompletion::detect_provider()?;
    /// # Ok::<(), openai_tools::common::errors::OpenAIToolError>(())
    /// ```
    pub fn detect_provider() -> Result<Self> {
        let auth = AuthProvider::from_env()?;
        Ok(Self { auth, request_body: Body::default(), timeout: None })
    }

    /// Creates a new ChatCompletion instance with URL-based provider detection
    ///
    /// Analyzes the URL pattern to determine the provider:
    /// - URLs containing `.openai.azure.com` → Azure
    /// - All other URLs → OpenAI-compatible
    ///
    /// # Arguments
    ///
    /// * `base_url` - The complete base URL for API requests
    /// * `api_key` - The API key or token
    ///
    /// # Returns
    ///
    /// `ChatCompletion` - Configured client
    ///
    /// # Example
    ///
    /// ```rust
    /// use openai_tools::chat::request::ChatCompletion;
    ///
    /// // OpenAI-compatible API (e.g., local Ollama)
    /// let chat = ChatCompletion::with_url(
    ///     "http://localhost:11434/v1",
    ///     "ollama",
    /// );
    ///
    /// // Azure OpenAI (complete base URL)
    /// let azure_chat = ChatCompletion::with_url(
    ///     "https://my-resource.openai.azure.com/openai/deployments/gpt-4o?api-version=2024-08-01-preview",
    ///     "azure-key",
    /// );
    /// ```
    pub fn with_url<S: Into<String>>(base_url: S, api_key: S) -> Self {
        let auth = AuthProvider::from_url_with_key(base_url, api_key);
        Self { auth, request_body: Body::default(), timeout: None }
    }

    /// Creates a new ChatCompletion instance from URL using environment variables
    ///
    /// Analyzes the URL pattern to determine the provider, then loads
    /// credentials from the appropriate environment variables.
    ///
    /// # Arguments
    ///
    /// * `base_url` - The complete base URL for API requests
    ///
    /// # Environment Variables
    ///
    /// For Azure URLs (`*.openai.azure.com`):
    /// - `AZURE_OPENAI_API_KEY` (required)
    ///
    /// For other URLs:
    /// - `OPENAI_API_KEY` (required)
    ///
    /// # Returns
    ///
    /// `Result<ChatCompletion>` - Configured client or error
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::chat::request::ChatCompletion;
    ///
    /// // Uses OPENAI_API_KEY from environment
    /// let chat = ChatCompletion::from_url("https://api.openai.com/v1")?;
    ///
    /// // Uses AZURE_OPENAI_API_KEY from environment (complete base URL)
    /// let azure = ChatCompletion::from_url(
    ///     "https://my-resource.openai.azure.com/openai/deployments/gpt-4o?api-version=2024-08-01-preview"
    /// )?;
    /// # Ok::<(), openai_tools::common::errors::OpenAIToolError>(())
    /// ```
    pub fn from_url<S: Into<String>>(base_url: S) -> Result<Self> {
        let auth = AuthProvider::from_url(base_url)?;
        Ok(Self { auth, request_body: Body::default(), timeout: None })
    }

    /// Returns the authentication provider
    ///
    /// # Returns
    ///
    /// Reference to the authentication provider
    pub fn auth(&self) -> &AuthProvider {
        &self.auth
    }

    /// Sets a custom API endpoint URL (OpenAI only)
    ///
    /// Use this to point to alternative OpenAI-compatible APIs (e.g., proxy servers).
    /// For Azure, use `azure()` or `with_auth()` instead.
    ///
    /// # Arguments
    ///
    /// * `url` - The base URL (e.g., "https://my-proxy.example.com/v1")
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    ///
    /// # Note
    ///
    /// This method only works with OpenAI authentication. For Azure, the endpoint
    /// is constructed from resource name and deployment name.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::chat::request::ChatCompletion;
    ///
    /// let mut chat = ChatCompletion::new();
    /// chat.base_url("https://my-proxy.example.com/v1");
    /// ```
    pub fn base_url<T: AsRef<str>>(&mut self, url: T) -> &mut Self {
        // Only modify if OpenAI provider
        if let AuthProvider::OpenAI(ref openai_auth) = self.auth {
            let new_auth = OpenAIAuth::new(openai_auth.api_key()).with_base_url(url.as_ref());
            self.auth = AuthProvider::OpenAI(new_auth);
        } else {
            tracing::warn!("base_url() is only supported for OpenAI provider. Use azure() or with_auth() for Azure.");
        }
        self
    }

    /// Sets the model to use for chat completion.
    ///
    /// # Arguments
    ///
    /// * `model` - The model to use (e.g., `ChatModel::Gpt4oMini`, `ChatModel::Gpt4o`)
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::chat::request::ChatCompletion;
    /// use openai_tools::common::models::ChatModel;
    ///
    /// let mut chat = ChatCompletion::new();
    /// chat.model(ChatModel::Gpt4oMini);
    /// ```
    pub fn model(&mut self, model: ChatModel) -> &mut Self {
        self.request_body.model = model;
        self
    }

    /// Sets the model using a string ID (for backward compatibility).
    ///
    /// Prefer using [`model`] with `ChatModel` enum for type safety.
    ///
    /// # Arguments
    ///
    /// * `model_id` - OpenAI model ID string (e.g., "gpt-4o-mini")
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::chat::request::ChatCompletion;
    ///
    /// let mut chat = ChatCompletion::new();
    /// chat.model_id("gpt-4o-mini");
    /// ```
    #[deprecated(since = "0.2.0", note = "Use `model(ChatModel)` instead for type safety")]
    pub fn model_id<T: AsRef<str>>(&mut self, model_id: T) -> &mut Self {
        self.request_body.model = ChatModel::from(model_id.as_ref());
        self
    }

    /// Sets the request timeout duration
    ///
    /// # Arguments
    ///
    /// * `timeout` - The maximum time to wait for a response
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use std::time::Duration;
    /// use openai_tools::chat::request::ChatCompletion;
    ///
    /// let mut chat = ChatCompletion::new();
    /// chat.model_id("gpt-4o-mini")
    ///     .timeout(Duration::from_secs(30));
    /// ```
    pub fn timeout(&mut self, timeout: Duration) -> &mut Self {
        self.timeout = Some(timeout);
        self
    }

    /// Sets the chat message history
    ///
    /// # Arguments
    ///
    /// * `messages` - Vector of chat messages representing the conversation history
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    pub fn messages(&mut self, messages: Vec<Message>) -> &mut Self {
        self.request_body.messages = messages;
        self
    }

    /// Adds a single message to the conversation history
    ///
    /// This method appends a new message to the existing conversation history.
    /// It's useful for building conversations incrementally.
    ///
    /// # Arguments
    ///
    /// * `message` - The message to add to the conversation
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use openai_tools::chat::request::ChatCompletion;
    /// use openai_tools::common::message::Message;
    /// use openai_tools::common::role::Role;
    ///
    /// let mut chat = ChatCompletion::new();
    /// chat.add_message(Message::from_string(Role::User, "Hello!"))
    ///     .add_message(Message::from_string(Role::Assistant, "Hi there!"))
    ///     .add_message(Message::from_string(Role::User, "How are you?"));
    /// ```
    pub fn add_message(&mut self, message: Message) -> &mut Self {
        self.request_body.messages.push(message);
        self
    }
    /// Sets whether to store the request and response at OpenAI
    ///
    /// # Arguments
    ///
    /// * `store` - `true` to store, `false` to not store
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    pub fn store(&mut self, store: bool) -> &mut Self {
        self.request_body.store = Option::from(store);
        self
    }

    /// Sets the frequency penalty
    ///
    /// A parameter that penalizes based on word frequency to reduce repetition.
    /// Positive values decrease repetition, negative values increase it.
    ///
    /// **Note:** Reasoning models (GPT-5, o-series) only support frequency_penalty=0.
    /// For these models, non-zero values will be ignored with a warning.
    ///
    /// # Arguments
    ///
    /// * `frequency_penalty` - Frequency penalty value (range: -2.0 to 2.0)
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    pub fn frequency_penalty(&mut self, frequency_penalty: f32) -> &mut Self {
        let support = self.request_body.model.parameter_support();
        match support.frequency_penalty {
            ParameterRestriction::FixedValue(fixed) => {
                if (frequency_penalty as f64 - fixed).abs() > f64::EPSILON {
                    tracing::warn!(
                        "Model '{}' only supports frequency_penalty={}. Ignoring frequency_penalty={}.",
                        self.request_body.model,
                        fixed,
                        frequency_penalty
                    );
                    return self;
                }
            }
            ParameterRestriction::NotSupported => {
                tracing::warn!("Model '{}' does not support frequency_penalty parameter. Ignoring.", self.request_body.model);
                return self;
            }
            ParameterRestriction::Any => {}
        }
        self.request_body.frequency_penalty = Some(frequency_penalty);
        self
    }

    /// Sets logit bias to adjust the probability of specific tokens
    ///
    /// **Note:** Reasoning models (GPT-5, o-series) do not support logit_bias.
    /// For these models, this parameter will be ignored with a warning.
    ///
    /// # Arguments
    ///
    /// * `logit_bias` - A map of token IDs to adjustment values
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    pub fn logit_bias<T: AsRef<str>>(&mut self, logit_bias: HashMap<T, i32>) -> &mut Self {
        let support = self.request_body.model.parameter_support();
        if !support.logit_bias {
            tracing::warn!("Model '{}' does not support logit_bias parameter. Ignoring.", self.request_body.model);
            return self;
        }
        self.request_body.logit_bias = Some(logit_bias.into_iter().map(|(k, v)| (k.as_ref().to_string(), v)).collect::<HashMap<String, i32>>());
        self
    }

    /// Sets whether to include probability information for each token
    ///
    /// **Note:** Reasoning models (GPT-5, o-series) do not support logprobs.
    /// For these models, this parameter will be ignored with a warning.
    ///
    /// # Arguments
    ///
    /// * `logprobs` - `true` to include probability information
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    pub fn logprobs(&mut self, logprobs: bool) -> &mut Self {
        let support = self.request_body.model.parameter_support();
        if !support.logprobs {
            tracing::warn!("Model '{}' does not support logprobs parameter. Ignoring.", self.request_body.model);
            return self;
        }
        self.request_body.logprobs = Some(logprobs);
        self
    }

    /// Sets the number of top probabilities to return for each token
    ///
    /// **Note:** Reasoning models (GPT-5, o-series) do not support top_logprobs.
    /// For these models, this parameter will be ignored with a warning.
    ///
    /// # Arguments
    ///
    /// * `top_logprobs` - Number of top probabilities (range: 0-20)
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    pub fn top_logprobs(&mut self, top_logprobs: u8) -> &mut Self {
        let support = self.request_body.model.parameter_support();
        if !support.top_logprobs {
            tracing::warn!("Model '{}' does not support top_logprobs parameter. Ignoring.", self.request_body.model);
            return self;
        }
        self.request_body.top_logprobs = Some(top_logprobs);
        self
    }

    /// Sets the maximum number of tokens to generate
    ///
    /// # Arguments
    ///
    /// * `max_completion_tokens` - Maximum number of tokens
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    pub fn max_completion_tokens(&mut self, max_completion_tokens: u64) -> &mut Self {
        self.request_body.max_completion_tokens = Option::from(max_completion_tokens);
        self
    }

    /// Sets the number of responses to generate
    ///
    /// **Note:** Reasoning models (GPT-5, o-series) only support n=1.
    /// For these models, values other than 1 will be ignored with a warning.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of responses to generate
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    pub fn n(&mut self, n: u32) -> &mut Self {
        let support = self.request_body.model.parameter_support();
        if !support.n_multiple && n != 1 {
            tracing::warn!("Model '{}' only supports n=1. Ignoring n={}.", self.request_body.model, n);
            return self;
        }
        self.request_body.n = Some(n);
        self
    }

    /// Sets the available modalities for the response
    ///
    /// # Arguments
    ///
    /// * `modalities` - List of modalities (e.g., `["text", "audio"]`)
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    pub fn modalities<T: AsRef<str>>(&mut self, modalities: Vec<T>) -> &mut Self {
        self.request_body.modalities = Option::from(modalities.into_iter().map(|m| m.as_ref().to_string()).collect::<Vec<String>>());
        self
    }

    /// Sets the presence penalty
    ///
    /// A parameter that controls the tendency to include new content in the document.
    /// Positive values encourage talking about new topics, negative values encourage
    /// staying on existing topics.
    ///
    /// **Note:** Reasoning models (GPT-5, o-series) only support presence_penalty=0.
    /// For these models, non-zero values will be ignored with a warning.
    ///
    /// # Arguments
    ///
    /// * `presence_penalty` - Presence penalty value (range: -2.0 to 2.0)
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    pub fn presence_penalty(&mut self, presence_penalty: f32) -> &mut Self {
        let support = self.request_body.model.parameter_support();
        match support.presence_penalty {
            ParameterRestriction::FixedValue(fixed) => {
                if (presence_penalty as f64 - fixed).abs() > f64::EPSILON {
                    tracing::warn!(
                        "Model '{}' only supports presence_penalty={}. Ignoring presence_penalty={}.",
                        self.request_body.model,
                        fixed,
                        presence_penalty
                    );
                    return self;
                }
            }
            ParameterRestriction::NotSupported => {
                tracing::warn!("Model '{}' does not support presence_penalty parameter. Ignoring.", self.request_body.model);
                return self;
            }
            ParameterRestriction::Any => {}
        }
        self.request_body.presence_penalty = Some(presence_penalty);
        self
    }

    /// Sets the temperature parameter to control response randomness
    ///
    /// Higher values (e.g., 1.0) produce more creative and diverse outputs,
    /// while lower values (e.g., 0.2) produce more deterministic and consistent outputs.
    ///
    /// **Note:** Reasoning models (GPT-5, o-series) only support temperature=1.0.
    /// For these models, other values will be ignored with a warning.
    ///
    /// # Arguments
    ///
    /// * `temperature` - Temperature parameter (range: 0.0 to 2.0)
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    pub fn temperature(&mut self, temperature: f32) -> &mut Self {
        let support = self.request_body.model.parameter_support();
        match support.temperature {
            ParameterRestriction::FixedValue(fixed) => {
                if (temperature as f64 - fixed).abs() > f64::EPSILON {
                    tracing::warn!("Model '{}' only supports temperature={}. Ignoring temperature={}.", self.request_body.model, fixed, temperature);
                    return self;
                }
            }
            ParameterRestriction::NotSupported => {
                tracing::warn!("Model '{}' does not support temperature parameter. Ignoring.", self.request_body.model);
                return self;
            }
            ParameterRestriction::Any => {}
        }
        self.request_body.temperature = Some(temperature);
        self
    }

    /// Sets structured output using JSON schema
    ///
    /// Enables receiving responses in a structured JSON format according to the
    /// specified JSON schema.
    ///
    /// # Arguments
    ///
    /// * `json_schema` - JSON schema defining the response structure
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    pub fn json_schema(&mut self, json_schema: Schema) -> &mut Self {
        self.request_body.response_format = Option::from(Format::new(String::from("json_schema"), json_schema));
        self
    }

    /// Sets the tools that can be called by the model
    ///
    /// Enables function calling by providing a list of tools that the model can choose to call.
    /// When tools are provided, the model may generate tool calls instead of or in addition to
    /// regular text responses.
    ///
    /// # Arguments
    ///
    /// * `tools` - Vector of tools available for the model to use
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    pub fn tools(&mut self, tools: Vec<Tool>) -> &mut Self {
        self.request_body.tools = Option::from(tools);
        self
    }

    /// Sets the safety identifier for end-user tracking
    ///
    /// A stable identifier used to help OpenAI detect users of your application
    /// that may be violating usage policies. This enables per-user safety
    /// monitoring and abuse detection.
    ///
    /// # Arguments
    ///
    /// * `safety_id` - A unique, stable identifier for the end user
    ///   (recommended: hash of email or internal user ID)
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    ///
    /// # Examples
    ///
    /// ```rust
    /// use openai_tools::chat::request::ChatCompletion;
    ///
    /// let mut chat = ChatCompletion::new();
    /// chat.safety_identifier("user_abc123");
    /// ```
    pub fn safety_identifier<T: AsRef<str>>(&mut self, safety_id: T) -> &mut Self {
        self.request_body.safety_identifier = Some(safety_id.as_ref().to_string());
        self
    }

    /// Sets a unique identifier representing your end-user, which can help to monitor and detect abuse
    ///
    /// A unique identifier representing your end-user, which can help to monitor and detect abuse.
    ///
    /// # Arguments
    ///
    /// * `user` - A identifier representing your end-user.
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    ///
    /// # Examples
    ///
    /// ```rust
    /// use openai_tools::chat::request::ChatCompletion;
    ///
    /// let mut chat = ChatCompletion::new();
    /// chat.user("abc123");
    /// ```
    pub fn user<T: AsRef<str>>(&mut self, user: T) -> &mut Self {
        self.request_body.user = Some(user.as_ref().to_string());
        self
    }

    /// Gets the current message history
    ///
    /// # Returns
    ///
    /// A vector containing the message history
    pub fn get_message_history(&self) -> Vec<Message> {
        self.request_body.messages.clone()
    }

    /// Checks if the model is a reasoning model that doesn't support custom temperature
    ///
    /// Reasoning models (o1, o3, o4 series) only support the default temperature value of 1.0.
    /// This method checks if the current model is one of these reasoning models.
    ///
    /// # Returns
    ///
    /// `true` if the model is a reasoning model, `false` otherwise
    ///
    /// # Supported Reasoning Models
    ///
    /// - `o1`, `o1-pro`, and variants
    /// - `o3`, `o3-mini`, and variants
    /// - `o4-mini` and variants
    fn is_reasoning_model(&self) -> bool {
        self.request_body.model.is_reasoning_model()
    }

    /// Sends the chat completion request to OpenAI API
    ///
    /// This method validates the request parameters, constructs the HTTP request,
    /// and sends it to the OpenAI Chat Completions endpoint.
    ///
    /// # Returns
    ///
    /// A `Result` containing the API response on success, or an error on failure.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - API key is not set
    /// - Model ID is not set
    /// - Messages are empty
    /// - Network request fails
    /// - Response parsing fails
    ///
    /// # Parameter Validation
    ///
    /// For reasoning models (GPT-5, o-series), certain parameters have restrictions:
    /// - `temperature`: only 1.0 supported
    /// - `frequency_penalty`: only 0 supported
    /// - `presence_penalty`: only 0 supported
    /// - `logprobs`, `top_logprobs`, `logit_bias`: not supported
    /// - `n`: only 1 supported
    ///
    /// **Validation occurs at two points:**
    /// 1. At setter time (when using `with_model()` constructor) - immediate warning
    /// 2. At API call time (fallback) - for cases where model is changed after setting params
    ///
    /// Unsupported parameter values are ignored with a warning and the request proceeds.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::chat::request::ChatCompletion;
    /// use openai_tools::common::message::Message;
    /// use openai_tools::common::role::Role;
    ///
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>>
    /// # {
    /// let mut chat = ChatCompletion::new();
    /// let messages = vec![Message::from_string(Role::User, "Hello!")];
    ///
    /// let response = chat
    ///     .model_id("gpt-4o-mini")
    ///     .messages(messages)
    ///     .temperature(1.0)
    ///     .chat()
    ///     .await?;
    ///
    /// println!("{}", response.choices[0].message.content.as_ref().unwrap().text.as_ref().unwrap());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// # }
    /// ```
    pub async fn chat(&mut self) -> Result<Response> {
        // Validate that messages are set
        if self.request_body.messages.is_empty() {
            return Err(OpenAIToolError::Error("Messages are not set.".into()));
        }

        // Handle reasoning models that don't support certain parameters
        // See: https://platform.openai.com/docs/guides/reasoning
        if self.is_reasoning_model() {
            let model = &self.request_body.model;

            // Temperature: only default (1.0) is supported
            if let Some(temp) = self.request_body.temperature {
                if (temp - 1.0).abs() > f32::EPSILON {
                    tracing::warn!(
                        "Reasoning model '{}' does not support custom temperature. \
                         Ignoring temperature={} and using default (1.0).",
                        model,
                        temp
                    );
                    self.request_body.temperature = None;
                }
            }

            // Frequency penalty: only 0 is supported
            if let Some(fp) = self.request_body.frequency_penalty {
                if fp.abs() > f32::EPSILON {
                    tracing::warn!(
                        "Reasoning model '{}' does not support frequency_penalty. \
                         Ignoring frequency_penalty={} and using default (0).",
                        model,
                        fp
                    );
                    self.request_body.frequency_penalty = None;
                }
            }

            // Presence penalty: only 0 is supported
            if let Some(pp) = self.request_body.presence_penalty {
                if pp.abs() > f32::EPSILON {
                    tracing::warn!(
                        "Reasoning model '{}' does not support presence_penalty. \
                         Ignoring presence_penalty={} and using default (0).",
                        model,
                        pp
                    );
                    self.request_body.presence_penalty = None;
                }
            }

            // Logprobs: not supported
            if self.request_body.logprobs.is_some() {
                tracing::warn!("Reasoning model '{}' does not support logprobs. Ignoring logprobs parameter.", model);
                self.request_body.logprobs = None;
            }

            // Top logprobs: not supported
            if self.request_body.top_logprobs.is_some() {
                tracing::warn!("Reasoning model '{}' does not support top_logprobs. Ignoring top_logprobs parameter.", model);
                self.request_body.top_logprobs = None;
            }

            // Logit bias: not supported
            if self.request_body.logit_bias.is_some() {
                tracing::warn!("Reasoning model '{}' does not support logit_bias. Ignoring logit_bias parameter.", model);
                self.request_body.logit_bias = None;
            }

            // N: only 1 is supported
            if let Some(n) = self.request_body.n {
                if n != 1 {
                    tracing::warn!(
                        "Reasoning model '{}' does not support n != 1. \
                         Ignoring n={} and using default (1).",
                        model,
                        n
                    );
                    self.request_body.n = None;
                }
            }
        }

        let body = serde_json::to_string(&self.request_body)?;

        let client = create_http_client(self.timeout)?;
        let mut headers = request::header::HeaderMap::new();
        headers.insert("Content-Type", request::header::HeaderValue::from_static("application/json"));
        headers.insert("User-Agent", request::header::HeaderValue::from_static("openai-tools-rust"));

        // Apply provider-specific authentication headers
        self.auth.apply_headers(&mut headers)?;

        if cfg!(debug_assertions) {
            // Replace API key with a placeholder in debug mode
            let body_for_debug = serde_json::to_string_pretty(&self.request_body).unwrap().replace(self.auth.api_key(), "*************");
            tracing::info!("Request body: {}", body_for_debug);
        }

        // Get the endpoint URL from the auth provider
        let endpoint = self.auth.endpoint(CHAT_COMPLETIONS_PATH);

        let response = client.post(&endpoint).headers(headers).body(body).send().await.map_err(OpenAIToolError::RequestError)?;
        let status = response.status();
        let content = response.text().await.map_err(OpenAIToolError::RequestError)?;

        if cfg!(debug_assertions) {
            tracing::info!("Response content: {}", content);
        }

        if !status.is_success() {
            if let Ok(error_resp) = serde_json::from_str::<ErrorResponse>(&content) {
                return Err(OpenAIToolError::Error(error_resp.error.message.unwrap_or_default()));
            }
            return Err(OpenAIToolError::Error(format!("API error ({}): {}", status, content)));
        }

        serde_json::from_str::<Response>(&content).map_err(OpenAIToolError::SerdeJsonError)
    }

    /// Creates a test-only ChatCompletion instance without authentication
    ///
    /// This is only available in test mode and bypasses API key requirements.
    #[cfg(test)]
    pub(crate) fn test_new_with_model(model: ChatModel) -> Self {
        use crate::common::auth::OpenAIAuth;
        Self { auth: AuthProvider::OpenAI(OpenAIAuth::new("test-key")), request_body: Body { model, ..Default::default() }, timeout: None }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::models::ChatModel;
    use std::collections::HashMap;

    // =============================================================================
    // Standard Model Parameter Tests
    // =============================================================================

    #[test]
    fn test_standard_model_accepts_all_parameters() {
        let mut chat = ChatCompletion::test_new_with_model(ChatModel::Gpt4oMini);

        // Standard models should accept all parameters
        chat.temperature(0.7);
        chat.frequency_penalty(0.5);
        chat.presence_penalty(0.5);
        chat.logprobs(true);
        chat.top_logprobs(5);
        chat.n(3);

        let logit_bias: HashMap<&str, i32> = [("1234", 10)].iter().cloned().collect();
        chat.logit_bias(logit_bias);

        assert_eq!(chat.request_body.temperature, Some(0.7));
        assert_eq!(chat.request_body.frequency_penalty, Some(0.5));
        assert_eq!(chat.request_body.presence_penalty, Some(0.5));
        assert_eq!(chat.request_body.logprobs, Some(true));
        assert_eq!(chat.request_body.top_logprobs, Some(5));
        assert_eq!(chat.request_body.n, Some(3));
        assert!(chat.request_body.logit_bias.is_some());
    }

    #[test]
    fn test_gpt4o_accepts_all_parameters() {
        let mut chat = ChatCompletion::test_new_with_model(ChatModel::Gpt4o);

        chat.temperature(0.3);
        chat.frequency_penalty(-1.0);
        chat.presence_penalty(1.5);

        assert_eq!(chat.request_body.temperature, Some(0.3));
        assert_eq!(chat.request_body.frequency_penalty, Some(-1.0));
        assert_eq!(chat.request_body.presence_penalty, Some(1.5));
    }

    #[test]
    fn test_gpt4_1_accepts_all_parameters() {
        let mut chat = ChatCompletion::test_new_with_model(ChatModel::Gpt4_1);

        chat.temperature(1.5);
        chat.frequency_penalty(0.8);
        chat.n(2);

        assert_eq!(chat.request_body.temperature, Some(1.5));
        assert_eq!(chat.request_body.frequency_penalty, Some(0.8));
        assert_eq!(chat.request_body.n, Some(2));
    }

    // =============================================================================
    // O-Series Reasoning Model Tests
    // =============================================================================

    #[test]
    fn test_o1_ignores_non_default_temperature() {
        let mut chat = ChatCompletion::test_new_with_model(ChatModel::O1);

        // Non-default temperature should be ignored
        chat.temperature(0.5);
        assert_eq!(chat.request_body.temperature, None);

        // Default temperature (1.0) should be accepted
        chat.temperature(1.0);
        assert_eq!(chat.request_body.temperature, Some(1.0));
    }

    #[test]
    fn test_o3_mini_ignores_non_default_temperature() {
        let mut chat = ChatCompletion::test_new_with_model(ChatModel::O3Mini);

        chat.temperature(0.3);
        assert_eq!(chat.request_body.temperature, None);
    }

    #[test]
    fn test_o4_mini_ignores_non_default_temperature() {
        let mut chat = ChatCompletion::test_new_with_model(ChatModel::O4Mini);

        chat.temperature(0.7);
        assert_eq!(chat.request_body.temperature, None);
    }

    #[test]
    fn test_o1_ignores_frequency_penalty() {
        let mut chat = ChatCompletion::test_new_with_model(ChatModel::O1);

        // Non-zero frequency_penalty should be ignored
        chat.frequency_penalty(0.5);
        assert_eq!(chat.request_body.frequency_penalty, None);

        // Zero value should be accepted
        chat.frequency_penalty(0.0);
        assert_eq!(chat.request_body.frequency_penalty, Some(0.0));
    }

    #[test]
    fn test_o3_ignores_presence_penalty() {
        let mut chat = ChatCompletion::test_new_with_model(ChatModel::O3);

        // Non-zero presence_penalty should be ignored
        chat.presence_penalty(0.5);
        assert_eq!(chat.request_body.presence_penalty, None);

        // Zero value should be accepted
        chat.presence_penalty(0.0);
        assert_eq!(chat.request_body.presence_penalty, Some(0.0));
    }

    #[test]
    fn test_o1_ignores_logprobs() {
        let mut chat = ChatCompletion::test_new_with_model(ChatModel::O1);

        chat.logprobs(true);
        assert_eq!(chat.request_body.logprobs, None);
    }

    #[test]
    fn test_o3_mini_ignores_top_logprobs() {
        let mut chat = ChatCompletion::test_new_with_model(ChatModel::O3Mini);

        chat.top_logprobs(5);
        assert_eq!(chat.request_body.top_logprobs, None);
    }

    #[test]
    fn test_o1_ignores_logit_bias() {
        let mut chat = ChatCompletion::test_new_with_model(ChatModel::O1);

        let logit_bias: HashMap<&str, i32> = [("1234", 10)].iter().cloned().collect();
        chat.logit_bias(logit_bias);
        assert_eq!(chat.request_body.logit_bias, None);
    }

    #[test]
    fn test_o1_ignores_n_greater_than_1() {
        let mut chat = ChatCompletion::test_new_with_model(ChatModel::O1);

        // n > 1 should be ignored
        chat.n(3);
        assert_eq!(chat.request_body.n, None);

        // n = 1 should be accepted
        chat.n(1);
        assert_eq!(chat.request_body.n, Some(1));
    }

    // =============================================================================
    // GPT-5 Series Reasoning Model Tests
    // =============================================================================

    #[test]
    fn test_gpt5_2_ignores_non_default_temperature() {
        let mut chat = ChatCompletion::test_new_with_model(ChatModel::Gpt5_2);

        chat.temperature(0.5);
        assert_eq!(chat.request_body.temperature, None);

        chat.temperature(1.0);
        assert_eq!(chat.request_body.temperature, Some(1.0));
    }

    #[test]
    fn test_gpt5_1_ignores_non_default_temperature() {
        let mut chat = ChatCompletion::test_new_with_model(ChatModel::Gpt5_1);

        chat.temperature(0.3);
        assert_eq!(chat.request_body.temperature, None);
    }

    #[test]
    fn test_gpt5_mini_ignores_frequency_penalty() {
        let mut chat = ChatCompletion::test_new_with_model(ChatModel::Gpt5Mini);

        chat.frequency_penalty(0.5);
        assert_eq!(chat.request_body.frequency_penalty, None);
    }

    #[test]
    fn test_gpt5_2_pro_ignores_presence_penalty() {
        let mut chat = ChatCompletion::test_new_with_model(ChatModel::Gpt5_2Pro);

        chat.presence_penalty(0.8);
        assert_eq!(chat.request_body.presence_penalty, None);
    }

    #[test]
    fn test_gpt5_1_codex_max_ignores_logprobs() {
        let mut chat = ChatCompletion::test_new_with_model(ChatModel::Gpt5_1CodexMax);

        chat.logprobs(true);
        assert_eq!(chat.request_body.logprobs, None);
    }

    #[test]
    fn test_gpt5_2_chat_latest_ignores_n_greater_than_1() {
        let mut chat = ChatCompletion::test_new_with_model(ChatModel::Gpt5_2ChatLatest);

        chat.n(5);
        assert_eq!(chat.request_body.n, None);
    }

    // =============================================================================
    // Multiple Restricted Parameters Tests
    // =============================================================================

    #[test]
    fn test_o1_ignores_all_restricted_parameters_at_once() {
        let mut chat = ChatCompletion::test_new_with_model(ChatModel::O1);

        // Set all restricted parameters
        chat.temperature(0.5);
        chat.frequency_penalty(0.5);
        chat.presence_penalty(0.5);
        chat.logprobs(true);
        chat.top_logprobs(5);
        chat.n(3);

        let logit_bias: HashMap<&str, i32> = [("1234", 10)].iter().cloned().collect();
        chat.logit_bias(logit_bias);

        // All should be ignored
        assert_eq!(chat.request_body.temperature, None);
        assert_eq!(chat.request_body.frequency_penalty, None);
        assert_eq!(chat.request_body.presence_penalty, None);
        assert_eq!(chat.request_body.logprobs, None);
        assert_eq!(chat.request_body.top_logprobs, None);
        assert_eq!(chat.request_body.n, None);
        assert_eq!(chat.request_body.logit_bias, None);
    }

    #[test]
    fn test_gpt5_2_ignores_all_restricted_parameters_at_once() {
        let mut chat = ChatCompletion::test_new_with_model(ChatModel::Gpt5_2);

        chat.temperature(0.5);
        chat.frequency_penalty(0.5);
        chat.presence_penalty(0.5);
        chat.logprobs(true);
        chat.top_logprobs(5);
        chat.n(3);

        let logit_bias: HashMap<&str, i32> = [("1234", 10)].iter().cloned().collect();
        chat.logit_bias(logit_bias);

        assert_eq!(chat.request_body.temperature, None);
        assert_eq!(chat.request_body.frequency_penalty, None);
        assert_eq!(chat.request_body.presence_penalty, None);
        assert_eq!(chat.request_body.logprobs, None);
        assert_eq!(chat.request_body.top_logprobs, None);
        assert_eq!(chat.request_body.n, None);
        assert_eq!(chat.request_body.logit_bias, None);
    }

    // =============================================================================
    // Custom Model Tests
    // =============================================================================

    #[test]
    fn test_custom_gpt5_model_detected_as_reasoning() {
        let mut chat = ChatCompletion::test_new_with_model(ChatModel::custom("gpt-5.3-preview"));

        // Custom GPT-5 models should be treated as reasoning models
        chat.temperature(0.5);
        assert_eq!(chat.request_body.temperature, None);
    }

    #[test]
    fn test_custom_o1_model_detected_as_reasoning() {
        let mut chat = ChatCompletion::test_new_with_model(ChatModel::custom("o1-pro-2025-01-15"));

        // Custom o1-series models should be treated as reasoning models
        chat.temperature(0.5);
        assert_eq!(chat.request_body.temperature, None);
    }

    #[test]
    fn test_custom_o3_model_detected_as_reasoning() {
        let mut chat = ChatCompletion::test_new_with_model(ChatModel::custom("o3-high"));

        // Custom o3-series models should be treated as reasoning models
        chat.temperature(0.5);
        assert_eq!(chat.request_body.temperature, None);
    }

    #[test]
    fn test_custom_o4_model_detected_as_reasoning() {
        let mut chat = ChatCompletion::test_new_with_model(ChatModel::custom("o4-mini-preview"));

        // Custom o4-series models should be treated as reasoning models
        chat.temperature(0.5);
        assert_eq!(chat.request_body.temperature, None);
    }

    #[test]
    fn test_custom_standard_model_accepts_all_parameters() {
        let mut chat = ChatCompletion::test_new_with_model(ChatModel::custom("ft:gpt-4o-mini:org::123"));

        // Fine-tuned standard models should accept all parameters
        chat.temperature(0.7);
        chat.frequency_penalty(0.5);
        chat.n(2);

        assert_eq!(chat.request_body.temperature, Some(0.7));
        assert_eq!(chat.request_body.frequency_penalty, Some(0.5));
        assert_eq!(chat.request_body.n, Some(2));
    }

    // =============================================================================
    // Parameter Boundary Tests
    // =============================================================================

    #[test]
    fn test_temperature_boundary_values() {
        let mut chat = ChatCompletion::test_new_with_model(ChatModel::Gpt4oMini);

        // Minimum value
        chat.temperature(0.0);
        assert_eq!(chat.request_body.temperature, Some(0.0));

        // Maximum value
        chat.temperature(2.0);
        assert_eq!(chat.request_body.temperature, Some(2.0));
    }

    #[test]
    fn test_frequency_penalty_boundary_values() {
        let mut chat = ChatCompletion::test_new_with_model(ChatModel::Gpt4oMini);

        // Minimum value
        chat.frequency_penalty(-2.0);
        assert_eq!(chat.request_body.frequency_penalty, Some(-2.0));

        // Maximum value
        chat.frequency_penalty(2.0);
        assert_eq!(chat.request_body.frequency_penalty, Some(2.0));
    }

    #[test]
    fn test_presence_penalty_boundary_values() {
        let mut chat = ChatCompletion::test_new_with_model(ChatModel::Gpt4oMini);

        // Minimum value
        chat.presence_penalty(-2.0);
        assert_eq!(chat.request_body.presence_penalty, Some(-2.0));

        // Maximum value
        chat.presence_penalty(2.0);
        assert_eq!(chat.request_body.presence_penalty, Some(2.0));
    }

    // =============================================================================
    // Model-Specific Unrestricted Parameters Tests
    // =============================================================================

    #[test]
    fn test_max_completion_tokens_accepted_by_all_models() {
        // Standard model
        let mut chat_standard = ChatCompletion::test_new_with_model(ChatModel::Gpt4oMini);
        chat_standard.max_completion_tokens(1000);
        assert_eq!(chat_standard.request_body.max_completion_tokens, Some(1000));

        // Reasoning model
        let mut chat_reasoning = ChatCompletion::test_new_with_model(ChatModel::O1);
        chat_reasoning.max_completion_tokens(2000);
        assert_eq!(chat_reasoning.request_body.max_completion_tokens, Some(2000));

        // GPT-5 model
        let mut chat_gpt5 = ChatCompletion::test_new_with_model(ChatModel::Gpt5_2);
        chat_gpt5.max_completion_tokens(3000);
        assert_eq!(chat_gpt5.request_body.max_completion_tokens, Some(3000));
    }

    #[test]
    fn test_store_accepted_by_all_models() {
        let mut chat_standard = ChatCompletion::test_new_with_model(ChatModel::Gpt4oMini);
        chat_standard.store(true);
        assert_eq!(chat_standard.request_body.store, Some(true));

        let mut chat_reasoning = ChatCompletion::test_new_with_model(ChatModel::O1);
        chat_reasoning.store(false);
        assert_eq!(chat_reasoning.request_body.store, Some(false));
    }

    // =============================================================================
    // Chat API Content Serialization Tests
    // =============================================================================

    #[test]
    fn test_chat_text_content_serialization() {
        use crate::common::message::Content;

        let content = Content::from_text("Hello, world!");
        let wrapper = ChatContentRef(&content);
        let json = serde_json::to_value(&wrapper).unwrap();

        assert_eq!(json["type"], "text");
        assert_eq!(json["text"], "Hello, world!");
        assert!(json.get("image_url").is_none());
    }

    #[test]
    fn test_chat_image_content_serialization() {
        use crate::common::message::Content;

        let content = Content::from_image_url("https://example.com/image.png");
        let wrapper = ChatContentRef(&content);
        let json = serde_json::to_value(&wrapper).unwrap();

        assert_eq!(json["type"], "image_url");
        assert_eq!(json["image_url"]["url"], "https://example.com/image.png");
    }

    #[test]
    fn test_chat_multimodal_message_serialization() {
        use crate::common::message::{Content, Message};
        use crate::common::role::Role;

        let contents = vec![Content::from_text("What's in this image?"), Content::from_image_url("https://example.com/image.png")];
        let message = Message::from_message_array(Role::User, contents);
        let wrapper = ChatMessageRef(&message);
        let json = serde_json::to_value(&wrapper).unwrap();

        assert_eq!(json["role"], "user");
        let content_arr = json["content"].as_array().unwrap();
        assert_eq!(content_arr.len(), 2);

        // First element: text
        assert_eq!(content_arr[0]["type"], "text");
        assert_eq!(content_arr[0]["text"], "What's in this image?");

        // Second element: image_url with nested object
        assert_eq!(content_arr[1]["type"], "image_url");
        assert_eq!(content_arr[1]["image_url"]["url"], "https://example.com/image.png");
    }

    #[test]
    fn test_chat_single_text_message_serialization() {
        use crate::common::message::Message;
        use crate::common::role::Role;

        let message = Message::from_string(Role::User, "Hello!");
        let wrapper = ChatMessageRef(&message);
        let json = serde_json::to_value(&wrapper).unwrap();

        assert_eq!(json["role"], "user");
        // Single text content should be serialized as a plain string, not an array
        assert_eq!(json["content"], "Hello!");
    }

    #[test]
    fn test_chat_body_messages_serialization() {
        use crate::common::message::{Content, Message};
        use crate::common::role::Role;

        let messages = vec![
            Message::from_string(Role::System, "You are a helpful assistant."),
            Message::from_message_array(
                Role::User,
                vec![Content::from_text("Describe this image"), Content::from_image_url("https://example.com/photo.jpg")],
            ),
        ];

        let body = Body { model: ChatModel::Gpt4oMini, messages, ..Default::default() };

        let json = serde_json::to_value(&body).unwrap();
        let msgs = json["messages"].as_array().unwrap();

        // System message: plain string content
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[0]["content"], "You are a helpful assistant.");

        // User multimodal message: array content with Chat API types
        assert_eq!(msgs[1]["role"], "user");
        let content_arr = msgs[1]["content"].as_array().unwrap();
        assert_eq!(content_arr[0]["type"], "text");
        assert_eq!(content_arr[1]["type"], "image_url");
        assert_eq!(content_arr[1]["image_url"]["url"], "https://example.com/photo.jpg");
    }

    #[test]
    fn test_safety_identifier() {
        let mut chat = ChatCompletion::test_new_with_model(ChatModel::Gpt4oMini);
        chat.safety_identifier("user_abc123");
        assert_eq!(chat.request_body.safety_identifier, Some("user_abc123".to_string()));

        // Verify serialization
        let json = serde_json::to_value(&chat.request_body).unwrap();
        assert_eq!(json["safety_identifier"], "user_abc123");
    }

    #[test]
    fn test_user() {
        let mut chat = ChatCompletion::test_new_with_model(ChatModel::Gpt4oMini);
        chat.user("abc123");
        assert_eq!(chat.request_body.user, Some("abc123".to_string()));

        // Verify serialization
        let json = serde_json::to_value(&chat.request_body).unwrap();
        assert_eq!(json["user"], "abc123");
    }

    #[test]
    fn test_safety_identifier_not_serialized_when_none() {
        let chat = ChatCompletion::test_new_with_model(ChatModel::Gpt4oMini);
        let json = serde_json::to_value(&chat.request_body).unwrap();
        assert!(json.get("safety_identifier").is_none());
    }
}
