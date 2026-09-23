//! OpenAI Embeddings API Request Module
//!
//! This module provides the functionality to build and send requests to the OpenAI Embeddings API.
//! It offers a builder pattern for constructing embedding requests, allowing you to convert text
//! into numerical vector representations that capture semantic meaning.
//!
//! # Key Features
//!
//! - **Builder Pattern**: Fluent API for constructing embedding requests
//! - **Single & Batch Input**: Support for single text or multiple texts at once
//! - **Encoding Formats**: Support for `float` and `base64` output formats
//! - **Error Handling**: Robust error management and validation
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use openai_tools::embedding::request::Embedding;
//! use openai_tools::common::models::EmbeddingModel;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Initialize the embedding client
//!     let mut embedding = Embedding::new()?;
//!
//!     // Generate embedding for a single text
//!     let response = embedding
//!         .model(EmbeddingModel::TextEmbedding3Small)
//!         .input_text("Hello, world!")
//!         .embed()
//!         .await?;
//!
//!     let vector = response.data[0].embedding.as_1d().unwrap();
//!     println!("Embedding dimension: {}", vector.len());
//!     Ok(())
//! }
//! ```
//!
//! # Batch Processing
//!
//! ```rust,no_run
//! use openai_tools::embedding::request::Embedding;
//! use openai_tools::common::models::EmbeddingModel;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut embedding = Embedding::new()?;
//!
//!     // Embed multiple texts in a single request
//!     let texts = vec!["First text", "Second text", "Third text"];
//!
//!     let response = embedding
//!         .model(EmbeddingModel::TextEmbedding3Small)
//!         .input_text_array(texts)
//!         .embed()
//!         .await?;
//!
//!     for data in &response.data {
//!         println!("Index {}: {} dimensions",
//!                  data.index,
//!                  data.embedding.as_1d().unwrap().len());
//!     }
//!     Ok(())
//! }
//! ```

use crate::common::auth::{AuthProvider, OpenAIAuth};
use crate::common::client::create_http_client;
use crate::common::errors::{ErrorResponse, OpenAIToolError, Result};
use crate::common::models::EmbeddingModel;
use crate::embedding::response::Response;
use core::str;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Internal structure for handling input text in embedding requests.
///
/// This struct supports two input formats:
/// - Single text string (`input_text`)
/// - Array of text strings (`input_text_array`)
///
/// The custom `Serialize` implementation ensures proper JSON formatting
/// based on which input type is provided.
#[derive(Debug, Clone, Deserialize, Default)]
struct Input {
    /// Single input text for embedding
    #[serde(skip_serializing_if = "String::is_empty")]
    input_text: String,
    /// Array of input texts for batch embedding
    #[serde(skip_serializing_if = "Vec::is_empty")]
    input_text_array: Vec<String>,
}

impl Input {
    /// Creates an Input from a single text string.
    ///
    /// # Arguments
    ///
    /// * `input_text` - The text to embed
    ///
    /// # Returns
    ///
    /// A new `Input` instance with the single text set
    pub fn from_text(input_text: &str) -> Self {
        Self { input_text: input_text.to_string(), input_text_array: vec![] }
    }

    /// Creates an Input from an array of text strings.
    ///
    /// # Arguments
    ///
    /// * `input_text_array` - Vector of texts to embed
    ///
    /// # Returns
    ///
    /// A new `Input` instance with the text array set
    pub fn from_text_array(input_text_array: Vec<String>) -> Self {
        Self { input_text: String::new(), input_text_array }
    }
}

/// Custom serialization for Input to match OpenAI API format.
///
/// The OpenAI Embeddings API accepts either a single string or an array of strings
/// for the `input` field. This implementation serializes to the appropriate format
/// based on which field is populated.
impl Serialize for Input {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        if !self.input_text.is_empty() && self.input_text_array.is_empty() {
            self.input_text.serialize(serializer)
        } else if self.input_text.is_empty() && !self.input_text_array.is_empty() {
            self.input_text_array.serialize(serializer)
        } else {
            // Default to empty string if both are empty
            "".serialize(serializer)
        }
    }
}

/// Request body structure for the OpenAI Embeddings API.
///
/// Contains all parameters that can be sent to the API endpoint.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
struct Body {
    /// The model to use for embedding generation
    model: EmbeddingModel,
    /// The input text(s) to embed
    input: Input,
    /// The format for the output embeddings ("float" or "base64")
    encoding_format: Option<String>,
}

/// Default API path for Embeddings
const EMBEDDINGS_PATH: &str = "embeddings";

/// Main struct for building and sending embedding requests to the OpenAI API.
///
/// This struct provides a builder pattern interface for constructing embedding
/// requests with various parameters. Use [`Embedding::new()`] to create a new
/// instance, then chain methods to configure the request before calling
/// [`embed()`](Embedding::embed).
///
/// # Providers
///
/// The client supports two providers:
/// - **OpenAI**: Standard OpenAI API (default)
/// - **Azure**: Azure OpenAI Service
///
/// # Example
///
/// ```rust,no_run
/// use openai_tools::embedding::request::Embedding;
/// use openai_tools::common::models::EmbeddingModel;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let mut embedding = Embedding::new()?;
///
///     let response = embedding
///         .model(EmbeddingModel::TextEmbedding3Small)
///         .input_text("Sample text")
///         .embed()
///         .await?;
///
///     Ok(())
/// }
/// ```
pub struct Embedding {
    /// Authentication provider (OpenAI or Azure)
    auth: AuthProvider,
    /// Request body containing model and input parameters
    body: Body,
    /// Optional request timeout duration
    timeout: Option<Duration>,
}

impl Embedding {
    /// Creates a new Embedding instance for OpenAI API.
    ///
    /// Initializes the embedding client by loading the OpenAI API key from
    /// the environment variable `OPENAI_API_KEY`. Supports `.env` file loading
    /// via dotenvy.
    ///
    /// # Returns
    ///
    /// * `Ok(Embedding)` - A new embedding instance ready for configuration
    /// * `Err(OpenAIToolError)` - If the API key is not found in the environment
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::embedding::request::Embedding;
    ///
    /// let embedding = Embedding::new().expect("API key should be set");
    /// ```
    pub fn new() -> Result<Self> {
        let auth = AuthProvider::openai_from_env()?;
        let body = Body::default();
        Ok(Self { auth, body, timeout: None })
    }

    /// Creates a new Embedding instance with a custom authentication provider
    ///
    /// Use this to explicitly configure OpenAI or Azure authentication.
    ///
    /// # Arguments
    ///
    /// * `auth` - The authentication provider
    ///
    /// # Returns
    ///
    /// A new Embedding instance with the specified auth provider
    pub fn with_auth(auth: AuthProvider) -> Self {
        Self { auth, body: Body::default(), timeout: None }
    }

    /// Creates a new Embedding instance for Azure OpenAI API
    ///
    /// Loads configuration from Azure-specific environment variables.
    ///
    /// # Returns
    ///
    /// `Result<Embedding>` - Configured for Azure or error if env vars missing
    pub fn azure() -> Result<Self> {
        let auth = AuthProvider::azure_from_env()?;
        Ok(Self { auth, body: Body::default(), timeout: None })
    }

    /// Creates a new Embedding instance by auto-detecting the provider
    ///
    /// Tries Azure first (if AZURE_OPENAI_API_KEY is set), then falls back to OpenAI.
    pub fn detect_provider() -> Result<Self> {
        let auth = AuthProvider::from_env()?;
        Ok(Self { auth, body: Body::default(), timeout: None })
    }

    /// Creates a new Embedding instance with URL-based provider detection
    ///
    /// Analyzes the URL pattern to determine the provider:
    /// - URLs containing `.openai.azure.com` → Azure
    /// - All other URLs → OpenAI-compatible
    ///
    /// # Arguments
    ///
    /// * `base_url` - The complete base URL for API requests
    /// * `api_key` - The API key or token
    pub fn with_url<S: Into<String>>(base_url: S, api_key: S) -> Self {
        let auth = AuthProvider::from_url_with_key(base_url, api_key);
        Self { auth, body: Body::default(), timeout: None }
    }

    /// Creates a new Embedding instance from URL using environment variables
    ///
    /// Analyzes the URL pattern to determine the provider, then loads
    /// credentials from the appropriate environment variables.
    pub fn from_url<S: Into<String>>(url: S) -> Result<Self> {
        let auth = AuthProvider::from_url(url)?;
        Ok(Self { auth, body: Body::default(), timeout: None })
    }

    /// Returns the authentication provider
    pub fn auth(&self) -> &AuthProvider {
        &self.auth
    }

    /// Sets a custom API endpoint URL (OpenAI only)
    ///
    /// Use this to point to alternative OpenAI-compatible APIs.
    ///
    /// # Arguments
    ///
    /// * `url` - The base URL (e.g., `https://my-proxy.example.com/v1`)
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    pub fn base_url<T: AsRef<str>>(&mut self, url: T) -> &mut Self {
        if let AuthProvider::OpenAI(ref openai_auth) = self.auth {
            let new_auth = OpenAIAuth::new(openai_auth.api_key()).with_base_url(url.as_ref());
            self.auth = AuthProvider::OpenAI(new_auth);
        } else {
            tracing::warn!("base_url() is only supported for OpenAI provider. Use azure() or with_auth() for Azure.");
        }
        self
    }

    /// Sets the model to use for embedding generation.
    ///
    /// # Arguments
    ///
    /// * `model` - The embedding model to use
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::embedding::request::Embedding;
    /// use openai_tools::common::models::EmbeddingModel;
    ///
    /// let mut embedding = Embedding::new().unwrap();
    /// embedding.model(EmbeddingModel::TextEmbedding3Small);
    /// ```
    pub fn model(&mut self, model: EmbeddingModel) -> &mut Self {
        self.body.model = model;
        self
    }

    /// Sets the model using a string ID (for backward compatibility).
    ///
    /// Prefer using [`model`](Self::model) with `EmbeddingModel` enum for type safety.
    ///
    /// # Arguments
    ///
    /// * `model_id` - The model identifier string (e.g., "text-embedding-3-small")
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    #[deprecated(since = "0.2.0", note = "Use `model(EmbeddingModel)` instead for type safety")]
    pub fn model_id<T: AsRef<str>>(&mut self, model_id: T) -> &mut Self {
        self.body.model = EmbeddingModel::from(model_id.as_ref());
        self
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
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use std::time::Duration;
    /// use openai_tools::embedding::request::Embedding;
    /// use openai_tools::common::models::EmbeddingModel;
    ///
    /// let mut embedding = Embedding::new().unwrap();
    /// embedding.model(EmbeddingModel::TextEmbedding3Small)
    ///     .timeout(Duration::from_secs(30));
    /// ```
    pub fn timeout(&mut self, timeout: Duration) -> &mut Self {
        self.timeout = Some(timeout);
        self
    }

    /// Sets a single text input for embedding.
    ///
    /// Use this method when you want to embed a single piece of text.
    /// For multiple texts, use [`input_text_array`](Self::input_text_array) instead.
    ///
    /// # Arguments
    ///
    /// * `input_text` - The text to convert into an embedding vector
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use openai_tools::embedding::request::Embedding;
    /// # let mut embedding = Embedding::new().unwrap();
    /// embedding.input_text("Hello, world!");
    /// ```
    pub fn input_text<T: AsRef<str>>(&mut self, input_text: T) -> &mut Self {
        self.body.input = Input::from_text(input_text.as_ref());
        self
    }

    /// Sets multiple text inputs for batch embedding.
    ///
    /// Use this method when you want to embed multiple texts in a single API call.
    /// This is more efficient than making separate requests for each text.
    ///
    /// # Arguments
    ///
    /// * `input_text_array` - Vector of texts to convert into embedding vectors
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use openai_tools::embedding::request::Embedding;
    /// # let mut embedding = Embedding::new().unwrap();
    /// let texts = vec!["First text", "Second text", "Third text"];
    /// embedding.input_text_array(texts);
    /// ```
    pub fn input_text_array<T: AsRef<str>>(&mut self, input_text_array: Vec<T>) -> &mut Self {
        let input_strings = input_text_array.into_iter().map(|s| s.as_ref().to_string()).collect();
        self.body.input = Input::from_text_array(input_strings);
        self
    }

    /// Sets the encoding format for the output embeddings.
    ///
    /// # Arguments
    ///
    /// * `encoding_format` - Either "float" (default) or "base64"
    ///   - `"float"`: Returns embeddings as arrays of floating point numbers
    ///   - `"base64"`: Returns embeddings as base64-encoded strings (more compact)
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    ///
    /// # Panics
    ///
    /// Panics if `encoding_format` is not "float" or "base64"
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use openai_tools::embedding::request::Embedding;
    /// # let mut embedding = Embedding::new().unwrap();
    /// embedding.encoding_format("float");
    /// ```
    pub fn encoding_format<T: AsRef<str>>(&mut self, encoding_format: T) -> &mut Self {
        let encoding_format = encoding_format.as_ref();
        assert!(encoding_format == "float" || encoding_format == "base64", "encoding_format must be either 'float' or 'base64'");
        self.body.encoding_format = Some(encoding_format.to_string());
        self
    }

    /// Sends the embedding request to the OpenAI API.
    ///
    /// This method validates the request parameters, constructs the HTTP request,
    /// sends it to the OpenAI Embeddings API endpoint, and parses the response.
    ///
    /// # Returns
    ///
    /// * `Ok(Response)` - The embedding response containing vectors and metadata
    /// * `Err(OpenAIToolError)` - If validation fails, the request fails, or parsing fails
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - API key is not set
    /// - Model ID is not set
    /// - Input text is not set
    /// - Network request fails
    /// - Response parsing fails
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use openai_tools::embedding::request::Embedding;
    /// # use openai_tools::common::models::EmbeddingModel;
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut embedding = Embedding::new()?;
    /// let response = embedding
    ///     .model(EmbeddingModel::TextEmbedding3Small)
    ///     .input_text("Hello, world!")
    ///     .embed()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn embed(&self) -> Result<Response> {
        // Validate that input text is set
        if self.body.input.input_text.is_empty() && self.body.input.input_text_array.is_empty() {
            return Err(OpenAIToolError::Error("Input text is not set.".into()));
        }

        let body = serde_json::to_string(&self.body)?;

        let client = create_http_client(self.timeout)?;
        let mut headers = request::header::HeaderMap::new();
        headers.insert("Content-Type", request::header::HeaderValue::from_static("application/json"));
        headers.insert("User-Agent", request::header::HeaderValue::from_static("openai-tools-rust"));

        // Apply provider-specific authentication headers
        self.auth.apply_headers(&mut headers)?;

        if cfg!(test) {
            // Replace API key with a placeholder in debug mode
            let body_for_debug = serde_json::to_string_pretty(&self.body).unwrap().replace(self.auth.api_key(), "*************");
            tracing::info!("Request body: {}", body_for_debug);
        }

        // Get the endpoint URL from the auth provider
        let endpoint = self.auth.endpoint(EMBEDDINGS_PATH);

        let response = client.post(&endpoint).headers(headers).body(body).send().await.map_err(OpenAIToolError::RequestError)?;
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

        serde_json::from_str::<Response>(&content).map_err(OpenAIToolError::SerdeJsonError)
    }
}
