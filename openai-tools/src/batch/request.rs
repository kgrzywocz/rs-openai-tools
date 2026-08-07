//! OpenAI Batch API Request Module
//!
//! This module provides the functionality to interact with the OpenAI Batch API.
//! It allows you to create, list, retrieve, and cancel batch jobs.
//!
//! # Key Features
//!
//! - **Create Batch**: Submit a batch of requests for asynchronous processing
//! - **Retrieve Batch**: Get the status and details of a batch job
//! - **List Batches**: List all batch jobs
//! - **Cancel Batch**: Cancel an in-progress batch job
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use openai_tools::batch::request::{Batches, CreateBatchRequest, BatchEndpoint, CompletionWindow};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let batches = Batches::new()?;
//!
//!     // List all batches
//!     let response = batches.list(None, None).await?;
//!     for batch in &response.data {
//!         println!("{}: {:?}", batch.id, batch.status);
//!     }
//!
//!     Ok(())
//! }
//! ```

use crate::batch::response::{BatchListResponse, BatchObject};
use crate::common::auth::AuthProvider;
use crate::common::client::create_http_client;
use crate::common::errors::{OpenAIToolError, Result};
use serde::Serialize;
use std::collections::HashMap;
use std::time::Duration;

/// Default API path for Batches
const BATCHES_PATH: &str = "batches";

/// The API endpoint to use for batch requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[non_exhaustive]
pub enum BatchEndpoint {
    /// Chat Completions API (/v1/chat/completions)
    #[serde(rename = "/v1/chat/completions")]
    ChatCompletions,
    /// Embeddings API (/v1/embeddings)
    #[serde(rename = "/v1/embeddings")]
    Embeddings,
    /// Completions API (/v1/completions)
    #[serde(rename = "/v1/completions")]
    Completions,
    /// Responses API (/v1/responses)
    #[serde(rename = "/v1/responses")]
    Responses,
    /// Moderations API (/v1/moderations)
    #[serde(rename = "/v1/moderations")]
    Moderations,
}

impl BatchEndpoint {
    /// Returns the string representation of the endpoint.
    pub fn as_str(&self) -> &'static str {
        match self {
            BatchEndpoint::ChatCompletions => "/v1/chat/completions",
            BatchEndpoint::Embeddings => "/v1/embeddings",
            BatchEndpoint::Completions => "/v1/completions",
            BatchEndpoint::Responses => "/v1/responses",
            BatchEndpoint::Moderations => "/v1/moderations",
        }
    }
}

/// The time window in which the batch must be completed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Default)]
#[non_exhaustive]
pub enum CompletionWindow {
    /// 24 hours
    #[serde(rename = "24h")]
    #[default]
    Hours24,
}

impl CompletionWindow {
    /// Returns the string representation of the completion window.
    pub fn as_str(&self) -> &'static str {
        match self {
            CompletionWindow::Hours24 => "24h",
        }
    }
}

/// Request to create a new batch job.
#[derive(Debug, Clone, Serialize)]
pub struct CreateBatchRequest {
    /// The ID of an uploaded file that contains requests for the batch.
    /// The file must be uploaded with purpose "batch".
    pub input_file_id: String,

    /// The endpoint to use for all requests in the batch.
    pub endpoint: BatchEndpoint,

    /// The time window in which the batch must be completed.
    pub completion_window: CompletionWindow,

    /// Optional metadata to attach to the batch.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, String>>,
}

impl CreateBatchRequest {
    /// Creates a new batch request with the given input file ID and endpoint.
    ///
    /// # Arguments
    ///
    /// * `input_file_id` - The ID of the uploaded input file
    /// * `endpoint` - The API endpoint to use for the batch
    ///
    /// # Example
    ///
    /// ```rust
    /// use openai_tools::batch::request::{CreateBatchRequest, BatchEndpoint};
    ///
    /// let request = CreateBatchRequest::new("file-abc123", BatchEndpoint::ChatCompletions);
    /// ```
    pub fn new(input_file_id: impl Into<String>, endpoint: BatchEndpoint) -> Self {
        Self { input_file_id: input_file_id.into(), endpoint, completion_window: CompletionWindow::default(), metadata: None }
    }

    /// Sets the metadata for the batch.
    ///
    /// # Arguments
    ///
    /// * `metadata` - Key-value pairs to attach to the batch
    pub fn with_metadata(mut self, metadata: HashMap<String, String>) -> Self {
        self.metadata = Some(metadata);
        self
    }
}

/// Client for interacting with the OpenAI Batch API.
///
/// This struct provides methods to create, list, retrieve, and cancel batch jobs.
/// Use [`Batches::new()`] to create a new instance.
///
/// # Example
///
/// ```rust,no_run
/// use openai_tools::batch::request::{Batches, CreateBatchRequest, BatchEndpoint};
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let batches = Batches::new()?;
///
///     // Create a batch job
///     let request = CreateBatchRequest::new("file-abc123", BatchEndpoint::ChatCompletions);
///     let batch = batches.create(request).await?;
///     println!("Created batch: {} ({:?})", batch.id, batch.status);
///
///     Ok(())
/// }
/// ```
pub struct Batches {
    /// Authentication provider (OpenAI or Azure)
    auth: AuthProvider,
    /// Optional request timeout duration
    timeout: Option<Duration>,
}

impl Batches {
    /// Creates a new Batches client for OpenAI API.
    ///
    /// Initializes the client by loading the OpenAI API key from
    /// the environment variable `OPENAI_API_KEY`. Supports `.env` file loading
    /// via dotenvy.
    ///
    /// # Returns
    ///
    /// * `Ok(Batches)` - A new Batches client ready for use
    /// * `Err(OpenAIToolError)` - If the API key is not found in the environment
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::batch::request::Batches;
    ///
    /// let batches = Batches::new().expect("API key should be set");
    /// ```
    pub fn new() -> Result<Self> {
        let auth = AuthProvider::openai_from_env()?;
        Ok(Self { auth, timeout: None })
    }

    /// Creates a new Batches client with a custom authentication provider
    pub fn with_auth(auth: AuthProvider) -> Self {
        Self { auth, timeout: None }
    }

    /// Creates a new Batches client for Azure OpenAI API
    pub fn azure() -> Result<Self> {
        let auth = AuthProvider::azure_from_env()?;
        Ok(Self { auth, timeout: None })
    }

    /// Creates a new Batches client by auto-detecting the provider
    pub fn detect_provider() -> Result<Self> {
        let auth = AuthProvider::from_env()?;
        Ok(Self { auth, timeout: None })
    }

    /// Creates a new Batches client with URL-based provider detection
    pub fn with_url<S: Into<String>>(base_url: S, api_key: S) -> Self {
        let auth = AuthProvider::from_url_with_key(base_url, api_key);
        Self { auth, timeout: None }
    }

    /// Creates a new Batches client from URL using environment variables
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
        headers.insert("Content-Type", request::header::HeaderValue::from_static("application/json"));
        headers.insert("User-Agent", request::header::HeaderValue::from_static("openai-tools-rust"));
        Ok((client, headers))
    }

    /// Creates a new batch job.
    ///
    /// # Arguments
    ///
    /// * `request` - The batch creation request
    ///
    /// # Returns
    ///
    /// * `Ok(BatchObject)` - The created batch object
    /// * `Err(OpenAIToolError)` - If the request fails
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::batch::request::{Batches, CreateBatchRequest, BatchEndpoint};
    /// use std::collections::HashMap;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let batches = Batches::new()?;
    ///
    ///     let mut metadata = HashMap::new();
    ///     metadata.insert("customer_id".to_string(), "user_123".to_string());
    ///
    ///     let request = CreateBatchRequest::new("file-abc123", BatchEndpoint::ChatCompletions)
    ///         .with_metadata(metadata);
    ///
    ///     let batch = batches.create(request).await?;
    ///     println!("Created batch: {}", batch.id);
    ///     Ok(())
    /// }
    /// ```
    pub async fn create(&self, request: CreateBatchRequest) -> Result<BatchObject> {
        let (client, headers) = self.create_client()?;

        let body = serde_json::to_string(&request).map_err(OpenAIToolError::SerdeJsonError)?;

        let url = self.auth.endpoint(BATCHES_PATH);
        let response = client.post(&url).headers(headers).body(body).send().await.map_err(OpenAIToolError::RequestError)?;

        let content = response.text().await.map_err(OpenAIToolError::RequestError)?;

        if cfg!(test) {
            tracing::info!("Response content: {}", content);
        }

        serde_json::from_str::<BatchObject>(&content).map_err(OpenAIToolError::SerdeJsonError)
    }

    /// Retrieves details of a specific batch job.
    ///
    /// # Arguments
    ///
    /// * `batch_id` - The ID of the batch to retrieve
    ///
    /// # Returns
    ///
    /// * `Ok(BatchObject)` - The batch details
    /// * `Err(OpenAIToolError)` - If the batch is not found or the request fails
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::batch::request::Batches;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let batches = Batches::new()?;
    ///     let batch = batches.retrieve("batch_abc123").await?;
    ///
    ///     println!("Status: {:?}", batch.status);
    ///     if let Some(counts) = &batch.request_counts {
    ///         println!("Completed: {}/{}", counts.completed, counts.total);
    ///     }
    ///     Ok(())
    /// }
    /// ```
    pub async fn retrieve(&self, batch_id: &str) -> Result<BatchObject> {
        let (client, headers) = self.create_client()?;
        let url = format!("{}/{}", self.auth.endpoint(BATCHES_PATH), batch_id);

        let response = client.get(&url).headers(headers).send().await.map_err(OpenAIToolError::RequestError)?;

        let content = response.text().await.map_err(OpenAIToolError::RequestError)?;

        if cfg!(test) {
            tracing::info!("Response content: {}", content);
        }

        serde_json::from_str::<BatchObject>(&content).map_err(OpenAIToolError::SerdeJsonError)
    }

    /// Cancels an in-progress batch job.
    ///
    /// The batch will transition to "cancelling" and eventually "cancelled".
    ///
    /// # Arguments
    ///
    /// * `batch_id` - The ID of the batch to cancel
    ///
    /// # Returns
    ///
    /// * `Ok(BatchObject)` - The updated batch object
    /// * `Err(OpenAIToolError)` - If the batch cannot be cancelled or the request fails
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::batch::request::Batches;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let batches = Batches::new()?;
    ///     let batch = batches.cancel("batch_abc123").await?;
    ///
    ///     println!("Batch status: {:?}", batch.status);
    ///     Ok(())
    /// }
    /// ```
    pub async fn cancel(&self, batch_id: &str) -> Result<BatchObject> {
        let (client, headers) = self.create_client()?;
        let url = format!("{}/{}/cancel", self.auth.endpoint(BATCHES_PATH), batch_id);

        let response = client.post(&url).headers(headers).send().await.map_err(OpenAIToolError::RequestError)?;

        let content = response.text().await.map_err(OpenAIToolError::RequestError)?;

        if cfg!(test) {
            tracing::info!("Response content: {}", content);
        }

        serde_json::from_str::<BatchObject>(&content).map_err(OpenAIToolError::SerdeJsonError)
    }

    /// Lists all batch jobs.
    ///
    /// Supports pagination through `limit` and `after` parameters.
    ///
    /// # Arguments
    ///
    /// * `limit` - Maximum number of batches to return (default: 20)
    /// * `after` - Cursor for pagination (batch ID to start after)
    ///
    /// # Returns
    ///
    /// * `Ok(BatchListResponse)` - The list of batch jobs
    /// * `Err(OpenAIToolError)` - If the request fails
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::batch::request::Batches;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let batches = Batches::new()?;
    ///
    ///     // Get first page
    ///     let response = batches.list(Some(10), None).await?;
    ///     for batch in &response.data {
    ///         println!("{}: {:?}", batch.id, batch.status);
    ///     }
    ///
    ///     // Get next page if available
    ///     if response.has_more {
    ///         if let Some(last_id) = &response.last_id {
    ///             let next_page = batches.list(Some(10), Some(last_id)).await?;
    ///             // ...
    ///         }
    ///     }
    ///
    ///     Ok(())
    /// }
    /// ```
    pub async fn list(&self, limit: Option<u32>, after: Option<&str>) -> Result<BatchListResponse> {
        let (client, headers) = self.create_client()?;

        let mut url = self.auth.endpoint(BATCHES_PATH);
        let mut params = Vec::new();

        if let Some(l) = limit {
            params.push(format!("limit={}", l));
        }
        if let Some(a) = after {
            params.push(format!("after={}", a));
        }

        if !params.is_empty() {
            url.push('?');
            url.push_str(&params.join("&"));
        }

        let response = client.get(&url).headers(headers).send().await.map_err(OpenAIToolError::RequestError)?;

        let content = response.text().await.map_err(OpenAIToolError::RequestError)?;

        if cfg!(test) {
            tracing::info!("Response content: {}", content);
        }

        serde_json::from_str::<BatchListResponse>(&content).map_err(OpenAIToolError::SerdeJsonError)
    }
}
