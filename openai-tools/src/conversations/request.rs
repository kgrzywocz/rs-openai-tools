//! OpenAI Conversations API Request Module
//!
//! This module provides the functionality to interact with the OpenAI Conversations API.
//! The Conversations API allows you to create and manage long-running conversations
//! with the Responses API.
//!
//! # Key Features
//!
//! - **Create Conversations**: Create new conversations with optional metadata and items
//! - **Retrieve Conversations**: Get details of a specific conversation
//! - **Update Conversations**: Modify conversation metadata
//! - **Delete Conversations**: Remove conversations
//! - **Manage Items**: Add and list conversation items
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use openai_tools::conversations::request::Conversations;
//! use openai_tools::conversations::response::InputItem;
//! use std::collections::HashMap;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let conversations = Conversations::new()?;
//!
//!     // Create a new conversation
//!     let mut metadata = HashMap::new();
//!     metadata.insert("topic".to_string(), "demo".to_string());
//!
//!     let conversation = conversations.create(Some(metadata), None).await?;
//!     println!("Created conversation: {}", conversation.id);
//!
//!     // Add items to the conversation
//!     let items = vec![InputItem::user_message("Hello!")];
//!     let added_items = conversations.create_items(&conversation.id, items).await?;
//!
//!     Ok(())
//! }
//! ```

use crate::common::auth::AuthProvider;
use crate::common::client::create_http_client;
use crate::common::errors::{ErrorResponse, OpenAIToolError, Result};
use crate::conversations::response::{Conversation, ConversationItemListResponse, ConversationListResponse, DeleteConversationResponse, InputItem};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Default API path for Conversations
const CONVERSATIONS_PATH: &str = "conversations";

/// Specifies additional data to include in conversation item responses.
///
/// This enum defines various types of additional information that can be
/// included when listing conversation items.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[non_exhaustive]
pub enum ConversationInclude {
    /// Include web search call action sources
    #[serde(rename = "web_search_call.action.sources")]
    WebSearchCallSources,
    /// Include code interpreter call outputs
    #[serde(rename = "code_interpreter_call.outputs")]
    CodeInterpreterCallOutputs,
    /// Include file search call results
    #[serde(rename = "file_search_call.results")]
    FileSearchCallResults,
    /// Include image URLs from input messages
    #[serde(rename = "message.input_image.image_url")]
    MessageInputImageUrl,
    /// Include encrypted reasoning content
    #[serde(rename = "reasoning.encrypted_content")]
    ReasoningEncryptedContent,
}

impl ConversationInclude {
    /// Returns the string representation for API requests.
    pub fn as_str(&self) -> &'static str {
        match self {
            ConversationInclude::WebSearchCallSources => "web_search_call.action.sources",
            ConversationInclude::CodeInterpreterCallOutputs => "code_interpreter_call.outputs",
            ConversationInclude::FileSearchCallResults => "file_search_call.results",
            ConversationInclude::MessageInputImageUrl => "message.input_image.image_url",
            ConversationInclude::ReasoningEncryptedContent => "reasoning.encrypted_content",
        }
    }
}

/// Request body for creating a conversation.
#[derive(Debug, Clone, Serialize)]
struct CreateConversationRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<HashMap<String, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    items: Option<Vec<InputItem>>,
}

/// Request body for updating a conversation.
#[derive(Debug, Clone, Serialize)]
struct UpdateConversationRequest {
    metadata: HashMap<String, String>,
}

/// Request body for creating conversation items.
#[derive(Debug, Clone, Serialize)]
struct CreateItemsRequest {
    items: Vec<InputItem>,
}

/// Client for interacting with the OpenAI Conversations API.
///
/// This struct provides methods to create, retrieve, update, delete conversations,
/// and manage conversation items. Use [`Conversations::new()`] to create a new instance.
///
/// # Example
///
/// ```rust,no_run
/// use openai_tools::conversations::request::Conversations;
/// use std::collections::HashMap;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let conversations = Conversations::new()?;
///
///     // Create a conversation with metadata
///     let mut metadata = HashMap::new();
///     metadata.insert("user_id".to_string(), "user123".to_string());
///
///     let conv = conversations.create(Some(metadata), None).await?;
///     println!("Created: {}", conv.id);
///
///     // Retrieve the conversation
///     let retrieved = conversations.retrieve(&conv.id).await?;
///     println!("Retrieved: {:?}", retrieved.metadata);
///
///     Ok(())
/// }
/// ```
pub struct Conversations {
    /// Authentication provider (OpenAI or Azure)
    auth: AuthProvider,
    /// Optional request timeout duration
    timeout: Option<Duration>,
}

impl Conversations {
    /// Creates a new Conversations client for OpenAI API.
    ///
    /// Initializes the client by loading the OpenAI API key from
    /// the environment variable `OPENAI_API_KEY`. Supports `.env` file loading
    /// via dotenvy.
    ///
    /// # Returns
    ///
    /// * `Ok(Conversations)` - A new Conversations client ready for use
    /// * `Err(OpenAIToolError)` - If the API key is not found in the environment
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::conversations::request::Conversations;
    ///
    /// let conversations = Conversations::new().expect("API key should be set");
    /// ```
    pub fn new() -> Result<Self> {
        let auth = AuthProvider::openai_from_env()?;
        Ok(Self { auth, timeout: None })
    }

    /// Creates a new Conversations client with a custom authentication provider
    pub fn with_auth(auth: AuthProvider) -> Self {
        Self { auth, timeout: None }
    }

    /// Creates a new Conversations client for Azure OpenAI API
    pub fn azure() -> Result<Self> {
        let auth = AuthProvider::azure_from_env()?;
        Ok(Self { auth, timeout: None })
    }

    /// Creates a new Conversations client by auto-detecting the provider
    pub fn detect_provider() -> Result<Self> {
        let auth = AuthProvider::from_env()?;
        Ok(Self { auth, timeout: None })
    }

    /// Creates a new Conversations client with URL-based provider detection
    pub fn with_url<S: Into<String>>(base_url: S, api_key: S) -> Self {
        let auth = AuthProvider::from_url_with_key(base_url, api_key);
        Self { auth, timeout: None }
    }

    /// Creates a new Conversations client from URL using environment variables
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

    /// Handles API error responses.
    fn handle_error(status: request::StatusCode, content: &str) -> OpenAIToolError {
        if let Ok(error_resp) = serde_json::from_str::<ErrorResponse>(content) {
            OpenAIToolError::Error(error_resp.error.message.unwrap_or_default())
        } else {
            OpenAIToolError::Error(format!("API error ({}): {}", status, content))
        }
    }

    /// Creates a new conversation.
    ///
    /// You can optionally provide metadata and initial items to include
    /// in the conversation.
    ///
    /// # Arguments
    ///
    /// * `metadata` - Optional key-value pairs for storing additional information
    /// * `items` - Optional initial items to add to the conversation (up to 20 items)
    ///
    /// # Returns
    ///
    /// * `Ok(Conversation)` - The created conversation object
    /// * `Err(OpenAIToolError)` - If the request fails
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::conversations::request::Conversations;
    /// use openai_tools::conversations::response::InputItem;
    /// use std::collections::HashMap;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let conversations = Conversations::new()?;
    ///
    ///     // Create with metadata and initial message
    ///     let mut metadata = HashMap::new();
    ///     metadata.insert("topic".to_string(), "greeting".to_string());
    ///
    ///     let items = vec![InputItem::user_message("Hello!")];
    ///
    ///     let conv = conversations.create(Some(metadata), Some(items)).await?;
    ///     println!("Created conversation: {}", conv.id);
    ///     Ok(())
    /// }
    /// ```
    pub async fn create(&self, metadata: Option<HashMap<String, String>>, items: Option<Vec<InputItem>>) -> Result<Conversation> {
        let (client, headers) = self.create_client()?;

        let request_body = CreateConversationRequest { metadata, items };
        let body = serde_json::to_string(&request_body)?;

        let url = self.auth.endpoint(CONVERSATIONS_PATH);
        let response = client.post(&url).headers(headers).body(body).send().await.map_err(OpenAIToolError::RequestError)?;

        let status = response.status();
        let content = response.text().await.map_err(OpenAIToolError::RequestError)?;

        if cfg!(test) {
            tracing::info!("Response content: {}", content);
        }

        if !status.is_success() {
            return Err(Self::handle_error(status, &content));
        }

        serde_json::from_str::<Conversation>(&content).map_err(OpenAIToolError::SerdeJsonError)
    }

    /// Retrieves a specific conversation.
    ///
    /// # Arguments
    ///
    /// * `conversation_id` - The ID of the conversation to retrieve
    ///
    /// # Returns
    ///
    /// * `Ok(Conversation)` - The conversation object
    /// * `Err(OpenAIToolError)` - If the conversation is not found or the request fails
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::conversations::request::Conversations;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let conversations = Conversations::new()?;
    ///     let conv = conversations.retrieve("conv_abc123").await?;
    ///
    ///     println!("Conversation: {}", conv.id);
    ///     println!("Created at: {}", conv.created_at);
    ///     Ok(())
    /// }
    /// ```
    pub async fn retrieve(&self, conversation_id: &str) -> Result<Conversation> {
        let (client, headers) = self.create_client()?;
        let url = format!("{}/{}", self.auth.endpoint(CONVERSATIONS_PATH), conversation_id);

        let response = client.get(&url).headers(headers).send().await.map_err(OpenAIToolError::RequestError)?;

        let status = response.status();
        let content = response.text().await.map_err(OpenAIToolError::RequestError)?;

        if cfg!(test) {
            tracing::info!("Response content: {}", content);
        }

        if !status.is_success() {
            return Err(Self::handle_error(status, &content));
        }

        serde_json::from_str::<Conversation>(&content).map_err(OpenAIToolError::SerdeJsonError)
    }

    /// Updates a conversation's metadata.
    ///
    /// # Arguments
    ///
    /// * `conversation_id` - The ID of the conversation to update
    /// * `metadata` - The new metadata to set
    ///
    /// # Returns
    ///
    /// * `Ok(Conversation)` - The updated conversation object
    /// * `Err(OpenAIToolError)` - If the request fails
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::conversations::request::Conversations;
    /// use std::collections::HashMap;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let conversations = Conversations::new()?;
    ///
    ///     let mut metadata = HashMap::new();
    ///     metadata.insert("topic".to_string(), "updated-topic".to_string());
    ///
    ///     let conv = conversations.update("conv_abc123", metadata).await?;
    ///     println!("Updated: {:?}", conv.metadata);
    ///     Ok(())
    /// }
    /// ```
    pub async fn update(&self, conversation_id: &str, metadata: HashMap<String, String>) -> Result<Conversation> {
        let (client, headers) = self.create_client()?;
        let url = format!("{}/{}", self.auth.endpoint(CONVERSATIONS_PATH), conversation_id);

        let request_body = UpdateConversationRequest { metadata };
        let body = serde_json::to_string(&request_body)?;

        let response = client.post(&url).headers(headers).body(body).send().await.map_err(OpenAIToolError::RequestError)?;

        let status = response.status();
        let content = response.text().await.map_err(OpenAIToolError::RequestError)?;

        if cfg!(test) {
            tracing::info!("Response content: {}", content);
        }

        if !status.is_success() {
            return Err(Self::handle_error(status, &content));
        }

        serde_json::from_str::<Conversation>(&content).map_err(OpenAIToolError::SerdeJsonError)
    }

    /// Deletes a conversation.
    ///
    /// # Arguments
    ///
    /// * `conversation_id` - The ID of the conversation to delete
    ///
    /// # Returns
    ///
    /// * `Ok(DeleteConversationResponse)` - Confirmation of deletion
    /// * `Err(OpenAIToolError)` - If the request fails
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::conversations::request::Conversations;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let conversations = Conversations::new()?;
    ///     let result = conversations.delete("conv_abc123").await?;
    ///
    ///     if result.deleted {
    ///         println!("Conversation {} was deleted", result.id);
    ///     }
    ///     Ok(())
    /// }
    /// ```
    pub async fn delete(&self, conversation_id: &str) -> Result<DeleteConversationResponse> {
        let (client, headers) = self.create_client()?;
        let url = format!("{}/{}", self.auth.endpoint(CONVERSATIONS_PATH), conversation_id);

        let response = client.delete(&url).headers(headers).send().await.map_err(OpenAIToolError::RequestError)?;

        let status = response.status();
        let content = response.text().await.map_err(OpenAIToolError::RequestError)?;

        if cfg!(test) {
            tracing::info!("Response content: {}", content);
        }

        if !status.is_success() {
            return Err(Self::handle_error(status, &content));
        }

        serde_json::from_str::<DeleteConversationResponse>(&content).map_err(OpenAIToolError::SerdeJsonError)
    }

    /// Creates items in a conversation.
    ///
    /// You can add up to 20 items at a time.
    ///
    /// # Arguments
    ///
    /// * `conversation_id` - The ID of the conversation
    /// * `items` - The items to add to the conversation
    ///
    /// # Returns
    ///
    /// * `Ok(ConversationItemListResponse)` - The created items
    /// * `Err(OpenAIToolError)` - If the request fails
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::conversations::request::Conversations;
    /// use openai_tools::conversations::response::InputItem;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let conversations = Conversations::new()?;
    ///
    ///     let items = vec![
    ///         InputItem::user_message("What is the weather like?"),
    ///         InputItem::assistant_message("I'd be happy to help with weather information!"),
    ///     ];
    ///
    ///     let result = conversations.create_items("conv_abc123", items).await?;
    ///     println!("Added {} items", result.data.len());
    ///     Ok(())
    /// }
    /// ```
    pub async fn create_items(&self, conversation_id: &str, items: Vec<InputItem>) -> Result<ConversationItemListResponse> {
        let (client, headers) = self.create_client()?;
        let url = format!("{}/{}/items", self.auth.endpoint(CONVERSATIONS_PATH), conversation_id);

        let request_body = CreateItemsRequest { items };
        let body = serde_json::to_string(&request_body)?;

        let response = client.post(&url).headers(headers).body(body).send().await.map_err(OpenAIToolError::RequestError)?;

        let status = response.status();
        let content = response.text().await.map_err(OpenAIToolError::RequestError)?;

        if cfg!(test) {
            tracing::info!("Response content: {}", content);
        }

        if !status.is_success() {
            return Err(Self::handle_error(status, &content));
        }

        serde_json::from_str::<ConversationItemListResponse>(&content).map_err(OpenAIToolError::SerdeJsonError)
    }

    /// Lists items in a conversation.
    ///
    /// # Arguments
    ///
    /// * `conversation_id` - The ID of the conversation
    /// * `limit` - Maximum number of items to return (1-100, default 20)
    /// * `after` - Cursor for pagination (item ID to start after)
    /// * `order` - Sort order ("asc" or "desc", default "desc")
    /// * `include` - Additional data to include in the response
    ///
    /// # Returns
    ///
    /// * `Ok(ConversationItemListResponse)` - The list of items
    /// * `Err(OpenAIToolError)` - If the request fails
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::conversations::request::{Conversations, ConversationInclude};
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let conversations = Conversations::new()?;
    ///
    ///     // List items with pagination
    ///     let items = conversations.list_items(
    ///         "conv_abc123",
    ///         Some(20),
    ///         None,
    ///         Some("desc"),
    ///         None,
    ///     ).await?;
    ///
    ///     for item in &items.data {
    ///         println!("Item: {} ({})", item.id, item.item_type);
    ///     }
    ///     Ok(())
    /// }
    /// ```
    pub async fn list_items(
        &self,
        conversation_id: &str,
        limit: Option<u32>,
        after: Option<&str>,
        order: Option<&str>,
        include: Option<Vec<ConversationInclude>>,
    ) -> Result<ConversationItemListResponse> {
        let (client, headers) = self.create_client()?;

        // Build query parameters
        let mut params = Vec::new();
        if let Some(l) = limit {
            params.push(format!("limit={}", l));
        }
        if let Some(a) = after {
            params.push(format!("after={}", a));
        }
        if let Some(o) = order {
            params.push(format!("order={}", o));
        }
        if let Some(inc) = include {
            for i in inc {
                params.push(format!("include[]={}", i.as_str()));
            }
        }

        let url = if params.is_empty() {
            format!("{}/{}/items", self.auth.endpoint(CONVERSATIONS_PATH), conversation_id)
        } else {
            format!("{}/{}/items?{}", self.auth.endpoint(CONVERSATIONS_PATH), conversation_id, params.join("&"))
        };

        let response = client.get(&url).headers(headers).send().await.map_err(OpenAIToolError::RequestError)?;

        let status = response.status();
        let content = response.text().await.map_err(OpenAIToolError::RequestError)?;

        if cfg!(test) {
            tracing::info!("Response content: {}", content);
        }

        if !status.is_success() {
            return Err(Self::handle_error(status, &content));
        }

        serde_json::from_str::<ConversationItemListResponse>(&content).map_err(OpenAIToolError::SerdeJsonError)
    }

    /// Lists all conversations (if available).
    ///
    /// Note: This endpoint may not be available in all API versions.
    ///
    /// # Arguments
    ///
    /// * `limit` - Maximum number of conversations to return (1-100, default 20)
    /// * `after` - Cursor for pagination (conversation ID to start after)
    ///
    /// # Returns
    ///
    /// * `Ok(ConversationListResponse)` - The list of conversations
    /// * `Err(OpenAIToolError)` - If the request fails
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::conversations::request::Conversations;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let conversations = Conversations::new()?;
    ///
    ///     let response = conversations.list(Some(10), None).await?;
    ///     for conv in &response.data {
    ///         println!("Conversation: {} (created: {})", conv.id, conv.created_at);
    ///     }
    ///     Ok(())
    /// }
    /// ```
    pub async fn list(&self, limit: Option<u32>, after: Option<&str>) -> Result<ConversationListResponse> {
        let (client, headers) = self.create_client()?;

        // Build query parameters
        let mut params = Vec::new();
        if let Some(l) = limit {
            params.push(format!("limit={}", l));
        }
        if let Some(a) = after {
            params.push(format!("after={}", a));
        }

        let url = if params.is_empty() {
            self.auth.endpoint(CONVERSATIONS_PATH)
        } else {
            format!("{}?{}", self.auth.endpoint(CONVERSATIONS_PATH), params.join("&"))
        };

        let response = client.get(&url).headers(headers).send().await.map_err(OpenAIToolError::RequestError)?;

        let status = response.status();
        let content = response.text().await.map_err(OpenAIToolError::RequestError)?;

        if cfg!(test) {
            tracing::info!("Response content: {}", content);
        }

        if !status.is_success() {
            return Err(Self::handle_error(status, &content));
        }

        serde_json::from_str::<ConversationListResponse>(&content).map_err(OpenAIToolError::SerdeJsonError)
    }
}
