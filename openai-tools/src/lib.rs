//! # OpenAI Tools for Rust
//!
//! A comprehensive Rust library for interacting with OpenAI's APIs, providing easy-to-use
//! interfaces for chat completions, responses, and various AI-powered functionalities.
//! This crate offers both high-level convenience methods and low-level control for
//! advanced use cases.
//!
//! ## Features
//!
//! ### Core APIs
//! - **Chat Completions API**: Chat with streaming, function calling, and structured output
//! - **Responses API**: Assistant-style interactions with multi-modal input
//! - **Conversations API**: Long-running conversation state management
//! - **Embedding API**: Text to vector embeddings for semantic search
//! - **Realtime API**: WebSocket-based real-time audio/text streaming
//! - **Live API**: Full-duplex speech-to-speech streaming over WebSocket
//!
//! ### Content & Media APIs
//! - **Images API**: DALL-E image generation, editing, and variations
//! - **Audio API**: Text-to-speech, transcription, and translation
//! - **Moderations API**: Content policy violation detection
//!
//! ### Management APIs
//! - **Models API**: List and retrieve available models
//! - **Files API**: Upload and manage files for fine-tuning/batch
//! - **Batch API**: Async bulk processing with 50% cost savings
//! - **Fine-tuning API**: Custom model training
//!
//! ## Quick Start
//!
//! Add this to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! openai-tools = "1.0"
//! tokio = { version = "1.0", features = ["full"] }
//! serde = { version = "1.0", features = ["derive"] }
//! ```
//!
//! Set up your API key:
//!
//! ```bash
//! export OPENAI_API_KEY="your-api-key-here"
//! ```
//!
//! ## Basic Chat Completion
//!
//! ```rust,no_run
//! use openai_tools::chat::request::ChatCompletion;
//! use openai_tools::common::message::Message;
//! use openai_tools::common::role::Role;
//! use openai_tools::common::models::ChatModel;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut chat = ChatCompletion::new();
//!     let messages = vec![
//!         Message::from_string(Role::User, "Hello! How are you?")
//!     ];
//!
//!     let response = chat
//!         .model(ChatModel::Gpt4oMini)  // Type-safe model selection
//!         .messages(messages)
//!         .temperature(0.7)
//!         .chat()
//!         .await?;
//!
//!     println!("AI: {}", response.choices[0].message.content.as_ref().unwrap().text.as_ref().unwrap());
//!     Ok(())
//! }
//! ```
//!
//! ## Structured Output with JSON Schema
//!
//! ```rust,no_run
//! use openai_tools::chat::request::ChatCompletion;
//! use openai_tools::common::{message::Message, role::Role, structured_output::Schema, models::ChatModel};
//! use serde::{Deserialize, Serialize};
//!
//! #[derive(Debug, Serialize, Deserialize)]
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
//!     // Create JSON schema
//!     let mut schema = Schema::chat_json_schema("person_info");
//!     schema.add_property("name", "string", "Person's full name");
//!     schema.add_property("age", "number", "Person's age");
//!     schema.add_property("occupation", "string", "Person's job");
//!
//!     let messages = vec![
//!         Message::from_string(Role::User,
//!             "Extract info: John Smith, 30, Software Engineer")
//!     ];
//!
//!     let response = chat
//!         .model(ChatModel::Gpt4oMini)
//!         .messages(messages)
//!         .json_schema(schema)
//!         .chat()
//!         .await?;
//!
//!     let person: PersonInfo = serde_json::from_str(
//!         response.choices[0].message.content.as_ref().unwrap().text.as_ref().unwrap()
//!     )?;
//!
//!     println!("Extracted: {} ({}), {}", person.name, person.age, person.occupation);
//!     Ok(())
//! }
//! ```
//!
//! ## Function Calling with Tools
//!
//! ```rust,no_run
//! use openai_tools::chat::request::ChatCompletion;
//! use openai_tools::common::{message::Message, role::Role, tool::Tool, parameters::ParameterProperty, models::ChatModel};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut chat = ChatCompletion::new();
//!
//!     // Define a weather tool
//!     let weather_tool = Tool::function(
//!         "get_weather",
//!         "Get current weather for a location",
//!         vec![
//!             ("location", ParameterProperty::from_string("City name")),
//!             ("unit", ParameterProperty::from_string("Temperature unit (celsius/fahrenheit)")),
//!         ],
//!         false,
//!     );
//!
//!     let messages = vec![
//!         Message::from_string(Role::User, "What's the weather in Tokyo?")
//!     ];
//!
//!     let response = chat
//!         .model(ChatModel::Gpt4oMini)
//!         .messages(messages)
//!         .tools(vec![weather_tool])
//!         .chat()
//!         .await?;
//!
//!     // Handle tool calls
//!     if let Some(tool_calls) = &response.choices[0].message.tool_calls {
//!         for call in tool_calls {
//!             println!("Tool: {}", call.function.name);
//!             if let Ok(args) = call.function.arguments_as_map() {
//!                 println!("Args: {:?}", args);
//!             }
//!             // Execute the function and continue conversation...
//!         }
//!     }
//!     Ok(())
//! }
//! ```
//!
//! ## Multi-modal Input (Text + Image)
//!
//! Both Chat Completions API and Responses API support multi-modal messages.
//! The same `Content` and `Message` types work with both APIs - serialization
//! format differences are handled automatically.
//!
//! ### Chat Completions API
//!
//! ```rust,no_run
//! use openai_tools::chat::request::ChatCompletion;
//! use openai_tools::common::{message::{Message, Content}, role::Role, models::ChatModel};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut chat = ChatCompletion::new();
//!
//!     let message = Message::from_message_array(
//!         Role::User,
//!         vec![
//!             Content::from_text("What do you see in this image?"),
//!             Content::from_image_url("https://example.com/image.jpg"),
//!         ],
//!     );
//!
//!     let response = chat
//!         .model(ChatModel::Gpt4oMini)
//!         .messages(vec![message])
//!         .chat()
//!         .await?;
//!
//!     println!("AI: {}", response.choices[0].message.content.as_ref().unwrap().text.as_ref().unwrap());
//!     Ok(())
//! }
//! ```
//!
//! ### Responses API
//!
//! ```rust,no_run
//! use openai_tools::responses::request::Responses;
//! use openai_tools::common::{message::{Message, Content}, role::Role, models::ChatModel};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut responses = Responses::new();
//!
//!     responses
//!         .model(ChatModel::Gpt4oMini)
//!         .instructions("You are an image analysis assistant.");
//!
//!     let message = Message::from_message_array(
//!         Role::User,
//!         vec![
//!             Content::from_text("What do you see in this image?"),
//!             Content::from_image_file("path/to/image.jpg"),
//!         ],
//!     );
//!
//!     responses.messages(vec![message]);
//!
//!     let response = responses.complete().await?;
//!     let text = response.output_text().unwrap();
//!     println!("Response: {}", text);
//!     Ok(())
//! }
//! ```
//!
//! ## Choosing the Right API
//!
//! | Use Case | Recommended API | Module |
//! |----------|-----------------|--------|
//! | Simple Q&A, chatbot | Chat Completions | [`chat`] |
//! | Multi-turn assistant with state | Responses + Conversations | [`responses`], [`conversations`] |
//! | Real-time voice interaction | Realtime | [`realtime`] |
//! | Semantic search, similarity | Embeddings | [`embedding`] |
//! | Image generation (DALL-E) | Images | [`images`] |
//! | Speech-to-text, TTS | Audio | [`audio`] |
//! | Content moderation | Moderations | [`moderations`] |
//! | Bulk processing (50% off) | Batch | [`batch`] |
//! | Custom model training | Fine-tuning | [`fine_tuning`] |
//!
//! ## Module Structure
//!
//! ### Core APIs
//!
//! - [`chat`] - Chat Completions API (`/v1/chat/completions`)
//!   - [`chat::request`] - `ChatCompletion` builder
//!   - [`chat::response`] - Response types
//!
//! - [`responses`] - Responses API (`/v1/responses`)
//!   - [`responses::request`] - `Responses` builder with CRUD operations
//!   - [`responses::response`] - Response types
//!
//! - [`conversations`] - Conversations API (`/v1/conversations`)
//!   - [`conversations::request`] - `Conversations` client
//!   - [`conversations::response`] - Conversation and item types
//!
//! - [`embedding`] - Embeddings API (`/v1/embeddings`)
//!   - [`embedding::request`] - `Embedding` builder
//!   - [`embedding::response`] - Vector response types
//!
//! - [`realtime`] - Realtime API (WebSocket)
//!   - [`RealtimeClient`](realtime::RealtimeClient) and
//!     [`RealtimeSession`](realtime::RealtimeSession)
//!   - [`realtime::events`] - Client/server event types
//!
//! ### Content & Media APIs
//!
//! - [`images`] - Images API (`/v1/images`)
//!   - Generate, edit, create variations with the GPT Image models
//!
//! - [`videos`] - Videos API (`/v1/videos`)
//!   - Generate, poll, download and remix Sora video clips
//!
//! - [`audio`] - Audio API (`/v1/audio`)
//!   - Text-to-speech, transcription, translation
//!
//! - [`moderations`] - Moderations API (`/v1/moderations`)
//!   - Content policy violation detection
//!
//! ### Management APIs
//!
//! - [`models`] - Models API (`/v1/models`)
//!   - List and retrieve available models
//!
//! - [`files`] - Files API (`/v1/files`)
//!   - Upload/download files for fine-tuning and batch
//!
//! - [`batch`] - Batch API (`/v1/batches`)
//!   - Async bulk processing with 50% cost savings
//!
//! - [`fine_tuning`] - Fine-tuning API (`/v1/fine_tuning/jobs`)
//!   - Custom model training and management
//!
//! ### Shared Utilities
//!
//! - [`common`] - Shared types across all APIs
//!   - [`common::models`] - Type-safe model enums (`ChatModel`, `EmbeddingModel`, etc.)
//!   - [`common::message`] - Message and content structures
//!   - [`common::role`] - User roles (User, Assistant, System, Tool)
//!   - [`common::tool`] - Function calling definitions
//!   - [`common::auth`] - Authentication (OpenAI, Azure, custom)
//!   - [`common::errors`] - Error types
//!   - [`common::structured_output`] - JSON schema utilities
//!
//! ## Error Handling
//!
//! All operations return `Result<T, OpenAIToolError>`:
//!
//! ```rust,no_run
//! use openai_tools::common::errors::OpenAIToolError;
//! # use openai_tools::chat::request::ChatCompletion;
//!
//! # #[tokio::main]
//! # async fn main() {
//! # let mut chat = ChatCompletion::new();
//! match chat.chat().await {
//!     Ok(response) => {
//!         println!("Success: {:?}", response.choices[0].message.content);
//!     },
//!     // Network/HTTP errors (connection failed, timeout, etc.)
//!     Err(OpenAIToolError::RequestError(e)) => {
//!         eprintln!("Network error: {}", e);
//!     },
//!     // JSON parsing errors (unexpected response format)
//!     Err(OpenAIToolError::SerdeJsonError(e)) => {
//!         eprintln!("JSON parse error: {}", e);
//!     },
//!     // WebSocket errors (Realtime API)
//!     Err(OpenAIToolError::WebSocketError(msg)) => {
//!         eprintln!("WebSocket error: {}", msg);
//!     },
//!     // Realtime API specific errors
//!     Err(OpenAIToolError::RealtimeError { code, message }) => {
//!         eprintln!("Realtime error [{}]: {}", code, message);
//!     },
//!     // Other errors
//!     Err(e) => eprintln!("Error: {}", e),
//! }
//! # }
//! ```
//!
//! For API errors (rate limits, invalid requests), check the HTTP response status
//! in `RequestError`.
//!
//! ## Provider Configuration
//!
//! This library supports multiple providers: OpenAI, Azure OpenAI, and OpenAI-compatible APIs.
//!
//! ### OpenAI (Default)
//!
//! ```bash
//! export OPENAI_API_KEY="sk-..."
//! ```
//!
//! ```rust,no_run
//! use openai_tools::chat::request::ChatCompletion;
//!
//! let chat = ChatCompletion::new();  // Uses OPENAI_API_KEY
//! ```
//!
//! ### Azure OpenAI
//!
//! ```bash
//! export AZURE_OPENAI_API_KEY="..."
//! export AZURE_OPENAI_BASE_URL="https://my-resource.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview"
//! ```
//!
//! ```rust,no_run
//! use openai_tools::chat::request::ChatCompletion;
//!
//! // From environment variables
//! let chat = ChatCompletion::azure().unwrap();
//!
//! // Or with explicit URL
//! let chat = ChatCompletion::with_url(
//!     "https://my-resource.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview",
//!     "api-key"
//! );
//! ```
//!
//! ### OpenAI-Compatible APIs (Ollama, vLLM, LocalAI)
//!
//! ```rust,no_run
//! use openai_tools::chat::request::ChatCompletion;
//!
//! let chat = ChatCompletion::with_url("http://localhost:11434/v1", "ollama");
//! ```
//!
//! ### Auto-Detect Provider
//!
//! ```rust,no_run
//! use openai_tools::chat::request::ChatCompletion;
//!
//! // Uses Azure if AZURE_OPENAI_API_KEY is set, otherwise OpenAI
//! let chat = ChatCompletion::detect_provider().unwrap();
//! ```
//!
//! ## Type-Safe Model Selection
//!
//! All APIs use enum-based model selection for compile-time validation:
//!
//! ```rust,no_run
//! use openai_tools::common::models::{ChatModel, EmbeddingModel, RealtimeModel, FineTuningModel};
//! use openai_tools::chat::request::ChatCompletion;
//! use openai_tools::embedding::request::Embedding;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Chat/Responses API
//! let mut chat = ChatCompletion::new();
//! chat.model(ChatModel::Gpt4oMini);      // Cost-effective
//! chat.model(ChatModel::Gpt4o);          // Most capable
//! chat.model(ChatModel::O3Mini);         // Reasoning model
//!
//! // Embedding API
//! let mut embedding = Embedding::new()?;
//! embedding.model(EmbeddingModel::TextEmbedding3Small);
//!
//! // Custom/fine-tuned models
//! chat.model(ChatModel::custom("ft:gpt-4o-mini:my-org::abc123"));
//! # Ok(())
//! # }
//! ```
//!

pub mod audio;
pub mod batch;
pub mod chat;
pub mod common;
pub mod conversations;
pub mod embedding;
pub mod files;
pub mod fine_tuning;
pub mod images;
pub mod live;
pub mod models;
pub mod moderations;
pub mod realtime;
pub mod responses;
pub mod videos;
