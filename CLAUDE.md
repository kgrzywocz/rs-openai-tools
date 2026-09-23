# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Rust library (`openai-tools`) providing a type-safe API wrapper for OpenAI's APIs. Published on crates.io.

## Build & Test Commands

```bash
# Build
cargo build
cargo make build-tools  # release build

# Run all tests (RECOMMENDED - use cargo-nextest for parallel execution)
cargo nextest run

# Run unit tests only (no API key required)
cargo nextest run --lib
cargo test --lib        # fallback if nextest not installed
cargo make test-unit

# Run integration tests (requires OPENAI_API_KEY)
cargo nextest run --test '*_integration'
cargo make test-integration
# or individual tests:
cargo nextest run --test chat_integration
cargo nextest run --test responses_integration
cargo nextest run --test embedding_integration
cargo nextest run --test realtime_integration
cargo nextest run --test live_integration
cargo nextest run --test models_integration
cargo nextest run --test models_by_version_integration
cargo nextest run --test files_integration
cargo nextest run --test moderations_integration
cargo nextest run --test images_integration
cargo nextest run --test audio_integration
cargo nextest run --test batch_integration
cargo nextest run --test fine_tuning_integration
cargo nextest run --test conversations_integration

# Run a specific test
cargo nextest run test_chat_completion

# Run tests with logging output
RUST_LOG=info cargo nextest run --nocapture

# Run all tests (alternative)
cargo make test

# Check without building
cargo check

# Format code
cargo fmt
cargo make format-all  # includes clippy and taplo

# Lint
cargo clippy
```

**Note:** Always prefer `cargo nextest run` over `cargo test` for faster parallel test execution.

## Test Structure

- **Unit tests** (`src/**/mod.rs`): Test builders, serialization, and validation without API calls
- **Integration tests** (`tests/*_integration.rs`): Test actual API calls, require `OPENAI_API_KEY`

## Architecture

### Workspace Structure

```
rs-openai-tools/
├── Cargo.toml          # Workspace root
└── openai-tools/       # Main library crate
    └── src/
        ├── lib.rs      # Crate entry point
        ├── chat/       # Chat Completions API
        ├── responses/  # Responses API (assistant-style)
        ├── conversations/ # Conversations API
        ├── embedding/  # Embeddings API
        ├── realtime/   # Realtime API (WebSocket)
        ├── live/       # Live API (Primary WebSocket and Fork)
        ├── models/     # Models API
        ├── files/      # Files API
        ├── moderations/ # Moderations API
        ├── images/     # Images API (DALL-E)
        ├── audio/      # Audio API (TTS, STT)
        ├── batch/      # Batch API
        ├── fine_tuning/ # Fine-tuning API
        ├── videos/      # Videos API (Sora)
        └── common/     # Shared types
```

### Module Responsibilities

- **`chat/`**: Chat Completions API (`/v1/chat/completions`)
  - `request.rs`: `ChatCompletion` builder for requests
  - `response.rs`: Response types

- **`responses/`**: Responses API (`/v1/responses`) - newer assistant-style API
  - `request.rs`: `Responses` builder with multi-modal support, `Include`, `ReasoningEffort`, `ReasoningSummary`, `Reasoning`, `Truncation`, `ToolChoice`, `ToolChoiceMode`, `NamedFunctionChoice`, `Prompt`
  - `response.rs`: `Response`, `DeleteResponseResult`, `ResponseInputItem`, `InputItemsListResponse`, `CompactedResponse`, `InputTokensResponse`

- **`conversations/`**: Conversations API (`/v1/conversations`) - long-running conversation management
  - `request.rs`: `Conversations` client for create, retrieve, update, delete, items
  - `response.rs`: `Conversation`, `ConversationItem`, `InputItem`

- **`embedding/`**: Embeddings API (`/v1/embeddings`)
  - `request.rs`: `Embedding` builder
  - `response.rs`: Vector response types (1D/2D/3D)

- **`realtime/`**: Realtime API (WebSocket, `wss://api.openai.com/v1/realtime`)
  - `client.rs`: `RealtimeClient` builder and `RealtimeSession` handle
  - `session.rs`: `SessionConfig`, `Modality`, `ToolChoice`
  - `audio.rs`: `AudioFormat`, `Voice` (Alloy, Ash, Ballad, Coral, Echo, Sage, Shimmer, Verse), `TranscriptionModel` (Whisper1), `InputAudioTranscription`, `InputAudioNoiseReduction`
  - `vad.rs`: `TurnDetection`, `ServerVadConfig`, `SemanticVadConfig`
  - `conversation.rs`: `ConversationItem`, `ContentPart`
  - `events/client.rs`: Client-to-server events (9 types)
  - `events/server.rs`: Server-to-client events (28 types)
  - `stream.rs`: `EventHandler` for callback-based processing

- **`live/`**: Live API (Primary WebSocket and Fork, `wss://api.openai.com/v1/live/sessions`)
  - `client.rs`: `LiveClient` builder, Primary WebSocket/Fork connections, and `LiveSession`/`LiveSender`/`LiveReceiver`
  - `session.rs`: Session configuration, audio formats, voices, and delegation configuration
  - `events.rs`: Client-to-server and server-to-client events
  - Supports OpenAI authentication only; WebRTC session creation, sideband attach, SIP, `transport.*` events, and recording download are not supported

- **`models/`**: Models API (`/v1/models`)
  - `request.rs`: `Models` client for list, retrieve, delete
  - `response.rs`: `Model`, `ModelsListResponse`, `DeleteResponse`

- **`files/`**: Files API (`/v1/files`)
  - `request.rs`: `Files` client with multipart upload support
  - `response.rs`: `File`, `FileListResponse`, `DeleteResponse`

- **`moderations/`**: Moderations API (`/v1/moderations`)
  - `request.rs`: `Moderations` client for content policy classification
  - `response.rs`: `ModerationResponse`, `ModerationCategories`, `ModerationCategoryScores`

- **`images/`**: Images API (`/v1/images`)
  - `request.rs`: `Images` client for generate, edit, variation
  - `response.rs`: `ImageResponse`, `ImageData`

- **`audio/`**: Audio API (`/v1/audio`)
  - `request.rs`: `Audio` client for TTS, transcription, translation; `TtsModel` (Tts1, Tts1Hd, Gpt4oMiniTts), `SttModel` (Whisper1, Gpt4oTranscribe, Gpt4oMiniTranscribe, GptTranscribe, GptLiveTranscribe, GptRealtimeWhisper), `Voice` (Alloy, Ash, Ballad, Cedar, Coral, Echo, Fable, Marin, Nova, Onyx, Sage, Shimmer, Verse), `AudioFormat`, `TtsOptions` (with `instructions` field for Gpt4oMiniTts)
  - `response.rs`: `TranscriptionResponse`, `Word`, `Segment`

- **`batch/`**: Batch API (`/v1/batches`)
  - `request.rs`: `Batches` client for create, list, retrieve, cancel
  - `response.rs`: `BatchObject`, `BatchStatus`, `RequestCounts`

- **`fine_tuning/`**: Fine-tuning API (`/v1/fine_tuning/jobs`)
  - `request.rs`: `FineTuning` client for jobs, events, checkpoints
  - `response.rs`: `FineTuningJob`, `FineTuningEvent`, `FineTuningCheckpoint`

- **`videos/`**: Videos API (`/v1/videos`) - Sora video generation
  - `request.rs`: `Videos` client for create, retrieve, list, delete, content, remix; `VideoModel` (Sora2, Sora2Pro), `VideoSize`, `VideoSeconds`, `VideoVariant`, `InputReference`, `CreateVideoOptions`, `SortOrder`
  - `response.rs`: `Video`, `VideoStatus`, `VideoError`, `VideoListResponse`, `DeleteVideoResponse`

- **`common/`**: Shared types across all APIs
  - `auth.rs`: Authentication abstraction (`AuthProvider`, `OpenAIAuth`, `AzureAuth`) for OpenAI and Azure OpenAI APIs
  - `client.rs`: HTTP client utilities with timeout configuration (`create_http_client`)
  - `message.rs`: `Message`, `Content`, `ToolCall`
  - `role.rs`: `Role` enum (User, Assistant, System, Tool)
  - `models.rs`: Type-safe model enums (`ChatModel`, `EmbeddingModel`, `RealtimeModel`, `LiveModel`, `FineTuningModel`) and `ParameterSupport`/`ParameterRestriction` for model parameter validation
  - `tool.rs`: `Tool` definition for function calling
  - `function.rs`: `Function` struct (internal function metadata used by `Tool`)
  - `parameters.rs`: `ParameterProperty` for tool parameters
  - `structured_output.rs`: `Schema` for JSON schema responses
  - `errors.rs`: `OpenAIToolError` error type
  - `usage.rs`: Token usage tracking

### Key Patterns

**Builder Pattern**: All API clients use builder-style configuration:
```rust
use openai_tools::common::models::ChatModel;

let mut chat = ChatCompletion::new();
chat.model(ChatModel::Gpt5Mini)  // Type-safe model selection
    .messages(messages)
    .temperature(0.7)
    .chat()
    .await?;
```

**Multi-modal Input (Chat API)**: Text and image content in Chat Completions API:
```rust
use openai_tools::chat::request::ChatCompletion;
use openai_tools::common::{message::{Message, Content}, role::Role, models::ChatModel};

let message = Message::from_message_array(
    Role::User,
    vec![
        Content::from_text("What do you see?"),
        Content::from_image_url("https://example.com/image.jpg"),
    ],
);

let mut chat = ChatCompletion::new();
let response = chat
    .model(ChatModel::Gpt5Mini)
    .messages(vec![message])
    .chat()
    .await?;
```

Note: The same `Content` and `Message` types work with both Chat API and Responses API.
Serialization format differences (`input_text`/`input_image` vs `text`/`image_url`) are
handled automatically by API-specific serialization wrappers.

**Timeout Configuration**: All HTTP-based API clients support optional request timeouts:
```rust
use std::time::Duration;
use openai_tools::common::models::ChatModel;

// Builder pattern clients
let mut chat = ChatCompletion::new();
chat.model(ChatModel::Gpt5Mini)
    .timeout(Duration::from_secs(30))  // Set 30 second timeout
    .messages(messages)
    .chat()
    .await?;

// Service pattern clients
let mut files = Files::new()?;
files.timeout(Duration::from_secs(120));  // Longer timeout for file uploads
let file = files.upload_path("data.jsonl", FilePurpose::FineTune).await?;
```

Timeout is optional - if not set, requests have no timeout limit (default reqwest behavior).

**Safety Identifier**: End-user identification for abuse detection (Chat and Responses APIs only):
```rust
use openai_tools::chat::request::ChatCompletion;
use openai_tools::common::models::ChatModel;

let mut chat = ChatCompletion::new();
chat.model(ChatModel::Gpt5Mini)
    .messages(messages)
    .safety_identifier("hashed-user-id")  // SHA-256 hash recommended
    .chat()
    .await?;
```

Note: `safety_identifier` is the successor to the legacy `user` parameter. It is supported by
Chat Completions API and Responses API. Images API does **not** support this parameter (use the
`user` field in `GenerateOptions`/`EditOptions`/`VariationOptions` instead).

**Enum Forward Compatibility** (since v3.0.0): OpenAI adds enum values continuously, so the public enums use two distinct mechanisms. Do not confuse them:

| Mechanism | Problem solved | When it applies |
|-----------|----------------|-----------------|
| `#[non_exhaustive]` | Adding a variant is not a breaking change | Compile time / semver |
| `Other(String)` / `Unknown` | Unrecognised API values still deserialize | Runtime |

`#[non_exhaustive]` does **nothing** at runtime — an unknown value from the API still fails to deserialize without a catch-all variant. Rules when adding a new enum:

- **Request-side value enums** (models, voices, sizes, formats): mark `#[non_exhaustive]`.
- **Response-side enums** (statuses, events): mark `#[non_exhaustive]` **and** add a catch-all variant (`#[serde(untagged)] Other(String)` for string enums; `#[serde(other)] Unknown` for internally tagged enums like `ServerEvent`).
- **Library-owned closed types** (`ParameterRestriction`, `AuthProvider`, `SortOrder`, …): leave as-is.

`tests/enum_forward_compat.rs` audits the source and fails if a designated enum loses `#[non_exhaustive]`. Add new enums to its `REQUIRED` list.

Note: a catch-all variant holding a `String` is incompatible with `#[derive(Copy)]`.

**Type-Safe Model Selection**: All APIs use enum-based model selection for compile-time validation:

| Enum | API | Available Variants |
|------|-----|-------------------|
| `ChatModel` | Chat, Responses | **GPT-6**: `Gpt6Astra`, `Gpt6Sol`, `Gpt6Luna` / **GPT-5.6**: `Gpt5_6` (alias→Sol), `Gpt5_6Sol`, `Gpt5_6Terra`, `Gpt5_6Luna` / **GPT-5.5**: `Gpt5_5`, `Gpt5_5Pro` / **GPT-5.4**: `Gpt5_4`, `Gpt5_4Pro`, `Gpt5_4Mini`, `Gpt5_4Nano` / **GPT-5.3**: `Gpt5_3ChatLatest` / **Codex**: `Gpt5_3Codex`, `Gpt5_2Codex`, `Gpt5_1Codex`, `Gpt5_1CodexMax` / **GPT-5**: `Gpt5`, `Gpt5Pro`, `Gpt5_2`, `Gpt5_2ChatLatest`, `Gpt5_2Pro`, `Gpt5_1`, `Gpt5_1ChatLatest`, `Gpt5Mini`, `Gpt5Nano` / **Search**: `Gpt5SearchApi`, `Gpt4oSearchPreview`, `Gpt4oMiniSearchPreview` / **Audio**: `GptAudio`, `GptAudio1_5`, `GptAudioMini` / **GPT-4.1**: `Gpt4_1`, `Gpt4_1Mini`, `Gpt4_1Nano` / **GPT-4o**: `Gpt4o`, `Gpt4oMini`, `Gpt4oAudioPreview` / **GPT-4/3.5**: `Gpt4Turbo`, `Gpt4`, `Gpt3_5Turbo`, `Gpt3_5Turbo16k` / **Reasoning**: `O1`, `O1Pro`, `O3`, `O3Pro`, `O3Mini`, `O4Mini` / `Custom(String)` |
| `EmbeddingModel` | Embedding | `TextEmbedding3Small`, `TextEmbedding3Large`, `TextEmbeddingAda002` |
| `RealtimeModel` | Realtime | `GptRealtime2_1`, `GptRealtime2_1Mini`, `GptRealtime2`, `GptRealtime1_5`, `GptRealtime`, `GptRealtimeMini`, `GptRealtimeTranslate`, `GptRealtime_2025_08_28` (default), `Custom(String)` |
| `LiveModel` | Live | `GptLive1` (default), `Custom(String)` |
| `FineTuningModel` | Fine-tuning | `Gpt41_2025_04_14`, `Gpt41Mini_2025_04_14`, `Gpt41Nano_2025_04_14`, `Gpt4oMini_2024_07_18`, `Gpt4o_2024_08_06`, `Gpt4_0613`, `Gpt35Turbo_0125`, `Babbage002`, `Davinci002`, etc. |
| `ImageModel` | Images | `GptImage1` (default), `GptImage1Mini`, `GptImage1_5`, `GptImage2`, `GptImage2_5Flare`, `GptImage2_5Sunburst`, `ChatGptImageLatest`, `DallE2`/`DallE3` (retired on OpenAI; Azure only) |
| `VideoModel` | Videos | `Sora2` (default), `Sora2Pro` |

```rust
use openai_tools::common::models::{ChatModel, EmbeddingModel, RealtimeModel, FineTuningModel};

// Chat/Responses API - GPT-5 series (latest flagship)
chat.model(ChatModel::Gpt5_2);          // GPT-5.2 Thinking - coding & agentic tasks
responses.model(ChatModel::Gpt5_2Pro);  // GPT-5.2 Pro - most capable (Responses API only)
chat.model(ChatModel::Gpt5_1);          // GPT-5.1 - configurable reasoning

// Chat/Responses API - Other models
chat.model(ChatModel::Gpt5Mini);       // Cost-effective
responses.model(ChatModel::O3Mini);     // Reasoning model

// Embedding API
embedding.model(EmbeddingModel::TextEmbedding3Small);

// Realtime API
client.model(RealtimeModel::GptRealtime_2025_08_28);

// Fine-tuning API
CreateFineTuningJobRequest::new(FineTuningModel::Gpt41Mini_2025_04_14, "file-id");

// Custom models (for fine-tuned models or new models)
chat.model(ChatModel::custom("ft:gpt-4.1-mini:my-org::abc123"));

// Check if model is a reasoning model (GPT-5 series and o-series)
if ChatModel::Gpt5_2.is_reasoning_model() {
    // Handle reasoning model restrictions
}
```

**Schema for Structured Output**: Two factory methods depending on API:
- `Schema::chat_json_schema("name")` - for Chat Completions
- `Schema::responses_json_schema("name")` - for Responses API

**Tool Definition**: Function tools follow this pattern:
```rust
Tool::function(
    "name",
    "description",
    vec![("param", ParameterProperty::from_string("desc"))],
    false, // strict mode
)
```

**Realtime API**: WebSocket-based real-time communication:
```rust
use openai_tools::common::models::RealtimeModel;

let mut client = RealtimeClient::new();
client
    .model(RealtimeModel::GptRealtime_2025_08_28)  // Type-safe model selection
    .modalities(vec![Modality::Text, Modality::Audio])
    .voice(Voice::Alloy)
    .server_vad(ServerVadConfig::default());

let mut session = client.connect().await?;
session.send_text("Hello!").await?;
session.create_response(None).await?;

while let Some(event) = session.recv().await? {
    match event {
        ServerEvent::ResponseTextDelta(e) => print!("{}", e.delta),
        ServerEvent::ResponseDone(_) => break,
        _ => {}
    }
}
session.close().await?;
```

**Live API**: Primary WebSocket sessions require input audio to keep advancing:
```rust
use openai_tools::live::{LiveClient, LiveVoice};
use std::time::Duration;

let mut client = LiveClient::new();
client
    .instructions("Greet the user briefly as soon as the session starts.")
    .voice(LiveVoice::Marin);

let session = client.connect().await?;
let (sender, mut receiver) = session.split();

// Keep real-time input flowing: 100 ms of 24 kHz PCM16 mono silence is 4,800 bytes.
let audio_sender = sender.clone();
let audio_task = tokio::spawn(async move {
    let silence = [0_u8; 4_800];
    loop {
        if audio_sender.append_audio_bytes(&silence).await.is_err() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
});

tokio::time::sleep(Duration::from_secs(2)).await;
sender
    .append_commentary("Tell the user: Hello from the Live API.", None)
    .await?;

// Continue receiving output events while the audio task keeps the model advancing.
if let Some(event) = tokio::time::timeout(Duration::from_secs(15), receiver.recv()).await?? {
    println!("{event:?}");
}

audio_task.abort();
drop(sender);
receiver.close().await?;
```

Sending commentary without streaming input audio can leave context injection incomplete. Also,
`session.usage.updated.usage.seconds` is cumulative: keep the latest value rather than adding
successive updates.

**Conversations API**: Manage long-running conversations with the Responses API:
```rust
use openai_tools::conversations::request::Conversations;
use openai_tools::conversations::response::InputItem;
use openai_tools::common::models::ChatModel;
use std::collections::HashMap;

let conversations = Conversations::new()?;

// Create a conversation with metadata
let mut metadata = HashMap::new();
metadata.insert("user_id".to_string(), "user123".to_string());

let conv = conversations.create(Some(metadata), None).await?;
println!("Created conversation: {}", conv.id);

// Add items to the conversation
let items = vec![InputItem::user_message("Hello!")];
conversations.create_items(&conv.id, items).await?;

// List conversation items
let items = conversations.list_items(&conv.id, Some(10), None, None, None).await?;
for item in &items.data {
    println!("Item: {} ({})", item.id, item.item_type);
}

// Use with Responses API
let mut client = Responses::new();
client.model(ChatModel::Gpt4_1).conversation(&conv.id).str_message("How are you?").complete().await?;

// Delete conversation when done
conversations.delete(&conv.id).await?;
```

**Responses API CRUD Operations**: Manage responses with full CRUD support:
```rust
use openai_tools::responses::request::{Responses, ToolChoice, ToolChoiceMode, Prompt};
use openai_tools::common::models::ChatModel;

let mut client = Responses::new();

// Create a response with tool_choice and prompt caching
client.model(ChatModel::Gpt5Mini)
    .str_message("Hello!")
    .tool_choice(ToolChoice::Simple(ToolChoiceMode::Auto))
    .prompt_cache_key("my-cache-key")
    .prompt_cache_retention("24h")
    .store(true);  // Store for later retrieval

let response = client.complete().await?;
let response_id = response.id.as_ref().unwrap();

// Retrieve a stored response
let retrieved = client.retrieve(response_id).await?;
println!("Status: {:?}", retrieved.status);

// List input items with pagination
let items = client.list_input_items(response_id, Some(10), None, None).await?;
for item in items.data {
    println!("Input: {} ({})", item.id, item.item_type);
}

// Count input tokens before sending a request
let tokens = client.get_input_tokens("gpt-5-mini", serde_json::json!("Hello!")).await?;
println!("Input tokens: {}", tokens.input_tokens);

// Compact a response to reduce token usage
let compacted = client.compact(response_id, None).await?;

// Cancel an in-progress response
// client.cancel(response_id).await?;

// Delete a response when done
client.delete(response_id).await?;
```

**Models API**: List and retrieve available models:
```rust
let models = Models::new()?;

// List all models
let response = models.list().await?;
for model in &response.data {
    println!("{}: owned by {}", model.id, model.owned_by);
}

// Retrieve a specific model
let model = models.retrieve("gpt-5-mini").await?;
```

**Files API**: Upload, manage, and retrieve files:
```rust
let files = Files::new()?;

// Upload a file for fine-tuning
let file = files.upload_path("training.jsonl", FilePurpose::FineTune).await?;

// Or upload from bytes
let content = b"jsonl content here";
let file = files.upload_bytes(content, "data.jsonl", FilePurpose::Batch).await?;

// List files
let response = files.list(Some(FilePurpose::FineTune)).await?;

// Get file content
let content = files.content(&file.id).await?;

// Delete file
files.delete(&file.id).await?;
```

**Moderations API**: Check content for policy violations:
```rust
let moderations = Moderations::new()?;

// Check a single text
let response = moderations.moderate_text("Hello, world!", None).await?;
if response.results[0].flagged {
    println!("Content was flagged!");
}

// Check multiple texts at once
let texts = vec!["Text 1".to_string(), "Text 2".to_string()];
let response = moderations.moderate_texts(texts, None).await?;
```

**Images API**: Generate images with DALL-E:
```rust
let images = Images::new()?;

// Generate an image
let options = GenerateOptions {
    model: Some(ImageModel::DallE3),
    size: Some(ImageSize::Size1024x1024),
    quality: Some(ImageQuality::Hd),
    ..Default::default()
};
let response = images.generate("A sunset over mountains", options).await?;
println!("Image URL: {:?}", response.data[0].url);

// Get base64-encoded image
let options = GenerateOptions {
    response_format: Some(ResponseFormat::B64Json),
    ..Default::default()
};
let response = images.generate("A cute robot", options).await?;
let bytes = response.data[0].as_bytes().unwrap()?;
std::fs::write("robot.png", bytes)?;
```

**Audio API**: Text-to-speech and transcription:
```rust
let audio = Audio::new()?;

// Text-to-speech (basic)
let options = TtsOptions {
    model: TtsModel::Tts1Hd,
    voice: Voice::Nova,
    ..Default::default()
};
let bytes = audio.text_to_speech("Hello!", options).await?;
std::fs::write("hello.mp3", bytes)?;

// Transcribe audio
let options = TranscribeOptions {
    language: Some("en".to_string()),
    ..Default::default()
};
let response = audio.transcribe("audio.mp3", options).await?;
println!("Transcript: {}", response.text);

// Translate audio to English
let response = audio.translate("foreign_audio.mp3", TranslateOptions::default()).await?;
```

**Batch API**: Process large volumes of requests asynchronously with 50% cost savings:
```rust
let batches = Batches::new()?;

// Create a batch job (input file must be uploaded via Files API with purpose "batch")
let request = CreateBatchRequest::new("file-abc123", BatchEndpoint::ChatCompletions);
let batch = batches.create(request).await?;

// Check batch status
let batch = batches.retrieve(&batch.id).await?;
match batch.status {
    BatchStatus::Completed => {
        println!("Output file: {:?}", batch.output_file_id);
    }
    _ => println!("Status: {:?}", batch.status),
}

// List all batches
let response = batches.list(Some(20), None).await?;
```

**Fine-tuning API**: Customize models with your training data:
```rust
use openai_tools::common::models::FineTuningModel;

let fine_tuning = FineTuning::new()?;

// Create a fine-tuning job
let hyperparams = Hyperparameters {
    n_epochs: Some(3),
    ..Default::default()
};
let request = CreateFineTuningJobRequest::new(FineTuningModel::Gpt41Mini_2025_04_14, "file-abc123")
    .with_suffix("my-model")
    .with_supervised_method(Some(hyperparams));

let job = fine_tuning.create(request).await?;

// Check job status
let job = fine_tuning.retrieve(&job.id).await?;
if job.status == FineTuningJobStatus::Succeeded {
    println!("Model: {:?}", job.fine_tuned_model);
}

// List training events
let events = fine_tuning.list_events(&job.id, Some(10), None).await?;
for event in &events.data {
    println!("[{}] {}", event.level, event.message);
}
```

## Environment Setup

### OpenAI API

Requires `OPENAI_API_KEY` environment variable or in `.env` file:
```
OPENAI_API_KEY=sk-...
```

### Azure OpenAI API

| Variable | Required | Description |
|----------|----------|-------------|
| `AZURE_OPENAI_API_KEY` | Yes | Azure API key |
| `AZURE_OPENAI_BASE_URL` | Yes | Complete endpoint URL including deployment, API path, and api-version |

Example `AZURE_OPENAI_BASE_URL` format (for Chat API):
```
https://my-resource.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview
```

Note: Each API (Chat, Embedding, etc.) requires its own complete endpoint URL.

### Provider Detection

All API clients support multiple ways to configure authentication:

```rust
use openai_tools::chat::request::ChatCompletion;
use openai_tools::common::auth::{AuthProvider, AzureAuth};

// 1. OpenAI (default) - from OPENAI_API_KEY env var
let chat = ChatCompletion::new();

// 2. Azure - from AZURE_OPENAI_API_KEY and AZURE_OPENAI_BASE_URL env vars
let chat = ChatCompletion::azure()?;

// 3. Auto-detect - Azure if AZURE_OPENAI_API_KEY is set, otherwise OpenAI
let chat = ChatCompletion::detect_provider()?;

// 4. URL-based detection - auto-detects provider from URL pattern
//    *.openai.azure.com → Azure, others → OpenAI-compatible
let chat = ChatCompletion::with_url(
    "https://my-resource.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview",
    "azure-key"
);

// 5. URL-based with env var credentials
let chat = ChatCompletion::from_url("https://api.openai.com/v1")?;

// 6. Explicit auth configuration with complete endpoint URL
let auth = AuthProvider::Azure(
    AzureAuth::new(
        "api-key",
        "https://my-resource.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview"
    )
);
let chat = ChatCompletion::with_auth(auth);

// 7. OpenAI-compatible APIs (Ollama, vLLM, LocalAI, etc.)
let chat = ChatCompletion::with_url(
    "http://localhost:11434/v1",
    "ollama"
);
```

### URL-based Provider Detection

The `with_url()` and `from_url()` methods automatically detect the provider based on URL pattern:

| URL Pattern | Detected Provider |
|-------------|-------------------|
| `*.openai.azure.com*` | Azure OpenAI |
| All other URLs | OpenAI-compatible |

This is useful for:
- Supporting multiple providers with a single configuration
- Connecting to local OpenAI-compatible servers (Ollama, vLLM, LocalAI)
- Dynamic provider switching based on environment

### Azure URL Construction

Azure OpenAI uses a complete endpoint URL that includes the deployment, API path, and api-version. The `endpoint()` method returns this URL as-is:

```rust
use openai_tools::common::auth::AzureAuth;

// For Chat API
let chat_auth = AzureAuth::new(
    "api-key",
    "https://my-resource.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview"
);

// For Embedding API
let embedding_auth = AzureAuth::new(
    "api-key",
    "https://my-resource.openai.azure.com/openai/deployments/text-embedding/embeddings?api-version=2024-08-01-preview"
);
```

Each API (Chat, Embedding, etc.) requires its own complete endpoint URL including the API path.

## Feature Status

| API | Endpoint | Features |
|-----|----------|----------|
| **Chat** | `/v1/chat/completions` | Structured Output, Function Calling, Multi-modal Input (Text + Image), Safety Identifier |
| **Responses** | `/v1/responses` | CRUD, Structured Output, Function Calling, Image Input, Reasoning, Tool Choice, Prompt Templates, Safety Identifier |
| **Conversations** | `/v1/conversations` | CRUD |
| **Embedding** | `/v1/embeddings` | Basic |
| **Realtime** | `wss://api.openai.com/v1/realtime` | Function Calling, Audio I/O, VAD, WebSocket |
| **Live** | `wss://api.openai.com/v1/live/sessions` | Primary WebSocket, Fork, Audio I/O, Delegation |
| **Models** | `/v1/models` | CRUD |
| **Files** | `/v1/files` | CRUD, Multipart Upload |
| **Moderations** | `/v1/moderations` | Basic |
| **Images** | `/v1/images` | Multipart Upload |
| **Audio** | `/v1/audio` | Audio I/O, Multipart Upload |
| **Batch** | `/v1/batches` | CRUD |
| **Fine-tuning** | `/v1/fine_tuning/jobs` | CRUD |
| **Videos** | `/v1/videos` | CRUD, Content Download, Remix |

**Chat Completions API Features:**
- Structured output (JSON schema)
- Function calling with tools
- Multi-modal input (text + image)
- Safety identifier (`safety_identifier`) for end-user abuse detection
  - Successor to the legacy `user` parameter
  - Recommended: hash user identifiers (e.g., SHA-256 of email) rather than sending PII

**Responses API Features:**
- Multi-modal input support (text, images)
- Configurable reasoning with `ReasoningEffort` (none, minimal, low, medium, high, xhigh, max) and `ReasoningSummary` (auto, concise, detailed)
- Input truncation behavior with `Truncation` (Auto, Disabled)
- Tool choice control with `ToolChoice` (auto, none, required, or specific function)
- Prompt templates with `Prompt` (ID and variable substitution)
- Prompt caching with `prompt_cache_key` and `prompt_cache_retention`
- Response data inclusion with `Include`:
  - `WebSearchCall` - web search results
  - `CodeInterpreterCall` - code execution outputs
  - `FileSearchCall` - file search results
  - `ImageUrlInInputMessages` - image URLs from input
  - `ImageUrlInComputerCallOutput` - computer call output images
  - `LogprobsInOutput` - token log probabilities
  - `ReasoningEncryptedContent` - encrypted reasoning content
- Safety identifier (`safety_identifier`) for end-user abuse detection
- Parallel tool calls support
- Conversation integration
- Metadata tracking
- CRUD endpoints:
  - `retrieve(id)` - GET response by ID
  - `delete(id)` - DELETE response
  - `cancel(id)` - Cancel in-progress response
  - `list_input_items(id)` - List input items with pagination
  - `compact(id)` - Compact response for token reduction
  - `get_input_tokens(model, input)` - Count input tokens

**Conversations API Features:**
- Create conversations with optional metadata and initial items
- Retrieve, update, and delete conversations
- Add items (messages, tool calls, etc.) to conversations
- List conversation items with pagination
- Integration with Responses API for multi-turn interactions
- Metadata support (key-value pairs for tracking)

**Models API Features:**
- List all available models
- Retrieve specific model details
- Delete fine-tuned models

**Files API Features:**
- Upload files (from path or bytes)
- List files (with purpose filter)
- Retrieve file details
- Delete files
- Get file content

**Moderations API Features:**
- Single text moderation
- Batch text moderation
- Multiple models (omni-moderation, text-moderation)
- Detailed category scores

**Images API Features:**
- Image generation (GPT Image 1 / 1 Mini / 1.5 / 2 / 2.5 Flare / 2.5 Sunburst)
- **DALL-E retired**: `dall-e-2`/`dall-e-3` no longer exist on `api.openai.com` (the API returns "The model 'dall-e-3' does not exist."). The enum variants remain for Azure deployments. `ImageModel::default()` is `GptImage1`.
- GPT Image models reject `response_format` and `style` (`unknown_parameter`) and always return base64; `quality` must be `low`/`medium`/`high`/`auto` (GPT Image 2.5 also accepts `max`), not `standard`/`hd`
- Image editing with masks
- Image variations (DALL-E 2 only)
- URL or base64 response format
- Quality and style options

**Realtime API Features:**
- WebSocket-based real-time communication
- Text and audio modalities
- Voice options: Alloy, Ash, Ballad, Coral, Echo, Sage, Shimmer, Verse
- Transcription models: Whisper1
- Voice Activity Detection (VAD): Server VAD and Semantic VAD
- Audio formats: PCM16, G711Ulaw, G711Alaw
- Input audio transcription with language hints
- Noise reduction (near-field, far-field)
- Function calling support

**Live API Features:**
- Primary WebSocket sessions and session Fork
- PCM16, G.711 μ-law, and G.711 A-law audio
- Client and Responses API delegation
- The model advances only while input audio is streaming in real time. Keep sending audio continuously (for example, 4,800 bytes of 24 kHz PCM16 silence every 100 ms) while waiting for speech or transcript events.
- `session.usage.updated.usage.seconds` is cumulative; use the latest value and do not add successive updates.
- WebRTC session creation, sideband attach, SIP, `transport.*` events, recording download, and Azure authentication are not supported.

**Audio API Features:**
- Text-to-speech (TTS) with multiple voices: Alloy, Ash, Ballad, Cedar, Coral, Echo, Fable, Marin, Nova, Onyx, Sage, Shimmer, Verse
- TTS models: Tts1, Tts1Hd, Gpt4oMiniTts
- `instructions` parameter for Gpt4oMiniTts (control voice tone, emotion, and pacing with natural language)
- STT models: Whisper1, Gpt4oTranscribe, Gpt4oMiniTranscribe, GptTranscribe
- Realtime-only STT models: GptLiveTranscribe, GptRealtimeWhisper (check `SttModel::supports_file_transcription()` before calling `transcribe`)
- Audio transcription (Whisper, GPT-4o Transcribe)
- Audio translation to English
- Multiple audio formats (MP3, WAV, FLAC, Opus, AAC, PCM)
- Word and segment timestamps

**Batch API Features:**
- Create async batch jobs with 50% cost savings
- Support for Chat Completions, Embeddings, Responses, Moderations endpoints
- Retrieve batch status and progress
- List all batches with pagination
- Cancel in-progress batches
- 24-hour completion window

**Fine-tuning API Features:**
- Create fine-tuning jobs (Supervised, DPO methods)
- Configurable hyperparameters (epochs, batch size, learning rate)
- Monitor training with events
- Access training checkpoints
- List and retrieve job details
- Cancel in-progress jobs
- Support for GPT-4.1, GPT-4.1 Mini, GPT-4.1 Nano, GPT-4o, GPT-4o Mini, GPT-4, GPT-3.5 Turbo

## Model-Specific Parameter Restrictions

### Reasoning Models (GPT-6 Series, GPT-5 Series, o1, o3, o4 Series)

OpenAI's reasoning models (GPT-6 series: `gpt-6-*`, GPT-5 series: `gpt-5.6-*`, `gpt-5.5`, `gpt-5.4`, `gpt-5.2`, `gpt-5.1`, `gpt-5-mini`, the `*-codex` models, and o-series: `o1`, `o3`, `o3-mini`, `o4-mini`, etc.) have specific parameter restrictions. This library automatically handles these restrictions by ignoring unsupported parameters and logging warnings.

**Exception — `*-chat-latest` models**: `gpt-5.1-chat-latest`, `gpt-5.2-chat-latest` and `gpt-5.3-chat-latest` point at the non-reasoning "Instant" snapshots used in ChatGPT. They accept the **full standard parameter set** (`temperature`, `top_p`, `n`, `logprobs`, …), so `is_reasoning_model()` returns `false` for them and no parameters are dropped.

**Responses API only** (no Chat Completions endpoint): `gpt-5.5-pro`, `gpt-5.4-pro`, `gpt-5.2-pro`, `gpt-5-pro`, `o3-pro`, and all `*-codex` models.

**Search models** (`gpt-5-search-api`, `gpt-4o-search-preview`, `gpt-4o-mini-search-preview`) are a third category: they reject the entire sampling parameter set (`temperature`, `top_p`, `n`, `logprobs`, both penalties) exactly like reasoning models, but they expose **no** `reasoning` parameter. Use `ChatModel::is_search_model()`; `parameter_support()` returns `ParameterSupport::search_model()` for them.

#### GPT-5 Series Reasoning Configuration

GPT-5 models support configurable reasoning effort via the `reasoning` parameter (Responses API only):

```rust
use openai_tools::responses::request::{Responses, ReasoningEffort, ReasoningSummary};
use openai_tools::common::models::ChatModel;

let mut responses = Responses::new();
responses.model(ChatModel::Gpt5_2)
    .reasoning(
        ReasoningEffort::High,  // none, minimal, low, medium, high, xhigh, max
        ReasoningSummary::Auto, // auto, concise, detailed
    )
    .str_message("Solve this complex problem...")
    .complete()
    .await?;
```

**ReasoningEffort levels:**
| Level | Description | Supported Models |
|-------|-------------|------------------|
| `None` | No reasoning tokens (fastest) | GPT-5.1, GPT-5.2 |
| `Minimal` | Very few reasoning tokens | GPT-5 Mini |
| `Low` | Light reasoning | GPT-5.1, GPT-5.2, o-series |
| `Medium` | Balanced reasoning | All reasoning models |
| `High` | Deep reasoning | All reasoning models |
| `Xhigh` | Maximum reasoning | GPT-5.2 Pro, GPT-5.1 Codex Max |
| `Max` | Maximum reasoning (Responses API only) | GPT-6 series |

#### Chat Completions API Parameter Restrictions

| Parameter | Restriction | Library Behavior |
|-----------|-------------|------------------|
| `temperature` | Only `1.0` supported | Values ≠ 1.0 are ignored with warning |
| `top_p` | Only `1.0` supported | Not implemented in Chat API |
| `frequency_penalty` | Only `0` supported | Values ≠ 0 are ignored with warning |
| `presence_penalty` | Only `0` supported | Values ≠ 0 are ignored with warning |
| `logprobs` | Not supported | Ignored with warning if set |
| `top_logprobs` | Not supported | Ignored with warning if set |
| `logit_bias` | Not supported | Ignored with warning if set |
| `n` | Only `1` supported | Values ≠ 1 are ignored with warning |

#### Responses API Parameter Restrictions

| Parameter | Restriction | Library Behavior |
|-----------|-------------|------------------|
| `temperature` | Only `1.0` supported | Values ≠ 1.0 are ignored with warning |
| `top_p` | Only `1.0` supported | Values ≠ 1.0 are ignored with warning |
| `top_logprobs` | Not supported | Ignored with warning if set |

#### Example Usage with Reasoning Models

```rust
use openai_tools::chat::request::ChatCompletion;
use openai_tools::responses::request::Responses;
use openai_tools::common::models::ChatModel;

// Chat API with reasoning model
let mut chat = ChatCompletion::new();
chat.model(ChatModel::O1)           // Type-safe reasoning model selection
    .temperature(0.3)               // Warning: ignored, using default 1.0
    .frequency_penalty(0.5)         // Warning: ignored, using default 0
    .messages(messages)
    .chat()
    .await?;  // Request succeeds without API error

// Responses API with reasoning model
let mut responses = Responses::new();
responses.model(ChatModel::O3Mini)  // Type-safe reasoning model selection
    .temperature(0.7)               // Warning: ignored, using default 1.0
    .top_p(0.9)                     // Warning: ignored, using default 1.0
    .str_message("Hello!")
    .complete()
    .await?;  // Request succeeds without API error
```

#### Warning Log Examples

When using unsupported parameters with reasoning models, warnings are logged via `tracing::warn!`:

```
WARN: Reasoning model 'o1-preview' does not support custom temperature. Ignoring temperature=0.3 and using default (1.0).
WARN: Reasoning model 'o1-preview' does not support frequency_penalty. Ignoring frequency_penalty=0.5 and using default (0).
WARN: Reasoning model 'o3-mini' does not support top_p. Ignoring top_p=0.9 and using default (1.0).
```

### Standard Models (GPT-4o, GPT-4.1, GPT-4, GPT-3.5, etc.)

Standard models support all available parameters without restrictions:

| API | Supported Parameters |
|-----|---------------------|
| Chat Completions | `temperature`, `frequency_penalty`, `presence_penalty`, `logprobs`, `top_logprobs`, `logit_bias`, `n`, `max_completion_tokens`, `modalities`, `store`, `tools`, `response_format` |
| Responses | `temperature`, `top_p`, `top_logprobs`, `max_output_tokens`, `max_tool_calls`, `parallel_tool_calls`, `truncation`, `reasoning`, `tools`, `structured_output`, `metadata`, `include`, `background`, `conversation`, `store`, `stream` |

### API Parameter Comparison

| Parameter | Chat API | Responses API | Notes |
|-----------|:--------:|:-------------:|-------|
| `temperature` | ✅ | ✅ | Controls randomness (0.0-2.0) |
| `top_p` | ❌ | ✅ | Nucleus sampling |
| `frequency_penalty` | ✅ | ❌ | Reduces repetition |
| `presence_penalty` | ✅ | ❌ | Encourages new topics |
| `logprobs` | ✅ | ❌ | Token probabilities |
| `top_logprobs` | ✅ | ✅ | Top N probabilities |
| `logit_bias` | ✅ | ❌ | Token probability adjustment |
| `n` | ✅ | ❌ | Number of completions |
| `max_tokens` | `max_completion_tokens` | `max_output_tokens` | Different parameter names |
| `tools` | ✅ | ✅ | Function calling |
| `structured_output` | `response_format` | ✅ | JSON schema output |
| `reasoning` | ❌ | ✅ | Reasoning effort/summary |
| `truncation` | ❌ | ✅ | Input truncation behavior |
| `parallel_tool_calls` | ❌ | ✅ | Concurrent tool execution |
| `metadata` | ❌ | ✅ | Request tracking |

### References

- [OpenAI Reasoning Models Guide](https://platform.openai.com/docs/guides/reasoning)
- [Chat Completions API Reference](https://platform.openai.com/docs/api-reference/chat)
- [Responses API Reference](https://platform.openai.com/docs/api-reference/responses)

---
*This file was generated by Claude Code.*
