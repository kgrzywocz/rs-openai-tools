use crate::{
    common::{
        auth::{AuthProvider, OpenAIAuth},
        client::create_http_client,
        errors::{OpenAIToolError, Result},
        message::Message,
        models::{ChatModel, ParameterRestriction},
        structured_output::Schema,
        tool::Tool,
    },
    responses::response::{CompactedResponse, DeleteResponseResult, InputItemsListResponse, InputTokensResponse, Response},
};
use derive_new::new;
use request;
use serde::{ser::SerializeStruct, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use strum::{Display, EnumString};

/// Specifies additional data to include in the response output
///
/// This enum defines various types of additional information that can be
/// included in the API response output, such as web search results, code
/// interpreter outputs, image URLs, and other metadata.
///
/// # API Reference
///
/// Corresponds to the `include` parameter in the OpenAI Responses API:
/// <https://platform.openai.com/docs/api-reference/responses/create>
#[derive(Debug, Clone, EnumString, Display, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Include {
    /// Include web search call results in the output  
    ///
    /// When included, the response will contain information about web search
    /// results that were used during the response generation process.
    #[strum(serialize = "web_search_call.results")]
    #[serde(rename = "web_search_call.results")]
    WebSearchCall,

    /// Include code interpreter call outputs in the output
    ///
    /// When included, the response will contain outputs from any code
    /// that was executed during the response generation process.
    #[strum(serialize = "code_interpreter_call.outputs")]
    #[serde(rename = "code_interpreter_call.outputs")]
    CodeInterpreterCall,

    /// Include computer call output image URLs in the output
    ///
    /// When included, the response will contain image URLs from any
    /// computer interaction calls that were made.
    #[strum(serialize = "computer_call_output.output.image_url")]
    #[serde(rename = "computer_call_output.output.image_url")]
    ImageUrlInComputerCallOutput,

    /// Include file search call results in the output
    ///
    /// When included, the response will contain results from any
    /// file search operations that were performed.
    #[strum(serialize = "file_search_call.results")]
    #[serde(rename = "file_search_call.results")]
    FileSearchCall,

    /// Include image URLs from input messages in the output
    ///
    /// When included, the response will contain image URLs that were
    /// present in the input messages.
    #[strum(serialize = "message.input_image.image_url")]
    #[serde(rename = "message.input_image.image_url")]
    ImageUrlInInputMessages,

    /// Include log probabilities in the output
    ///
    /// When included, the response will contain log probability information
    /// for the generated text tokens.
    #[strum(serialize = "message.output_text.logprobs")]
    #[serde(rename = "message.output_text.logprobs")]
    LogprobsInOutput,

    /// Include reasoning encrypted content in the output
    ///
    /// When included, the response will contain encrypted reasoning
    /// content that shows the model's internal reasoning process.
    #[strum(serialize = "reasoning.encrypted_content")]
    #[serde(rename = "reasoning.encrypted_content")]
    ReasoningEncryptedContent,
}

/// Defines the level of reasoning effort the model should apply
///
/// This enum controls how much computational effort the model invests
/// in reasoning through complex problems before generating a response.
///
/// # Model Support
///
/// | Model | Supported Values |
/// |-------|-----------------|
/// | GPT-6 Astra | `low`, `medium`, `high`, `xhigh`, `max` |
/// | GPT-6 Sol, GPT-6 Luna | `none`, `low`, `medium`, `high`, `xhigh`, `max` |
/// | GPT-5.2, GPT-5.2-pro | `none`, `low`, `medium`, `high`, `xhigh` |
/// | GPT-5.1 | `none`, `low`, `medium`, `high` |
/// | GPT-5-mini | `minimal`, `medium`, `high` |
/// | o1, o3, o4 series | `low`, `medium`, `high` |
///
/// # API Reference
///
/// Corresponds to the `reasoning.effort` parameter in the OpenAI Responses API.
#[derive(Debug, Clone, Serialize, EnumString, Display, PartialEq)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ReasoningEffort {
    /// No reasoning tokens - fastest response (GPT-5.1/5.2 default)
    ///
    /// Use this when you don't need reasoning capabilities and want
    /// the fastest possible response time.
    #[strum(serialize = "none")]
    #[serde(rename = "none")]
    None,

    /// Minimal reasoning effort - fastest response time
    ///
    /// Use this for simple queries that don't require deep analysis.
    #[strum(serialize = "minimal")]
    #[serde(rename = "minimal")]
    Minimal,

    /// Low reasoning effort - balanced performance
    ///
    /// Use this for moderately complex queries that benefit from some reasoning.
    #[strum(serialize = "low")]
    #[serde(rename = "low")]
    Low,

    /// Medium reasoning effort - comprehensive analysis
    ///
    /// Use this for complex queries that require thorough consideration.
    #[strum(serialize = "medium")]
    #[serde(rename = "medium")]
    Medium,

    /// High reasoning effort - maximum thoughtfulness
    ///
    /// Use this for very complex queries requiring deep, careful analysis.
    #[strum(serialize = "high")]
    #[serde(rename = "high")]
    High,

    /// Extra-high reasoning effort - quality-critical work (GPT-5.2 only)
    ///
    /// Use this for the most demanding tasks requiring maximum reasoning
    /// quality. Only available on GPT-5.2 and GPT-5.2-pro models.
    #[strum(serialize = "xhigh")]
    #[serde(rename = "xhigh")]
    Xhigh,

    /// Maximum reasoning effort for GPT-6 models
    ///
    /// Responses API only. Accepted by GPT-6 Astra, GPT-6 Sol, and GPT-6 Luna
    /// (`gpt-6-astra`, `gpt-6-sol`, and `gpt-6-luna`). Chat Completions rejects
    /// `max` in its `reasoning_effort` parameter, including for GPT-6 models.
    #[strum(serialize = "max")]
    #[serde(rename = "max")]
    Max,
}

/// Defines the format of reasoning summary to include in the response
///
/// This enum controls how the model's reasoning process is summarized
/// and presented in the response output.
///
/// # API Reference
///
/// Corresponds to the `reasoning.summary` parameter in the OpenAI Responses API.
#[derive(Debug, Clone, Serialize, EnumString, Display, PartialEq)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ReasoningSummary {
    /// Automatically determine the appropriate summary format
    ///
    /// The model will choose the most suitable summary format based on the query.
    #[strum(serialize = "auto")]
    #[serde(rename = "auto")]
    Auto,

    /// Provide a concise summary of the reasoning process
    ///
    /// Use this for shorter, more focused reasoning explanations.
    #[strum(serialize = "concise")]
    #[serde(rename = "concise")]
    Concise,

    /// Provide a detailed summary of the reasoning process
    ///
    /// Use this for comprehensive reasoning explanations with full detail.
    #[strum(serialize = "detailed")]
    #[serde(rename = "detailed")]
    Detailed,
}

/// Configuration for reasoning behavior in responses
///
/// This struct allows you to control how the model approaches reasoning
/// for complex queries, including the effort level and summary format.
///
/// # API Reference
///
/// Corresponds to the `reasoning` parameter in the OpenAI Responses API.
#[derive(Debug, Clone, Serialize)]
pub struct Reasoning {
    /// The level of reasoning effort to apply
    pub effort: Option<ReasoningEffort>,
    /// The format for the reasoning summary
    pub summary: Option<ReasoningSummary>,
}

/// Defines the verbosity level for text output
///
/// This enum controls how detailed and lengthy the model's text responses
/// should be. Available on GPT-5.2 and newer models.
///
/// # API Reference
///
/// Corresponds to the `text.verbosity` parameter in the OpenAI Responses API.
#[derive(Debug, Clone, Serialize, EnumString, Display, PartialEq)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum TextVerbosity {
    /// Low verbosity - concise responses
    ///
    /// Use this for brief, to-the-point answers.
    #[strum(serialize = "low")]
    #[serde(rename = "low")]
    Low,

    /// Medium verbosity - balanced responses (default)
    ///
    /// Use this for standard-length responses with appropriate detail.
    #[strum(serialize = "medium")]
    #[serde(rename = "medium")]
    Medium,

    /// High verbosity - comprehensive responses
    ///
    /// Use this for detailed explanations with thorough coverage.
    #[strum(serialize = "high")]
    #[serde(rename = "high")]
    High,
}

/// Configuration for text output behavior
///
/// This struct allows you to control the characteristics of the generated
/// text output, such as verbosity level.
///
/// # API Reference
///
/// Corresponds to the `text` parameter in the OpenAI Responses API.
#[derive(Debug, Clone, Serialize)]
pub struct TextConfig {
    /// The verbosity level for text output
    pub verbosity: Option<TextVerbosity>,
}

/// Defines how the model should choose and use tools
///
/// This enum controls the model's behavior regarding tool usage during
/// response generation.
///
/// # API Reference
///
/// Corresponds to the `tool_choice` parameter in the OpenAI Responses API.
#[derive(Debug, Clone, Serialize, EnumString, Display, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoiceMode {
    /// Disable tool usage completely
    ///
    /// The model will not use any tools and will generate responses
    /// based solely on its training data.
    #[strum(serialize = "none")]
    #[serde(rename = "none")]
    None,

    /// Automatically decide when to use tools
    ///
    /// The model will automatically determine when tools are needed
    /// and which tools to use based on the query context.
    #[strum(serialize = "auto")]
    #[serde(rename = "auto")]
    Auto,

    /// Require the use of tools
    ///
    /// The model must use at least one of the provided tools in its response.
    #[strum(serialize = "required")]
    #[serde(rename = "required")]
    Required,
}

/// Controls truncation behavior for long inputs
///
/// This enum defines how the system should handle inputs that exceed
/// the maximum context length.
///
/// # API Reference
///
/// Corresponds to the `truncation` parameter in the OpenAI Responses API.
#[derive(Debug, Clone, Serialize, EnumString, Display, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Truncation {
    /// Automatically truncate inputs to fit context length
    ///
    /// The system will automatically trim inputs to ensure they fit
    /// within the model's context window.
    #[strum(serialize = "auto")]
    #[serde(rename = "auto")]
    Auto,

    /// Disable truncation - return error if input is too long
    ///
    /// The system will return an error rather than truncating
    /// inputs that exceed the context length.
    #[strum(serialize = "disabled")]
    #[serde(rename = "disabled")]
    Disabled,
}

/// Specifies a specific function to force the model to call
///
/// When you want the model to call a specific function rather than
/// choosing which tool to use, use this structure to specify the
/// function name.
///
/// # Example
///
/// ```rust
/// use openai_tools::responses::request::NamedFunctionChoice;
///
/// let function_choice = NamedFunctionChoice::new("get_weather");
/// ```
#[derive(Debug, Clone, Serialize)]
pub struct NamedFunctionChoice {
    /// The type of tool choice, always "function" for named functions
    #[serde(rename = "type")]
    pub type_name: String,
    /// The name of the function to call
    pub name: String,
}

impl NamedFunctionChoice {
    /// Creates a new NamedFunctionChoice for the specified function
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the function to call
    ///
    /// # Returns
    ///
    /// A new NamedFunctionChoice instance
    pub fn new<S: AsRef<str>>(name: S) -> Self {
        Self { type_name: "function".to_string(), name: name.as_ref().to_string() }
    }
}

/// Controls how the model selects tools
///
/// This enum allows you to specify whether the model should automatically
/// choose tools, be forced to use specific tools, or be prevented from
/// using tools altogether.
///
/// # API Reference
///
/// Corresponds to the `tool_choice` parameter in the OpenAI Responses API:
/// <https://platform.openai.com/docs/api-reference/responses/create>
///
/// # Examples
///
/// ```rust
/// use openai_tools::responses::request::{ToolChoice, ToolChoiceMode, NamedFunctionChoice};
///
/// // Let the model decide
/// let auto_choice = ToolChoice::Simple(ToolChoiceMode::Auto);
///
/// // Force a specific function
/// let function_choice = ToolChoice::Function(NamedFunctionChoice::new("get_weather"));
/// ```
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum ToolChoice {
    /// Simple mode selection (auto, none, required)
    Simple(ToolChoiceMode),
    /// Force a specific function to be called
    Function(NamedFunctionChoice),
}

/// Reference to a prompt template with variables
///
/// Allows you to use pre-defined prompt templates stored in the OpenAI
/// platform, optionally with variable substitution.
///
/// # API Reference
///
/// Corresponds to the `prompt` parameter in the OpenAI Responses API:
/// <https://platform.openai.com/docs/api-reference/responses/create>
///
/// # Examples
///
/// ```rust
/// use openai_tools::responses::request::Prompt;
/// use std::collections::HashMap;
///
/// // Simple prompt reference
/// let prompt = Prompt::new("prompt-abc123");
///
/// // Prompt with variables
/// let mut vars = HashMap::new();
/// vars.insert("name".to_string(), "Alice".to_string());
/// let prompt = Prompt::with_variables("prompt-abc123", vars);
/// ```
#[derive(Debug, Clone, Serialize)]
pub struct Prompt {
    /// The ID of the prompt template
    pub id: String,
    /// Optional variables to substitute in the template
    #[serde(skip_serializing_if = "Option::is_none")]
    pub variables: Option<HashMap<String, String>>,
}

impl Prompt {
    /// Creates a new Prompt reference without variables
    ///
    /// # Arguments
    ///
    /// * `id` - The ID of the prompt template
    ///
    /// # Returns
    ///
    /// A new Prompt instance
    pub fn new<S: AsRef<str>>(id: S) -> Self {
        Self { id: id.as_ref().to_string(), variables: None }
    }

    /// Creates a new Prompt reference with variables
    ///
    /// # Arguments
    ///
    /// * `id` - The ID of the prompt template
    /// * `variables` - Variables to substitute in the template
    ///
    /// # Returns
    ///
    /// A new Prompt instance with variables
    pub fn with_variables<S: AsRef<str>>(id: S, variables: HashMap<String, String>) -> Self {
        Self { id: id.as_ref().to_string(), variables: Some(variables) }
    }
}

/// Options for streaming responses
///
/// This struct configures how streaming responses should behave,
/// including whether to include obfuscated content.
///
/// # API Reference
///
/// Corresponds to the `stream_options` parameter in the OpenAI Responses API.
#[derive(Debug, Clone, Serialize)]
pub struct StreamOptions {
    /// Whether to include obfuscated content in streaming responses
    ///
    /// When enabled, streaming responses may include placeholder or
    /// obfuscated content that gets replaced as the real content is generated.
    pub include_obfuscation: bool,
}
/// Represents the format configuration for structured output in responses
///
/// This struct is used to specify the schema format for structured text output
/// when making requests to the OpenAI Responses API.
#[derive(Debug, Clone, Default, Serialize, new)]
pub struct Format {
    /// The schema definition that specifies the structure of the expected output
    pub format: Schema,
}

/// Represents the body of a request to the OpenAI Responses API
///
/// This struct contains all the parameters for making requests to the OpenAI Responses API.
/// It supports both plain text and structured message input, along with extensive configuration
/// options for tools, reasoning, output formatting, and response behavior.
///
/// # Required Parameters
///
/// - `model`: The ID of the model to use
/// - Either `plain_text_input` OR `messages_input` (mutually exclusive)
///
/// # API Reference
///
/// Based on the OpenAI Responses API specification:
/// <https://platform.openai.com/docs/api-reference/responses/create>
///
/// # Examples
///
/// ## Simple Text Input
///
/// ```rust
/// use openai_tools::responses::request::Body;
/// use openai_tools::common::models::ChatModel;
///
/// let body = Body {
///     model: ChatModel::Gpt4o,
///     plain_text_input: Some("What is the weather like?".to_string()),
///     ..Default::default()
/// };
/// ```
///
/// ## With Messages and Tools
///
/// ```rust
/// use openai_tools::responses::request::Body;
/// use openai_tools::common::message::Message;
/// use openai_tools::common::role::Role;
/// use openai_tools::common::models::ChatModel;
///
/// let messages = vec![
///     Message::from_string(Role::User, "Help me with coding")
/// ];
///
/// let body = Body {
///     model: ChatModel::Gpt4o,
///     messages_input: Some(messages),
///     instructions: Some("You are a helpful coding assistant".to_string()),
///     max_output_tokens: Some(1000),
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, Default, new)]
#[allow(clippy::too_many_arguments)]
pub struct Body {
    /// The model to use for generating responses
    ///
    /// Specifies which OpenAI model to use for response generation.
    ///
    /// # Required
    ///
    /// This field is required for all requests.
    ///
    /// # Examples
    ///
    /// - `ChatModel::Gpt4o` - Latest GPT-4o model
    /// - `ChatModel::Gpt4oMini` - Cost-effective option
    /// - `ChatModel::O3Mini` - Reasoning model
    pub model: ChatModel,

    /// Optional instructions to guide the model's behavior and response style
    ///
    /// Provides system-level instructions that define how the model should
    /// behave, its personality, response format, or any other behavioral guidance.
    ///
    /// # Examples
    ///
    /// - `"You are a helpful assistant that provides concise answers"`
    /// - `"Respond only with JSON formatted data"`
    /// - `"Act as a professional code reviewer"`
    pub instructions: Option<String>,

    /// Plain text input for simple text-based requests
    ///
    /// Use this for straightforward text input when you don't need the structure
    /// of messages with roles. This is mutually exclusive with `messages_input`.
    ///
    /// # Mutually Exclusive
    ///
    /// Cannot be used together with `messages_input`. Choose one based on your needs:
    /// - Use `plain_text_input` for simple, single-turn interactions
    /// - Use `messages_input` for conversation history or role-based interactions
    ///
    /// # Examples
    ///
    /// - `"What is the capital of France?"`
    /// - `"Summarize this article: [article content]"`
    /// - `"Write a haiku about programming"`
    pub plain_text_input: Option<String>,

    /// Structured message input for conversation-style interactions
    ///
    /// Use this when you need conversation history, different message roles
    /// (user, assistant, system), or structured dialogue. This is mutually
    /// exclusive with `plain_text_input`.
    ///
    /// # Mutually Exclusive
    ///
    /// Cannot be used together with `plain_text_input`.
    ///
    /// # Message Roles
    ///
    /// - `System`: Instructions for the model's behavior
    /// - `User`: User input or questions
    /// - `Assistant`: Previous model responses (for conversation history)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use openai_tools::common::message::Message;
    /// use openai_tools::common::role::Role;
    ///
    /// let messages = vec![
    ///     Message::from_string(Role::System, "You are a helpful assistant"),
    ///     Message::from_string(Role::User, "Hello!"),
    ///     Message::from_string(Role::Assistant, "Hi there! How can I help you?"),
    ///     Message::from_string(Role::User, "What's 2+2?"),
    /// ];
    /// ```
    pub messages_input: Option<Vec<Message>>,

    /// Optional tools that the model can use during response generation
    ///
    /// Provides the model with access to external tools like web search,
    /// code execution, file access, or custom functions. The model will
    /// automatically decide when and how to use these tools based on the query.
    ///
    /// # Tool Types
    ///
    /// - Web search tools for finding current information
    /// - Code interpreter for running and analyzing code
    /// - File search tools for accessing document collections
    /// - Custom function tools for specific business logic
    ///
    /// # Examples
    ///
    /// ```rust
    /// use openai_tools::common::tool::Tool;
    /// use openai_tools::common::parameters::ParameterProperty;
    ///
    /// let tools = vec![
    ///     Tool::function("search", "Search the web", Vec::<(&str, ParameterProperty)>::new(), false),
    ///     Tool::function("calculate", "Perform calculations", Vec::<(&str, ParameterProperty)>::new(), false),
    /// ];
    /// ```
    pub tools: Option<Vec<Tool>>,

    /// Optional tool choice configuration
    ///
    /// Controls how the model selects which tool to use when tools are available.
    /// Can be set to auto (let model decide), none (no tools), required (must use tools),
    /// or a specific function name to force calling that function.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use openai_tools::responses::request::{ToolChoice, ToolChoiceMode, NamedFunctionChoice};
    ///
    /// // Let the model decide
    /// let auto_choice = ToolChoice::Simple(ToolChoiceMode::Auto);
    ///
    /// // Force a specific function
    /// let function_choice = ToolChoice::Function(NamedFunctionChoice::new("get_weather"));
    /// ```
    pub tool_choice: Option<ToolChoice>,

    /// Optional prompt template reference
    ///
    /// Allows you to use pre-defined prompt templates stored in the OpenAI
    /// platform, optionally with variable substitution.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use openai_tools::responses::request::Prompt;
    ///
    /// let prompt = Prompt::new("prompt-abc123");
    /// ```
    pub prompt: Option<Prompt>,

    /// Optional prompt cache key for caching
    ///
    /// A unique key to use for prompt caching. When provided, the same
    /// prompt will be cached and reused for subsequent requests with
    /// the same cache key.
    pub prompt_cache_key: Option<String>,

    /// Optional prompt cache retention duration
    ///
    /// Controls how long cached prompts should be retained.
    /// Format: duration string (e.g., "1h", "24h", "7d")
    pub prompt_cache_retention: Option<String>,

    /// Optional structured output format specification
    ///
    /// Defines the structure and format for the model's response output.
    /// Use this when you need the response in a specific JSON schema format
    /// or other structured format for programmatic processing.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use openai_tools::common::structured_output::Schema;
    /// use openai_tools::responses::request::Format;
    ///
    /// let format = Format::new(Schema::responses_json_schema("response_schema"));
    /// ```
    pub structured_output: Option<Format>,

    /// Optional sampling temperature for controlling response randomness
    ///
    /// Controls the randomness and creativity of the model's responses.
    /// Higher values make the output more random and creative, while lower
    /// values make it more focused, deterministic, and consistent.
    ///
    /// # Range
    ///
    /// - **Range**: 0.0 to 2.0
    /// - **Default**: 1.0 (if not specified)
    /// - **Minimum**: 0.0 (most deterministic, least creative)
    /// - **Maximum**: 2.0 (most random, most creative)
    ///
    /// # Recommended Values
    ///
    /// - **0.0 - 0.3**: Highly focused and deterministic
    ///   - Best for: Factual questions, code generation, translations
    ///   - Behavior: Very consistent, predictable responses
    ///
    /// - **0.3 - 0.7**: Balanced creativity and consistency
    ///   - Best for: General conversation, explanations, analysis
    ///   - Behavior: Good balance between creativity and reliability
    ///
    /// - **0.7 - 1.2**: More creative and varied responses
    ///   - Best for: Creative writing, brainstorming, ideation
    ///   - Behavior: More diverse and interesting outputs
    ///
    /// - **1.2 - 2.0**: Highly creative and unpredictable
    ///   - Best for: Experimental creative tasks, humor, unconventional ideas
    ///   - Behavior: Very diverse but potentially less coherent
    ///
    /// # Usage Guidelines
    ///
    /// - **Start with 0.7** for most applications as a good default
    /// - **Use 0.0-0.3** when you need consistent, reliable responses
    /// - **Use 0.8-1.2** for creative tasks that still need coherence
    /// - **Avoid values above 1.5** unless you specifically want very random outputs
    ///
    /// # API Reference
    ///
    /// Corresponds to the `temperature` parameter in the OpenAI Responses API:
    /// <https://platform.openai.com/docs/api-reference/responses/create>
    ///
    /// # Examples
    ///
    /// ```rust
    /// use openai_tools::responses::request::Responses;
    ///
    /// // Deterministic, factual responses
    /// let mut client_factual = Responses::new();
    /// client_factual.temperature(0.2);
    ///
    /// // Balanced creativity and consistency
    /// let mut client_balanced = Responses::new();
    /// client_balanced.temperature(0.7);
    ///
    /// // High creativity for brainstorming
    /// let mut client_creative = Responses::new();
    /// client_creative.temperature(1.1);
    /// ```
    pub temperature: Option<f64>,

    /// Optional maximum number of tokens to generate in the response
    ///
    /// Controls the maximum length of the generated response. The actual response
    /// may be shorter if the model naturally concludes or hits other stopping conditions.
    ///
    /// # Range
    ///
    /// - Minimum: 1
    /// - Maximum: Depends on the model (typically 4096-8192 for most models)
    ///
    /// # Default Behavior
    ///
    /// If not specified, the model will use its default maximum output length.
    ///
    /// # Examples
    ///
    /// - `Some(100)` - Short responses, good for summaries or brief answers
    /// - `Some(1000)` - Medium responses, suitable for detailed explanations
    /// - `Some(4000)` - Long responses, for comprehensive analysis or long-form content
    pub max_output_tokens: Option<usize>,

    /// Optional maximum number of tool calls to make
    ///
    /// Limits how many tools the model can invoke during response generation.
    /// This helps control cost and response time when using multiple tools.
    ///
    /// # Range
    ///
    /// - Minimum: 0 (no tool calls allowed)
    /// - Maximum: Implementation-dependent
    ///
    /// # Use Cases
    ///
    /// - Set to `Some(1)` for single tool usage
    /// - Set to `Some(0)` to disable tool usage entirely
    /// - Leave as `None` for unlimited tool usage (subject to other constraints)
    pub max_tool_calls: Option<usize>,

    /// Optional metadata to include with the request
    ///
    /// Arbitrary key-value pairs that can be attached to the request for
    /// tracking, logging, or passing additional context that doesn't affect
    /// the model's behavior.
    ///
    /// # Common Use Cases
    ///
    /// - Request tracking: `{"request_id": "req_123", "user_id": "user_456"}`
    /// - A/B testing: `{"experiment": "variant_a", "test_group": "control"}`
    /// - Analytics: `{"session_id": "sess_789", "feature": "chat"}`
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::collections::HashMap;
    /// use serde_json::Value;
    ///
    /// let mut metadata = HashMap::new();
    /// metadata.insert("user_id".to_string(), Value::String("user123".to_string()));
    /// metadata.insert("session_id".to_string(), Value::String("sess456".to_string()));
    /// metadata.insert("priority".to_string(), Value::Number(serde_json::Number::from(1)));
    /// ```
    pub metadata: Option<HashMap<String, serde_json::Value>>,

    /// Optional flag to enable parallel tool calls
    ///
    /// When enabled, the model can make multiple tool calls simultaneously
    /// rather than sequentially. This can significantly improve response time
    /// when multiple independent tools need to be used.
    ///
    /// # Default
    ///
    /// If not specified, defaults to the model's default behavior (usually `true`).
    ///
    /// # When to Use
    ///
    /// - `Some(true)`: Enable when tools are independent and can run in parallel
    /// - `Some(false)`: Disable when tools have dependencies or order matters
    ///
    /// # Examples
    ///
    /// - Weather + Stock prices: Can run in parallel (`true`)
    /// - File read + File analysis: Should run sequentially (`false`)
    pub parallel_tool_calls: Option<bool>,

    /// Optional fields to include in the output
    ///
    /// Specifies additional metadata and information to include in the response
    /// beyond the main generated content. This can include tool call details,
    /// reasoning traces, log probabilities, and more.
    ///
    /// # Available Inclusions
    ///
    /// - Web search call sources and results
    /// - Code interpreter execution outputs
    /// - Image URLs from various sources
    /// - Log probabilities for generated tokens
    /// - Reasoning traces and encrypted content
    ///
    /// # Examples
    ///
    /// ```rust
    /// use openai_tools::responses::request::Include;
    ///
    /// let includes = vec![
    ///     Include::WebSearchCall,
    ///     Include::LogprobsInOutput,
    ///     Include::ReasoningEncryptedContent,
    /// ];
    /// ```
    pub include: Option<Vec<Include>>,

    /// Optional flag to enable background processing
    ///
    /// When enabled, allows the request to be processed in the background,
    /// potentially improving throughput for non-urgent requests.
    ///
    /// # Use Cases
    ///
    /// - `Some(true)`: Batch processing, non-interactive requests
    /// - `Some(false)` or `None`: Real-time, interactive requests
    ///
    /// # Trade-offs
    ///
    /// - Background processing may have lower latency guarantees
    /// - May be more cost-effective for bulk operations
    /// - May have different rate limiting behavior
    pub background: Option<bool>,

    /// Optional conversation ID for tracking
    ///
    /// Identifier for grouping related requests as part of the same conversation
    /// or session. This helps with context management and analytics.
    ///
    /// # Format
    ///
    /// Typically a UUID or other unique identifier string.
    ///
    /// # Examples
    ///
    /// - `Some("conv_123e4567-e89b-12d3-a456-426614174000".to_string())`
    /// - `Some("user123_session456".to_string())`
    pub conversation: Option<String>,

    /// Optional ID of the previous response for context
    ///
    /// References a previous response in the same conversation to maintain
    /// context and enable features like response chaining or follow-up handling.
    ///
    /// # Use Cases
    ///
    /// - Multi-turn conversations with context preservation
    /// - Follow-up questions or clarifications
    /// - Response refinement or iteration
    ///
    /// # Examples
    ///
    /// - `Some("resp_abc123def456".to_string())`
    pub previous_response_id: Option<String>,

    /// Optional reasoning configuration
    ///
    /// Controls how the model approaches complex reasoning tasks, including
    /// the effort level and format of reasoning explanations.
    ///
    /// # Use Cases
    ///
    /// - Complex problem-solving requiring deep analysis
    /// - Mathematical or logical reasoning tasks
    /// - When you need insight into the model's reasoning process
    ///
    /// # Examples
    ///
    /// ```rust
    /// use openai_tools::responses::request::{Reasoning, ReasoningEffort, ReasoningSummary};
    ///
    /// let reasoning = Reasoning {
    ///     effort: Some(ReasoningEffort::High),
    ///     summary: Some(ReasoningSummary::Detailed),
    /// };
    /// ```
    pub reasoning: Option<Reasoning>,

    /// Optional text output configuration
    ///
    /// Controls the characteristics of the generated text output,
    /// such as verbosity level. Available on GPT-5.2 and newer models.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use openai_tools::responses::request::{TextConfig, TextVerbosity};
    ///
    /// let text = TextConfig {
    ///     verbosity: Some(TextVerbosity::High),
    /// };
    /// ```
    pub text: Option<TextConfig>,

    /// Optional safety identifier
    ///
    /// Identifier for safety and content filtering configurations.
    /// Used to specify which safety policies should be applied to the request.
    ///
    /// # Examples
    ///
    /// - `Some("strict".to_string())` - Apply strict content filtering
    /// - `Some("moderate".to_string())` - Apply moderate content filtering
    /// - `Some("permissive".to_string())` - Apply permissive content filtering
    pub safety_identifier: Option<String>,

    /// Optional service tier specification
    ///
    /// Specifies the service tier for the request, which may affect
    /// processing priority, rate limits, and pricing.
    ///
    /// # Common Values
    ///
    /// - `Some("default".to_string())` - Standard service tier
    /// - `Some("scale".to_string())` - High-throughput tier
    /// - `Some("premium".to_string())` - Premium service tier with enhanced features
    pub service_tier: Option<String>,

    /// Optional flag to store the conversation
    ///
    /// When enabled, the conversation may be stored for future reference,
    /// training, or analytics purposes (subject to privacy policies).
    ///
    /// # Privacy Considerations
    ///
    /// - `Some(true)`: Allow storage (check privacy policies)
    /// - `Some(false)`: Explicitly opt-out of storage
    /// - `None`: Use default storage policy
    pub store: Option<bool>,

    /// Optional flag to enable streaming responses
    ///
    /// When enabled, the response will be streamed back in chunks as it's
    /// generated, allowing for real-time display of partial results.
    ///
    /// # Use Cases
    ///
    /// - `Some(true)`: Real-time chat interfaces, live text generation
    /// - `Some(false)`: Batch processing, when you need the complete response
    ///
    /// # Considerations
    ///
    /// - Streaming responses require different handling in client code
    /// - May affect some response features or formatting options
    pub stream: Option<bool>,

    /// Optional streaming configuration options
    ///
    /// Additional options for controlling streaming response behavior,
    /// such as whether to include obfuscated placeholder content.
    ///
    /// # Only Relevant When Streaming
    ///
    /// This field is only meaningful when `stream` is `Some(true)`.
    pub stream_options: Option<StreamOptions>,

    /// Optional number of top log probabilities to include
    ///
    /// Specifies how many of the most likely alternative tokens to include
    /// with their log probabilities for each generated token.
    ///
    /// # Range
    ///
    /// - Minimum: 0 (no log probabilities)
    /// - Maximum: Model-dependent (typically 5-20)
    ///
    /// # Use Cases
    ///
    /// - Model analysis and debugging
    /// - Confidence estimation
    /// - Alternative response exploration
    ///
    /// # Examples
    ///
    /// - `Some(1)` - Include the top alternative for each token
    /// - `Some(5)` - Include top 5 alternatives for detailed analysis
    pub top_logprobs: Option<usize>,

    /// Optional nucleus sampling parameter
    ///
    /// Controls the randomness of the model's responses by limiting the
    /// cumulative probability of considered tokens.
    ///
    /// # Range
    ///
    /// - 0.0 to 1.0
    /// - Lower values (e.g., 0.1) make responses more focused and deterministic
    /// - Higher values (e.g., 0.9) make responses more diverse and creative
    ///
    /// # Default
    ///
    /// If not specified, uses the model's default value (typically around 1.0).
    ///
    /// # Examples
    ///
    /// - `Some(0.1)` - Very focused, deterministic responses
    /// - `Some(0.7)` - Balanced creativity and focus
    /// - `Some(0.95)` - High creativity and diversity
    pub top_p: Option<f64>,

    /// Optional truncation behavior configuration
    ///
    /// Controls how the system handles inputs that exceed the maximum
    /// context length supported by the model.
    ///
    /// # Options
    ///
    /// - `Some(Truncation::Auto)` - Automatically truncate long inputs
    /// - `Some(Truncation::Disabled)` - Return error for long inputs
    /// - `None` - Use system default behavior
    ///
    /// # Use Cases
    ///
    /// - `Auto`: When you want to handle long documents gracefully
    /// - `Disabled`: When you need to ensure complete input processing
    pub truncation: Option<Truncation>,
}

impl Serialize for Body {
    /// Custom serialization implementation for the request body
    ///
    /// This implementation handles the conversion of either plain text input
    /// or messages input into the appropriate "input" field format required
    /// by the OpenAI API. It also conditionally includes optional fields
    /// like tools and text formatting.
    ///
    /// # Errors
    ///
    /// Returns a serialization error if neither plain_text_input nor
    /// messages_input is set, as one of them is required.
    fn serialize<S>(&self, serializer: S) -> anyhow::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut state = serializer.serialize_struct("ResponsesBody", 4)?;
        state.serialize_field("model", &self.model)?;

        // Set input
        if self.plain_text_input.is_some() {
            state.serialize_field("input", &self.plain_text_input.clone().unwrap())?;
        } else if self.messages_input.is_some() {
            state.serialize_field("input", &self.messages_input.clone().unwrap())?;
        } else {
            return Err(serde::ser::Error::custom("Either plain_text_input or messages_input must be set."));
        };

        // Optional fields
        if self.temperature.is_some() {
            state.serialize_field("temperature", &self.temperature)?;
        }
        if self.instructions.is_some() {
            state.serialize_field("instructions", &self.instructions)?;
        }
        if self.tools.is_some() {
            state.serialize_field("tools", &self.tools)?;
        }
        if self.tool_choice.is_some() {
            state.serialize_field("tool_choice", &self.tool_choice)?;
        }
        if self.prompt.is_some() {
            state.serialize_field("prompt", &self.prompt)?;
        }
        if self.prompt_cache_key.is_some() {
            state.serialize_field("prompt_cache_key", &self.prompt_cache_key)?;
        }
        if self.prompt_cache_retention.is_some() {
            state.serialize_field("prompt_cache_retention", &self.prompt_cache_retention)?;
        }
        if self.structured_output.is_some() {
            state.serialize_field("text", &self.structured_output)?;
        }
        if self.max_output_tokens.is_some() {
            state.serialize_field("max_output_tokens", &self.max_output_tokens)?;
        }
        if self.max_tool_calls.is_some() {
            state.serialize_field("max_tool_calls", &self.max_tool_calls)?;
        }
        if self.metadata.is_some() {
            state.serialize_field("metadata", &self.metadata)?;
        }
        if self.parallel_tool_calls.is_some() {
            state.serialize_field("parallel_tool_calls", &self.parallel_tool_calls)?;
        }
        if self.include.is_some() {
            state.serialize_field("include", &self.include)?;
        }
        if self.background.is_some() {
            state.serialize_field("background", &self.background)?;
        }
        if self.conversation.is_some() {
            state.serialize_field("conversation", &self.conversation)?;
        }
        if self.previous_response_id.is_some() {
            state.serialize_field("previous_response_id", &self.previous_response_id)?;
        }
        if self.reasoning.is_some() {
            state.serialize_field("reasoning", &self.reasoning)?;
        }
        if self.text.is_some() {
            state.serialize_field("text", &self.text)?;
        }
        if self.safety_identifier.is_some() {
            state.serialize_field("safety_identifier", &self.safety_identifier)?;
        }
        if self.service_tier.is_some() {
            state.serialize_field("service_tier", &self.service_tier)?;
        }
        if self.store.is_some() {
            state.serialize_field("store", &self.store)?;
        }
        if self.stream.is_some() {
            state.serialize_field("stream", &self.stream)?;
        }
        if self.stream_options.is_some() {
            state.serialize_field("stream_options", &self.stream_options)?;
        }
        if self.top_logprobs.is_some() {
            state.serialize_field("top_logprobs", &self.top_logprobs)?;
        }
        if self.top_p.is_some() {
            state.serialize_field("top_p", &self.top_p)?;
        }
        if self.truncation.is_some() {
            state.serialize_field("truncation", &self.truncation)?;
        }
        state.end()
    }
}

/// Default API path for Responses
const RESPONSES_PATH: &str = "responses";

/// Client for making requests to the OpenAI Responses API
///
/// This struct provides a convenient interface for building and executing requests
/// to the OpenAI Responses API and Azure OpenAI API. It handles authentication,
/// request formatting, and response parsing automatically.
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
/// use openai_tools::responses::request::Responses;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let mut client = Responses::new();
/// let response = client
///     .model_id("gpt-4")
///     .instructions("You are a helpful assistant.")
///     .str_message("Hello, how are you?")
///     .complete()
///     .await?;
/// # Ok(())
/// # }
/// ```
///
/// ## Azure OpenAI
///
/// ```rust,no_run
/// use openai_tools::responses::request::Responses;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let mut client = Responses::azure()?;
/// let response = client
///     .str_message("Hello!")
///     .complete()
///     .await?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct Responses {
    /// Authentication provider (OpenAI or Azure)
    auth: AuthProvider,
    /// The User-Agent string to include in requests
    user_agent: String,
    /// The request body containing all parameters for the API call
    pub request_body: Body,
    /// Optional request timeout duration
    timeout: Option<Duration>,
}

impl Default for Responses {
    fn default() -> Self {
        Self::new()
    }
}

impl Responses {
    /// Creates a new instance of the Responses client for OpenAI API
    ///
    /// This method initializes a new client by loading the OpenAI API key from
    /// the `OPENAI_API_KEY` environment variable. Make sure to set this environment
    /// variable before calling this method.
    ///
    /// # Panics
    ///
    /// Panics if the `OPENAI_API_KEY` environment variable is not set.
    pub fn new() -> Self {
        let auth = AuthProvider::openai_from_env().map_err(|e| OpenAIToolError::Error(format!("Failed to load OpenAI auth: {}", e))).unwrap();
        Self { auth, user_agent: "".into(), request_body: Body::default(), timeout: None }
    }

    /// Creates a new instance of the Responses client with a custom endpoint
    #[deprecated(since = "0.3.0", note = "Use `with_auth()` with custom OpenAIAuth for custom endpoints")]
    pub fn from_endpoint<T: AsRef<str>>(endpoint: T) -> Self {
        let auth = AuthProvider::openai_from_env().map_err(|e| OpenAIToolError::Error(format!("Failed to load OpenAI auth: {}", e))).unwrap();
        // Extract the path from the endpoint and use it
        let mut responses = Self { auth, user_agent: "".into(), request_body: Body::default(), timeout: None };
        responses.base_url(endpoint.as_ref().trim_end_matches("/responses"));
        responses
    }

    /// Creates a new Responses client with a specified model.
    ///
    /// This is the recommended constructor as it enables parameter validation
    /// at setter time. When you set parameters like `temperature()` or `top_p()`,
    /// the model's parameter support is checked and warnings are logged for
    /// unsupported values.
    ///
    /// # Arguments
    ///
    /// * `model` - The model to use for response generation
    ///
    /// # Panics
    ///
    /// Panics if the `OPENAI_API_KEY` environment variable is not set.
    ///
    /// # Returns
    ///
    /// A new Responses instance with the specified model
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::responses::request::Responses;
    /// use openai_tools::common::models::ChatModel;
    ///
    /// // Recommended: specify model at creation time
    /// let mut responses = Responses::with_model(ChatModel::Gpt4oMini);
    ///
    /// // For reasoning models, unsupported parameters are validated at setter time
    /// let mut reasoning_responses = Responses::with_model(ChatModel::O3Mini);
    /// reasoning_responses.temperature(0.5); // Warning logged, value ignored
    /// ```
    pub fn with_model(model: ChatModel) -> Self {
        let auth = AuthProvider::openai_from_env().map_err(|e| OpenAIToolError::Error(format!("Failed to load OpenAI auth: {}", e))).unwrap();
        Self { auth, user_agent: "".into(), request_body: Body { model, ..Default::default() }, timeout: None }
    }

    /// Creates a new Responses client with a custom authentication provider
    ///
    /// Use this to explicitly configure OpenAI or Azure authentication.
    ///
    /// # Arguments
    ///
    /// * `auth` - The authentication provider
    ///
    /// # Returns
    ///
    /// A new Responses instance with the specified auth provider
    ///
    /// # Example
    ///
    /// ```rust
    /// use openai_tools::responses::request::Responses;
    /// use openai_tools::common::auth::{AuthProvider, AzureAuth};
    ///
    /// // Explicit Azure configuration with complete base URL
    /// let auth = AuthProvider::Azure(
    ///     AzureAuth::new(
    ///         "api-key",
    ///         "https://my-resource.openai.azure.com/openai/deployments/gpt-4o?api-version=2024-08-01-preview"
    ///     )
    /// );
    /// let mut responses = Responses::with_auth(auth);
    /// ```
    pub fn with_auth(auth: AuthProvider) -> Self {
        Self { auth, user_agent: "".into(), request_body: Body::default(), timeout: None }
    }

    /// Creates a new Responses client for Azure OpenAI API
    ///
    /// Loads configuration from Azure-specific environment variables.
    ///
    /// # Returns
    ///
    /// `Result<Responses>` - Configured for Azure or error if env vars missing
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
    /// use openai_tools::responses::request::Responses;
    ///
    /// // With environment variables:
    /// // AZURE_OPENAI_API_KEY=xxx
    /// // AZURE_OPENAI_BASE_URL=https://my-resource.openai.azure.com/openai/deployments/gpt-4o/responses?api-version=2024-08-01-preview
    /// let mut responses = Responses::azure()?;
    /// # Ok::<(), openai_tools::common::errors::OpenAIToolError>(())
    /// ```
    pub fn azure() -> Result<Self> {
        let auth = AuthProvider::azure_from_env()?;
        Ok(Self { auth, user_agent: "".into(), request_body: Body::default(), timeout: None })
    }

    /// Creates a new Responses client by auto-detecting the provider
    ///
    /// Tries Azure first (if AZURE_OPENAI_API_KEY is set), then falls back to OpenAI.
    ///
    /// # Returns
    ///
    /// `Result<Responses>` - Auto-configured client or error
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::responses::request::Responses;
    ///
    /// // Uses Azure if AZURE_OPENAI_API_KEY is set, otherwise OpenAI
    /// let mut responses = Responses::detect_provider()?;
    /// # Ok::<(), openai_tools::common::errors::OpenAIToolError>(())
    /// ```
    pub fn detect_provider() -> Result<Self> {
        let auth = AuthProvider::from_env()?;
        Ok(Self { auth, user_agent: "".into(), request_body: Body::default(), timeout: None })
    }

    /// Creates a new Responses instance with URL-based provider detection
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
        Self { auth, user_agent: "".into(), request_body: Body::default(), timeout: None }
    }

    /// Creates a new Responses instance from URL using environment variables
    ///
    /// Analyzes the URL pattern to determine the provider, then loads
    /// credentials from the appropriate environment variables.
    pub fn from_url<S: Into<String>>(url: S) -> Result<Self> {
        let auth = AuthProvider::from_url(url)?;
        Ok(Self { auth, user_agent: "".into(), request_body: Body::default(), timeout: None })
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
    /// * `url` - The base URL (e.g., `https://my-proxy.example.com/v1`)
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
    /// use openai_tools::responses::request::Responses;
    ///
    /// let mut responses = Responses::new();
    /// responses.base_url("https://my-proxy.example.com/v1");
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

    /// Sets the model for the request.
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
    /// use openai_tools::responses::request::Responses;
    /// use openai_tools::common::models::ChatModel;
    ///
    /// let mut responses = Responses::new();
    /// responses.model(ChatModel::Gpt4oMini);
    /// ```
    pub fn model(&mut self, model: ChatModel) -> &mut Self {
        self.request_body.model = model;
        self
    }

    /// Sets the model using a string ID (for backward compatibility).
    ///
    /// Prefer using [`model`](Self::model) with `ChatModel` enum for type safety.
    ///
    /// # Arguments
    ///
    /// * `model_id` - The ID of the model to use (e.g., "gpt-4o-mini")
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
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
    /// use openai_tools::responses::request::Responses;
    ///
    /// let mut responses = Responses::new();
    /// responses.model_id("gpt-4o")
    ///     .timeout(Duration::from_secs(30));
    /// ```
    pub fn timeout(&mut self, timeout: Duration) -> &mut Self {
        self.timeout = Some(timeout);
        self
    }

    /// Sets the User-Agent string for the request
    ///
    /// # Arguments
    ///
    /// * `user_agent` - The User-Agent string to include in the request headers
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    pub fn user_agent<T: AsRef<str>>(&mut self, user_agent: T) -> &mut Self {
        self.user_agent = user_agent.as_ref().to_string();
        self
    }

    /// Sets instructions to guide the model's behavior
    ///
    /// # Arguments
    ///
    /// * `instructions` - Instructions that define how the model should behave
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    pub fn instructions<T: AsRef<str>>(&mut self, instructions: T) -> &mut Self {
        self.request_body.instructions = Some(instructions.as_ref().to_string());
        self
    }

    /// Sets plain text input for simple text-based requests
    ///
    /// This method is mutually exclusive with `messages()`. Use this for simple
    /// text-based interactions where you don't need conversation history.
    ///
    /// # Arguments
    ///
    /// * `input` - The plain text input to send to the model
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    pub fn str_message<T: AsRef<str>>(&mut self, input: T) -> &mut Self {
        self.request_body.plain_text_input = Some(input.as_ref().to_string());
        self
    }

    /// Sets structured message input for conversation-style interactions
    ///
    /// This method is mutually exclusive with `plain_text_input()`. Use this
    /// for complex conversations with message history and different roles.
    ///
    /// # Arguments
    ///
    /// * `messages` - A vector of messages representing the conversation history
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    pub fn messages(&mut self, messages: Vec<Message>) -> &mut Self {
        self.request_body.messages_input = Some(messages);
        self
    }

    /// Sets tools that the model can use during response generation
    ///
    /// # Arguments
    ///
    /// * `tools` - A vector of tools available to the model
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    pub fn tools(&mut self, tools: Vec<Tool>) -> &mut Self {
        self.request_body.tools = Some(tools);
        self
    }

    /// Sets the tool choice configuration
    ///
    /// Controls how the model selects which tool to use when tools are available.
    /// Can be set to auto (let model decide), none (no tools), required (must use tools),
    /// or a specific function name to force calling that function.
    ///
    /// # Arguments
    ///
    /// * `tool_choice` - The tool choice configuration
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    ///
    /// # Examples
    ///
    /// ```rust
    /// use openai_tools::responses::request::{Responses, ToolChoice, ToolChoiceMode, NamedFunctionChoice};
    ///
    /// let mut client = Responses::new();
    ///
    /// // Let the model decide
    /// client.tool_choice(ToolChoice::Simple(ToolChoiceMode::Auto));
    ///
    /// // Force a specific function
    /// client.tool_choice(ToolChoice::Function(NamedFunctionChoice::new("get_weather")));
    /// ```
    pub fn tool_choice(&mut self, tool_choice: ToolChoice) -> &mut Self {
        self.request_body.tool_choice = Some(tool_choice);
        self
    }

    /// Sets a prompt template reference
    ///
    /// Allows you to use pre-defined prompt templates stored in the OpenAI
    /// platform, optionally with variable substitution.
    ///
    /// # Arguments
    ///
    /// * `prompt` - The prompt template reference
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    ///
    /// # Examples
    ///
    /// ```rust
    /// use openai_tools::responses::request::{Responses, Prompt};
    /// use std::collections::HashMap;
    ///
    /// let mut client = Responses::new();
    ///
    /// // Simple prompt reference
    /// client.prompt(Prompt::new("prompt-abc123"));
    ///
    /// // Prompt with variables
    /// let mut vars = HashMap::new();
    /// vars.insert("name".to_string(), "Alice".to_string());
    /// client.prompt(Prompt::with_variables("prompt-abc123", vars));
    /// ```
    pub fn prompt(&mut self, prompt: Prompt) -> &mut Self {
        self.request_body.prompt = Some(prompt);
        self
    }

    /// Sets the prompt cache key for caching
    ///
    /// A unique key to use for prompt caching. When provided, the same
    /// prompt will be cached and reused for subsequent requests with
    /// the same cache key.
    ///
    /// # Arguments
    ///
    /// * `key` - The cache key to use
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    ///
    /// # Examples
    ///
    /// ```rust
    /// use openai_tools::responses::request::Responses;
    ///
    /// let mut client = Responses::new();
    /// client.prompt_cache_key("my-cache-key");
    /// ```
    pub fn prompt_cache_key<T: AsRef<str>>(&mut self, key: T) -> &mut Self {
        self.request_body.prompt_cache_key = Some(key.as_ref().to_string());
        self
    }

    /// Sets the prompt cache retention duration
    ///
    /// Controls how long cached prompts should be retained.
    ///
    /// # Arguments
    ///
    /// * `retention` - The retention duration string (e.g., "1h", "24h", "7d")
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    ///
    /// # Examples
    ///
    /// ```rust
    /// use openai_tools::responses::request::Responses;
    ///
    /// let mut client = Responses::new();
    /// client.prompt_cache_retention("24h");
    /// ```
    pub fn prompt_cache_retention<T: AsRef<str>>(&mut self, retention: T) -> &mut Self {
        self.request_body.prompt_cache_retention = Some(retention.as_ref().to_string());
        self
    }

    /// Sets structured output format specification
    ///
    /// This allows you to specify the exact format and structure of the
    /// model's response output.
    ///
    /// # Arguments
    ///
    /// * `text_format` - The schema defining the expected output structure
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    pub fn structured_output(&mut self, text_format: Schema) -> &mut Self {
        self.request_body.structured_output = Option::from(Format::new(text_format));
        self
    }

    /// Sets the sampling temperature for controlling response randomness
    ///
    /// Controls the randomness and creativity of the model's responses.
    /// Higher values make the output more random and creative, while lower
    /// values make it more focused and deterministic.
    ///
    /// # Arguments
    ///
    /// * `temperature` - The temperature value (0.0 to 2.0)
    ///   - 0.0: Most deterministic and focused responses
    ///   - 1.0: Default balanced behavior
    ///   - 2.0: Most random and creative responses
    ///
    /// **Note:** Reasoning models (GPT-6, GPT-5, o-series) only support temperature=1.0.
    /// For these models, other values will be ignored with a warning.
    ///
    /// # Panics
    ///
    /// This method will panic if the temperature value is outside the valid
    /// range of 0.0 to 2.0, as this would result in an API error.
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    ///
    /// # Examples
    ///
    /// ```rust
    /// use openai_tools::responses::request::Responses;
    ///
    /// // Deterministic responses for factual queries
    /// let mut client = Responses::new();
    /// client.temperature(0.2);
    ///
    /// // Creative responses for brainstorming
    /// let mut client = Responses::new();
    /// client.temperature(1.1);
    /// ```
    pub fn temperature(&mut self, temperature: f64) -> &mut Self {
        assert!((0.0..=2.0).contains(&temperature), "Temperature must be between 0.0 and 2.0, got {}", temperature);
        let support = self.request_body.model.parameter_support();
        match support.temperature {
            ParameterRestriction::FixedValue(fixed) => {
                if (temperature - fixed).abs() > f64::EPSILON {
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

    /// Sets the maximum number of tokens to generate in the response
    ///
    /// Controls the maximum length of the generated response. The actual response
    /// may be shorter if the model naturally concludes or hits other stopping conditions.
    ///
    /// # Arguments
    ///
    /// * `max_tokens` - Maximum number of tokens to generate (minimum: 1)
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    ///
    /// # Examples
    ///
    /// ```rust
    /// use openai_tools::responses::request::Responses;
    ///
    /// let mut client = Responses::new();
    /// client.max_output_tokens(100);  // Limit response to 100 tokens
    /// ```
    pub fn max_output_tokens(&mut self, max_tokens: usize) -> &mut Self {
        self.request_body.max_output_tokens = Some(max_tokens);
        self
    }

    /// Sets the maximum number of tool calls allowed during response generation
    ///
    /// Limits how many tools the model can invoke during response generation.
    /// This helps control cost and response time when using multiple tools.
    ///
    /// # Arguments
    ///
    /// * `max_tokens` - Maximum number of tool calls allowed (0 = no tool calls)
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    ///
    /// # Examples
    ///
    /// ```rust
    /// use openai_tools::responses::request::Responses;
    ///
    /// let mut client = Responses::new();
    /// client.max_tool_calls(3);  // Allow up to 3 tool calls
    /// client.max_tool_calls(0);  // Disable tool usage
    /// ```
    pub fn max_tool_calls(&mut self, max_tokens: usize) -> &mut Self {
        self.request_body.max_tool_calls = Some(max_tokens);
        self
    }

    /// Adds or updates a metadata key-value pair for the request
    ///
    /// Metadata provides arbitrary key-value pairs that can be attached to the request
    /// for tracking, logging, or passing additional context that doesn't affect
    /// the model's behavior.
    ///
    /// # Arguments
    ///
    /// * `key` - The metadata key (string identifier)
    /// * `value` - The metadata value (can be string, number, boolean, etc.)
    ///
    /// # Behavior
    ///
    /// - If the key already exists, the old value is replaced with the new one
    /// - If metadata doesn't exist yet, a new metadata map is created
    /// - Values are stored as `serde_json::Value` for flexibility
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    ///
    /// # Examples
    ///
    /// ```rust
    /// use openai_tools::responses::request::Responses;
    /// use serde_json::Value;
    ///
    /// let mut client = Responses::new();
    /// client.metadata("user_id".to_string(), Value::String("user123".to_string()));
    /// client.metadata("priority".to_string(), Value::Number(serde_json::Number::from(1)));
    /// client.metadata("debug".to_string(), Value::Bool(true));
    /// ```
    pub fn metadata(&mut self, key: String, value: serde_json::Value) -> &mut Self {
        if self.request_body.metadata.is_none() {
            self.request_body.metadata = Some(HashMap::new());
        }
        if self.request_body.metadata.as_ref().unwrap().keys().any(|k| k == &key) {
            self.request_body.metadata.as_mut().unwrap().remove(&key);
        }
        self.request_body.metadata.as_mut().unwrap().insert(key, value);
        self
    }

    /// Enables or disables parallel tool calls
    ///
    /// When enabled, the model can make multiple tool calls simultaneously
    /// rather than sequentially. This can significantly improve response time
    /// when multiple independent tools need to be used.
    ///
    /// # Arguments
    ///
    /// * `enable` - Whether to enable parallel tool calls
    ///   - `true`: Tools can be called in parallel (faster for independent tools)
    ///   - `false`: Tools are called sequentially (better for dependent operations)
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    ///
    /// # When to Use
    ///
    /// - **Enable (true)**: When tools are independent (e.g., weather + stock prices)
    /// - **Disable (false)**: When tools have dependencies (e.g., read file → analyze content)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use openai_tools::responses::request::Responses;
    ///
    /// let mut client = Responses::new();
    /// client.parallel_tool_calls(true);   // Enable parallel execution
    /// client.parallel_tool_calls(false);  // Force sequential execution
    /// ```
    pub fn parallel_tool_calls(&mut self, enable: bool) -> &mut Self {
        self.request_body.parallel_tool_calls = Some(enable);
        self
    }

    /// Specifies additional data to include in the response output
    ///
    /// Defines various types of additional information that can be included
    /// in the API response output, such as web search results, code interpreter
    /// outputs, image URLs, log probabilities, and reasoning traces.
    ///
    /// # Arguments
    ///
    /// * `includes` - A vector of `Include` enum values specifying what to include
    ///
    /// # Available Inclusions
    ///
    /// - `Include::WebSearchCall` - Web search results and sources
    /// - `Include::CodeInterpreterCall` - Code execution outputs
    /// - `Include::FileSearchCall` - File search operation results
    /// - `Include::LogprobsInOutput` - Token log probabilities
    /// - `Include::ReasoningEncryptedContent` - Reasoning process traces
    /// - `Include::ImageUrlInInputMessages` - Image URLs from input
    /// - `Include::ImageUrlInComputerCallOutput` - Computer interaction screenshots
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    ///
    /// # Examples
    ///
    /// ```rust
    /// use openai_tools::responses::request::{Responses, Include};
    ///
    /// let mut client = Responses::new();
    /// client.include(vec![
    ///     Include::WebSearchCall,
    ///     Include::LogprobsInOutput,
    ///     Include::ReasoningEncryptedContent,
    /// ]);
    /// ```
    pub fn include(&mut self, includes: Vec<Include>) -> &mut Self {
        self.request_body.include = Some(includes);
        self
    }

    /// Enables or disables background processing for the request
    ///
    /// When enabled, allows the request to be processed in the background,
    /// potentially improving throughput for non-urgent requests at the cost
    /// of potentially higher latency.
    ///
    /// # Arguments
    ///
    /// * `enable` - Whether to enable background processing
    ///   - `true`: Process in background (lower priority, potentially longer latency)
    ///   - `false`: Process with standard priority (default behavior)
    ///
    /// # Trade-offs
    ///
    /// - **Background processing**: Better for batch operations, non-interactive requests
    /// - **Standard processing**: Better for real-time, interactive applications
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    ///
    /// # Examples
    ///
    /// ```rust
    /// use openai_tools::responses::request::Responses;
    ///
    /// let mut client = Responses::new();
    /// client.background(true);   // Enable background processing
    /// client.background(false);  // Use standard processing
    /// ```
    pub fn background(&mut self, enable: bool) -> &mut Self {
        self.request_body.background = Some(enable);
        self
    }

    /// Sets the conversation ID for grouping related requests
    ///
    /// Identifier for grouping related requests as part of the same conversation
    /// or session. This helps with context management, analytics, and conversation
    /// tracking across multiple API calls.
    ///
    /// # Arguments
    ///
    /// * `conversation_id` - The conversation identifier
    ///   - Must start with "conv-" prefix according to API requirements
    ///   - Should be a unique identifier (UUID recommended)
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    ///
    /// # Format Requirements
    ///
    /// The conversation ID must follow the format: `conv-{identifier}`
    ///
    /// # Examples
    ///
    /// ```rust
    /// use openai_tools::responses::request::Responses;
    ///
    /// let mut client = Responses::new();
    /// client.conversation("conv-123e4567-e89b-12d3-a456-426614174000");
    /// client.conversation("conv-user123-session456");
    /// ```
    pub fn conversation<T: AsRef<str>>(&mut self, conversation_id: T) -> &mut Self {
        self.request_body.conversation = Some(conversation_id.as_ref().to_string());
        self
    }

    /// Sets the ID of the previous response for context continuation
    ///
    /// References a previous response in the same conversation to maintain
    /// context and enable features like response chaining, follow-up handling,
    /// or response refinement.
    ///
    /// # Arguments
    ///
    /// * `response_id` - The ID of the previous response to reference
    ///
    /// # Use Cases
    ///
    /// - **Multi-turn conversations**: Maintaining context across multiple exchanges
    /// - **Follow-up questions**: Building on previous responses
    /// - **Response refinement**: Iterating on or clarifying previous answers
    /// - **Context chaining**: Creating connected sequences of responses
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    ///
    /// # Examples
    ///
    /// ```rust
    /// use openai_tools::responses::request::Responses;
    ///
    /// let mut client = Responses::new();
    /// client.previous_response_id("resp_abc123def456");
    /// client.previous_response_id("response-uuid-here");
    /// ```
    pub fn previous_response_id<T: AsRef<str>>(&mut self, response_id: T) -> &mut Self {
        self.request_body.previous_response_id = Some(response_id.as_ref().to_string());
        self
    }

    /// Configures reasoning behavior for complex problem-solving
    ///
    /// Controls how the model approaches complex reasoning tasks, including
    /// the computational effort level and format of reasoning explanations.
    /// This is particularly useful for mathematical, logical, or analytical tasks.
    ///
    /// # Arguments
    ///
    /// * `effort` - The level of reasoning effort to apply:
    ///   - `ReasoningEffort::Minimal` - Fastest, for simple queries
    ///   - `ReasoningEffort::Low` - Balanced, for moderate complexity
    ///   - `ReasoningEffort::Medium` - Thorough, for complex queries
    ///   - `ReasoningEffort::High` - Maximum analysis, for very complex problems
    ///
    /// * `summary` - The format for reasoning explanations:
    ///   - `ReasoningSummary::Auto` - Let the model choose the format
    ///   - `ReasoningSummary::Concise` - Brief, focused explanations
    ///   - `ReasoningSummary::Detailed` - Comprehensive, step-by-step explanations
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    ///
    /// # Use Cases
    ///
    /// - Mathematical problem-solving with step-by-step explanations
    /// - Complex logical reasoning tasks
    /// - Analysis requiring deep consideration
    /// - Tasks where understanding the reasoning process is important
    ///
    /// # Examples
    ///
    /// ```rust
    /// use openai_tools::responses::request::{Responses, ReasoningEffort, ReasoningSummary};
    ///
    /// let mut client = Responses::new();
    ///
    /// // High effort with detailed explanations for complex problems
    /// client.reasoning(ReasoningEffort::High, ReasoningSummary::Detailed);
    ///
    /// // Medium effort with concise explanations for balanced approach
    /// client.reasoning(ReasoningEffort::Medium, ReasoningSummary::Concise);
    /// ```
    pub fn reasoning(&mut self, effort: ReasoningEffort, summary: ReasoningSummary) -> &mut Self {
        self.request_body.reasoning = Some(Reasoning { effort: Some(effort), summary: Some(summary) });
        self
    }

    /// Sets the text output verbosity level
    ///
    /// Controls how detailed and lengthy the model's text responses should be.
    /// This is particularly useful for controlling response length and detail level
    /// based on your use case requirements.
    ///
    /// # Arguments
    ///
    /// * `verbosity` - The verbosity level for text output:
    ///   - `TextVerbosity::Low` - Concise, brief responses
    ///   - `TextVerbosity::Medium` - Balanced responses (default)
    ///   - `TextVerbosity::High` - Comprehensive, detailed responses
    ///
    /// # Model Support
    ///
    /// This parameter is available on GPT-5.2 and newer models.
    ///
    /// # Use Cases
    ///
    /// - **Low verbosity**: Quick answers, summaries, yes/no questions
    /// - **Medium verbosity**: Standard explanations, general queries
    /// - **High verbosity**: Detailed tutorials, comprehensive analysis
    ///
    /// # Examples
    ///
    /// ```rust
    /// use openai_tools::responses::request::{Responses, TextVerbosity};
    ///
    /// let mut client = Responses::new();
    ///
    /// // Concise responses for simple queries
    /// client.text_verbosity(TextVerbosity::Low);
    ///
    /// // Detailed responses for complex explanations
    /// client.text_verbosity(TextVerbosity::High);
    /// ```
    pub fn text_verbosity(&mut self, verbosity: TextVerbosity) -> &mut Self {
        self.request_body.text = Some(TextConfig { verbosity: Some(verbosity) });
        self
    }

    /// Sets the safety identifier for content filtering configuration
    ///
    /// Specifies which safety and content filtering policies should be applied
    /// to the request. Different safety levels provide varying degrees of content
    /// restriction and filtering.
    ///
    /// # Arguments
    ///
    /// * `safety_id` - The safety configuration identifier
    ///
    /// # Common Safety Levels
    ///
    /// - `"strict"` - Apply strict content filtering (highest safety)
    /// - `"moderate"` - Apply moderate content filtering (balanced approach)
    /// - `"permissive"` - Apply permissive content filtering (minimal restrictions)
    /// - `"default"` - Use system default safety settings
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    ///
    /// # Use Cases
    ///
    /// - Educational content requiring strict filtering
    /// - Business applications with moderate restrictions
    /// - Research applications needing broader content access
    ///
    /// # Examples
    ///
    /// ```rust
    /// use openai_tools::responses::request::Responses;
    ///
    /// let mut client = Responses::new();
    /// client.safety_identifier("strict");     // High safety for education
    /// client.safety_identifier("moderate");   // Balanced for general use
    /// client.safety_identifier("permissive"); // Minimal restrictions
    /// ```
    pub fn safety_identifier<T: AsRef<str>>(&mut self, safety_id: T) -> &mut Self {
        self.request_body.safety_identifier = Some(safety_id.as_ref().to_string());
        self
    }

    /// Sets the service tier for request processing priority and features
    ///
    /// Specifies the service tier for the request, which affects processing
    /// priority, rate limits, pricing, and available features. Different tiers
    /// provide different levels of service quality and capabilities.
    ///
    /// # Arguments
    ///
    /// * `tier` - The service tier identifier
    ///
    /// # Common Service Tiers
    ///
    /// - `"default"` - Standard service tier with regular priority
    /// - `"scale"` - High-throughput tier optimized for bulk processing
    /// - `"premium"` - Premium service tier with enhanced features and priority
    /// - `"enterprise"` - Enterprise tier with dedicated resources
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    ///
    /// # Considerations
    ///
    /// - Higher tiers may have different pricing structures
    /// - Some features may only be available in certain tiers
    /// - Rate limits and quotas may vary by tier
    ///
    /// # Examples
    ///
    /// ```rust
    /// use openai_tools::responses::request::Responses;
    ///
    /// let mut client = Responses::new();
    /// client.service_tier("default");   // Standard service
    /// client.service_tier("scale");     // High-throughput processing
    /// client.service_tier("premium");   // Premium features and priority
    /// ```
    pub fn service_tier<T: AsRef<str>>(&mut self, tier: T) -> &mut Self {
        self.request_body.service_tier = Some(tier.as_ref().to_string());
        self
    }

    /// Enables or disables conversation storage
    ///
    /// Controls whether the conversation may be stored for future reference,
    /// training, or analytics purposes. This setting affects data retention
    /// and privacy policies.
    ///
    /// # Arguments
    ///
    /// * `enable` - Whether to allow conversation storage
    ///   - `true`: Allow storage for training, analytics, etc.
    ///   - `false`: Explicitly opt-out of storage
    ///
    /// # Privacy Considerations
    ///
    /// - **Enabled storage**: Conversation may be retained according to service policies
    /// - **Disabled storage**: Request explicit deletion after processing
    /// - **Default behavior**: Varies by service configuration
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    ///
    /// # Use Cases
    ///
    /// - **Enable**: Contributing to model improvement, analytics
    /// - **Disable**: Sensitive data, privacy-critical applications
    ///
    /// # Examples
    ///
    /// ```rust
    /// use openai_tools::responses::request::Responses;
    ///
    /// let mut client = Responses::new();
    /// client.store(false);  // Opt-out of storage for privacy
    /// client.store(true);   // Allow storage for improvement
    /// ```
    pub fn store(&mut self, enable: bool) -> &mut Self {
        self.request_body.store = Some(enable);
        self
    }

    /// Enables or disables streaming responses
    ///
    /// When enabled, the response will be streamed back in chunks as it's
    /// generated, allowing for real-time display of partial results instead
    /// of waiting for the complete response.
    ///
    /// # Arguments
    ///
    /// * `enable` - Whether to enable streaming
    ///   - `true`: Stream response in real-time chunks
    ///   - `false`: Wait for complete response before returning
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    ///
    /// # Use Cases
    ///
    /// - **Enable streaming**: Real-time chat interfaces, live text generation
    /// - **Disable streaming**: Batch processing, when complete response is needed
    ///
    /// # Implementation Notes
    ///
    /// - Streaming responses require different handling in client code
    /// - May affect some response features or formatting options
    /// - Typically used with `stream_options()` for additional configuration
    ///
    /// # Examples
    ///
    /// ```rust
    /// use openai_tools::responses::request::Responses;
    ///
    /// let mut client = Responses::new();
    /// client.stream(true);   // Enable real-time streaming
    /// client.stream(false);  // Wait for complete response
    /// ```
    pub fn stream(&mut self, enable: bool) -> &mut Self {
        self.request_body.stream = Some(enable);
        self
    }

    /// Configures streaming response options
    ///
    /// Additional options for controlling streaming response behavior,
    /// such as whether to include obfuscated placeholder content during
    /// the streaming process.
    ///
    /// # Arguments
    ///
    /// * `include_obfuscation` - Whether to include obfuscated content
    ///   - `true`: Include placeholder/obfuscated content in streams
    ///   - `false`: Only include final, non-obfuscated content
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    ///
    /// # Relevance
    ///
    /// This setting is only meaningful when `stream(true)` is also set.
    /// It has no effect on non-streaming responses.
    ///
    /// # Use Cases
    ///
    /// - **Include obfuscation**: Better user experience with placeholder content
    /// - **Exclude obfuscation**: Cleaner streams with only final content
    ///
    /// # Examples
    ///
    /// ```rust
    /// use openai_tools::responses::request::Responses;
    ///
    /// let mut client = Responses::new();
    /// client.stream(true);                    // Enable streaming
    /// client.stream_options(true);            // Include placeholder content
    /// client.stream_options(false);           // Only final content
    /// ```
    pub fn stream_options(&mut self, include_obfuscation: bool) -> &mut Self {
        self.request_body.stream_options = Some(StreamOptions { include_obfuscation });
        self
    }

    /// Sets the number of top log probabilities to include in the response
    ///
    /// Specifies how many of the most likely alternative tokens to include
    /// with their log probabilities for each generated token. This provides
    /// insight into the model's confidence and alternative choices.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of top alternatives to include (typically 1-20)
    ///   - `0`: No log probabilities included
    ///   - `1-5`: Common range for most use cases
    ///   - `>5`: Detailed analysis scenarios
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    ///
    /// # Use Cases
    ///
    /// - **Model analysis**: Understanding model decision-making
    /// - **Confidence estimation**: Measuring response certainty
    /// - **Alternative exploration**: Seeing what else the model considered
    /// - **Debugging**: Analyzing unexpected model behavior
    ///
    /// # Performance Note
    ///
    /// Higher values increase response size and may affect latency.
    ///
    /// **Note:** Reasoning models (GPT-6, GPT-5, o-series) do not support top_logprobs.
    /// For these models, this parameter will be ignored with a warning.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use openai_tools::responses::request::Responses;
    ///
    /// let mut client = Responses::new();
    /// client.top_logprobs(1);   // Include top alternative for each token
    /// client.top_logprobs(5);   // Include top 5 alternatives (detailed analysis)
    /// client.top_logprobs(0);   // No log probabilities
    /// ```
    pub fn top_logprobs(&mut self, n: usize) -> &mut Self {
        let support = self.request_body.model.parameter_support();
        if !support.top_logprobs {
            tracing::warn!("Model '{}' does not support top_logprobs parameter. Ignoring.", self.request_body.model);
            return self;
        }
        self.request_body.top_logprobs = Some(n);
        self
    }

    /// Sets the nucleus sampling parameter for controlling response diversity
    ///
    /// Controls the randomness of the model's responses by limiting the
    /// cumulative probability of considered tokens. This is an alternative
    /// to temperature-based sampling that can provide more stable results.
    ///
    /// # Arguments
    ///
    /// * `p` - The nucleus sampling parameter (0.0 to 1.0)
    ///   - `0.1`: Very focused, deterministic responses
    ///   - `0.7`: Balanced creativity and focus (good default)
    ///   - `0.9`: More diverse and creative responses
    ///   - `1.0`: Consider all possible tokens (no truncation)
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    ///
    /// # How It Works
    ///
    /// The model considers only the tokens whose cumulative probability
    /// reaches the specified threshold, filtering out unlikely options.
    ///
    /// # Interaction with Temperature
    ///
    /// Can be used together with `temperature()` for fine-tuned control:
    /// - Low top_p + Low temperature = Very focused responses
    /// - High top_p + High temperature = Very creative responses
    ///
    /// **Note:** Reasoning models (GPT-6, GPT-5, o-series) only support top_p=1.0.
    /// For these models, other values will be ignored with a warning.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use openai_tools::responses::request::Responses;
    ///
    /// let mut client = Responses::new();
    /// client.top_p(0.1);   // Very focused responses
    /// client.top_p(0.7);   // Balanced (recommended default)
    /// client.top_p(0.95);  // High diversity
    /// ```
    pub fn top_p(&mut self, p: f64) -> &mut Self {
        let support = self.request_body.model.parameter_support();
        match support.top_p {
            ParameterRestriction::FixedValue(fixed) => {
                if (p - fixed).abs() > f64::EPSILON {
                    tracing::warn!("Model '{}' only supports top_p={}. Ignoring top_p={}.", self.request_body.model, fixed, p);
                    return self;
                }
            }
            ParameterRestriction::NotSupported => {
                tracing::warn!("Model '{}' does not support top_p parameter. Ignoring.", self.request_body.model);
                return self;
            }
            ParameterRestriction::Any => {}
        }
        self.request_body.top_p = Some(p);
        self
    }

    /// Sets the truncation behavior for handling long inputs
    ///
    /// Controls how the system handles inputs that exceed the maximum
    /// context length supported by the model. This helps manage cases
    /// where input content is too large to process entirely.
    ///
    /// # Arguments
    ///
    /// * `truncation` - The truncation mode to use:
    ///   - `Truncation::Auto`: Automatically truncate long inputs to fit
    ///   - `Truncation::Disabled`: Return error if input exceeds context length
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    ///
    /// # Use Cases
    ///
    /// - **Auto truncation**: When you want to handle long documents gracefully
    /// - **Disabled truncation**: When you need to ensure complete input processing
    ///
    /// # Considerations
    ///
    /// - Auto truncation may remove important context from long inputs
    /// - Disabled truncation ensures complete processing but may cause errors
    /// - Consider breaking long inputs into smaller chunks when possible
    ///
    /// # Examples
    ///
    /// ```rust
    /// use openai_tools::responses::request::{Responses, Truncation};
    ///
    /// let mut client = Responses::new();
    /// client.truncation(Truncation::Auto);      // Handle long inputs gracefully
    /// client.truncation(Truncation::Disabled);  // Ensure complete processing
    /// ```
    pub fn truncation(&mut self, truncation: Truncation) -> &mut Self {
        self.request_body.truncation = Some(truncation);
        self
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

    /// Executes the request and returns the response
    ///
    /// This method sends the configured request to the OpenAI Responses API
    /// and returns the parsed response. It performs validation of required
    /// fields before sending the request.
    ///
    /// # Returns
    ///
    /// A `Result` containing the `Response` on success, or an `OpenAIToolError` on failure
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The API key is not set or is empty
    /// - The model ID is not set or is empty
    /// - Neither messages nor plain text input is provided
    /// - Both messages and plain text input are provided (mutually exclusive)
    /// - The HTTP request fails
    /// - The response cannot be parsed
    ///
    /// # Parameter Validation
    ///
    /// For reasoning models (GPT-6, GPT-5, o-series), certain parameters have restrictions:
    /// - `temperature`: only 1.0 supported
    /// - `top_p`: only 1.0 supported
    /// - `top_logprobs`: not supported
    ///
    /// **Validation occurs at two points:**
    /// 1. At setter time (when using `with_model()` constructor) - immediate warning
    /// 2. At API call time (fallback) - for cases where model is changed after setting params
    ///
    /// Unsupported parameter values are ignored with a warning and the request proceeds.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use openai_tools::responses::request::Responses;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut client = Responses::new();
    /// let response = client
    ///     .model_id("gpt-4")
    ///     .str_message("Hello!")
    ///     .complete()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn complete(&self) -> Result<Response> {
        // Validate that either messages or plain text input is set
        if self.request_body.messages_input.is_none() && self.request_body.plain_text_input.is_none() {
            return Err(OpenAIToolError::Error("Messages are not set.".into()));
        } else if self.request_body.plain_text_input.is_none() && self.request_body.messages_input.is_none() {
            return Err(OpenAIToolError::Error("Both plain text input and messages are set. Please use one of them.".into()));
        }

        // Handle reasoning models that don't support certain parameters
        // See: https://platform.openai.com/docs/guides/reasoning
        let mut request_body = self.request_body.clone();
        if self.is_reasoning_model() {
            let model = &self.request_body.model;

            // Temperature: only default (1.0) is supported
            if let Some(temp) = request_body.temperature {
                if (temp - 1.0).abs() > f64::EPSILON {
                    tracing::warn!(
                        "Reasoning model '{}' does not support custom temperature. \
                         Ignoring temperature={} and using default (1.0).",
                        model,
                        temp
                    );
                    request_body.temperature = None;
                }
            }

            // Top P: only default (1.0) is supported
            if let Some(top_p) = request_body.top_p {
                if (top_p - 1.0).abs() > f64::EPSILON {
                    tracing::warn!(
                        "Reasoning model '{}' does not support custom top_p. \
                         Ignoring top_p={} and using default (1.0).",
                        model,
                        top_p
                    );
                    request_body.top_p = None;
                }
            }

            // Top logprobs: not supported
            if request_body.top_logprobs.is_some() {
                tracing::warn!("Reasoning model '{}' does not support top_logprobs. Ignoring top_logprobs parameter.", model);
                request_body.top_logprobs = None;
            }
        }

        let body = serde_json::to_string(&request_body)?;

        let client = create_http_client(self.timeout)?;

        // Set up headers
        let mut headers = request::header::HeaderMap::new();
        headers.insert("Content-Type", request::header::HeaderValue::from_static("application/json"));
        if !self.user_agent.is_empty() {
            headers.insert("User-Agent", request::header::HeaderValue::from_str(&self.user_agent).unwrap());
        }

        // Apply provider-specific authentication headers
        self.auth.apply_headers(&mut headers)?;

        // Get the endpoint URL from the auth provider
        let endpoint = self.auth.endpoint(RESPONSES_PATH);

        if cfg!(test) {
            tracing::info!("Endpoint: {}", endpoint);
            // Replace API key with a placeholder for security
            let body_for_debug = serde_json::to_string_pretty(&request_body).unwrap().replace(self.auth.api_key(), "*************");
            // Log the request body for debugging purposes
            tracing::info!("Request body: {}", body_for_debug);
        }

        // Send the request and handle the response
        match client.post(&endpoint).headers(headers).body(body).send().await.map_err(OpenAIToolError::RequestError) {
            Err(e) => {
                tracing::error!("Request error: {}", e);
                Err(e)
            }
            Ok(response) if !response.status().is_success() => {
                let status = response.status();
                let error_text = response.text().await.unwrap_or_else(|_| "Failed to read error response".to_string());
                tracing::error!("API error (status: {}): {}", status, error_text);
                Err(OpenAIToolError::Error(format!("API request failed with status {}: {}", status, error_text)))
            }
            Ok(response) => {
                let content = response.text().await.map_err(OpenAIToolError::RequestError)?;

                if cfg!(test) {
                    tracing::info!("Response content: {}", content);
                }

                serde_json::from_str::<Response>(&content).map_err(OpenAIToolError::SerdeJsonError)
            }
        }
    }

    // ========================================
    // CRUD Endpoint Methods
    // ========================================

    /// Creates the HTTP client and headers for API requests
    ///
    /// This is a helper method that sets up the common HTTP client and headers
    /// needed for all API requests.
    ///
    /// # Returns
    ///
    /// A tuple of the HTTP client and headers
    fn create_api_client(&self) -> Result<(request::Client, request::header::HeaderMap)> {
        let client = create_http_client(self.timeout)?;
        let mut headers = request::header::HeaderMap::new();
        headers.insert("Content-Type", request::header::HeaderValue::from_static("application/json"));
        if !self.user_agent.is_empty() {
            headers.insert(
                "User-Agent",
                request::header::HeaderValue::from_str(&self.user_agent).map_err(|e| OpenAIToolError::Error(format!("Invalid user agent: {}", e)))?,
            );
        }
        self.auth.apply_headers(&mut headers)?;
        Ok((client, headers))
    }

    /// Handles API error responses
    ///
    /// This is a helper method that formats API error responses into a
    /// standardized error type.
    ///
    /// # Arguments
    ///
    /// * `status` - The HTTP status code
    /// * `content` - The error response content
    ///
    /// # Returns
    ///
    /// An OpenAIToolError containing the error details
    fn handle_api_error(status: request::StatusCode, content: &str) -> OpenAIToolError {
        tracing::error!("API error (status: {}): {}", status, content);
        OpenAIToolError::Error(format!("API request failed with status {}: {}", status, content))
    }

    /// Retrieves a response by its ID
    ///
    /// Fetches the details of a specific response, including its output,
    /// status, and metadata.
    ///
    /// # Arguments
    ///
    /// * `response_id` - The ID of the response to retrieve
    ///
    /// # Returns
    ///
    /// A `Result` containing the `Response` on success
    ///
    /// # API Reference
    ///
    /// <https://platform.openai.com/docs/api-reference/responses/get>
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use openai_tools::responses::request::Responses;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = Responses::new();
    /// let response = client.retrieve("resp_abc123").await?;
    /// println!("Status: {:?}", response.status);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn retrieve(&self, response_id: &str) -> Result<Response> {
        let (client, headers) = self.create_api_client()?;
        let endpoint = format!("{}/{}", self.auth.endpoint(RESPONSES_PATH), response_id);

        match client.get(&endpoint).headers(headers).send().await.map_err(OpenAIToolError::RequestError) {
            Err(e) => {
                tracing::error!("Request error: {}", e);
                Err(e)
            }
            Ok(response) if !response.status().is_success() => {
                let status = response.status();
                let error_text = response.text().await.unwrap_or_else(|_| "Failed to read error response".to_string());
                Err(Self::handle_api_error(status, &error_text))
            }
            Ok(response) => {
                let content = response.text().await.map_err(OpenAIToolError::RequestError)?;
                serde_json::from_str::<Response>(&content).map_err(OpenAIToolError::SerdeJsonError)
            }
        }
    }

    /// Deletes a response by its ID
    ///
    /// Permanently removes a response from the OpenAI platform.
    ///
    /// # Arguments
    ///
    /// * `response_id` - The ID of the response to delete
    ///
    /// # Returns
    ///
    /// A `Result` containing `DeleteResponseResult` on success
    ///
    /// # API Reference
    ///
    /// <https://platform.openai.com/docs/api-reference/responses/delete>
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use openai_tools::responses::request::Responses;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = Responses::new();
    /// let result = client.delete("resp_abc123").await?;
    /// assert!(result.deleted);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn delete(&self, response_id: &str) -> Result<DeleteResponseResult> {
        let (client, headers) = self.create_api_client()?;
        let endpoint = format!("{}/{}", self.auth.endpoint(RESPONSES_PATH), response_id);

        match client.delete(&endpoint).headers(headers).send().await.map_err(OpenAIToolError::RequestError) {
            Err(e) => {
                tracing::error!("Request error: {}", e);
                Err(e)
            }
            Ok(response) if !response.status().is_success() => {
                let status = response.status();
                let error_text = response.text().await.unwrap_or_else(|_| "Failed to read error response".to_string());
                Err(Self::handle_api_error(status, &error_text))
            }
            Ok(response) => {
                let content = response.text().await.map_err(OpenAIToolError::RequestError)?;
                serde_json::from_str::<DeleteResponseResult>(&content).map_err(OpenAIToolError::SerdeJsonError)
            }
        }
    }

    /// Cancels an in-progress response
    ///
    /// Cancels a response that is currently being generated. This is useful
    /// for background responses that are taking too long.
    ///
    /// # Arguments
    ///
    /// * `response_id` - The ID of the response to cancel
    ///
    /// # Returns
    ///
    /// A `Result` containing the cancelled `Response` on success
    ///
    /// # API Reference
    ///
    /// <https://platform.openai.com/docs/api-reference/responses/cancel>
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use openai_tools::responses::request::Responses;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = Responses::new();
    /// let response = client.cancel("resp_abc123").await?;
    /// println!("Response cancelled: {:?}", response.status);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn cancel(&self, response_id: &str) -> Result<Response> {
        let (client, headers) = self.create_api_client()?;
        let endpoint = format!("{}/{}/cancel", self.auth.endpoint(RESPONSES_PATH), response_id);

        match client.post(&endpoint).headers(headers).send().await.map_err(OpenAIToolError::RequestError) {
            Err(e) => {
                tracing::error!("Request error: {}", e);
                Err(e)
            }
            Ok(response) if !response.status().is_success() => {
                let status = response.status();
                let error_text = response.text().await.unwrap_or_else(|_| "Failed to read error response".to_string());
                Err(Self::handle_api_error(status, &error_text))
            }
            Ok(response) => {
                let content = response.text().await.map_err(OpenAIToolError::RequestError)?;
                serde_json::from_str::<Response>(&content).map_err(OpenAIToolError::SerdeJsonError)
            }
        }
    }

    /// Lists input items for a response
    ///
    /// Retrieves a paginated list of input items that were part of the request
    /// that generated the response.
    ///
    /// # Arguments
    ///
    /// * `response_id` - The ID of the response to get input items for
    /// * `limit` - Maximum number of items to return (default: 20)
    /// * `after` - Cursor for pagination (return items after this ID)
    /// * `before` - Cursor for pagination (return items before this ID)
    ///
    /// # Returns
    ///
    /// A `Result` containing `InputItemsListResponse` on success
    ///
    /// # API Reference
    ///
    /// <https://platform.openai.com/docs/api-reference/responses/list-input-items>
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use openai_tools::responses::request::Responses;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = Responses::new();
    /// let items = client.list_input_items("resp_abc123", Some(10), None, None).await?;
    /// for item in items.data {
    ///     println!("Item: {} (type: {})", item.id, item.item_type);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn list_input_items(
        &self,
        response_id: &str,
        limit: Option<u32>,
        after: Option<&str>,
        before: Option<&str>,
    ) -> Result<InputItemsListResponse> {
        let (client, headers) = self.create_api_client()?;
        let base_endpoint = format!("{}/{}/input_items", self.auth.endpoint(RESPONSES_PATH), response_id);

        // Build query parameters
        let mut query_params = Vec::new();
        if let Some(limit) = limit {
            query_params.push(format!("limit={}", limit));
        }
        if let Some(after) = after {
            query_params.push(format!("after={}", after));
        }
        if let Some(before) = before {
            query_params.push(format!("before={}", before));
        }

        let endpoint = if query_params.is_empty() { base_endpoint } else { format!("{}?{}", base_endpoint, query_params.join("&")) };

        match client.get(&endpoint).headers(headers).send().await.map_err(OpenAIToolError::RequestError) {
            Err(e) => {
                tracing::error!("Request error: {}", e);
                Err(e)
            }
            Ok(response) if !response.status().is_success() => {
                let status = response.status();
                let error_text = response.text().await.unwrap_or_else(|_| "Failed to read error response".to_string());
                Err(Self::handle_api_error(status, &error_text))
            }
            Ok(response) => {
                let content = response.text().await.map_err(OpenAIToolError::RequestError)?;
                serde_json::from_str::<InputItemsListResponse>(&content).map_err(OpenAIToolError::SerdeJsonError)
            }
        }
    }

    /// Compacts a response to reduce its size
    ///
    /// Creates a compacted version of a response, which can be useful
    /// for long-running conversations to reduce token usage.
    ///
    /// # Arguments
    ///
    /// * `previous_response_id` - The ID of the response to compact
    /// * `model` - Optional model to use for compaction (defaults to original model)
    ///
    /// # Returns
    ///
    /// A `Result` containing `CompactedResponse` on success
    ///
    /// # API Reference
    ///
    /// <https://platform.openai.com/docs/api-reference/responses/compact>
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use openai_tools::responses::request::Responses;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = Responses::new();
    /// let compacted = client.compact("resp_abc123", None).await?;
    /// println!("Compacted response ID: {}", compacted.id);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn compact(&self, previous_response_id: &str, model: Option<&str>) -> Result<CompactedResponse> {
        let (client, headers) = self.create_api_client()?;
        let endpoint = format!("{}/compact", self.auth.endpoint(RESPONSES_PATH));

        // Build request body
        let mut body = serde_json::json!({
            "previous_response_id": previous_response_id
        });
        if let Some(model) = model {
            body["model"] = serde_json::json!(model);
        }

        match client.post(&endpoint).headers(headers).body(serde_json::to_string(&body)?).send().await.map_err(OpenAIToolError::RequestError) {
            Err(e) => {
                tracing::error!("Request error: {}", e);
                Err(e)
            }
            Ok(response) if !response.status().is_success() => {
                let status = response.status();
                let error_text = response.text().await.unwrap_or_else(|_| "Failed to read error response".to_string());
                Err(Self::handle_api_error(status, &error_text))
            }
            Ok(response) => {
                let content = response.text().await.map_err(OpenAIToolError::RequestError)?;
                serde_json::from_str::<CompactedResponse>(&content).map_err(OpenAIToolError::SerdeJsonError)
            }
        }
    }

    /// Counts the number of input tokens for a potential request
    ///
    /// Useful for estimating token usage before sending a request.
    ///
    /// # Arguments
    ///
    /// * `model` - The model to use for token counting
    /// * `input` - The input to count tokens for (can be a string or messages array)
    ///
    /// # Returns
    ///
    /// A `Result` containing `InputTokensResponse` on success
    ///
    /// # API Reference
    ///
    /// <https://platform.openai.com/docs/api-reference/responses/input-tokens>
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use openai_tools::responses::request::Responses;
    /// use serde_json::json;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = Responses::new();
    /// let tokens = client.get_input_tokens("gpt-4o-mini", json!("Hello, world!")).await?;
    /// println!("Input tokens: {}", tokens.input_tokens);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn get_input_tokens(&self, model: &str, input: serde_json::Value) -> Result<InputTokensResponse> {
        let (client, headers) = self.create_api_client()?;
        let endpoint = format!("{}/input_tokens", self.auth.endpoint(RESPONSES_PATH));

        let body = serde_json::json!({
            "model": model,
            "input": input
        });

        match client.post(&endpoint).headers(headers).body(serde_json::to_string(&body)?).send().await.map_err(OpenAIToolError::RequestError) {
            Err(e) => {
                tracing::error!("Request error: {}", e);
                Err(e)
            }
            Ok(response) if !response.status().is_success() => {
                let status = response.status();
                let error_text = response.text().await.unwrap_or_else(|_| "Failed to read error response".to_string());
                Err(Self::handle_api_error(status, &error_text))
            }
            Ok(response) => {
                let content = response.text().await.map_err(OpenAIToolError::RequestError)?;
                serde_json::from_str::<InputTokensResponse>(&content).map_err(OpenAIToolError::SerdeJsonError)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ReasoningEffort;

    #[test]
    fn reasoning_effort_max_serializes_as_max() {
        assert_eq!(serde_json::to_string(&ReasoningEffort::Max).unwrap(), "\"max\"");
        assert_eq!(ReasoningEffort::Max.to_string(), "max");
        assert_eq!("max".parse::<ReasoningEffort>().unwrap(), ReasoningEffort::Max);
    }
}
