//! Authentication abstraction for OpenAI and Azure OpenAI APIs
//!
//! This module provides a unified interface for authenticating with both
//! OpenAI API and Azure OpenAI API. It handles the differences in:
//! - Header format (Bearer token vs api-key)
//! - Endpoint URL construction
//! - Environment variable names
//!
//! # Quick Start
//!
//! ## OpenAI API (existing users - no changes needed)
//!
//! ```rust,no_run
//! use openai_tools::common::auth::AuthProvider;
//!
//! // From environment variable OPENAI_API_KEY
//! let auth = AuthProvider::openai_from_env()?;
//! # Ok::<(), openai_tools::common::errors::OpenAIToolError>(())
//! ```
//!
//! ## Azure OpenAI API
//!
//! ```rust,no_run
//! use openai_tools::common::auth::{AuthProvider, AzureAuth};
//!
//! // From environment variables (AZURE_OPENAI_API_KEY, AZURE_OPENAI_BASE_URL)
//! let auth = AuthProvider::azure_from_env()?;
//!
//! // Or explicit configuration with complete base URL
//! let auth = AuthProvider::Azure(
//!     AzureAuth::new(
//!         "your-api-key",
//!         "https://my-resource.openai.azure.com/openai/deployments/gpt-4o?api-version=2024-08-01-preview"
//!     )
//! );
//! # Ok::<(), openai_tools::common::errors::OpenAIToolError>(())
//! ```
//!
//! ## URL-based Provider Detection
//!
//! Use `from_url_with_key` to auto-detect the provider based on URL pattern:
//!
//! ```rust
//! use openai_tools::common::auth::AuthProvider;
//!
//! // Azure URL detected automatically (*.openai.azure.com)
//! let auth = AuthProvider::from_url_with_key(
//!     "https://my-resource.openai.azure.com/openai/deployments/gpt-4o?api-version=2024-08-01-preview",
//!     "your-api-key"
//! );
//!
//! // OpenAI-compatible URL (non-Azure URLs)
//! let auth = AuthProvider::from_url_with_key(
//!     "http://localhost:11434/v1",  // Ollama, vLLM, etc.
//!     "ollama"
//! );
//! ```
//!
//! ## Auto-detection from Environment
//!
//! ```rust,no_run
//! use openai_tools::common::auth::AuthProvider;
//!
//! // Uses Azure if AZURE_OPENAI_API_KEY is set, otherwise OpenAI
//! let auth = AuthProvider::from_env()?;
//! # Ok::<(), openai_tools::common::errors::OpenAIToolError>(())
//! ```

use crate::common::errors::{OpenAIToolError, Result};
use dotenvy::dotenv;
use request::header::{HeaderMap, HeaderValue};
use std::env;

/// Default OpenAI API base URL
const OPENAI_DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// Authentication provider for OpenAI APIs
///
/// This enum encapsulates the authentication strategy for different API providers.
/// It handles both endpoint URL construction and HTTP header application.
///
/// # Example
///
/// ```rust,no_run
/// use openai_tools::common::auth::AuthProvider;
///
/// // Auto-detect from environment
/// let auth = AuthProvider::from_env()?;
///
/// // Get endpoint for chat completions
/// let endpoint = auth.endpoint("chat/completions");
///
/// // Apply auth headers to a request
/// let mut headers = request::header::HeaderMap::new();
/// auth.apply_headers(&mut headers)?;
/// # Ok::<(), openai_tools::common::errors::OpenAIToolError>(())
/// ```
#[derive(Debug, Clone)]
pub enum AuthProvider {
    /// OpenAI API authentication
    OpenAI(OpenAIAuth),
    /// Azure OpenAI API authentication
    Azure(AzureAuth),
}

/// OpenAI API authentication configuration
///
/// Handles authentication for the standard OpenAI API using Bearer tokens.
///
/// # Header Format
///
/// ```text
/// Authorization: Bearer sk-...
/// ```
///
/// # Endpoint Format
///
/// ```text
/// https://api.openai.com/v1/{path}
/// ```
#[derive(Debug, Clone)]
pub struct OpenAIAuth {
    /// The API key (sk-...)
    api_key: String,
    /// Base URL for API requests (default: `https://api.openai.com/v1`)
    base_url: String,
}

impl OpenAIAuth {
    /// Creates a new OpenAI authentication configuration
    ///
    /// # Arguments
    ///
    /// * `api_key` - OpenAI API key (typically starts with "sk-")
    ///
    /// # Returns
    ///
    /// A new `OpenAIAuth` instance with default base URL
    ///
    /// # Example
    ///
    /// ```rust
    /// use openai_tools::common::auth::OpenAIAuth;
    ///
    /// let auth = OpenAIAuth::new("sk-your-api-key");
    /// ```
    pub fn new<T: Into<String>>(api_key: T) -> Self {
        Self { api_key: api_key.into(), base_url: OPENAI_DEFAULT_BASE_URL.to_string() }
    }

    /// Sets a custom base URL
    ///
    /// Use this for proxies or alternative OpenAI-compatible APIs.
    ///
    /// # Arguments
    ///
    /// * `url` - Custom base URL (without trailing slash)
    ///
    /// # Returns
    ///
    /// Self for method chaining
    ///
    /// # Example
    ///
    /// ```rust
    /// use openai_tools::common::auth::OpenAIAuth;
    ///
    /// let auth = OpenAIAuth::new("sk-key")
    ///     .with_base_url("https://my-proxy.example.com/v1");
    /// ```
    pub fn with_base_url<T: Into<String>>(mut self, url: T) -> Self {
        self.base_url = url.into();
        self
    }

    /// Returns the API key
    pub fn api_key(&self) -> &str {
        &self.api_key
    }

    /// Returns the base URL
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Constructs the full endpoint URL for a given path
    ///
    /// # Arguments
    ///
    /// * `path` - API path (e.g., "chat/completions")
    ///
    /// # Returns
    ///
    /// Full URL string
    fn endpoint(&self, path: &str) -> String {
        format!("{}/{}", self.base_url.trim_end_matches('/'), path.trim_start_matches('/'))
    }

    /// Applies authentication headers to a request
    ///
    /// Adds the `Authorization: Bearer {key}` header.
    fn apply_headers(&self, headers: &mut HeaderMap) -> Result<()> {
        headers.insert(
            "Authorization",
            HeaderValue::from_str(&format!("Bearer {}", self.api_key)).map_err(|e| OpenAIToolError::Error(format!("Invalid header value: {}", e)))?,
        );
        Ok(())
    }
}

/// Azure OpenAI API authentication configuration
///
/// Handles authentication for Azure OpenAI Service, which uses different
/// header names and endpoint URL patterns than the standard OpenAI API.
///
/// # Header Format
///
/// ```text
/// api-key: {key}
/// ```
///
/// # Endpoint Format
///
/// The `base_url` should be a complete endpoint URL including deployment path,
/// API path (e.g., `/chat/completions`), and query parameters (e.g., `?api-version=...`).
/// The `endpoint()` method returns this URL as-is.
///
/// # Example
///
/// ```rust
/// use openai_tools::common::auth::AzureAuth;
///
/// // For Chat API
/// let auth = AzureAuth::new(
///     "your-api-key",
///     "https://my-resource.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview"
/// );
///
/// // For Embedding API
/// let auth = AzureAuth::new(
///     "your-api-key",
///     "https://my-resource.openai.azure.com/openai/deployments/text-embedding/embeddings?api-version=2024-08-01-preview"
/// );
/// ```
#[derive(Debug, Clone)]
pub struct AzureAuth {
    /// API key
    api_key: String,
    /// Complete endpoint URL for API requests
    base_url: String,
}

impl AzureAuth {
    /// Creates a new Azure OpenAI authentication configuration
    ///
    /// # Arguments
    ///
    /// * `api_key` - Azure OpenAI API key
    /// * `base_url` - Complete endpoint URL including deployment path, API path, and api-version
    ///
    /// # Returns
    ///
    /// A new `AzureAuth` instance
    ///
    /// # Example
    ///
    /// ```rust
    /// use openai_tools::common::auth::AzureAuth;
    ///
    /// // Complete endpoint URL for Chat API
    /// let auth = AzureAuth::new(
    ///     "your-api-key",
    ///     "https://my-resource.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview"
    /// );
    /// ```
    pub fn new<T: Into<String>>(api_key: T, base_url: T) -> Self {
        Self { api_key: api_key.into(), base_url: base_url.into() }
    }

    /// Returns the API key
    pub fn api_key(&self) -> &str {
        &self.api_key
    }

    /// Returns the base URL
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Returns the endpoint URL
    ///
    /// # Arguments
    ///
    /// * `_path` - Ignored (for API compatibility with OpenAIAuth)
    ///
    /// # Returns
    ///
    /// The complete base URL as-is
    fn endpoint(&self, _path: &str) -> String {
        self.base_url.clone()
    }

    /// Applies authentication headers to a request
    ///
    /// Uses `api-key` header for Azure OpenAI authentication.
    fn apply_headers(&self, headers: &mut HeaderMap) -> Result<()> {
        headers.insert("api-key", HeaderValue::from_str(&self.api_key).map_err(|e| OpenAIToolError::Error(format!("Invalid header value: {}", e)))?);
        Ok(())
    }
}

impl AuthProvider {
    /// Creates an OpenAI authentication provider from environment variables
    ///
    /// Reads the API key from `OPENAI_API_KEY` environment variable.
    ///
    /// # Returns
    ///
    /// `Result<AuthProvider>` - OpenAI auth provider or error if env var not set
    ///
    /// # Environment Variables
    ///
    /// | Variable | Required | Description |
    /// |----------|----------|-------------|
    /// | `OPENAI_API_KEY` | Yes | OpenAI API key |
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::common::auth::AuthProvider;
    ///
    /// let auth = AuthProvider::openai_from_env()?;
    /// # Ok::<(), openai_tools::common::errors::OpenAIToolError>(())
    /// ```
    pub fn openai_from_env() -> Result<Self> {
        dotenv().ok();
        let api_key = env::var("OPENAI_API_KEY").map_err(|_| OpenAIToolError::Error("OPENAI_API_KEY environment variable not set".into()))?;
        Ok(Self::OpenAI(OpenAIAuth::new(api_key)))
    }

    /// Creates an Azure OpenAI authentication provider from environment variables
    ///
    /// # Returns
    ///
    /// `Result<AuthProvider>` - Azure auth provider or error if required vars not set
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
    /// use openai_tools::common::auth::AuthProvider;
    ///
    /// // With environment variables:
    /// // AZURE_OPENAI_API_KEY=xxx
    /// // AZURE_OPENAI_BASE_URL=https://my-resource.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview
    /// let auth = AuthProvider::azure_from_env()?;
    /// # Ok::<(), openai_tools::common::errors::OpenAIToolError>(())
    /// ```
    pub fn azure_from_env() -> Result<Self> {
        dotenv().ok();

        // Get API key
        let api_key =
            env::var("AZURE_OPENAI_API_KEY").map_err(|_| OpenAIToolError::Error("AZURE_OPENAI_API_KEY environment variable not set".into()))?;

        // Get base URL (required)
        let base_url =
            env::var("AZURE_OPENAI_BASE_URL").map_err(|_| OpenAIToolError::Error("AZURE_OPENAI_BASE_URL environment variable not set".into()))?;

        Ok(Self::Azure(AzureAuth::new(api_key, base_url)))
    }

    /// Creates an authentication provider by auto-detecting from environment
    ///
    /// Tries Azure first (if `AZURE_OPENAI_API_KEY` is set), then falls back to OpenAI.
    ///
    /// # Returns
    ///
    /// `Result<AuthProvider>` - Detected auth provider or error
    ///
    /// # Detection Order
    ///
    /// 1. If `AZURE_OPENAI_API_KEY` is set → Azure
    /// 2. If `OPENAI_API_KEY` is set → OpenAI
    /// 3. Otherwise → Error
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::common::auth::AuthProvider;
    ///
    /// let auth = AuthProvider::from_env()?;
    /// match &auth {
    ///     AuthProvider::OpenAI(_) => println!("Using OpenAI"),
    ///     AuthProvider::Azure(_) => println!("Using Azure"),
    /// }
    /// # Ok::<(), openai_tools::common::errors::OpenAIToolError>(())
    /// ```
    pub fn from_env() -> Result<Self> {
        dotenv().ok();

        // Try Azure first if its key is present
        if env::var("AZURE_OPENAI_API_KEY").is_ok() {
            return Self::azure_from_env();
        }

        // Fall back to OpenAI
        Self::openai_from_env()
    }

    /// Constructs the full endpoint URL for a given API path
    ///
    /// # Arguments
    ///
    /// * `path` - API path (e.g., "chat/completions", "embeddings")
    ///
    /// # Returns
    ///
    /// Full endpoint URL appropriate for the provider
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::common::auth::AuthProvider;
    ///
    /// let auth = AuthProvider::openai_from_env()?;
    /// let url = auth.endpoint("chat/completions");
    /// // OpenAI: https://api.openai.com/v1/chat/completions
    /// // Azure: https://{resource}.openai.azure.com/openai/deployments/{deployment}/chat/completions?api-version={ver}
    /// # Ok::<(), openai_tools::common::errors::OpenAIToolError>(())
    /// ```
    pub fn endpoint(&self, path: &str) -> String {
        match self {
            Self::OpenAI(auth) => auth.endpoint(path),
            Self::Azure(auth) => auth.endpoint(path),
        }
    }

    /// Applies authentication headers to a request
    ///
    /// # Arguments
    ///
    /// * `headers` - Mutable reference to header map
    ///
    /// # Returns
    ///
    /// `Result<()>` - Success or error if header value is invalid
    ///
    /// # Header Differences
    ///
    /// | Provider | Header Name | Value Format |
    /// |----------|-------------|--------------|
    /// | OpenAI | `Authorization` | `Bearer {key}` |
    /// | Azure | `api-key` | `{key}` |
    pub fn apply_headers(&self, headers: &mut HeaderMap) -> Result<()> {
        match self {
            Self::OpenAI(auth) => auth.apply_headers(headers),
            Self::Azure(auth) => auth.apply_headers(headers),
        }
    }

    /// Returns the API key (for backward compatibility)
    ///
    /// # Returns
    ///
    /// The API key or token string
    pub fn api_key(&self) -> &str {
        match self {
            Self::OpenAI(auth) => auth.api_key(),
            Self::Azure(auth) => auth.api_key(),
        }
    }

    /// Returns whether this is an Azure provider
    ///
    /// # Returns
    ///
    /// `true` if Azure, `false` if OpenAI
    pub fn is_azure(&self) -> bool {
        matches!(self, Self::Azure(_))
    }

    /// Returns whether this is an OpenAI provider
    ///
    /// # Returns
    ///
    /// `true` if OpenAI, `false` if Azure
    pub fn is_openai(&self) -> bool {
        matches!(self, Self::OpenAI(_))
    }

    /// Creates an authentication provider by detecting the provider from URL pattern.
    ///
    /// This method analyzes the URL to determine whether it's an Azure OpenAI endpoint
    /// or a standard OpenAI (or compatible) endpoint.
    ///
    /// # Arguments
    ///
    /// * `base_url` - The complete base URL for API requests
    /// * `api_key` - The API key or token
    ///
    /// # URL Detection
    ///
    /// | URL Pattern | Detected Provider |
    /// |-------------|-------------------|
    /// | `*.openai.azure.com*` | Azure |
    /// | All others | OpenAI (or compatible) |
    ///
    /// # Returns
    ///
    /// `AuthProvider` - Detected provider
    ///
    /// # Example
    ///
    /// ```rust
    /// use openai_tools::common::auth::AuthProvider;
    ///
    /// // OpenAI compatible API
    /// let openai = AuthProvider::from_url_with_key(
    ///     "https://api.openai.com/v1",
    ///     "sk-key",
    /// );
    /// assert!(openai.is_openai());
    ///
    /// // Azure OpenAI (complete base URL)
    /// let azure = AuthProvider::from_url_with_key(
    ///     "https://my-resource.openai.azure.com/openai/deployments/gpt-4o?api-version=2024-08-01-preview",
    ///     "azure-key",
    /// );
    /// assert!(azure.is_azure());
    ///
    /// // Local API (Ollama, LocalAI, etc.)
    /// let local = AuthProvider::from_url_with_key(
    ///     "http://localhost:11434/v1",
    ///     "dummy-key",
    /// );
    /// assert!(local.is_openai());  // Treated as OpenAI-compatible
    /// ```
    pub fn from_url_with_key<S: Into<String>>(base_url: S, api_key: S) -> Self {
        let url_str = base_url.into();
        let api_key_str = api_key.into();

        // Check if URL matches Azure pattern
        if url_str.contains(".openai.azure.com") {
            Self::Azure(AzureAuth::new(api_key_str, url_str))
        } else {
            // OpenAI or compatible API
            Self::OpenAI(OpenAIAuth::new(api_key_str).with_base_url(url_str))
        }
    }

    /// Creates an authentication provider from URL, using environment variables for credentials.
    ///
    /// This is a convenience method that combines URL-based detection with
    /// environment variable-based credential loading.
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
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::common::auth::AuthProvider;
    ///
    /// // Uses OPENAI_API_KEY from environment
    /// let auth = AuthProvider::from_url("https://api.openai.com/v1")?;
    ///
    /// // Uses AZURE_OPENAI_API_KEY from environment
    /// let azure = AuthProvider::from_url(
    ///     "https://my-resource.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview"
    /// )?;
    /// # Ok::<(), openai_tools::common::errors::OpenAIToolError>(())
    /// ```
    pub fn from_url<S: Into<String>>(base_url: S) -> Result<Self> {
        let url_str = base_url.into();
        dotenv().ok();

        if url_str.contains(".openai.azure.com") {
            // Azure: get credentials from Azure env vars
            let api_key = env::var("AZURE_OPENAI_API_KEY")
                .map_err(|_| OpenAIToolError::Error("Azure URL detected but AZURE_OPENAI_API_KEY is not set".into()))?;

            Ok(Self::Azure(AzureAuth::new(api_key, url_str)))
        } else {
            // OpenAI: get credentials from OpenAI env var
            let api_key = env::var("OPENAI_API_KEY").map_err(|_| OpenAIToolError::Error("OPENAI_API_KEY environment variable not set".into()))?;

            Ok(Self::OpenAI(OpenAIAuth::new(api_key).with_base_url(url_str)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // OpenAI Auth Tests

    #[test]
    fn test_openai_auth_new() {
        let auth = OpenAIAuth::new("sk-test-key");
        assert_eq!(auth.api_key(), "sk-test-key");
        assert_eq!(auth.base_url(), OPENAI_DEFAULT_BASE_URL);
    }

    #[test]
    fn test_openai_auth_with_base_url() {
        let auth = OpenAIAuth::new("sk-test-key").with_base_url("https://custom.example.com/v1");
        assert_eq!(auth.base_url(), "https://custom.example.com/v1");
    }

    #[test]
    fn test_openai_endpoint() {
        let auth = OpenAIAuth::new("sk-test-key");
        assert_eq!(auth.endpoint("chat/completions"), "https://api.openai.com/v1/chat/completions");
        assert_eq!(auth.endpoint("/chat/completions"), "https://api.openai.com/v1/chat/completions");
    }

    #[test]
    fn test_openai_apply_headers() {
        let auth = OpenAIAuth::new("sk-test-key");
        let mut headers = HeaderMap::new();
        auth.apply_headers(&mut headers).unwrap();

        assert_eq!(headers.get("Authorization").unwrap(), "Bearer sk-test-key");
    }

    #[test]
    fn test_openai_endpoint_trailing_slash_handling() {
        let auth = OpenAIAuth::new("key").with_base_url("https://example.com/v1/");
        assert_eq!(auth.endpoint("chat/completions"), "https://example.com/v1/chat/completions");
        assert_eq!(auth.endpoint("/chat/completions"), "https://example.com/v1/chat/completions");
    }

    // Azure Auth Tests (Simplified API - base_url only)

    #[test]
    fn test_azure_auth_new() {
        let auth = AzureAuth::new(
            "api-key",
            "https://my-resource.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview",
        );
        assert_eq!(auth.api_key(), "api-key");
        assert_eq!(auth.base_url(), "https://my-resource.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview");
    }

    #[test]
    fn test_azure_endpoint_returns_base_url() {
        // Azure endpoint() returns base_url as-is (path is ignored)
        let auth =
            AzureAuth::new("key", "https://my-resource.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview");
        let endpoint = auth.endpoint("ignored");
        assert_eq!(endpoint, "https://my-resource.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview");
    }

    #[test]
    fn test_azure_apply_headers() {
        let auth = AzureAuth::new("my-api-key", "https://my-resource.openai.azure.com");
        let mut headers = HeaderMap::new();
        auth.apply_headers(&mut headers).unwrap();

        assert_eq!(headers.get("api-key").unwrap(), "my-api-key");
        assert!(headers.get("Authorization").is_none());
    }

    // AuthProvider Tests

    #[test]
    fn test_auth_provider_openai() {
        let auth = AuthProvider::OpenAI(OpenAIAuth::new("sk-key"));
        assert!(auth.is_openai());
        assert!(!auth.is_azure());
        assert_eq!(auth.api_key(), "sk-key");
    }

    #[test]
    fn test_auth_provider_azure() {
        let auth = AuthProvider::Azure(AzureAuth::new("key", "https://my-resource.openai.azure.com/openai/deployments/gpt-4o"));
        assert!(auth.is_azure());
        assert!(!auth.is_openai());
        assert_eq!(auth.api_key(), "key");
    }

    #[test]
    fn test_auth_provider_endpoint_openai() {
        let auth = AuthProvider::OpenAI(OpenAIAuth::new("key"));
        assert_eq!(auth.endpoint("chat/completions"), "https://api.openai.com/v1/chat/completions");
    }

    #[test]
    fn test_auth_provider_endpoint_azure() {
        // Azure endpoint returns base_url as-is (path is ignored)
        let base_url = "https://my-resource.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview";
        let auth = AuthProvider::Azure(AzureAuth::new("key", base_url));
        let endpoint = auth.endpoint("ignored");
        assert_eq!(endpoint, base_url);
    }

    #[test]
    fn test_auth_provider_apply_headers() {
        // OpenAI
        let openai_auth = AuthProvider::OpenAI(OpenAIAuth::new("sk-key"));
        let mut headers = HeaderMap::new();
        openai_auth.apply_headers(&mut headers).unwrap();
        assert!(headers.get("Authorization").unwrap().to_str().unwrap().starts_with("Bearer"));

        // Azure
        let azure_auth = AuthProvider::Azure(AzureAuth::new("azure-key", "https://my-resource.openai.azure.com"));
        let mut headers = HeaderMap::new();
        azure_auth.apply_headers(&mut headers).unwrap();
        assert_eq!(headers.get("api-key").unwrap(), "azure-key");
    }

    #[test]
    fn test_from_env_returns_correct_provider_type() {
        // Test direct construction (which from_env uses internally)
        let openai = AuthProvider::OpenAI(OpenAIAuth::new("sk-test"));
        assert!(openai.is_openai());
        assert!(!openai.is_azure());

        let azure = AuthProvider::Azure(AzureAuth::new("key", "https://my-resource.openai.azure.com/openai/deployments/gpt-4o"));
        assert!(azure.is_azure());
        assert!(!azure.is_openai());
    }

    // URL-based provider detection tests (from_url_with_key)

    #[test]
    fn test_from_url_with_key_openai_api() {
        let auth = AuthProvider::from_url_with_key("https://api.openai.com/v1", "sk-test-key");

        assert!(auth.is_openai());
        assert!(!auth.is_azure());
        assert_eq!(auth.api_key(), "sk-test-key");
        assert_eq!(auth.endpoint("chat/completions"), "https://api.openai.com/v1/chat/completions");
    }

    #[test]
    fn test_from_url_with_key_azure() {
        let base_url = "https://my-resource.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview";
        let auth = AuthProvider::from_url_with_key(base_url, "azure-api-key");

        assert!(auth.is_azure());
        assert!(!auth.is_openai());
        assert_eq!(auth.api_key(), "azure-api-key");

        // Azure endpoint returns base_url as-is (path is ignored)
        let endpoint = auth.endpoint("ignored");
        assert_eq!(endpoint, base_url);
    }

    #[test]
    fn test_from_url_with_key_local_api_ollama() {
        // Local APIs like Ollama should be treated as OpenAI-compatible
        let auth = AuthProvider::from_url_with_key("http://localhost:11434/v1", "ollama");

        assert!(auth.is_openai());
        assert_eq!(auth.endpoint("chat/completions"), "http://localhost:11434/v1/chat/completions");
    }

    #[test]
    fn test_from_url_with_key_custom_openai_compatible() {
        // Custom OpenAI-compatible endpoints (e.g., vLLM, LocalAI)
        let auth = AuthProvider::from_url_with_key("https://my-proxy.example.com/openai/v1", "proxy-key");

        assert!(auth.is_openai());
        assert_eq!(auth.endpoint("embeddings"), "https://my-proxy.example.com/openai/v1/embeddings");
    }

    #[test]
    fn test_from_url_with_key_azure_various_patterns() {
        // Test various Azure URL patterns
        let patterns = [
            "https://eastus.openai.azure.com/openai/deployments/gpt-4o",
            "https://my-company-resource.openai.azure.com/openai/deployments/gpt-4o",
            "https://test.openai.azure.com/openai/deployments/gpt-4o?api-version=2024-08-01-preview",
        ];

        for url in patterns {
            let auth = AuthProvider::from_url_with_key(url, "key");
            assert!(auth.is_azure(), "Should be Azure provider for URL: {}", url);
        }
    }

    #[test]
    fn test_from_url_with_key_headers_openai() {
        let auth = AuthProvider::from_url_with_key("https://api.openai.com/v1", "sk-secret-key");

        let mut headers = HeaderMap::new();
        auth.apply_headers(&mut headers).unwrap();

        assert_eq!(headers.get("Authorization").unwrap(), "Bearer sk-secret-key");
    }

    #[test]
    fn test_from_url_with_key_headers_azure() {
        let auth = AuthProvider::from_url_with_key("https://resource.openai.azure.com/openai/deployments/gpt-4o", "azure-secret");

        let mut headers = HeaderMap::new();
        auth.apply_headers(&mut headers).unwrap();

        assert_eq!(headers.get("api-key").unwrap(), "azure-secret");
        assert!(headers.get("Authorization").is_none());
    }
}
