//! OpenAI Videos API Request Module
//!
//! This module provides the functionality to interact with the OpenAI Videos API
//! (`/v1/videos`) for generating video clips with the Sora models.
//!
//! # Key Features
//!
//! - **Create**: Queue a generation job from a text prompt
//! - **Retrieve**: Poll a job for status and progress
//! - **List**: Page through recently generated videos
//! - **Delete**: Remove a video and its assets
//! - **Content**: Download the rendered video, thumbnail or spritesheet
//! - **Remix**: Re-generate an existing video with an updated prompt
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use openai_tools::videos::request::{Videos, CreateVideoOptions};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let videos = Videos::new()?;
//!
//!     let job = videos.create("A red balloon over Tokyo", CreateVideoOptions::default()).await?;
//!     println!("Queued: {}", job.id);
//!
//!     Ok(())
//! }
//! ```

use crate::common::auth::{AuthProvider, OpenAIAuth};
use crate::common::client::create_http_client;
use crate::common::errors::{ErrorResponse, OpenAIToolError, Result};
use crate::videos::response::{DeleteVideoResponse, Video, VideoListResponse};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Default API path for Videos
const VIDEOS_PATH: &str = "videos";

/// Video generation models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum VideoModel {
    /// Sora 2 - standard video generation model
    #[serde(rename = "sora-2")]
    #[default]
    Sora2,
    /// Sora 2 Pro - higher quality, more expensive
    #[serde(rename = "sora-2-pro")]
    Sora2Pro,
}

impl VideoModel {
    /// Returns the model identifier string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Sora2 => "sora-2",
            Self::Sora2Pro => "sora-2-pro",
        }
    }
}

impl std::fmt::Display for VideoModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Output resolutions supported by the Videos API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum VideoSize {
    /// 720x1280 - portrait (default)
    #[serde(rename = "720x1280")]
    #[default]
    Size720x1280,
    /// 1280x720 - landscape
    #[serde(rename = "1280x720")]
    Size1280x720,
    /// 1024x1792 - tall portrait
    #[serde(rename = "1024x1792")]
    Size1024x1792,
    /// 1792x1024 - wide landscape
    #[serde(rename = "1792x1024")]
    Size1792x1024,
}

impl VideoSize {
    /// Returns the size string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Size720x1280 => "720x1280",
            Self::Size1280x720 => "1280x720",
            Self::Size1024x1792 => "1024x1792",
            Self::Size1792x1024 => "1792x1024",
        }
    }
}

impl std::fmt::Display for VideoSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Clip durations supported by the Videos API.
///
/// The API requires this value as a *string*; sending an integer is rejected
/// with `invalid_type`. The `serde` rename below produces the string form.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum VideoSeconds {
    /// 4 second clip (default)
    #[serde(rename = "4")]
    #[default]
    Four,
    /// 8 second clip
    #[serde(rename = "8")]
    Eight,
    /// 12 second clip
    #[serde(rename = "12")]
    Twelve,
}

impl VideoSeconds {
    /// Returns the duration string as the API expects it.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Four => "4",
            Self::Eight => "8",
            Self::Twelve => "12",
        }
    }
}

impl std::fmt::Display for VideoSeconds {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Downloadable asset variants for a completed video.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum VideoVariant {
    /// The rendered video file (default)
    #[default]
    Video,
    /// A single still frame
    Thumbnail,
    /// A grid of frames
    Spritesheet,
}

impl VideoVariant {
    /// Returns the variant string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Video => "video",
            Self::Thumbnail => "thumbnail",
            Self::Spritesheet => "spritesheet",
        }
    }
}

impl std::fmt::Display for VideoVariant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// A reference image guiding generation.
///
/// Supply either an uploaded file or an image URL - never both.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct InputReference {
    /// ID of a file uploaded via the Files API
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_id: Option<String>,
    /// A fully qualified URL, or a base64-encoded data URL
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_url: Option<String>,
}

impl InputReference {
    /// Builds a reference from an uploaded file ID.
    pub fn file_id<S: Into<String>>(file_id: S) -> Self {
        Self { file_id: Some(file_id.into()), image_url: None }
    }

    /// Builds a reference from an image URL or data URL.
    pub fn image_url<S: Into<String>>(image_url: S) -> Self {
        Self { file_id: None, image_url: Some(image_url.into()) }
    }
}

/// Optional settings for [`Videos::create`].
///
/// Unset fields are omitted from the request so the API applies its own
/// defaults (`sora-2`, `720x1280`, 4 seconds).
#[derive(Debug, Clone, Default)]
pub struct CreateVideoOptions {
    /// The generation model
    pub model: Option<VideoModel>,
    /// Clip duration
    pub seconds: Option<VideoSeconds>,
    /// Output resolution
    pub size: Option<VideoSize>,
    /// Reference image guiding generation
    pub input_reference: Option<InputReference>,
}

impl CreateVideoOptions {
    /// Combines these options with a prompt into a serializable request body.
    pub fn into_request<S: Into<String>>(self, prompt: S) -> CreateVideoRequest {
        CreateVideoRequest { prompt: prompt.into(), model: self.model, seconds: self.seconds, size: self.size, input_reference: self.input_reference }
    }
}

/// Request body for `POST /v1/videos`.
#[derive(Debug, Clone, Serialize)]
pub struct CreateVideoRequest {
    /// Describes the video to generate
    pub prompt: String,
    /// The generation model
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<VideoModel>,
    /// Clip duration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seconds: Option<VideoSeconds>,
    /// Output resolution
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<VideoSize>,
    /// Reference image guiding generation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_reference: Option<InputReference>,
}

/// Request body for `POST /v1/videos/{video_id}/remix`.
#[derive(Debug, Clone, Serialize)]
struct RemixVideoRequest {
    prompt: String,
}

/// Sort order for [`Videos::list`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SortOrder {
    /// Oldest first
    Asc,
    /// Newest first (default)
    #[default]
    Desc,
}

impl SortOrder {
    /// Returns the order string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Asc => "asc",
            Self::Desc => "desc",
        }
    }
}

/// OpenAI Videos API client.
///
/// Video generation is a long-running job: [`create`](Videos::create) queues
/// the work and returns immediately, so poll [`retrieve`](Videos::retrieve)
/// until the job settles before downloading with [`content`](Videos::content).
///
/// # Example
///
/// ```rust,no_run
/// use openai_tools::videos::request::{Videos, CreateVideoOptions, VideoSize};
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let videos = Videos::new()?;
///
///     let options = CreateVideoOptions { size: Some(VideoSize::Size1280x720), ..Default::default() };
///     let job = videos.create("A red balloon over Tokyo", options).await?;
///     println!("Queued: {}", job.id);
///
///     Ok(())
/// }
/// ```
pub struct Videos {
    /// Authentication provider (OpenAI or Azure)
    auth: AuthProvider,
    /// Optional request timeout duration
    timeout: Option<Duration>,
}

impl Videos {
    /// Creates a new Videos client from the `OPENAI_API_KEY` environment
    /// variable.
    pub fn new() -> Result<Self> {
        let auth = AuthProvider::openai_from_env()?;
        Ok(Self { auth, timeout: None })
    }

    /// Creates a new Videos client with a custom authentication provider.
    pub fn with_auth(auth: AuthProvider) -> Self {
        Self { auth, timeout: None }
    }

    /// Creates a new Videos client for Azure OpenAI API.
    pub fn azure() -> Result<Self> {
        let auth = AuthProvider::azure_from_env()?;
        Ok(Self { auth, timeout: None })
    }

    /// Creates a new Videos client by auto-detecting the provider.
    pub fn detect_provider() -> Result<Self> {
        let auth = AuthProvider::from_env()?;
        Ok(Self { auth, timeout: None })
    }

    /// Creates a new Videos client with URL-based provider detection.
    pub fn with_url<S: Into<String>>(base_url: S, api_key: S) -> Self {
        let auth = AuthProvider::from_url_with_key(base_url, api_key);
        Self { auth, timeout: None }
    }

    /// Creates a new Videos client from a URL using environment credentials.
    pub fn from_url<S: Into<String>>(url: S) -> Result<Self> {
        let auth = AuthProvider::from_url(url)?;
        Ok(Self { auth, timeout: None })
    }

    /// Returns the authentication provider.
    pub fn auth(&self) -> &AuthProvider {
        &self.auth
    }

    /// Sets a custom API endpoint URL (OpenAI only).
    pub fn base_url<T: AsRef<str>>(&mut self, url: T) -> &mut Self {
        if let AuthProvider::OpenAI(ref openai_auth) = self.auth {
            let new_auth = OpenAIAuth::new(openai_auth.api_key()).with_base_url(url.as_ref());
            self.auth = AuthProvider::OpenAI(new_auth);
        } else {
            tracing::warn!("base_url() is only supported for OpenAI provider. Use azure() or with_auth() for Azure.");
        }
        self
    }

    /// Sets the request timeout duration.
    ///
    /// Generation jobs are queued asynchronously, so the default (no timeout)
    /// is usually fine; a longer timeout mainly matters for
    /// [`content`](Videos::content) downloads.
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

    /// Reads a response body, turning API errors into [`OpenAIToolError`].
    async fn read_json<T: serde::de::DeserializeOwned>(response: request::Response) -> Result<T> {
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

        serde_json::from_str::<T>(&content).map_err(OpenAIToolError::SerdeJsonError)
    }

    /// Queues a video generation job.
    ///
    /// Returns as soon as the job is accepted - the returned [`Video`] will be
    /// [`Queued`](crate::videos::response::VideoStatus::Queued), not finished.
    ///
    /// # Arguments
    ///
    /// * `prompt` - Describes the video to generate
    /// * `options` - Model, duration, size and reference image
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::videos::request::{Videos, CreateVideoOptions, VideoSeconds};
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let videos = Videos::new()?;
    ///
    ///     let options = CreateVideoOptions { seconds: Some(VideoSeconds::Eight), ..Default::default() };
    ///     let job = videos.create("A red balloon over Tokyo", options).await?;
    ///     println!("{} -> {:?}", job.id, job.status);
    ///     Ok(())
    /// }
    /// ```
    pub async fn create<S: Into<String>>(&self, prompt: S, options: CreateVideoOptions) -> Result<Video> {
        let (client, headers) = self.create_client()?;
        let url = self.auth.endpoint(VIDEOS_PATH);
        let body = serde_json::to_string(&options.into_request(prompt))?;

        let response = client.post(&url).headers(headers).body(body).send().await.map_err(OpenAIToolError::RequestError)?;

        Self::read_json(response).await
    }

    /// Retrieves the current state of a video generation job.
    ///
    /// # Arguments
    ///
    /// * `video_id` - The video identifier
    pub async fn retrieve(&self, video_id: &str) -> Result<Video> {
        let (client, headers) = self.create_client()?;
        let url = format!("{}/{}", self.auth.endpoint(VIDEOS_PATH), video_id);

        let response = client.get(&url).headers(headers).send().await.map_err(OpenAIToolError::RequestError)?;

        Self::read_json(response).await
    }

    /// Lists recently generated videos for the current project.
    ///
    /// # Arguments
    ///
    /// * `limit` - Maximum number of videos to return
    /// * `after` - Pagination cursor: return items after this video ID
    /// * `order` - Sort order by creation time
    pub async fn list(&self, limit: Option<u32>, after: Option<&str>, order: Option<SortOrder>) -> Result<VideoListResponse> {
        let (client, headers) = self.create_client()?;

        let mut query: Vec<String> = Vec::new();
        if let Some(limit) = limit {
            query.push(format!("limit={}", limit));
        }
        if let Some(after) = after {
            query.push(format!("after={}", after));
        }
        if let Some(order) = order {
            query.push(format!("order={}", order.as_str()));
        }

        let endpoint = self.auth.endpoint(VIDEOS_PATH);
        let url = if query.is_empty() { endpoint } else { format!("{}?{}", endpoint, query.join("&")) };

        let response = client.get(&url).headers(headers).send().await.map_err(OpenAIToolError::RequestError)?;

        Self::read_json(response).await
    }

    /// Deletes a completed or failed video along with its assets.
    ///
    /// # Arguments
    ///
    /// * `video_id` - The video identifier
    pub async fn delete(&self, video_id: &str) -> Result<DeleteVideoResponse> {
        let (client, headers) = self.create_client()?;
        let url = format!("{}/{}", self.auth.endpoint(VIDEOS_PATH), video_id);

        let response = client.delete(&url).headers(headers).send().await.map_err(OpenAIToolError::RequestError)?;

        Self::read_json(response).await
    }

    /// Downloads the bytes of a generated asset.
    ///
    /// # Arguments
    ///
    /// * `video_id` - The video identifier
    /// * `variant` - Which asset to download; defaults to
    ///   [`VideoVariant::Video`]
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::videos::request::{Videos, VideoVariant};
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let videos = Videos::new()?;
    ///
    ///     let mp4 = videos.content("video_abc123", None).await?;
    ///     std::fs::write("out.mp4", mp4)?;
    ///
    ///     let thumbnail = videos.content("video_abc123", Some(VideoVariant::Thumbnail)).await?;
    ///     std::fs::write("thumb.jpg", thumbnail)?;
    ///     Ok(())
    /// }
    /// ```
    pub async fn content(&self, video_id: &str, variant: Option<VideoVariant>) -> Result<Vec<u8>> {
        let (client, headers) = self.create_client()?;
        let url = match variant {
            Some(variant) => format!("{}/{}/content?variant={}", self.auth.endpoint(VIDEOS_PATH), video_id, variant.as_str()),
            None => format!("{}/{}/content", self.auth.endpoint(VIDEOS_PATH), video_id),
        };

        let response = client.get(&url).headers(headers).send().await.map_err(OpenAIToolError::RequestError)?;

        let status = response.status();
        if !status.is_success() {
            let content = response.text().await.map_err(OpenAIToolError::RequestError)?;
            if let Ok(error_resp) = serde_json::from_str::<ErrorResponse>(&content) {
                return Err(OpenAIToolError::Error(error_resp.error.message.unwrap_or_default()));
            }
            return Err(OpenAIToolError::Error(format!("API error ({}): {}", status, content)));
        }

        let bytes = response.bytes().await.map_err(OpenAIToolError::RequestError)?;
        Ok(bytes.to_vec())
    }

    /// Creates a remix of an existing video using an updated prompt.
    ///
    /// # Arguments
    ///
    /// * `video_id` - The source video identifier
    /// * `prompt` - Updated prompt directing the remix
    pub async fn remix<S: Into<String>>(&self, video_id: &str, prompt: S) -> Result<Video> {
        let (client, headers) = self.create_client()?;
        let url = format!("{}/{}/remix", self.auth.endpoint(VIDEOS_PATH), video_id);
        let body = serde_json::to_string(&RemixVideoRequest { prompt: prompt.into() })?;

        let response = client.post(&url).headers(headers).body(body).send().await.map_err(OpenAIToolError::RequestError)?;

        Self::read_json(response).await
    }
}
