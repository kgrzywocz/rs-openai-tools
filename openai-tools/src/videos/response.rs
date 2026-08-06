//! OpenAI Videos API Response Module
//!
//! Response types returned by the Videos API (`/v1/videos`).

use serde::{Deserialize, Serialize};

/// Lifecycle status of a video generation job.
///
/// Unrecognised values deserialize into [`Other`](VideoStatus::Other) so that a
/// new server-side status does not break existing clients.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VideoStatus {
    /// Job accepted and waiting to start
    Queued,
    /// Job is currently generating
    InProgress,
    /// Job finished successfully; assets are downloadable
    Completed,
    /// Job failed; see [`Video::error`]
    Failed,
    /// A status this library does not know about yet
    #[serde(untagged)]
    Other(String),
}

/// Error details attached to a failed video job.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VideoError {
    /// Machine-readable error code
    pub code: String,
    /// Human-readable description
    pub message: String,
}

/// A video generation job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Video {
    /// Unique identifier for the job
    pub id: String,
    /// Object type, always "video"
    pub object: String,
    /// The model used for generation
    pub model: String,
    /// Current lifecycle status
    pub status: VideoStatus,
    /// Approximate completion percentage (0-100)
    #[serde(default)]
    pub progress: u32,
    /// The prompt used for generation
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    /// Unix timestamp (seconds) when the job was created
    pub created_at: i64,
    /// Unix timestamp (seconds) when the job finished, if it has
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<i64>,
    /// Unix timestamp (seconds) when the generated assets expire
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<i64>,
    /// Source video ID when this job is a remix
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub remixed_from_video_id: Option<String>,
    /// Clip duration in seconds, as returned by the API (a string)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seconds: Option<String>,
    /// Output resolution
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<String>,
    /// Error details when [`status`](Video::status) is
    /// [`Failed`](VideoStatus::Failed)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<VideoError>,
}

impl Video {
    /// Returns `true` once the job has settled - successfully or not.
    ///
    /// Use this to end a polling loop.
    ///
    /// # Example
    ///
    /// ```rust
    /// use openai_tools::videos::response::{Video, VideoStatus};
    ///
    /// let json = r#"{"id":"v1","object":"video","model":"sora-2",
    ///     "status":"completed","progress":100,"created_at":1}"#;
    /// let video: Video = serde_json::from_str(json).unwrap();
    /// assert!(video.is_terminal());
    /// ```
    pub fn is_terminal(&self) -> bool {
        matches!(self.status, VideoStatus::Completed | VideoStatus::Failed)
    }

    /// Returns `true` only when the job completed successfully and its assets
    /// can be downloaded.
    pub fn is_completed(&self) -> bool {
        self.status == VideoStatus::Completed
    }
}

/// A page of video jobs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoListResponse {
    /// Object type, always "list"
    pub object: String,
    /// The videos in this page
    pub data: Vec<Video>,
    /// ID of the first item, for pagination
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_id: Option<String>,
    /// ID of the last item, for pagination
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_id: Option<String>,
    /// Whether more items are available
    #[serde(default)]
    pub has_more: bool,
}

/// Result of deleting a video.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteVideoResponse {
    /// The deleted video identifier
    pub id: String,
    /// Object type, always "video.deleted"
    pub object: String,
    /// Whether the video was successfully deleted
    pub deleted: bool,
}
