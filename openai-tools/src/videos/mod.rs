//! # Videos Module
//!
//! This module provides functionality for interacting with the OpenAI Videos API
//! (`/v1/videos`), which generates video clips with the Sora models.
//!
//! ## Key Features
//!
//! - **Create Video**: Start a generation job from a text prompt
//! - **Retrieve Video**: Poll a job for status and progress
//! - **List Videos**: Page through recently generated videos
//! - **Delete Video**: Remove a completed or failed video and its assets
//! - **Download Content**: Fetch the rendered video, thumbnail or spritesheet
//! - **Remix Video**: Re-generate an existing video with an updated prompt
//!
//! ## Asynchronous by Design
//!
//! Video generation is a long-running job. [`create`](request::Videos::create)
//! returns as soon as the job is queued; poll
//! [`retrieve`](request::Videos::retrieve) until
//! [`Video::is_terminal`](response::Video::is_terminal) reports the job has
//! settled, then download the bytes with
//! [`content`](request::Videos::content).
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use openai_tools::videos::request::{Videos, CreateVideoOptions};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let videos = Videos::new()?;
//!
//!     // Queue a generation job
//!     let job = videos.create("A red balloon over Tokyo", CreateVideoOptions::default()).await?;
//!     println!("Queued: {} ({:?})", job.id, job.status);
//!
//!     // Poll until it settles
//!     let mut video = videos.retrieve(&job.id).await?;
//!     while !video.is_terminal() {
//!         tokio::time::sleep(std::time::Duration::from_secs(10)).await;
//!         video = videos.retrieve(&job.id).await?;
//!         println!("progress: {}%", video.progress);
//!     }
//!
//!     if video.is_completed() {
//!         let bytes = videos.content(&video.id, None).await?;
//!         std::fs::write("out.mp4", bytes)?;
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Choosing a Size and Duration
//!
//! | Enum | Allowed values |
//! |------|----------------|
//! | [`VideoSize`](request::VideoSize) | `720x1280`, `1280x720`, `1024x1792`, `1792x1024` |
//! | [`VideoSeconds`](request::VideoSeconds) | `4`, `8`, `12` |
//!
//! Note that the API requires `seconds` to be sent as a *string*; the
//! [`VideoSeconds`](request::VideoSeconds) enum handles that encoding.
//!
//! ## Costs
//!
//! Video generation is billed per second of output. Generating a clip is far
//! more expensive than a text completion, so guard calls accordingly.

pub mod request;
pub mod response;

#[cfg(test)]
mod tests {
    use crate::videos::request::{CreateVideoOptions, InputReference, VideoModel, VideoSeconds, VideoSize, VideoVariant};
    use crate::videos::response::{DeleteVideoResponse, Video, VideoListResponse, VideoStatus};

    // ========================================================================
    // Enums
    //
    // Values verified against the live API (August 2026): the `size` and
    // `seconds` errors enumerate exactly these values, and `seconds` is
    // rejected outright when sent as an integer.
    // ========================================================================

    #[test]
    fn test_video_model_as_str() {
        assert_eq!(VideoModel::Sora2.as_str(), "sora-2");
        assert_eq!(VideoModel::Sora2Pro.as_str(), "sora-2-pro");
    }

    #[test]
    fn test_video_model_default_is_sora2() {
        assert_eq!(VideoModel::default(), VideoModel::Sora2);
    }

    #[test]
    fn test_video_model_serialization() {
        for (model, expected) in [(VideoModel::Sora2, "sora-2"), (VideoModel::Sora2Pro, "sora-2-pro")] {
            let json = serde_json::to_string(&model).unwrap();
            assert_eq!(json, format!("\"{}\"", expected), "Wrong serialization for {:?}", model);
        }
    }

    #[test]
    fn test_video_size_as_str() {
        assert_eq!(VideoSize::Size720x1280.as_str(), "720x1280");
        assert_eq!(VideoSize::Size1280x720.as_str(), "1280x720");
        assert_eq!(VideoSize::Size1024x1792.as_str(), "1024x1792");
        assert_eq!(VideoSize::Size1792x1024.as_str(), "1792x1024");
    }

    #[test]
    fn test_video_size_default_is_portrait() {
        assert_eq!(VideoSize::default(), VideoSize::Size720x1280);
    }

    /// The API rejects `seconds` sent as an integer, so the enum must encode
    /// itself as a string.
    #[test]
    fn test_video_seconds_serializes_as_string() {
        for (seconds, expected) in [(VideoSeconds::Four, "4"), (VideoSeconds::Eight, "8"), (VideoSeconds::Twelve, "12")] {
            assert_eq!(seconds.as_str(), expected);
            let json = serde_json::to_string(&seconds).unwrap();
            assert_eq!(json, format!("\"{}\"", expected), "seconds must serialize as a JSON string, got {}", json);
        }
    }

    #[test]
    fn test_video_seconds_default_is_four() {
        assert_eq!(VideoSeconds::default(), VideoSeconds::Four);
    }

    #[test]
    fn test_video_variant_as_str() {
        assert_eq!(VideoVariant::Video.as_str(), "video");
        assert_eq!(VideoVariant::Thumbnail.as_str(), "thumbnail");
        assert_eq!(VideoVariant::Spritesheet.as_str(), "spritesheet");
    }

    // ========================================================================
    // Request serialization
    // ========================================================================

    /// Unset options must be omitted so the API applies its own defaults.
    #[test]
    fn test_create_request_omits_unset_options() {
        let body = CreateVideoOptions::default().into_request("A red balloon");
        let value = serde_json::to_value(&body).unwrap();

        assert_eq!(value.get("prompt").and_then(|v| v.as_str()), Some("A red balloon"));
        assert!(value.get("model").is_none(), "model must be omitted when unset");
        assert!(value.get("size").is_none(), "size must be omitted when unset");
        assert!(value.get("seconds").is_none(), "seconds must be omitted when unset");
        assert!(value.get("input_reference").is_none(), "input_reference must be omitted when unset");
    }

    #[test]
    fn test_create_request_serializes_all_options() {
        let options = CreateVideoOptions {
            model: Some(VideoModel::Sora2Pro),
            size: Some(VideoSize::Size1280x720),
            seconds: Some(VideoSeconds::Twelve),
            input_reference: None,
        };
        let value = serde_json::to_value(options.into_request("A cat")).unwrap();

        assert_eq!(value.get("model").and_then(|v| v.as_str()), Some("sora-2-pro"));
        assert_eq!(value.get("size").and_then(|v| v.as_str()), Some("1280x720"));
        // Must be a string, not the number 12.
        assert_eq!(value.get("seconds").and_then(|v| v.as_str()), Some("12"));
    }

    #[test]
    fn test_create_request_with_file_input_reference() {
        let options = CreateVideoOptions { input_reference: Some(InputReference::file_id("file-abc123")), ..Default::default() };
        let value = serde_json::to_value(options.into_request("A cat")).unwrap();

        let reference = value.get("input_reference").expect("input_reference should be present");
        assert_eq!(reference.get("file_id").and_then(|v| v.as_str()), Some("file-abc123"));
        assert!(reference.get("image_url").is_none(), "only the set variant should be serialized");
    }

    #[test]
    fn test_create_request_with_image_url_input_reference() {
        let options = CreateVideoOptions { input_reference: Some(InputReference::image_url("https://example.com/a.png")), ..Default::default() };
        let value = serde_json::to_value(options.into_request("A cat")).unwrap();

        let reference = value.get("input_reference").expect("input_reference should be present");
        assert_eq!(reference.get("image_url").and_then(|v| v.as_str()), Some("https://example.com/a.png"));
        assert!(reference.get("file_id").is_none(), "only the set variant should be serialized");
    }

    // ========================================================================
    // Response deserialization
    // ========================================================================

    #[test]
    fn test_video_deserialization_queued() {
        let json = r#"{
            "id": "video_abc123",
            "object": "video",
            "model": "sora-2",
            "status": "queued",
            "progress": 0,
            "prompt": "A red balloon",
            "created_at": 1760000000,
            "completed_at": null,
            "expires_at": null,
            "remixed_from_video_id": null,
            "seconds": "4",
            "size": "720x1280",
            "error": null
        }"#;

        let video: Video = serde_json::from_str(json).expect("Should deserialize Video");
        assert_eq!(video.id, "video_abc123");
        assert_eq!(video.object, "video");
        assert_eq!(video.status, VideoStatus::Queued);
        assert_eq!(video.progress, 0);
        assert_eq!(video.prompt.as_deref(), Some("A red balloon"));
        assert_eq!(video.seconds.as_deref(), Some("4"));
        assert!(video.error.is_none());
        assert!(!video.is_terminal(), "a queued job has not settled");
        assert!(!video.is_completed());
    }

    #[test]
    fn test_video_deserialization_completed() {
        let json = r#"{
            "id": "video_abc123",
            "object": "video",
            "model": "sora-2-pro",
            "status": "completed",
            "progress": 100,
            "prompt": "A red balloon",
            "created_at": 1760000000,
            "completed_at": 1760000600,
            "expires_at": 1760600000,
            "remixed_from_video_id": null,
            "seconds": "8",
            "size": "1280x720"
        }"#;

        let video: Video = serde_json::from_str(json).expect("Should deserialize completed Video");
        assert_eq!(video.status, VideoStatus::Completed);
        assert_eq!(video.progress, 100);
        assert_eq!(video.completed_at, Some(1760000600));
        assert!(video.is_terminal());
        assert!(video.is_completed());
    }

    #[test]
    fn test_video_deserialization_failed_with_error() {
        let json = r#"{
            "id": "video_abc123",
            "object": "video",
            "model": "sora-2",
            "status": "failed",
            "progress": 0,
            "created_at": 1760000000,
            "seconds": "4",
            "size": "720x1280",
            "error": { "code": "content_policy_violation", "message": "Rejected by the safety system" }
        }"#;

        let video: Video = serde_json::from_str(json).expect("Should deserialize failed Video");
        assert_eq!(video.status, VideoStatus::Failed);
        assert!(video.is_terminal(), "a failed job has settled");
        assert!(!video.is_completed());

        let error = video.error.expect("error details should be present");
        assert_eq!(error.code, "content_policy_violation");
        assert_eq!(error.message, "Rejected by the safety system");
    }

    #[test]
    fn test_video_status_in_progress() {
        let json = r#"{
            "id": "video_abc123",
            "object": "video",
            "model": "sora-2",
            "status": "in_progress",
            "progress": 42,
            "created_at": 1760000000,
            "seconds": "4",
            "size": "720x1280"
        }"#;

        let video: Video = serde_json::from_str(json).expect("Should deserialize in-progress Video");
        assert_eq!(video.status, VideoStatus::InProgress);
        assert_eq!(video.progress, 42);
        assert!(!video.is_terminal());
    }

    /// An unrecognised status must not break deserialization - the API can add
    /// lifecycle states without a library release.
    #[test]
    fn test_unknown_video_status_is_preserved() {
        let json = r#"{
            "id": "video_abc123",
            "object": "video",
            "model": "sora-2",
            "status": "cancelled",
            "progress": 0,
            "created_at": 1760000000,
            "seconds": "4",
            "size": "720x1280"
        }"#;

        let video: Video = serde_json::from_str(json).expect("Unknown status should still deserialize");
        assert_eq!(video.status, VideoStatus::Other("cancelled".to_string()));
        assert!(!video.is_completed());
    }

    /// Matches the shape returned by a live `GET /v1/videos` call.
    #[test]
    fn test_video_list_response_deserialization() {
        let json = r#"{
            "object": "list",
            "data": [],
            "first_id": null,
            "last_id": null,
            "has_more": false
        }"#;

        let response: VideoListResponse = serde_json::from_str(json).expect("Should deserialize VideoListResponse");
        assert_eq!(response.object, "list");
        assert!(response.data.is_empty());
        assert!(!response.has_more);
        assert!(response.first_id.is_none());
    }

    #[test]
    fn test_delete_video_response_deserialization() {
        let json = r#"{
            "id": "video_abc123",
            "object": "video.deleted",
            "deleted": true
        }"#;

        let response: DeleteVideoResponse = serde_json::from_str(json).expect("Should deserialize DeleteVideoResponse");
        assert_eq!(response.id, "video_abc123");
        assert_eq!(response.object, "video.deleted");
        assert!(response.deleted);
    }
}
