//! Integration tests for the OpenAI Videos API.
//!
//! These tests require a valid OPENAI_API_KEY environment variable.
//! Run with: cargo test --test videos_integration
//!
//! # Cost
//!
//! Video generation is billed per second of output and is far more expensive
//! than a text completion. The tests that run by default only exercise the
//! free read paths (list, error handling). The test that actually generates a
//! clip is marked `#[ignore]`; run it deliberately with:
//!
//! ```bash
//! cargo test --test videos_integration -- --ignored --nocapture
//! ```

use openai_tools::videos::request::{CreateVideoOptions, VideoSeconds, VideoSize, Videos};
use openai_tools::videos::response::VideoStatus;

/// Listing videos exercises auth, URL construction and response parsing
/// without generating anything.
#[tokio::test]
async fn test_list_videos() {
    let videos = Videos::new().expect("Should create Videos client");
    let response = videos.list(Some(5), None, None).await.expect("Should list videos");

    assert_eq!(response.object, "list");
    assert!(response.data.len() <= 5, "Should respect the limit");

    for video in &response.data {
        assert!(!video.id.is_empty(), "Video ID should not be empty");
        assert_eq!(video.object, "video");
        assert!(video.created_at > 0, "Created timestamp should be positive");
    }

    println!("Found {} videos", response.data.len());
}

/// The `order` and `limit` query parameters must be accepted by the API.
#[tokio::test]
async fn test_list_videos_with_order() {
    use openai_tools::videos::request::SortOrder;

    let videos = Videos::new().expect("Should create Videos client");
    let response = videos.list(Some(2), None, Some(SortOrder::Asc)).await.expect("Should list videos in ascending order");

    assert_eq!(response.object, "list");
}

/// Retrieving an unknown video must surface an API error rather than panic.
#[tokio::test]
async fn test_retrieve_nonexistent_video() {
    let videos = Videos::new().expect("Should create Videos client");
    let result = videos.retrieve("video_does_not_exist_000").await;

    assert!(result.is_err(), "Retrieving an unknown video should fail");
    println!("Error (expected): {:?}", result.err());
}

/// Downloading content for an unknown video must surface an API error rather
/// than returning the error body as if it were video bytes.
#[tokio::test]
async fn test_content_nonexistent_video() {
    let videos = Videos::new().expect("Should create Videos client");
    let result = videos.content("video_does_not_exist_000", None).await;

    assert!(result.is_err(), "Downloading an unknown video should fail");
}

/// End-to-end generation. Ignored by default because it costs real money and
/// takes minutes to complete.
#[tokio::test]
#[ignore = "generates a real video: billed per second and takes minutes"]
async fn test_create_and_poll_video() {
    let videos = Videos::new().expect("Should create Videos client");

    let options = CreateVideoOptions { seconds: Some(VideoSeconds::Four), size: Some(VideoSize::Size720x1280), ..Default::default() };

    let job = videos.create("A red balloon drifting over a quiet city at dawn", options).await.expect("Should queue a video job");

    assert!(!job.id.is_empty());
    assert_eq!(job.object, "video");
    assert!(!job.is_terminal(), "A freshly queued job should not be terminal");
    println!("Queued {} ({:?})", job.id, job.status);

    // Poll until the job settles, with a generous cap.
    let mut video = job.clone();
    for _ in 0..60 {
        if video.is_terminal() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_secs(10)).await;
        video = videos.retrieve(&job.id).await.expect("Should retrieve video");
        println!("  status={:?} progress={}", video.status, video.progress);
    }

    assert!(video.is_terminal(), "Video should settle within the polling window");

    if video.status == VideoStatus::Completed {
        let bytes = videos.content(&video.id, None).await.expect("Should download video content");
        assert!(!bytes.is_empty(), "Downloaded video should not be empty");
        println!("Downloaded {} bytes", bytes.len());
    } else {
        println!("Generation failed: {:?}", video.error);
    }

    // Clean up so the test does not leave assets behind.
    let deleted = videos.delete(&video.id).await.expect("Should delete video");
    assert!(deleted.deleted);
}
