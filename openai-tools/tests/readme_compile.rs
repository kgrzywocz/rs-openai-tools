//! Compile-check for the code samples in README.md.
//!
//! Nothing here talks to the network - the sample bodies are never called. The
//! file exists so that the test build fails if a README snippet drifts out of
//! sync with the actual API surface.

use openai_tools::audio::request::Audio;
use openai_tools::batch::request::Batches;
use openai_tools::chat::request::ChatCompletion;
use openai_tools::conversations::request::Conversations;
use openai_tools::embedding::request::Embedding;
use openai_tools::files::request::Files;
use openai_tools::fine_tuning::request::FineTuning;
use openai_tools::images::request::Images;
use openai_tools::models::request::Models;
use openai_tools::moderations::request::Moderations;
use openai_tools::realtime::RealtimeClient;
use openai_tools::responses::request::Responses;
use openai_tools::videos::request::Videos;

use openai_tools::common::models::ChatModel;
use openai_tools::images::request::{GenerateOptions, ImageModel, ImageQuality, ImageSize};
use openai_tools::videos::request::{CreateVideoOptions, SortOrder, VideoSeconds, VideoSize, VideoVariant};

#[allow(dead_code, unused_variables)]
async fn images_sample() -> Result<(), Box<dyn std::error::Error>> {
    let images = Images::new()?;

    let options = GenerateOptions {
        model: Some(ImageModel::GptImage1Mini),
        size: Some(ImageSize::Size1024x1024),
        quality: Some(ImageQuality::Low),
        ..Default::default()
    };
    let response = images.generate("A sunset over mountains", options).await?;

    let bytes = response.data[0].as_bytes().unwrap()?;
    std::fs::write("sunset.png", bytes)?;
    Ok(())
}

#[allow(dead_code, unused_variables)]
async fn videos_generate_sample() -> Result<(), Box<dyn std::error::Error>> {
    let videos = Videos::new()?;

    let options = CreateVideoOptions { seconds: Some(VideoSeconds::Four), size: Some(VideoSize::Size720x1280), ..Default::default() };
    let job = videos.create("A red balloon over Tokyo at dawn", options).await?;
    println!("Queued: {}", job.id);

    let mut video = videos.retrieve(&job.id).await?;
    while !video.is_terminal() {
        tokio::time::sleep(std::time::Duration::from_secs(10)).await;
        video = videos.retrieve(&job.id).await?;
        println!("progress: {}%", video.progress);
    }

    if video.is_completed() {
        let bytes = videos.content(&video.id, None).await?;
        std::fs::write("out.mp4", bytes)?;
    } else {
        eprintln!("Generation failed: {:?}", video.error);
    }

    videos.delete(&video.id).await?;
    Ok(())
}

#[allow(dead_code, unused_variables)]
async fn videos_other_sample() -> Result<(), Box<dyn std::error::Error>> {
    let videos = Videos::new()?;

    let page = videos.list(Some(10), None, Some(SortOrder::Desc)).await?;
    println!("{} jobs", page.data.len());

    let thumbnail = videos.content("video_abc123", Some(VideoVariant::Thumbnail)).await?;
    std::fs::write("thumb.jpg", thumbnail)?;

    let remixed = videos.remix("video_abc123", "...but at night").await?;
    println!("Remixed as {}", remixed.id);
    Ok(())
}

#[allow(dead_code, unused_variables)]
fn model_selection_sample() {
    let mut chat = ChatCompletion::new();
    chat.model(ChatModel::Gpt5_6Sol);
    chat.model(ChatModel::Gpt5_6Luna);
    chat.model(ChatModel::custom("ft:gpt-4.1-mini:my-org::abc123"));

    let mut responses = Responses::new();
    responses.model(ChatModel::Gpt5_5Pro);
}

#[allow(dead_code, unused_variables)]
fn module_imports_resolve() -> Result<(), Box<dyn std::error::Error>> {
    let _ = ChatCompletion::new();
    let _ = Responses::new();
    let _ = Embedding::new();
    let _ = RealtimeClient::new();
    let _ = Conversations::new()?;
    let _ = Models::new()?;
    let _ = Files::new()?;
    let _ = Moderations::new()?;
    let _ = Images::new()?;
    let _ = Audio::new()?;
    let _ = Batches::new()?;
    let _ = FineTuning::new()?;
    let _ = Videos::new()?;
    Ok(())
}

/// The samples above are compile-checked by virtue of being in this file; this
/// test just gives the harness something to report.
#[test]
fn readme_samples_compile() {}
