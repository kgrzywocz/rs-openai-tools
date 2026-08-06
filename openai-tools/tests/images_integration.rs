//! Integration tests for the OpenAI Images API.
//!
//! These tests require a valid OPENAI_API_KEY environment variable.
//! Run with: cargo test --test images_integration
//!
//! Note: Image generation tests incur API costs. These tests generate minimal
//! images to reduce costs while still validating the API integration.

use openai_tools::images::request::{GenerateOptions, ImageModel, ImageQuality, ImageSize, ImageStyle, Images, ResponseFormat};

/// Test basic image generation with GPT Image 1 Mini.
///
/// OpenAI retired `dall-e-2`/`dall-e-3`, so these tests target the cheapest
/// live model and use `low` quality to keep costs down.
#[tokio::test]
async fn test_generate_image_gpt_image_mini() {
    let images = Images::new().expect("Should create Images client");

    let options = GenerateOptions {
        model: Some(ImageModel::GptImage1Mini),
        size: Some(ImageSize::Size1024x1024),
        quality: Some(ImageQuality::Low),
        ..Default::default()
    };

    let response = images.generate("A simple red circle on white background", options).await.expect("Should generate image");

    assert!(response.created > 0, "Created timestamp should be positive");
    assert_eq!(response.data.len(), 1, "Should generate one image");

    // The GPT Image models always return base64 data rather than a URL.
    let image = &response.data[0];
    assert!(image.b64_json.is_some(), "GPT Image models return base64 data");

    let bytes = image.as_bytes().expect("Should have bytes").expect("Should decode base64");
    assert!(!bytes.is_empty(), "Decoded image should not be empty");
    println!("Generated {} bytes", bytes.len());
}

/// Test generation with a portrait size.
#[tokio::test]
async fn test_generate_image_portrait_size() {
    let images = Images::new().expect("Should create Images client");

    let options = GenerateOptions {
        model: Some(ImageModel::GptImage1Mini),
        size: Some(ImageSize::Size1024x1536),
        quality: Some(ImageQuality::Low),
        ..Default::default()
    };

    let response = images.generate("A blue square", options).await.expect("Should generate a portrait image");

    assert_eq!(response.data.len(), 1);
    assert!(response.data[0].b64_json.is_some());
}

/// The GPT Image models reject `response_format` outright - the library must
/// surface that as an error rather than silently dropping it.
#[tokio::test]
async fn test_gpt_image_rejects_response_format() {
    let images = Images::new().expect("Should create Images client");

    let options = GenerateOptions {
        model: Some(ImageModel::GptImage1Mini),
        response_format: Some(ResponseFormat::B64Json),
        size: Some(ImageSize::Size1024x1024),
        quality: Some(ImageQuality::Low),
        ..Default::default()
    };

    let result = images.generate("A small green dot", options).await;

    assert!(result.is_err(), "GPT Image models do not accept response_format");
    println!("Error (expected): {:?}", result.err());
}

/// Test ImageModel enum functionality.
#[test]
fn test_image_model_enum() {
    assert_eq!(ImageModel::DallE2.as_str(), "dall-e-2");
    assert_eq!(ImageModel::DallE3.as_str(), "dall-e-3");
    assert_eq!(ImageModel::GptImage1.as_str(), "gpt-image-1");

    assert_eq!(format!("{}", ImageModel::DallE3), "dall-e-3");
    // The default must name a model that still resolves on api.openai.com.
    assert_eq!(ImageModel::default(), ImageModel::GptImage1);
}

/// Test ImageSize enum functionality.
#[test]
fn test_image_size_enum() {
    assert_eq!(ImageSize::Size256x256.as_str(), "256x256");
    assert_eq!(ImageSize::Size512x512.as_str(), "512x512");
    assert_eq!(ImageSize::Size1024x1024.as_str(), "1024x1024");
    assert_eq!(ImageSize::Size1792x1024.as_str(), "1792x1024");
    assert_eq!(ImageSize::Size1024x1792.as_str(), "1024x1792");

    assert_eq!(ImageSize::default(), ImageSize::Size1024x1024);
}

/// Test ImageQuality and ImageStyle enums.
#[test]
fn test_image_quality_and_style_enums() {
    assert_eq!(ImageQuality::Standard.as_str(), "standard");
    assert_eq!(ImageQuality::Hd.as_str(), "hd");
    assert_eq!(ImageQuality::default(), ImageQuality::Standard);

    assert_eq!(ImageStyle::Vivid.as_str(), "vivid");
    assert_eq!(ImageStyle::Natural.as_str(), "natural");
    assert_eq!(ImageStyle::default(), ImageStyle::Vivid);
}

/// Test ResponseFormat enum.
#[test]
fn test_response_format_enum() {
    assert_eq!(ResponseFormat::Url.as_str(), "url");
    assert_eq!(ResponseFormat::B64Json.as_str(), "b64_json");
    assert_eq!(ResponseFormat::default(), ResponseFormat::Url);
}
