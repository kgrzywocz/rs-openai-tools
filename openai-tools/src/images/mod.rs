//! # Images Module
//!
//! This module provides functionality for interacting with the OpenAI Images API.
//! It allows you to generate, edit, and create variations of images using DALL-E models.
//!
//! ## Key Features
//!
//! - **Image Generation**: Create images from text prompts
//! - **Image Editing**: Modify existing images with masks and prompts
//! - **Image Variations**: Create variations of existing images
//! - **Multiple Models**: Support for DALL-E 2, DALL-E 3, and GPT Image models
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use openai_tools::images::request::{Images, GenerateOptions};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create an Images client
//!     let images = Images::new()?;
//!
//!     // Generate an image
//!     let response = images.generate("A beautiful sunset", GenerateOptions::default()).await?;
//!
//!     if let Some(url) = &response.data[0].url {
//!         println!("Image URL: {}", url);
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Usage Examples
//!
//! ### Generate with DALL-E 3
//!
//! ```rust,no_run
//! use openai_tools::images::request::{
//!     Images, GenerateOptions, ImageModel, ImageSize, ImageQuality, ImageStyle
//! };
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let images = Images::new()?;
//!
//!     let options = GenerateOptions {
//!         model: Some(ImageModel::DallE3),
//!         size: Some(ImageSize::Size1792x1024), // Landscape
//!         quality: Some(ImageQuality::Hd),
//!         style: Some(ImageStyle::Natural),
//!         ..Default::default()
//!     };
//!
//!     let response = images.generate("A peaceful mountain lake at sunrise", options).await?;
//!
//!     // DALL-E 3 may revise the prompt for better results
//!     if let Some(revised) = &response.data[0].revised_prompt {
//!         println!("Revised prompt: {}", revised);
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! ### Get Base64 Image Data
//!
//! ```rust,no_run
//! use openai_tools::images::request::{Images, GenerateOptions, ResponseFormat};
//! use std::fs;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let images = Images::new()?;
//!
//!     let options = GenerateOptions {
//!         response_format: Some(ResponseFormat::B64Json),
//!         ..Default::default()
//!     };
//!
//!     let response = images.generate("A cute robot", options).await?;
//!
//!     // Decode and save the image
//!     if let Some(bytes_result) = response.data[0].as_bytes() {
//!         let bytes = bytes_result?;
//!         fs::write("robot.png", bytes)?;
//!         println!("Saved image to robot.png");
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! ### Edit an Image
//!
//! ```rust,no_run
//! use openai_tools::images::request::{Images, EditOptions, ImageModel};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let images = Images::new()?;
//!
//!     let options = EditOptions {
//!         model: Some(ImageModel::DallE2),
//!         mask: Some("mask.png".to_string()), // Transparent areas will be edited
//!         n: Some(2),
//!         ..Default::default()
//!     };
//!
//!     let response = images.edit("original.png", "Add a red balloon", options).await?;
//!
//!     for (i, image) in response.data.iter().enumerate() {
//!         println!("Edit {}: {:?}", i + 1, image.url);
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! ### Create Variations
//!
//! ```rust,no_run
//! use openai_tools::images::request::{Images, VariationOptions, ImageModel, ImageSize};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let images = Images::new()?;
//!
//!     // Variations only work with DALL-E 2
//!     let options = VariationOptions {
//!         model: Some(ImageModel::DallE2),
//!         n: Some(4),
//!         size: Some(ImageSize::Size512x512),
//!         ..Default::default()
//!     };
//!
//!     let response = images.variation("photo.png", options).await?;
//!
//!     println!("Created {} variations", response.data.len());
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Model Comparison
//!
//! | Feature | DALL-E 2 | DALL-E 3 |
//! |---------|----------|----------|
//! | Sizes | 256x256, 512x512, 1024x1024 | 1024x1024, 1792x1024, 1024x1792 |
//! | Quality | Standard only | Standard, HD |
//! | Style | N/A | Vivid, Natural |
//! | Variations | Yes | No |
//! | Editing | Yes | No |
//! | N (count) | 1-10 | 1 only |
//! | Prompt revision | No | Yes |
//!
//! ## Response Format
//!
//! The API returns either URLs (valid for 60 minutes) or base64-encoded image data:
//! - Use `ResponseFormat::Url` (default) for temporary URLs
//! - Use `ResponseFormat::B64Json` to receive image data directly

pub mod request;
pub mod response;

#[cfg(test)]
mod tests {
    use crate::images::request::{ImageModel, ImageQuality, ImageSize, ImageStyle, ResponseFormat};
    use crate::images::response::ImageResponse;

    #[test]
    fn test_image_response_deserialization() {
        let json = r#"{
            "created": 1704067200,
            "data": [
                {
                    "url": "https://example.com/image.png"
                }
            ]
        }"#;

        let response: ImageResponse = serde_json::from_str(json).expect("Should deserialize ImageResponse");
        assert_eq!(response.created, 1704067200);
        assert_eq!(response.data.len(), 1);
        assert_eq!(response.data[0].url, Some("https://example.com/image.png".to_string()));
    }

    #[test]
    fn test_image_response_with_b64() {
        let json = r#"{
            "created": 1704067200,
            "data": [
                {
                    "b64_json": "aGVsbG8gd29ybGQ="
                }
            ]
        }"#;

        let response: ImageResponse = serde_json::from_str(json).expect("Should deserialize ImageResponse with b64");
        assert!(response.data[0].has_b64());
        assert!(!response.data[0].has_url());

        // Verify decoding
        let bytes = response.data[0].as_bytes().expect("Should have bytes").expect("Should decode");
        assert_eq!(bytes, b"hello world");
    }

    #[test]
    fn test_image_response_with_revised_prompt() {
        let json = r#"{
            "created": 1704067200,
            "data": [
                {
                    "url": "https://example.com/image.png",
                    "revised_prompt": "A detailed painting of a sunset over calm waters"
                }
            ]
        }"#;

        let response: ImageResponse = serde_json::from_str(json).expect("Should deserialize with revised_prompt");
        assert_eq!(response.data[0].revised_prompt, Some("A detailed painting of a sunset over calm waters".to_string()));
    }

    #[test]
    fn test_image_model_serialization() {
        assert_eq!(serde_json::to_string(&ImageModel::DallE2).unwrap(), "\"dall-e-2\"");
        assert_eq!(serde_json::to_string(&ImageModel::DallE3).unwrap(), "\"dall-e-3\"");
        assert_eq!(serde_json::to_string(&ImageModel::GptImage1).unwrap(), "\"gpt-image-1\"");
    }

    #[test]
    fn test_image_size_serialization() {
        assert_eq!(serde_json::to_string(&ImageSize::Size256x256).unwrap(), "\"256x256\"");
        assert_eq!(serde_json::to_string(&ImageSize::Size1024x1024).unwrap(), "\"1024x1024\"");
        assert_eq!(serde_json::to_string(&ImageSize::Size1792x1024).unwrap(), "\"1792x1024\"");
    }

    #[test]
    fn test_image_quality_serialization() {
        assert_eq!(serde_json::to_string(&ImageQuality::Standard).unwrap(), "\"standard\"");
        assert_eq!(serde_json::to_string(&ImageQuality::Hd).unwrap(), "\"hd\"");
    }

    #[test]
    fn test_image_style_serialization() {
        assert_eq!(serde_json::to_string(&ImageStyle::Vivid).unwrap(), "\"vivid\"");
        assert_eq!(serde_json::to_string(&ImageStyle::Natural).unwrap(), "\"natural\"");
    }

    #[test]
    fn test_response_format_serialization() {
        assert_eq!(serde_json::to_string(&ResponseFormat::Url).unwrap(), "\"url\"");
        assert_eq!(serde_json::to_string(&ResponseFormat::B64Json).unwrap(), "\"b64_json\"");
    }

    #[test]
    fn test_defaults() {
        // DALL-E was retired on api.openai.com, so the default names a live model.
        assert_eq!(ImageModel::default(), ImageModel::GptImage1);
        assert_eq!(ImageSize::default(), ImageSize::Size1024x1024);
        assert_eq!(ImageQuality::default(), ImageQuality::Standard);
        assert_eq!(ImageStyle::default(), ImageStyle::Vivid);
        assert_eq!(ResponseFormat::default(), ResponseFormat::Url);
    }

    #[test]
    fn test_multiple_images_response() {
        let json = r#"{
            "created": 1704067200,
            "data": [
                {"url": "https://example.com/image1.png"},
                {"url": "https://example.com/image2.png"},
                {"url": "https://example.com/image3.png"}
            ]
        }"#;

        let response: ImageResponse = serde_json::from_str(json).expect("Should deserialize multiple images");
        assert_eq!(response.data.len(), 3);
    }
}
