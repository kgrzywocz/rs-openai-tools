//! OpenAI Embeddings API Response Types
//!
//! This module provides data structures for deserializing responses from the
//! OpenAI Embeddings API. It includes all necessary types to handle complete
//! API responses, including embedding vectors, metadata, and usage statistics.
//!
//! # Response Structure
//!
//! The main response structure contains:
//! - `object`: Always "list" for embedding responses
//! - `data`: Array of embedding objects with vectors and indices
//! - `model`: The model used for generating embeddings
//! - `usage`: Token usage information
//!
//! # Examples
//!
//! ## Parsing a Simple Embedding Response
//!
//! ```rust,no_run
//! use openai_tools::embedding::response::Response;
//!
//! let json = r#"{
//!     "object": "list",
//!     "data": [{
//!         "object": "embedding",
//!         "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
//!         "index": 0
//!     }],
//!     "model": "text-embedding-3-small",
//!     "usage": {
//!         "prompt_tokens": 5,
//!         "total_tokens": 5
//!     }
//! }"#;
//!
//! let response: Response = serde_json::from_str(json).unwrap();
//! assert_eq!(response.object, "list");
//! assert_eq!(response.data.len(), 1);
//!
//! let embedding = response.data[0].embedding.as_1d().unwrap();
//! assert_eq!(embedding.len(), 5);
//! ```
//!
//! ## Working with Batch Embeddings
//!
//! ```rust,no_run
//! use openai_tools::embedding::response::Response;
//!
//! let json = r#"{
//!     "object": "list",
//!     "data": [
//!         {"object": "embedding", "embedding": [0.1, 0.2], "index": 0},
//!         {"object": "embedding", "embedding": [0.3, 0.4], "index": 1},
//!         {"object": "embedding", "embedding": [0.5, 0.6], "index": 2}
//!     ],
//!     "model": "text-embedding-3-small",
//!     "usage": {"prompt_tokens": 15, "total_tokens": 15}
//! }"#;
//!
//! let response: Response = serde_json::from_str(json).unwrap();
//!
//! for data in &response.data {
//!     let vector = data.embedding.as_1d().unwrap();
//!     println!("Index {}: {:?}", data.index, vector);
//! }
//! ```
//!
//! # Embedding Dimensions
//!
//! The [`Embedding`] enum supports multiple dimensionalities:
//! - **1D**: Standard embedding vector (most common)
//! - **2D**: Matrix of embeddings
//! - **3D**: Tensor of embeddings
//!
//! Use the `is_*d()` and `as_*d()` methods to check and access the appropriate dimension.

use serde::Deserialize;

/// Embedding vector that supports multiple dimensionalities.
///
/// The OpenAI API typically returns 1D vectors, but this enum provides
/// flexibility for future API changes or custom use cases. The `#[serde(untagged)]`
/// attribute allows automatic deserialization based on the JSON structure.
///
/// # Variants
///
/// * `OneDim` - Standard 1D embedding vector (e.g., `[0.1, 0.2, 0.3]`)
/// * `TwoDim` - 2D embedding matrix (e.g., `[[0.1, 0.2], [0.3, 0.4]]`)
/// * `ThreeDim` - 3D embedding tensor
///
/// # Examples
///
/// ```rust
/// use openai_tools::embedding::response::Embedding;
///
/// // Parse a 1D embedding
/// let json = r#"[0.1, 0.2, 0.3]"#;
/// let embedding: Embedding = serde_json::from_str(json).unwrap();
///
/// assert!(embedding.is_1d());
/// let vector = embedding.as_1d().unwrap();
/// assert_eq!(vector.len(), 3);
/// ```
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum Embedding {
    /// 1D embedding: `Vec<f32>`
    OneDim(Vec<f32>),
    /// 2D embedding: `Vec<Vec<f32>>`
    TwoDim(Vec<Vec<f32>>),
    /// 3D embedding: `Vec<Vec<Vec<f32>>>`
    ThreeDim(Vec<Vec<Vec<f32>>>),
}

impl Embedding {
    /// Returns the embedding as a 1D vector if it is 1D, otherwise returns None.
    ///
    /// # Returns
    ///
    /// * `Some(&Vec<f32>)` - Reference to the 1D vector if the embedding is 1D
    /// * `None` - If the embedding is 2D or 3D
    ///
    /// # Example
    ///
    /// ```rust
    /// use openai_tools::embedding::response::Embedding;
    ///
    /// let embedding: Embedding = serde_json::from_str("[0.1, 0.2, 0.3]").unwrap();
    /// if let Some(vec) = embedding.as_1d() {
    ///     println!("Dimension: {}", vec.len());
    /// }
    /// ```
    pub fn as_1d(&self) -> Option<&Vec<f32>> {
        match self {
            Embedding::OneDim(v) => Some(v),
            _ => None,
        }
    }

    /// Returns the embedding as a 2D vector if it is 2D, otherwise returns None.
    ///
    /// # Returns
    ///
    /// * `Some(&Vec<Vec<f32>>)` - Reference to the 2D vector if the embedding is 2D
    /// * `None` - If the embedding is 1D or 3D
    pub fn as_2d(&self) -> Option<&Vec<Vec<f32>>> {
        match self {
            Embedding::TwoDim(v) => Some(v),
            _ => None,
        }
    }

    /// Returns the embedding as a 3D vector if it is 3D, otherwise returns None.
    ///
    /// # Returns
    ///
    /// * `Some(&Vec<Vec<Vec<f32>>>)` - Reference to the 3D vector if the embedding is 3D
    /// * `None` - If the embedding is 1D or 2D
    pub fn as_3d(&self) -> Option<&Vec<Vec<Vec<f32>>>> {
        match self {
            Embedding::ThreeDim(v) => Some(v),
            _ => None,
        }
    }

    /// Returns true if the embedding is 1D.
    ///
    /// # Example
    ///
    /// ```rust
    /// use openai_tools::embedding::response::Embedding;
    ///
    /// let embedding: Embedding = serde_json::from_str("[0.1, 0.2]").unwrap();
    /// assert!(embedding.is_1d());
    /// ```
    pub fn is_1d(&self) -> bool {
        matches!(self, Embedding::OneDim(_))
    }

    /// Returns true if the embedding is 2D.
    pub fn is_2d(&self) -> bool {
        matches!(self, Embedding::TwoDim(_))
    }

    /// Returns true if the embedding is 3D.
    pub fn is_3d(&self) -> bool {
        matches!(self, Embedding::ThreeDim(_))
    }
}

/// Single embedding data item from the API response.
///
/// Each input text produces one `EmbeddingData` object containing
/// the embedding vector and its index in the input array.
///
/// # Fields
///
/// * `object` - Type identifier, always "embedding"
/// * `embedding` - The numerical vector representation of the input text
/// * `index` - The position of this embedding in the input array (0-indexed)
#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingData {
    /// Type identifier for this object (always "embedding")
    pub object: String,
    /// The embedding vector generated from the input text
    pub embedding: Embedding,
    /// Index of this embedding in the input array
    pub index: usize,
}

/// Token usage statistics for the embedding request.
///
/// Provides information about the number of tokens processed,
/// which is useful for cost estimation and rate limiting.
///
/// # Note
///
/// Unlike chat completions, embeddings only count prompt tokens
/// (the input text) since there are no completion tokens generated.
#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingUsage {
    /// Number of tokens in the input text(s)
    pub prompt_tokens: usize,
    /// Total tokens processed (equal to prompt_tokens for embeddings)
    pub total_tokens: usize,
}

/// Complete response from the OpenAI Embeddings API.
///
/// This is the top-level structure returned by the API, containing
/// all embedding data and metadata about the request.
///
/// # Example
///
/// ```rust,no_run
/// use openai_tools::embedding::request::Embedding;
/// use openai_tools::common::models::EmbeddingModel;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let mut embedding = Embedding::new()?;
///     let response = embedding
///         .model(EmbeddingModel::TextEmbedding3Small)
///         .input_text("Hello, world!")
///         .embed()
///         .await?;
///
///     println!("Model: {}", response.model);
///     println!("Embeddings: {}", response.data.len());
///     println!("Tokens used: {}", response.usage.total_tokens);
///     Ok(())
/// }
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct Response {
    /// Type identifier for this response (always "list")
    pub object: String,
    /// Array of embedding data, one per input text
    pub data: Vec<EmbeddingData>,
    /// The model used to generate the embeddings
    pub model: String,
    /// Token usage statistics for this request
    pub usage: EmbeddingUsage,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_variants() {
        // Test 1D embedding
        let json_1d = r#"[0.1, 0.2, 0.3]"#;
        let embedding_1d: Embedding = serde_json::from_str(json_1d).unwrap();
        assert!(embedding_1d.is_1d());
        assert!(!embedding_1d.is_2d());
        assert!(!embedding_1d.is_3d());
        assert_eq!(embedding_1d.as_1d(), Some(&vec![0.1, 0.2, 0.3]));
        assert_eq!(embedding_1d.as_2d(), None);
        assert_eq!(embedding_1d.as_3d(), None);

        // Test 2D embedding
        let json_2d = r#"[[0.1, 0.2], [0.3, 0.4]]"#;
        let embedding_2d: Embedding = serde_json::from_str(json_2d).unwrap();
        assert!(!embedding_2d.is_1d());
        assert!(embedding_2d.is_2d());
        assert!(!embedding_2d.is_3d());
        assert_eq!(embedding_2d.as_1d(), None);
        assert_eq!(embedding_2d.as_2d(), Some(&vec![vec![0.1, 0.2], vec![0.3, 0.4]]));
        assert_eq!(embedding_2d.as_3d(), None);

        // Test 3D embedding
        let json_3d = r#"[[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]"#;
        let embedding_3d: Embedding = serde_json::from_str(json_3d).unwrap();
        assert!(!embedding_3d.is_1d());
        assert!(!embedding_3d.is_2d());
        assert!(embedding_3d.is_3d());
        assert_eq!(embedding_3d.as_1d(), None);
        assert_eq!(embedding_3d.as_2d(), None);
        assert_eq!(embedding_3d.as_3d(), Some(&vec![vec![vec![0.1, 0.2], vec![0.3, 0.4]], vec![vec![0.5, 0.6], vec![0.7, 0.8]]]));
    }
}
