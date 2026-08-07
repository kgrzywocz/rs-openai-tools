//! OpenAI Batch API Response Types
//!
//! This module defines the response types for the OpenAI Batch API.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// The status of a batch job.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum BatchStatus {
    /// The batch is being validated.
    Validating,
    /// The batch failed validation.
    Failed,
    /// The batch is currently being processed.
    InProgress,
    /// The batch is being finalized.
    Finalizing,
    /// The batch has been completed successfully.
    Completed,
    /// The batch has expired (24h window passed).
    Expired,
    /// The batch is being cancelled.
    Cancelling,
    /// The batch has been cancelled.
    Cancelled,
    /// A status this version of the library does not know about.
    ///
    /// OpenAI can add lifecycle states at any time; capturing the raw value
    /// keeps the surrounding response parseable.
    #[serde(untagged)]
    Other(String),
}

/// Counts of requests in different states within the batch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestCounts {
    /// Total number of requests in the batch.
    pub total: u32,
    /// Number of requests that have been completed successfully.
    pub completed: u32,
    /// Number of requests that have failed.
    pub failed: u32,
}

/// An error that occurred during batch processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchError {
    /// A machine-readable error code.
    pub code: String,
    /// A human-readable error message.
    pub message: String,
    /// The parameter related to the error, if any.
    pub param: Option<String>,
    /// The line number in the input file, if applicable.
    pub line: Option<u32>,
}

/// A collection of errors from batch processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchErrors {
    /// The type of object (always "list").
    pub object: Option<String>,
    /// The list of errors.
    pub data: Vec<BatchError>,
}

/// A batch object representing an async batch job.
///
/// The Batch API allows you to send asynchronous groups of requests
/// with 50% lower costs and higher rate limits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchObject {
    /// The unique identifier for the batch.
    pub id: String,

    /// The object type (always "batch").
    pub object: String,

    /// The API endpoint used for the batch.
    pub endpoint: String,

    /// Any errors that occurred during batch processing.
    pub errors: Option<BatchErrors>,

    /// The ID of the input file containing the requests.
    pub input_file_id: String,

    /// The time window for batch completion (e.g., "24h").
    pub completion_window: String,

    /// The current status of the batch.
    pub status: BatchStatus,

    /// The ID of the output file containing the results.
    pub output_file_id: Option<String>,

    /// The ID of the error file containing failed requests.
    pub error_file_id: Option<String>,

    /// The Unix timestamp when the batch was created.
    pub created_at: i64,

    /// The Unix timestamp when processing started.
    pub in_progress_at: Option<i64>,

    /// The Unix timestamp when the batch expires.
    pub expires_at: Option<i64>,

    /// The Unix timestamp when finalization started.
    pub finalizing_at: Option<i64>,

    /// The Unix timestamp when the batch completed.
    pub completed_at: Option<i64>,

    /// The Unix timestamp when the batch failed.
    pub failed_at: Option<i64>,

    /// The Unix timestamp when the batch expired.
    pub expired_at: Option<i64>,

    /// The Unix timestamp when cancellation started.
    pub cancelling_at: Option<i64>,

    /// The Unix timestamp when the batch was cancelled.
    pub cancelled_at: Option<i64>,

    /// Counts of requests in different states.
    pub request_counts: Option<RequestCounts>,

    /// User-defined metadata attached to the batch.
    pub metadata: Option<HashMap<String, String>>,
}

/// Response for listing batch jobs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchListResponse {
    /// The object type (always "list").
    pub object: String,

    /// The list of batch objects.
    pub data: Vec<BatchObject>,

    /// The ID of the first batch in the list.
    pub first_id: Option<String>,

    /// The ID of the last batch in the list.
    pub last_id: Option<String>,

    /// Whether there are more batches to retrieve.
    pub has_more: bool,
}
