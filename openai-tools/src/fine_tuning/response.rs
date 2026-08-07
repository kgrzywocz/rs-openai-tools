//! OpenAI Fine-tuning API Response Types
//!
//! This module defines the response types for the OpenAI Fine-tuning API.

use serde::{Deserialize, Serialize};

/// The status of a fine-tuning job.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum FineTuningJobStatus {
    /// Files are being validated.
    ValidatingFiles,
    /// Job is queued for processing.
    Queued,
    /// Job is currently running.
    Running,
    /// Job completed successfully.
    Succeeded,
    /// Job failed.
    Failed,
    /// Job was cancelled.
    Cancelled,
    /// A status this version of the library does not know about.
    ///
    /// OpenAI can add lifecycle states at any time; capturing the raw value
    /// keeps the surrounding response parseable.
    #[serde(untagged)]
    Other(String),
}

/// Hyperparameters used for fine-tuning.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Hyperparameters {
    /// Number of epochs to train for.
    /// Can be "auto" in API but represented as Option here.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_epochs: Option<u32>,

    /// Batch size for training.
    /// Can be "auto" in API but represented as Option here.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub batch_size: Option<u32>,

    /// Learning rate multiplier.
    /// Can be "auto" in API but represented as Option here.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub learning_rate_multiplier: Option<f64>,
}

/// Error information for a failed fine-tuning job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FineTuningError {
    /// A machine-readable error code.
    pub code: String,
    /// A human-readable error message.
    pub message: String,
    /// The parameter related to the error, if any.
    pub param: Option<String>,
}

/// Integration configuration (e.g., Weights & Biases).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Integration {
    /// The type of integration (e.g., "wandb").
    #[serde(rename = "type")]
    pub integration_type: String,

    /// Integration-specific settings.
    #[serde(flatten)]
    pub settings: serde_json::Value,
}

/// Method configuration for fine-tuning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MethodConfig {
    /// The type of fine-tuning method (e.g., "supervised", "dpo").
    #[serde(rename = "type")]
    pub method_type: String,

    /// Supervised fine-tuning configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supervised: Option<SupervisedConfig>,

    /// DPO fine-tuning configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dpo: Option<DpoConfig>,
}

/// Configuration for supervised fine-tuning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupervisedConfig {
    /// Hyperparameters for supervised fine-tuning.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hyperparameters: Option<Hyperparameters>,
}

/// Configuration for DPO (Direct Preference Optimization) fine-tuning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DpoConfig {
    /// Hyperparameters for DPO fine-tuning.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hyperparameters: Option<Hyperparameters>,
}

/// A fine-tuning job object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FineTuningJob {
    /// The unique identifier for the job.
    pub id: String,

    /// The object type (always "fine_tuning.job").
    pub object: String,

    /// The base model being fine-tuned.
    pub model: String,

    /// The Unix timestamp when the job was created.
    pub created_at: i64,

    /// The Unix timestamp when the job finished.
    pub finished_at: Option<i64>,

    /// The name of the fine-tuned model (available after success).
    pub fine_tuned_model: Option<String>,

    /// The organization ID that owns the job.
    pub organization_id: String,

    /// Array of result file IDs.
    pub result_files: Vec<String>,

    /// The current status of the job.
    pub status: FineTuningJobStatus,

    /// The validation file ID, if provided.
    pub validation_file: Option<String>,

    /// The training file ID.
    pub training_file: String,

    /// The hyperparameters used for training.
    pub hyperparameters: Hyperparameters,

    /// The number of tokens trained on (null while running).
    pub trained_tokens: Option<u64>,

    /// Error information if the job failed.
    pub error: Option<FineTuningError>,

    /// The seed used for training.
    pub seed: u64,

    /// Estimated finish time (Unix timestamp).
    pub estimated_finish: Option<i64>,

    /// Configured integrations.
    pub integrations: Option<Vec<Integration>>,

    /// The fine-tuning method used.
    pub method: Option<MethodConfig>,

    /// User-provided suffix for the model name.
    pub user_provided_suffix: Option<String>,
}

/// Response for listing fine-tuning jobs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FineTuningJobListResponse {
    /// The object type (always "list").
    pub object: String,

    /// The list of fine-tuning jobs.
    pub data: Vec<FineTuningJob>,

    /// Whether there are more jobs to retrieve.
    pub has_more: bool,
}

/// A fine-tuning event object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FineTuningEvent {
    /// The unique identifier for the event.
    pub id: String,

    /// The object type (always "fine_tuning.job.event").
    pub object: String,

    /// The Unix timestamp when the event was created.
    pub created_at: i64,

    /// The level of the event ("info", "warn", "error").
    pub level: String,

    /// The event message.
    pub message: String,

    /// Additional data associated with the event.
    pub data: Option<serde_json::Value>,

    /// The type of event.
    #[serde(rename = "type")]
    pub event_type: String,
}

/// Response for listing fine-tuning events.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FineTuningEventListResponse {
    /// The object type (always "list").
    pub object: String,

    /// The list of events.
    pub data: Vec<FineTuningEvent>,

    /// Whether there are more events to retrieve.
    pub has_more: bool,
}

/// Metrics for a fine-tuning checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetrics {
    /// The training step number.
    pub step: u32,

    /// The training loss at this checkpoint.
    pub train_loss: f64,

    /// The mean token accuracy during training.
    pub train_mean_token_accuracy: f64,

    /// The validation loss at this checkpoint.
    pub valid_loss: Option<f64>,

    /// The mean token accuracy during validation.
    pub valid_mean_token_accuracy: Option<f64>,

    /// The full validation loss.
    pub full_valid_loss: Option<f64>,

    /// The full validation mean token accuracy.
    pub full_valid_mean_token_accuracy: Option<f64>,
}

/// A fine-tuning checkpoint object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FineTuningCheckpoint {
    /// The unique identifier for the checkpoint.
    pub id: String,

    /// The object type (always "fine_tuning.job.checkpoint").
    pub object: String,

    /// The Unix timestamp when the checkpoint was created.
    pub created_at: i64,

    /// The ID of the fine-tuning job.
    pub fine_tuning_job_id: String,

    /// The name of the checkpoint model.
    pub fine_tuned_model_checkpoint: String,

    /// The step number at which this checkpoint was created.
    pub step_number: u32,

    /// Training metrics at this checkpoint.
    pub metrics: CheckpointMetrics,
}

/// Response for listing fine-tuning checkpoints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FineTuningCheckpointListResponse {
    /// The object type (always "list").
    pub object: String,

    /// The list of checkpoints.
    pub data: Vec<FineTuningCheckpoint>,

    /// The ID of the first checkpoint in the list.
    pub first_id: Option<String>,

    /// The ID of the last checkpoint in the list.
    pub last_id: Option<String>,

    /// Whether there are more checkpoints to retrieve.
    pub has_more: bool,
}
