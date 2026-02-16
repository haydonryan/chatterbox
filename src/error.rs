use pyo3::PyErr;
use std::fmt;

/// Error type for Chatterbox TTS operations
#[derive(Debug)]
pub enum ChatterboxError {
    /// Python error from PyO3
    Python(PyErr),
    /// Conditionals not prepared before generate
    ConditionalsNotPrepared,
    /// Invalid device specification
    InvalidDevice(String),
    /// Audio file not found
    AudioFileNotFound(String),
}

impl fmt::Display for ChatterboxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChatterboxError::Python(e) => write!(f, "Python error: {}", e),
            ChatterboxError::ConditionalsNotPrepared => {
                write!(f, "Conditionals not prepared. Call prepare_conditionals() or provide audio_prompt_path")
            }
            ChatterboxError::InvalidDevice(d) => write!(f, "Invalid device: {}", d),
            ChatterboxError::AudioFileNotFound(p) => write!(f, "Audio file not found: {}", p),
        }
    }
}

impl std::error::Error for ChatterboxError {}

impl From<PyErr> for ChatterboxError {
    fn from(err: PyErr) -> Self {
        ChatterboxError::Python(err)
    }
}

pub type Result<T> = std::result::Result<T, ChatterboxError>;
