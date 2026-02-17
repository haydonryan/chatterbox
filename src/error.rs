use pyo3::PyErr;
use std::fmt;

/// Error type for Chatterbox TTS operations
#[derive(Debug)]
pub enum ChatterboxError {
    /// Python error from PyO3
    Python(PyErr),
    /// IO error
    Io(std::io::Error),
    /// Safetensors error
    Safetensors(String),
    /// Missing model weight
    MissingWeight(String),
    /// Shape mismatch when loading weights
    ShapeMismatch(String),
    /// Not implemented yet
    NotImplemented(&'static str),
    /// Tensor conversion error
    TensorConversion(String),
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
            ChatterboxError::Io(e) => write!(f, "IO error: {}", e),
            ChatterboxError::Safetensors(e) => write!(f, "Safetensors error: {}", e),
            ChatterboxError::MissingWeight(name) => write!(f, "Missing weight: {}", name),
            ChatterboxError::ShapeMismatch(msg) => write!(f, "Shape mismatch: {}", msg),
            ChatterboxError::NotImplemented(msg) => write!(f, "Not implemented: {}", msg),
            ChatterboxError::TensorConversion(msg) => write!(f, "Tensor conversion error: {}", msg),
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

impl From<std::io::Error> for ChatterboxError {
    fn from(err: std::io::Error) -> Self {
        ChatterboxError::Io(err)
    }
}

pub type Result<T> = std::result::Result<T, ChatterboxError>;
