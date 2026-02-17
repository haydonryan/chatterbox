//! T3 inference module.
//!
//! This module contains the inference-related components for the T3 model:
//!
//! - `AlignmentStreamAnalyzer` - Attention-based hallucination detection
//! - `T3HuggingfaceBackend` - HuggingFace-compatible inference backend

mod alignment_stream_analyzer;
mod t3_hf_backend;

pub use alignment_stream_analyzer::{AlignmentAnalysisResult, AlignmentStreamAnalyzer, LLAMA_ALIGNED_HEADS};
pub use t3_hf_backend::{T3HuggingfaceBackend, T3ForwardOutput};
