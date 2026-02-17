//! Rust wrapper for the AlignmentStreamAnalyzer Python class.
//!
//! AlignmentStreamAnalyzer exploits transformer attention patterns to detect
//! hallucinations during streaming TTS inference. It monitors alignment between
//! text and speech tokens to detect:
//! - False starts (hallucinations at the beginning)
//! - Long tails (hallucinations after text completion)
//! - Repetitions (repeating existing text content)
//! - Discontinuities (jumps in alignment position)

use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Result of alignment analysis for a single frame.
#[derive(Debug, Clone)]
pub struct AlignmentAnalysisResult {
    /// Was this frame detected as being part of a noisy beginning chunk with potential hallucinations?
    pub false_start: bool,
    /// Was this frame detected as being part of a long tail with potential hallucinations?
    pub long_tail: bool,
    /// Was this frame detected as repeating existing text content?
    pub repetition: bool,
    /// Was the alignment position of this frame too far from the previous frame?
    pub discontinuity: bool,
    /// Has inference reached the end of the text tokens?
    pub complete: bool,
    /// Approximate position in the text token sequence (for online timestamps).
    pub position: i64,
}

/// Heads in the LLaMA model that show alignment behavior.
/// Format: (layer_idx, head_idx)
pub const LLAMA_ALIGNED_HEADS: [(i32, i32); 3] = [(12, 15), (13, 11), (9, 2)];

/// Wrapper for the Python AlignmentStreamAnalyzer class.
///
/// This module exploits transformer attention patterns to perform online
/// integrity checks during streaming inference. It hooks into specified
/// attention layers and uses heuristics to determine alignment position,
/// repetition, and hallucinations.
pub struct AlignmentStreamAnalyzer {
    inner: Py<PyAny>,
}

impl AlignmentStreamAnalyzer {
    /// Create a new AlignmentStreamAnalyzer.
    ///
    /// # Arguments
    /// * `tfmr` - The transformer model (LlamaModel)
    /// * `text_tokens_slice` - Slice indices (start, end) for text tokens in the sequence
    /// * `alignment_layer_idx` - Which layer to use for alignment (default 9)
    /// * `eos_idx` - End of speech token index
    pub fn new(
        py: Python<'_>,
        tfmr: &Bound<'_, PyAny>,
        text_tokens_slice: (i64, i64),
        alignment_layer_idx: Option<i32>,
        eos_idx: i64,
    ) -> PyResult<Self> {
        let analyzer_mod = py.import("chatterbox.models.t3.inference.alignment_stream_analyzer")?;
        let analyzer_class = analyzer_mod.getattr("AlignmentStreamAnalyzer")?;

        let kwargs = PyDict::new(py);
        kwargs.set_item("text_tokens_slice", text_tokens_slice)?;
        kwargs.set_item("alignment_layer_idx", alignment_layer_idx.unwrap_or(9))?;
        kwargs.set_item("eos_idx", eos_idx)?;

        // queue is not used (passed as None)
        let instance = analyzer_class.call((tfmr, py.None()), Some(&kwargs))?;

        Ok(Self {
            inner: instance.into(),
        })
    }

    /// Get the underlying Python object.
    pub fn as_py(&self) -> &Py<PyAny> {
        &self.inner
    }

    /// Create from an existing Python AlignmentStreamAnalyzer object.
    pub fn from_py(obj: Py<PyAny>) -> Self {
        Self { inner: obj }
    }

    /// Process one frame and potentially modify logits to force EOS.
    ///
    /// This method:
    /// 1. Extracts alignment matrix chunk from attention weights
    /// 2. Updates alignment position tracking
    /// 3. Detects hallucination patterns (false starts, long tails, repetitions)
    /// 4. Modifies logits to force EOS if bad ending is detected
    ///
    /// # Arguments
    /// * `logits` - Current logits tensor (may be modified)
    /// * `next_token` - Optional last generated token for repetition tracking
    ///
    /// # Returns
    /// Modified logits tensor
    pub fn step(
        &self,
        py: Python<'_>,
        logits: &Bound<'_, PyAny>,
        next_token: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let kwargs = PyDict::new(py);
        if let Some(token) = next_token {
            kwargs.set_item("next_token", token)?;
        }
        let result = self.inner.call_method(py, "step", (logits,), Some(&kwargs))?;
        Ok(result)
    }

    /// Get current text position (approximate alignment position).
    pub fn text_position(&self, py: Python<'_>) -> PyResult<i64> {
        let pos = self.inner.getattr(py, "text_position")?;
        pos.extract(py)
    }

    /// Check if generation has started (past false start detection).
    pub fn has_started(&self, py: Python<'_>) -> PyResult<bool> {
        let started = self.inner.getattr(py, "started")?;
        started.extract(py)
    }

    /// Check if generation appears complete.
    pub fn is_complete(&self, py: Python<'_>) -> PyResult<bool> {
        let complete = self.inner.getattr(py, "complete")?;
        complete.extract(py)
    }

    /// Get the current frame position.
    pub fn current_frame_pos(&self, py: Python<'_>) -> PyResult<i64> {
        let pos = self.inner.getattr(py, "curr_frame_pos")?;
        pos.extract(py)
    }

    /// Get the EOS token index.
    pub fn eos_idx(&self, py: Python<'_>) -> PyResult<i64> {
        let idx = self.inner.getattr(py, "eos_idx")?;
        idx.extract(py)
    }
}
