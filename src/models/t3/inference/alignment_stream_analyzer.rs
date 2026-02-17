//! Pure Rust implementation of AlignmentStreamAnalyzer.
//!
//! This module exploits transformer attention patterns to detect hallucinations
//! during streaming TTS inference.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PySlice, PyTuple};

/// Result of alignment analysis for a single frame.
#[derive(Debug, Clone)]
pub struct AlignmentAnalysisResult {
    pub false_start: bool,
    pub long_tail: bool,
    pub repetition: bool,
    pub discontinuity: bool,
    pub complete: bool,
    pub position: i64,
}

/// Heads in the LLaMA model that show alignment behavior.
pub const LLAMA_ALIGNED_HEADS: [(i32, i32); 3] = [(12, 15), (13, 11), (9, 2)];

/// Pure Rust implementation of AlignmentStreamAnalyzer.
pub struct AlignmentStreamAnalyzer {
    text_tokens_slice: (isize, isize),
    eos_idx: isize,
    alignment: Py<PyAny>,
    curr_frame_pos: isize,
    text_position: isize,
    started: bool,
    started_at: Option<isize>,
    complete: bool,
    completed_at: Option<isize>,
    generated_tokens: Vec<i64>,
    last_aligned_attns: Vec<Option<Py<PyAny>>>,
    torch: Py<PyAny>,
}

impl AlignmentStreamAnalyzer {
    /// Create a new AlignmentStreamAnalyzer.
    pub fn new(
        py: Python<'_>,
        tfmr: &Bound<'_, PyAny>,
        text_tokens_slice: (i64, i64),
        _alignment_layer_idx: Option<i32>,
        eos_idx: i64,
    ) -> PyResult<Self> {
        let torch = py.import("torch")?;
        let (i, j) = (text_tokens_slice.0 as isize, text_tokens_slice.1 as isize);

        // Initialize alignment matrix as empty tensor
        let alignment = torch.call_method1("zeros", ((0, j - i),))?;

        // Initialize attention buffers
        let mut last_aligned_attns = Vec::with_capacity(LLAMA_ALIGNED_HEADS.len());
        for _ in 0..LLAMA_ALIGNED_HEADS.len() {
            last_aligned_attns.push(None);
        }

        // Enable output_attentions on the transformer config
        if let Ok(config) = tfmr.getattr("config") {
            if config.hasattr("output_attentions")? {
                config.setattr("output_attentions", true)?;
            }
        }

        Ok(Self {
            text_tokens_slice: (i, j),
            eos_idx: eos_idx as isize,
            alignment: alignment.unbind(),
            curr_frame_pos: 0,
            text_position: 0,
            started: false,
            started_at: None,
            complete: false,
            completed_at: None,
            generated_tokens: Vec::new(),
            last_aligned_attns,
            torch: torch.unbind().into(),
        })
    }

    /// Update attention buffers from model output.
    pub fn update_attentions(&mut self, py: Python<'_>, attentions: &Bound<'_, PyAny>) -> PyResult<()> {
        for (idx, &(layer_idx, head_idx)) in LLAMA_ALIGNED_HEADS.iter().enumerate() {
            let layer_attn = attentions.get_item(layer_idx as usize)?;
            let head_attn = layer_attn.get_item(0)?.get_item(head_idx as usize)?;
            self.last_aligned_attns[idx] = Some(head_attn.call_method0("cpu")?.unbind());
        }
        Ok(())
    }

    /// Process one frame and potentially modify logits to force EOS.
    pub fn step(
        &mut self,
        py: Python<'_>,
        logits: &Bound<'_, PyAny>,
        next_token: Option<i64>,
    ) -> PyResult<Py<PyAny>> {
        let torch = self.torch.bind(py);
        let (i, j) = self.text_tokens_slice;

        // Stack and average the aligned attention heads
        let valid_attns: Vec<_> = self.last_aligned_attns
            .iter()
            .filter_map(|a| a.as_ref())
            .collect();

        if valid_attns.is_empty() {
            return Ok(logits.clone().unbind());
        }

        let attns_list = pyo3::types::PyList::new(py, valid_attns)?;
        let stacked = torch.call_method1("stack", (attns_list,))?;
        let aligned_attn = stacked.call_method1("mean", (0,))?;

        // Extract alignment matrix chunk using proper slicing
        let a_chunk = if self.curr_frame_pos == 0 {
            // First chunk: [j:, i:j]
            let row_slice = PySlice::new(py, j, isize::MAX, 1);
            let col_slice = PySlice::new(py, i, j, 1);
            let idx = PyTuple::new(py, &[row_slice.as_any(), col_slice.as_any()])?;
            aligned_attn.get_item(idx)?.call_method0("clone")?.call_method0("cpu")?
        } else {
            // Subsequent chunks: [:, i:j]
            let row_slice = PySlice::new(py, 0isize, isize::MAX, 1);
            let col_slice = PySlice::new(py, i, j, 1);
            let idx = PyTuple::new(py, &[row_slice.as_any(), col_slice.as_any()])?;
            aligned_attn.get_item(idx)?.call_method0("clone")?.call_method0("cpu")?
        };

        // Monotonic masking: zero out positions after current frame
        let row_slice = PySlice::new(py, 0isize, isize::MAX, 1);
        let col_slice = PySlice::new(py, self.curr_frame_pos + 1, isize::MAX, 1);
        let mask_idx = PyTuple::new(py, &[row_slice.as_any(), col_slice.as_any()])?;
        a_chunk.set_item(mask_idx, 0)?;

        // Concatenate to alignment matrix
        let cat_list = pyo3::types::PyList::new(py, &[self.alignment.bind(py), &a_chunk])?;
        let cat_kwargs = PyDict::new(py);
        cat_kwargs.set_item("dim", 0)?;
        self.alignment = torch.call_method("cat", (cat_list,), Some(&cat_kwargs))?.unbind();

        let alignment = self.alignment.bind(py);
        let shape = alignment.getattr("shape")?;
        let t: isize = shape.get_item(0)?.extract()?;
        let s: isize = shape.get_item(1)?.extract()?;

        // Update position
        let last_row = a_chunk.get_item(-1)?;
        let cur_text_posn: isize = last_row.call_method0("argmax")?.extract()?;
        let discontinuity = !(-4 < cur_text_posn - self.text_position && cur_text_posn - self.text_position < 7);
        if !discontinuity {
            self.text_position = cur_text_posn;
        }

        // Detect false start: A[-2:, -2:].max() and A[:, :4].max()
        let row_slice = PySlice::new(py, -2isize, isize::MAX, 1);
        let col_slice = PySlice::new(py, -2isize, isize::MAX, 1);
        let idx = PyTuple::new(py, &[row_slice.as_any(), col_slice.as_any()])?;
        let last_two_section = alignment.get_item(idx)?;
        let max_last_two: f64 = last_two_section.call_method0("max")?.extract()?;

        let row_slice = PySlice::new(py, 0isize, isize::MAX, 1);
        let col_slice = PySlice::new(py, 0isize, 4isize, 1);
        let idx = PyTuple::new(py, &[row_slice.as_any(), col_slice.as_any()])?;
        let first_four_section = alignment.get_item(idx)?;
        let max_first_four: f64 = first_four_section.call_method0("max")?.extract()?;

        let false_start = !self.started && (max_last_two > 0.1 || max_first_four < 0.5);
        self.started = !false_start;
        if self.started && self.started_at.is_none() {
            self.started_at = Some(t);
        }

        // Is generation complete?
        self.complete = self.complete || self.text_position >= s - 3;
        if self.complete && self.completed_at.is_none() {
            self.completed_at = Some(t);
        }

        // Detect long tail
        let long_tail = if self.complete {
            if let Some(completed_at) = self.completed_at {
                let row_slice = PySlice::new(py, completed_at, isize::MAX, 1);
                let col_slice = PySlice::new(py, -3isize, isize::MAX, 1);
                let idx = PyTuple::new(py, &[row_slice.as_any(), col_slice.as_any()])?;
                let tail_section = alignment.get_item(idx)?;
                let sum_dim0 = tail_section.call_method1("sum", (0,))?;
                let max_val: f64 = sum_dim0.call_method0("max")?.extract()?;
                max_val >= 5.0
            } else {
                false
            }
        } else {
            false
        };

        // Detect alignment repetition
        let alignment_repetition = if self.complete {
            if let Some(completed_at) = self.completed_at {
                let row_slice = PySlice::new(py, completed_at, isize::MAX, 1);
                let col_slice = PySlice::new(py, 0isize, -5isize, 1);
                let idx = PyTuple::new(py, &[row_slice.as_any(), col_slice.as_any()])?;
                let section = alignment.get_item(idx)?;
                let max_dim1 = section.call_method1("max", (1,))?;
                let values = max_dim1.get_item(0)?;
                let sum_val: f64 = values.call_method0("sum")?.extract()?;
                sum_val > 5.0
            } else {
                false
            }
        } else {
            false
        };

        // Track generated tokens for repetition detection
        if let Some(token) = next_token {
            self.generated_tokens.push(token);
            if self.generated_tokens.len() > 8 {
                self.generated_tokens = self.generated_tokens[self.generated_tokens.len() - 8..].to_vec();
            }
        }

        // Check for token repetition
        let token_repetition = if self.generated_tokens.len() >= 3 {
            let last_two: std::collections::HashSet<_> =
                self.generated_tokens[self.generated_tokens.len() - 2..].iter().collect();
            last_two.len() == 1
        } else {
            false
        };

        // Clone logits for modification
        let mut result_logits = logits.clone().unbind();

        // Suppress EoS to prevent early termination
        if cur_text_posn < s - 3 && s > 5 {
            let ellipsis = py.import("builtins")?.getattr("Ellipsis")?;
            let eos_int = self.eos_idx as i64;
            let idx = PyTuple::new(py, &[ellipsis.as_any(), eos_int.into_pyobject(py)?.into_any().as_any()])?;
            result_logits.bind(py).set_item(idx, -(2_i64.pow(15)))?;
        }

        // If bad ending detected, force EOS
        if long_tail || alignment_repetition || token_repetition {
            let neg_val = -(2_i64.pow(15));
            let ones = torch.call_method1("ones_like", (result_logits.bind(py),))?;
            result_logits = ones.call_method1("__mul__", (neg_val,))?.unbind();
            let ellipsis = py.import("builtins")?.getattr("Ellipsis")?;
            let eos_int = self.eos_idx as i64;
            let idx = PyTuple::new(py, &[ellipsis.as_any(), eos_int.into_pyobject(py)?.into_any().as_any()])?;
            result_logits.bind(py).set_item(idx, 2_i64.pow(15))?;
        }

        self.curr_frame_pos += 1;
        Ok(result_logits)
    }

    pub fn text_position(&self) -> isize { self.text_position }
    pub fn has_started(&self) -> bool { self.started }
    pub fn is_complete(&self) -> bool { self.complete }
    pub fn current_frame_pos(&self) -> isize { self.curr_frame_pos }
    pub fn eos_idx(&self) -> isize { self.eos_idx }
}
