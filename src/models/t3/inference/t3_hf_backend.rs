//! Pure Rust implementation of T3HuggingfaceBackend.
//!
//! This module provides a HuggingFace-compatible interface for T3 inference,
//! using custom speech token embedding and logit projection layers.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PySlice, PyTuple};

use super::alignment_stream_analyzer::AlignmentStreamAnalyzer;

/// Pure Rust implementation of T3HuggingfaceBackend.
pub struct T3HuggingfaceBackend {
    model: Py<PyAny>,
    speech_enc: Py<PyAny>,
    speech_head: Py<PyAny>,
    added_cond: bool,
    alignment_analyzer: Option<AlignmentStreamAnalyzer>,
}

impl T3HuggingfaceBackend {
    /// Create a new T3HuggingfaceBackend.
    pub fn new(
        model: Py<PyAny>,
        speech_enc: Py<PyAny>,
        speech_head: Py<PyAny>,
        alignment_analyzer: Option<AlignmentStreamAnalyzer>,
    ) -> Self {
        Self {
            model,
            speech_enc,
            speech_head,
            added_cond: false,
            alignment_analyzer,
        }
    }

    /// Create from T3 model components.
    pub fn from_t3_components(
        tfmr: &Bound<'_, PyAny>,
        speech_emb: &Bound<'_, PyAny>,
        speech_head: &Bound<'_, PyAny>,
        alignment_analyzer: Option<AlignmentStreamAnalyzer>,
    ) -> Self {
        Self {
            model: tfmr.clone().unbind(),
            speech_enc: speech_emb.clone().unbind(),
            speech_head: speech_head.clone().unbind(),
            added_cond: false,
            alignment_analyzer,
        }
    }

    /// Reset the added_cond flag for reuse.
    pub fn reset(&mut self) {
        self.added_cond = false;
    }

    /// Check if conditioning has been added.
    pub fn added_cond(&self) -> bool {
        self.added_cond
    }

    /// Get the speech embedding layer.
    pub fn speech_enc(&self) -> &Py<PyAny> {
        &self.speech_enc
    }

    /// Get the speech head layer.
    pub fn speech_head(&self) -> &Py<PyAny> {
        &self.speech_head
    }

    /// Get mutable reference to alignment analyzer.
    pub fn alignment_analyzer_mut(&mut self) -> Option<&mut AlignmentStreamAnalyzer> {
        self.alignment_analyzer.as_mut()
    }

    /// Check if alignment analyzer is present.
    pub fn has_alignment_analyzer(&self) -> bool {
        self.alignment_analyzer.is_some()
    }

    /// Prepare inputs for generation (embedding + optional conditioning).
    pub fn prepare_inputs_for_generation(
        &mut self,
        py: Python<'_>,
        input_ids: &Bound<'_, PyAny>,
        decoder_cond: Option<&Bound<'_, PyAny>>,
        past_key_values: Option<&Bound<'_, PyAny>>,
        use_cache: bool,
    ) -> PyResult<(Py<PyAny>, Option<Py<PyAny>>)> {
        let torch = py.import("torch")?;

        // Trim input_ids if using cache (only last token is new)
        let input_ids = if past_key_values.is_some() && use_cache {
            let row_slice = PySlice::new(py, 0isize, isize::MAX, 1);
            let col_slice = PySlice::new(py, -1isize, isize::MAX, 1);
            let idx = PyTuple::new(py, &[row_slice.as_any(), col_slice.as_any()])?;
            input_ids.get_item(idx)?
        } else {
            input_ids.clone()
        };

        // Apply speech token embedding
        let mut inputs_embeds = self.speech_enc.call1(py, (&input_ids,))?;

        // Prefix decoder conditioning on first step
        if !self.added_cond {
            if let Some(cond) = decoder_cond {
                let cond_batch: i64 = cond.getattr("size")?.call1((0,))?.extract()?;
                let embed_batch: i64 = inputs_embeds.getattr(py, "size")?.call1(py, (0,))?.extract(py)?;

                let cond = if cond_batch != embed_batch {
                    cond.call_method1("expand", (embed_batch, -1, -1))?
                } else {
                    cond.clone()
                };

                let cat_list = pyo3::types::PyList::new(py, &[cond.unbind(), inputs_embeds])?;
                let cat_kwargs = PyDict::new(py);
                cat_kwargs.set_item("dim", 1)?;
                inputs_embeds = torch.call_method("cat", (cat_list,), Some(&cat_kwargs))?.unbind();
            }
            self.added_cond = true;
        }

        let past = past_key_values.map(|p| p.clone().unbind());
        Ok((inputs_embeds, past))
    }

    /// Forward pass through the model.
    pub fn forward(
        &mut self,
        py: Python<'_>,
        inputs_embeds: &Bound<'_, PyAny>,
        past_key_values: Option<&Bound<'_, PyAny>>,
        use_cache: bool,
        output_attentions: bool,
        output_hidden_states: bool,
    ) -> PyResult<T3ForwardOutput> {
        // Call the transformer model
        let kwargs = PyDict::new(py);
        kwargs.set_item("inputs_embeds", inputs_embeds)?;
        if let Some(past) = past_key_values {
            kwargs.set_item("past_key_values", past)?;
        }
        kwargs.set_item("use_cache", use_cache)?;
        kwargs.set_item("output_attentions", output_attentions)?;
        kwargs.set_item("output_hidden_states", output_hidden_states)?;
        kwargs.set_item("return_dict", true)?;

        let tfmr_out = self.model.call(py, (), Some(&kwargs))?;
        let tfmr_out = tfmr_out.bind(py);

        // Get hidden states from last layer
        let hidden_states_tuple = tfmr_out.getattr("hidden_states")?;
        let hidden_states = hidden_states_tuple.get_item(-1)?;

        // Apply speech head to get logits
        let logits = self.speech_head.call1(py, (&hidden_states,))?;

        // Extract other outputs
        let past_key_values = tfmr_out.getattr("past_key_values")?.unbind();
        let attentions = if output_attentions {
            Some(tfmr_out.getattr("attentions")?.unbind())
        } else {
            None
        };

        // Update alignment analyzer with attention weights if present
        if let (Some(analyzer), Some(attns)) = (&mut self.alignment_analyzer, &attentions) {
            analyzer.update_attentions(py, attns.bind(py))?;
        }

        Ok(T3ForwardOutput {
            logits,
            past_key_values,
            hidden_states: hidden_states.unbind(),
            attentions,
        })
    }

    /// Run alignment analysis step on logits.
    pub fn step_alignment(
        &mut self,
        py: Python<'_>,
        logits: &Bound<'_, PyAny>,
        next_token: Option<i64>,
    ) -> PyResult<Py<PyAny>> {
        if let Some(analyzer) = &mut self.alignment_analyzer {
            analyzer.step(py, logits, next_token)
        } else {
            Ok(logits.clone().unbind())
        }
    }
}

/// Output from T3HuggingfaceBackend forward pass.
pub struct T3ForwardOutput {
    pub logits: Py<PyAny>,
    pub past_key_values: Py<PyAny>,
    pub hidden_states: Py<PyAny>,
    pub attentions: Option<Py<PyAny>>,
}

impl T3ForwardOutput {
    pub fn logits(&self) -> &Py<PyAny> { &self.logits }
    pub fn past_key_values(&self) -> &Py<PyAny> { &self.past_key_values }
    pub fn hidden_states(&self) -> &Py<PyAny> { &self.hidden_states }
    pub fn attentions(&self) -> Option<&Py<PyAny>> { self.attentions.as_ref() }
}
