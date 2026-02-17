//! Rust wrapper for the T3HuggingfaceBackend Python class.
//!
//! T3HuggingfaceBackend overrides HuggingFace interface methods to use
//! custom embedding and logit layers for T3 inference. It extends
//! LlamaPreTrainedModel to avoid re-initializing weights.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::alignment_stream_analyzer::AlignmentStreamAnalyzer;

/// Wrapper for the Python T3HuggingfaceBackend class.
///
/// This class provides a HuggingFace-compatible interface for T3 inference,
/// overriding `prepare_inputs_for_generation` and `forward` to use custom
/// speech token embedding and logit projection layers.
pub struct T3HuggingfaceBackend {
    inner: Py<PyAny>,
}

impl T3HuggingfaceBackend {
    /// Create a new T3HuggingfaceBackend.
    ///
    /// # Arguments
    /// * `config` - LlamaConfig for the model
    /// * `llama` - The LlamaModel backbone
    /// * `speech_enc` - Speech token embedding layer
    /// * `speech_head` - Speech logit projection layer
    /// * `alignment_stream_analyzer` - Optional analyzer for hallucination detection
    pub fn new(
        py: Python<'_>,
        config: &Bound<'_, PyAny>,
        llama: &Bound<'_, PyAny>,
        speech_enc: &Bound<'_, PyAny>,
        speech_head: &Bound<'_, PyAny>,
        alignment_stream_analyzer: Option<&AlignmentStreamAnalyzer>,
    ) -> PyResult<Self> {
        let backend_mod = py.import("chatterbox.models.t3.inference.t3_hf_backend")?;
        let backend_class = backend_mod.getattr("T3HuggingfaceBackend")?;

        let kwargs = PyDict::new(py);
        kwargs.set_item("config", config)?;
        kwargs.set_item("llama", llama)?;
        kwargs.set_item("speech_enc", speech_enc)?;
        kwargs.set_item("speech_head", speech_head)?;

        if let Some(analyzer) = alignment_stream_analyzer {
            kwargs.set_item("alignment_stream_analyzer", analyzer.as_py())?;
        }

        let instance = backend_class.call((), Some(&kwargs))?;

        Ok(Self {
            inner: instance.into(),
        })
    }

    /// Get the underlying Python object.
    pub fn as_py(&self) -> &Py<PyAny> {
        &self.inner
    }

    /// Create from an existing Python T3HuggingfaceBackend object.
    pub fn from_py(obj: Py<PyAny>) -> Self {
        Self { inner: obj }
    }

    /// Forward pass through the model.
    ///
    /// # Arguments
    /// * `inputs_embeds` - Input embeddings tensor (B, S, C)
    /// * `past_key_values` - Optional KV cache from previous steps
    /// * `use_cache` - Whether to return updated KV cache
    /// * `output_attentions` - Whether to return attention weights
    /// * `output_hidden_states` - Whether to return hidden states
    ///
    /// # Returns
    /// CausalLMOutputWithCrossAttentions containing logits, past_key_values,
    /// hidden_states, and attentions.
    pub fn forward(
        &self,
        py: Python<'_>,
        inputs_embeds: &Bound<'_, PyAny>,
        past_key_values: Option<&Bound<'_, PyAny>>,
        use_cache: bool,
        output_attentions: bool,
        output_hidden_states: bool,
    ) -> PyResult<Py<PyAny>> {
        let kwargs = PyDict::new(py);
        kwargs.set_item("inputs_embeds", inputs_embeds)?;
        kwargs.set_item(
            "past_key_values",
            past_key_values.map(|p| p.clone()).unwrap_or_else(|| py.None().into_bound(py)),
        )?;
        kwargs.set_item("use_cache", use_cache)?;
        kwargs.set_item("output_attentions", output_attentions)?;
        kwargs.set_item("output_hidden_states", output_hidden_states)?;
        kwargs.set_item("return_dict", true)?;

        let result = self.inner.call(py, (), Some(&kwargs))?;
        Ok(result)
    }

    /// Get the alignment stream analyzer if one was provided.
    pub fn alignment_stream_analyzer(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let analyzer = self.inner.getattr(py, "alignment_stream_analyzer")?;
        if analyzer.is_none(py) {
            Ok(None)
        } else {
            Ok(Some(analyzer))
        }
    }

    /// Check if conditioning has been added to input embeddings.
    pub fn added_cond(&self, py: Python<'_>) -> PyResult<bool> {
        let added = self.inner.getattr(py, "_added_cond")?;
        added.extract(py)
    }

    /// Reset the added_cond flag (for reusing the backend).
    pub fn reset_added_cond(&self, py: Python<'_>) -> PyResult<()> {
        self.inner.setattr(py, "_added_cond", false)?;
        Ok(())
    }
}
