//! Rust wrapper for the Python T3 model module.
//!
//! T3 is a transformer-based text-to-speech token model that converts
//! text tokens into speech tokens using a LLaMA backbone.
//!
//! ## Submodules
//!
//! - `inference` - Inference-related components (AlignmentStreamAnalyzer, T3HuggingfaceBackend)

pub mod inference;

pub use inference::{AlignmentStreamAnalyzer, T3HuggingfaceBackend};

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// Configuration for the T3 model.
///
/// Mirrors the Python T3Config class.
#[derive(Debug, Clone)]
pub struct T3Config {
    // Text tokens
    pub start_text_token: u32,
    pub stop_text_token: u32,
    pub text_tokens_dict_size: u32,
    pub max_text_tokens: u32,

    // Speech tokens
    pub start_speech_token: u32,
    pub stop_speech_token: u32,
    pub speech_tokens_dict_size: u32,
    pub max_speech_tokens: u32,

    // Model config
    pub llama_config_name: String,
    pub input_pos_emb: String,
    pub speech_cond_prompt_len: u32,

    // Conditioning
    pub encoder_type: String,
    pub speaker_embed_size: u32,
    pub use_perceiver_resampler: bool,
    pub emotion_adv: bool,
}

impl Default for T3Config {
    fn default() -> Self {
        Self::english_only()
    }
}

impl T3Config {
    /// Create English-only configuration.
    pub fn english_only() -> Self {
        Self {
            start_text_token: 255,
            stop_text_token: 0,
            text_tokens_dict_size: 704,
            max_text_tokens: 2048,
            start_speech_token: 6561,
            stop_speech_token: 6562,
            speech_tokens_dict_size: 8194,
            max_speech_tokens: 4096,
            llama_config_name: "Llama_520M".to_string(),
            input_pos_emb: "learned".to_string(),
            speech_cond_prompt_len: 150,
            encoder_type: "voice_encoder".to_string(),
            speaker_embed_size: 256,
            use_perceiver_resampler: true,
            emotion_adv: true,
        }
    }

    /// Create multilingual configuration.
    pub fn multilingual() -> Self {
        Self {
            text_tokens_dict_size: 2454,
            ..Self::english_only()
        }
    }

    /// Check if this is a multilingual configuration.
    pub fn is_multilingual(&self) -> bool {
        self.text_tokens_dict_size == 2454
    }
}

/// Conditioning data for T3 model inference.
///
/// Wraps the Python T3Cond dataclass.
pub struct T3Cond {
    inner: Py<PyAny>,
}

impl T3Cond {
    /// Create T3Cond from Python object.
    pub fn from_py(obj: Py<PyAny>) -> Self {
        Self { inner: obj }
    }

    /// Get the underlying Python object.
    pub fn as_py(&self) -> &Py<PyAny> {
        &self.inner
    }

    /// Create a new T3Cond with speaker embedding.
    ///
    /// # Arguments
    /// * `speaker_emb` - Speaker embedding tensor (batch x 256)
    /// * `cond_prompt_speech_tokens` - Optional speech prompt tokens
    /// * `emotion_adv` - Emotion exaggeration factor (0.0-1.0)
    pub fn new(
        py: Python<'_>,
        speaker_emb: &Bound<'_, PyAny>,
        cond_prompt_speech_tokens: Option<&Bound<'_, PyAny>>,
        emotion_adv: Option<f32>,
    ) -> PyResult<Self> {
        let cond_enc_mod = py.import("chatterbox.models.t3.modules.cond_enc")?;
        let t3_cond_class = cond_enc_mod.getattr("T3Cond")?;
        let torch = py.import("torch")?;

        let kwargs = PyDict::new(py);
        kwargs.set_item("speaker_emb", speaker_emb)?;

        if let Some(tokens) = cond_prompt_speech_tokens {
            kwargs.set_item("cond_prompt_speech_tokens", tokens)?;
        }

        if let Some(emotion) = emotion_adv {
            // Create emotion tensor: emotion * torch.ones(1, 1, 1)
            let ones = torch.call_method1("ones", ((1, 1, 1),))?;
            let emotion_tensor = ones.call_method1("__mul__", (emotion,))?;
            kwargs.set_item("emotion_adv", emotion_tensor)?;
        }

        let instance = t3_cond_class.call((), Some(&kwargs))?;
        Ok(Self {
            inner: instance.into(),
        })
    }

    /// Move conditioning to a device.
    pub fn to_device(&self, py: Python<'_>, device: &str) -> PyResult<Self> {
        let kwargs = PyDict::new(py);
        kwargs.set_item("device", device)?;
        let moved = self.inner.call_method(py, "to", (), Some(&kwargs))?;
        Ok(Self { inner: moved })
    }

    /// Save conditioning to a file.
    pub fn save(&self, py: Python<'_>, path: &str) -> PyResult<()> {
        self.inner.call_method1(py, "save", (path,))?;
        Ok(())
    }

    /// Load conditioning from a file.
    pub fn load(py: Python<'_>, path: &str, map_location: Option<&str>) -> PyResult<Self> {
        let cond_enc_mod = py.import("chatterbox.models.t3.modules.cond_enc")?;
        let t3_cond_class = cond_enc_mod.getattr("T3Cond")?;

        let instance = if let Some(loc) = map_location {
            t3_cond_class.call_method1("load", (path, loc))?
        } else {
            t3_cond_class.call_method1("load", (path,))?
        };

        Ok(Self {
            inner: instance.into(),
        })
    }
}

/// Options for T3 inference.
#[derive(Debug, Clone)]
pub struct T3InferenceOptions {
    pub max_new_tokens: Option<u32>,
    pub temperature: f32,
    pub top_p: f32,
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub cfg_weight: f32,
    pub num_return_sequences: u32,
    pub stop_on_eos: bool,
    pub do_sample: bool,
}

impl Default for T3InferenceOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: None, // Uses model default (4096)
            temperature: 0.8,
            top_p: 0.95,
            min_p: 0.05,
            repetition_penalty: 1.2,
            cfg_weight: 0.5,
            num_return_sequences: 1,
            stop_on_eos: true,
            do_sample: true,
        }
    }
}

impl T3InferenceOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn temperature(mut self, value: f32) -> Self {
        self.temperature = value;
        self
    }

    pub fn top_p(mut self, value: f32) -> Self {
        self.top_p = value;
        self
    }

    pub fn min_p(mut self, value: f32) -> Self {
        self.min_p = value;
        self
    }

    pub fn repetition_penalty(mut self, value: f32) -> Self {
        self.repetition_penalty = value;
        self
    }

    pub fn cfg_weight(mut self, value: f32) -> Self {
        self.cfg_weight = value;
        self
    }

    pub fn max_new_tokens(mut self, value: u32) -> Self {
        self.max_new_tokens = Some(value);
        self
    }
}

/// Wrapper for the Python T3 transformer model.
///
/// T3 converts text tokens to speech tokens using a LLaMA backbone
/// with conditioning from speaker embeddings.
pub struct T3 {
    inner: Py<PyAny>,
}

impl T3 {
    fn tensor_size(py: Python<'_>, tensor: &Bound<'_, PyAny>, dim: i64) -> PyResult<i64> {
        tensor.getattr("size")?.call1((dim,))?.extract()
    }

    fn prepare_conditioning(
        &self,
        py: Python<'_>,
        t3_cond: &T3Cond,
        is_gpt: bool,
    ) -> PyResult<Py<PyAny>> {
        let t3_cond_obj = t3_cond.as_py().bind(py);
        let cond_prompt_speech_tokens = t3_cond_obj.getattr("cond_prompt_speech_tokens")?;
        let cond_prompt_speech_emb = t3_cond_obj.getattr("cond_prompt_speech_emb")?;

        if !cond_prompt_speech_tokens.is_none() && cond_prompt_speech_emb.is_none() {
            let speech_emb = self.inner.getattr(py, "speech_emb")?;
            let mut prompt_emb =
                speech_emb.bind(py).call1((cond_prompt_speech_tokens.clone(),))?;

            if !is_gpt {
                let speech_pos_emb = self.inner.getattr(py, "speech_pos_emb")?;
                let pos_emb = speech_pos_emb.bind(py).call1((cond_prompt_speech_tokens,))?;
                prompt_emb = prompt_emb.call_method1("__add__", (pos_emb,))?;
            }

            t3_cond_obj.setattr("cond_prompt_speech_emb", prompt_emb)?;
        }

        let cond_enc = self.inner.getattr(py, "cond_enc")?;
        let cond_emb = cond_enc.bind(py).call1((t3_cond_obj,))?;
        Ok(cond_emb.into())
    }

    fn prepare_input_embeds(
        &self,
        py: Python<'_>,
        t3_cond: &T3Cond,
        text_tokens: &Bound<'_, PyAny>,
        speech_tokens: &Bound<'_, PyAny>,
        cfg_weight: f32,
    ) -> PyResult<(Py<PyAny>, i64)> {
        let is_gpt: bool = self.inner.getattr(py, "is_gpt")?.extract(py)?;
        let hp = self.inner.getattr(py, "hp")?;
        let input_pos_emb: String = hp.getattr(py, "input_pos_emb")?.extract(py)?;

        let mut cond_emb = self.prepare_conditioning(py, t3_cond, is_gpt)?;
        let mut text_emb = self
            .inner
            .getattr(py, "text_emb")?
            .bind(py)
            .call1((text_tokens,))?;

        if cfg_weight > 0.0 && !is_gpt {
            if let Ok(item) = text_emb.get_item(1) {
                item.call_method0("zero_")?;
            }
        }

        let mut speech_emb = self
            .inner
            .getattr(py, "speech_emb")?
            .bind(py)
            .call1((speech_tokens,))?;

        if input_pos_emb == "learned" {
            let text_pos_emb = self.inner.getattr(py, "text_pos_emb")?;
            let speech_pos_emb = self.inner.getattr(py, "speech_pos_emb")?;
            let text_pos = text_pos_emb.bind(py).call1((text_tokens,))?;
            let speech_pos = speech_pos_emb.bind(py).call1((speech_tokens,))?;
            text_emb = text_emb.call_method1("__add__", (text_pos,))?;
            speech_emb = speech_emb.call_method1("__add__", (speech_pos,))?;
        }

        let len_cond = Self::tensor_size(py, cond_emb.bind(py), 1)?;
        let cond_batch = Self::tensor_size(py, cond_emb.bind(py), 0)?;
        let text_batch = Self::tensor_size(py, &text_emb, 0)?;
        if cond_batch != text_batch {
            cond_emb = cond_emb
                .bind(py)
                .call_method1("expand", (text_batch, -1, -1))?
                .unbind();
        }

        let torch = py.import("torch")?;
        let text_emb = text_emb.unbind();
        let speech_emb = speech_emb.unbind();
        let cat_inputs = PyList::new(
            py,
            &[cond_emb.clone_ref(py), text_emb.clone_ref(py), speech_emb.clone_ref(py)],
        )?;
        let cat_kwargs = PyDict::new(py);
        cat_kwargs.set_item("dim", 1)?;
        let embeds = torch.call_method("cat", (cat_inputs,), Some(&cat_kwargs))?;
        Ok((embeds.into(), len_cond))
    }
    /// Create a new T3 instance with default (English-only) config.
    pub fn new(py: Python<'_>) -> PyResult<Self> {
        let t3_mod = py.import("chatterbox.models.t3")?;
        let t3_class = t3_mod.getattr("T3")?;
        let instance = t3_class.call0()?;

        Ok(Self {
            inner: instance.into(),
        })
    }

    /// Create a new T3 instance with custom config.
    pub fn with_config(py: Python<'_>, multilingual: bool) -> PyResult<Self> {
        let t3_mod = py.import("chatterbox.models.t3")?;
        let t3_class = t3_mod.getattr("T3")?;
        let config_mod = py.import("chatterbox.models.t3.modules.t3_config")?;
        let config_class = config_mod.getattr("T3Config")?;

        let config = if multilingual {
            config_class.call_method0("multilingual")?
        } else {
            config_class.call_method0("english_only")?
        };

        let instance = t3_class.call1((config,))?;
        Ok(Self {
            inner: instance.into(),
        })
    }

    /// Get the underlying Python object.
    pub fn as_py(&self) -> &Py<PyAny> {
        &self.inner
    }

    /// Load state dict from a tensor dict.
    pub fn load_state_dict(&self, py: Python<'_>, state_dict: &Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.call_method1(py, "load_state_dict", (state_dict,))?;
        Ok(())
    }

    /// Move the model to a device.
    pub fn to_device(&self, py: Python<'_>, device: &str) -> PyResult<&Self> {
        self.inner.call_method1(py, "to", (device,))?;
        Ok(self)
    }

    /// Set the model to evaluation mode.
    pub fn eval(&self, py: Python<'_>) -> PyResult<&Self> {
        self.inner.call_method0(py, "eval")?;
        Ok(self)
    }

    /// Get the model's hyperparameters (T3Config).
    pub fn hp(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let hp = self.inner.getattr(py, "hp")?;
        Ok(hp)
    }

    /// Get the start text token value.
    pub fn start_text_token(&self, py: Python<'_>) -> PyResult<u32> {
        let hp = self.inner.getattr(py, "hp")?;
        let token = hp.getattr(py, "start_text_token")?;
        token.extract(py)
    }

    /// Get the stop text token value.
    pub fn stop_text_token(&self, py: Python<'_>) -> PyResult<u32> {
        let hp = self.inner.getattr(py, "hp")?;
        let token = hp.getattr(py, "stop_text_token")?;
        token.extract(py)
    }

    /// Run inference to generate speech tokens from text tokens.
    ///
    /// # Arguments
    /// * `t3_cond` - Conditioning data (speaker embedding, etc.)
    /// * `text_tokens` - Input text tokens (must include start/stop tokens)
    /// * `options` - Inference options (temperature, sampling, etc.)
    ///
    /// # Returns
    /// Generated speech token tensor
    pub fn inference(
        &self,
        py: Python<'_>,
        t3_cond: &T3Cond,
        text_tokens: &Bound<'_, PyAny>,
        options: &T3InferenceOptions,
    ) -> PyResult<Py<PyAny>> {
        let torch = py.import("torch")?;
        let hp = self.inner.getattr(py, "hp")?;
        let start_text: i64 = hp.getattr(py, "start_text_token")?.extract(py)?;
        let stop_text: i64 = hp.getattr(py, "stop_text_token")?.extract(py)?;
        let start_speech: i64 = hp.getattr(py, "start_speech_token")?.extract(py)?;
        let stop_speech: i64 = hp.getattr(py, "stop_speech_token")?.extract(py)?;
        let max_speech_tokens: i64 = hp.getattr(py, "max_speech_tokens")?.extract(py)?;

        // Ensure BOT/EOT tokens exist
        let batch_size: i64 = Self::tensor_size(py, text_tokens, 0)?;
        let start_matches = text_tokens.call_method1("eq", (start_text,))?;
        let start_count: i64 = start_matches
            .call_method0("int")?
            .call_method0("sum")?
            .extract()?;
        if start_count < batch_size {
            return Err(pyo3::exceptions::PyAssertionError::new_err(
                "missing start_text_token",
            ));
        }
        let stop_matches = text_tokens.call_method1("eq", (stop_text,))?;
        let stop_count: i64 = stop_matches
            .call_method0("int")?
            .call_method0("sum")?
            .extract()?;
        if stop_count < batch_size {
            return Err(pyo3::exceptions::PyAssertionError::new_err(
                "missing stop_text_token",
            ));
        }

        let device = self.inner.getattr(py, "device")?;
        let text_tokens = torch.call_method1("atleast_2d", (text_tokens,))?;
        let to_kwargs = PyDict::new(py);
        to_kwargs.set_item("dtype", torch.getattr("long")?)?;
        to_kwargs.set_item("device", device)?;
        let text_tokens = text_tokens.call_method("to", (), Some(&to_kwargs))?;

        // Default initial speech token: start_speech_token * ones_like(text_tokens[:, :1])
        let first_col = text_tokens.call_method1("narrow", (1, 0, 1))?;
        let initial_speech_tokens = torch
            .call_method1("ones_like", (first_col,))?
            .call_method1("__mul__", (start_speech,))?;

        let (embeds, len_cond) = self.prepare_input_embeds(
            py,
            t3_cond,
            &text_tokens,
            &initial_speech_tokens,
            options.cfg_weight,
        )?;
        let embeds = embeds;

        // Build / reuse patched model
        let compiled: bool = self.inner.getattr(py, "compiled")?.extract(py)?;
        if !compiled {
            let is_multilingual: bool = hp.getattr(py, "is_multilingual")?.extract(py)?;
            let alignment_stream_analyzer: Option<Bound<'_, PyAny>> = if is_multilingual {
                let analyzer_mod = py.import("chatterbox.models.t3.inference.alignment_stream_analyzer")?;
                let analyzer_class = analyzer_mod.getattr("AlignmentStreamAnalyzer")?;
                let text_len = Self::tensor_size(py, &text_tokens, -1)?;
                let slice = (len_cond, len_cond + text_len);
                let kwargs = PyDict::new(py);
                kwargs.set_item("text_tokens_slice", slice)?;
                kwargs.set_item("alignment_layer_idx", 9)?;
                kwargs.set_item("eos_idx", stop_speech)?;
                Some(analyzer_class.call(
                    (self.inner.getattr(py, "tfmr")?, py.None()),
                    Some(&kwargs),
                )?)
            } else {
                None
            };

            let backend_mod = py.import("chatterbox.models.t3.inference.t3_hf_backend")?;
            let backend_class = backend_mod.getattr("T3HuggingfaceBackend")?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("config", self.inner.getattr(py, "cfg")?)?;
            kwargs.set_item("llama", self.inner.getattr(py, "tfmr")?)?;
            kwargs.set_item("speech_enc", self.inner.getattr(py, "speech_emb")?)?;
            kwargs.set_item("speech_head", self.inner.getattr(py, "speech_head")?)?;
            if let Some(alignment) = alignment_stream_analyzer {
                kwargs.set_item("alignment_stream_analyzer", alignment)?;
            }
            let patched_model = backend_class.call((), Some(&kwargs))?;
            self.inner.setattr(py, "patched_model", patched_model)?;
            self.inner.setattr(py, "compiled", true)?;
        }

        let patched_model = self.inner.getattr(py, "patched_model")?;

        let bos_kwargs = PyDict::new(py);
        bos_kwargs.set_item("dtype", torch.getattr("long")?)?;
        bos_kwargs.set_item("device", embeds.bind(py).getattr("device")?)?;
        let bos_token = torch.call_method("tensor", (vec![vec![start_speech]],), Some(&bos_kwargs))?;

        let speech_emb = self.inner.getattr(py, "speech_emb")?;
        let mut bos_embed = speech_emb.bind(py).call1((bos_token.clone(),))?;
        let speech_pos_emb = self.inner.getattr(py, "speech_pos_emb")?;
        let fixed = speech_pos_emb.bind(py).call_method1("get_fixed_embedding", (0,))?;
        bos_embed = bos_embed.call_method1("__add__", (fixed,))?;
        let bos_embed = bos_embed.unbind();

        let cat_inputs = PyList::new(py, &[bos_embed.clone_ref(py), bos_embed.clone_ref(py)])?;
        let cat_kwargs = PyDict::new(py);
        cat_kwargs.set_item("dim", 0)?;
        let bos_embed = torch.call_method("cat", (cat_inputs,), Some(&cat_kwargs))?.unbind();

        let cat_inputs = PyList::new(py, &[embeds.clone_ref(py), bos_embed.clone_ref(py)])?;
        let cat_kwargs = PyDict::new(py);
        cat_kwargs.set_item("dim", 1)?;
        let inputs_embeds = torch.call_method("cat", (cat_inputs,), Some(&cat_kwargs))?;

        let mut generated_ids = bos_token.call_method0("clone")?.unbind();
        let mut predicted: Vec<Py<PyAny>> = Vec::new();

        let logits_mod = py.import("transformers.generation.logits_process")?;
        let top_p_warper = logits_mod.getattr("TopPLogitsWarper")?.call1((options.top_p,))?;
        let min_p_warper = logits_mod.getattr("MinPLogitsWarper")?.call1((options.min_p,))?;
        let repetition_penalty_processor =
            logits_mod.getattr("RepetitionPenaltyLogitsProcessor")?.call1((options.repetition_penalty,))?;

        let output = patched_model.bind(py).call((), Some(&{
            let kwargs = PyDict::new(py);
            kwargs.set_item("inputs_embeds", inputs_embeds)?;
            kwargs.set_item("past_key_values", py.None())?;
            kwargs.set_item("use_cache", true)?;
            kwargs.set_item("output_attentions", true)?;
            kwargs.set_item("output_hidden_states", true)?;
            kwargs.set_item("return_dict", true)?;
            kwargs
        }))?;
        let mut output = output.unbind();
        let mut past = output.bind(py).getattr("past_key_values")?.unbind();

        let max_new_tokens = options.max_new_tokens.map(|v| v as i64).unwrap_or(max_speech_tokens);

        for i in 0..max_new_tokens {
            let logits = output.bind(py).getattr("logits")?;
            let last_idx = Self::tensor_size(py, &logits, 1)? - 1;
            let logits_step = logits.call_method1("select", (1, last_idx))?;

            let cond = logits_step.call_method1("narrow", (0, 0, 1))?;
            let uncond = logits_step.call_method1("narrow", (0, 1, 1))?;
            let cfg = torch.call_method("as_tensor", (options.cfg_weight,), Some(&{
                let kwargs = PyDict::new(py);
                kwargs.set_item("device", cond.getattr("device")?)?;
                kwargs.set_item("dtype", cond.getattr("dtype")?)?;
                kwargs
            }))?;
            let diff = cond.call_method1("__sub__", (uncond,))?;
            let scaled = diff.call_method1("__mul__", (cfg,))?;
            let mut logits = cond.call_method1("__add__", (scaled,))?;

            let alignment_stream_analyzer = patched_model.getattr(py, "alignment_stream_analyzer")?;
            if !alignment_stream_analyzer.is_none(py) {
                let alignment_stream_analyzer = alignment_stream_analyzer.bind(py);
                let len = Self::tensor_size(py, generated_ids.bind(py), 1)?;
                let last_token = if len > 0 {
                    let item = generated_ids
                        .bind(py)
                        .call_method1("select", (0, 0))?
                        .call_method1("select", (0, len - 1))?
                        .call_method0("item")?;
                    Some(item)
                } else {
                    None
                };
                let kwargs = PyDict::new(py);
                if let Some(tok) = last_token {
                    kwargs.set_item("next_token", tok)?;
                }
                logits = alignment_stream_analyzer.call_method("step", (logits,), Some(&kwargs))?;
            }

            let ids_for_proc = generated_ids.bind(py).call_method1("narrow", (0, 0, 1))?;
            logits = repetition_penalty_processor.call1((ids_for_proc.clone(), logits))?;

            if (options.temperature - 1.0).abs() > f32::EPSILON {
                logits = logits.call_method1("__truediv__", (options.temperature,))?;
            }

            logits = min_p_warper.call1((ids_for_proc.clone(), logits))?;
            logits = top_p_warper.call1((ids_for_proc.clone(), logits))?;

            let probs = torch.call_method("softmax", (logits, -1), None)?;
            let next_token = torch.call_method1("multinomial", (probs, 1))?.unbind();

            predicted.push(next_token.clone_ref(py));
            let cat_inputs = PyList::new(py, &[generated_ids.bind(py), next_token.bind(py)])?;
            let cat_kwargs = PyDict::new(py);
            cat_kwargs.set_item("dim", 1)?;
            generated_ids = torch.call_method("cat", (cat_inputs,), Some(&cat_kwargs))?.unbind();

            let next_item = next_token.bind(py).call_method1("view", (-1,))?.get_item(0)?;
            let next_id: i64 = next_item.extract()?;
            if next_id == stop_speech {
                break;
            }

            let mut next_token_embed = speech_emb.bind(py).call1((next_token.bind(py),))?;
            let fixed = speech_pos_emb
                .bind(py)
                .call_method1("get_fixed_embedding", (i + 1,))?;
            next_token_embed = next_token_embed.call_method1("__add__", (fixed,))?;
            let cat_inputs = PyList::new(py, &[next_token_embed.clone(), next_token_embed.clone()])?;
            let cat_kwargs = PyDict::new(py);
            cat_kwargs.set_item("dim", 0)?;
            let next_token_embed = torch.call_method("cat", (cat_inputs,), Some(&cat_kwargs))?;

            let output_next = patched_model.bind(py).call((), Some(&{
                let kwargs = PyDict::new(py);
                kwargs.set_item("inputs_embeds", next_token_embed)?;
                kwargs.set_item("past_key_values", past.bind(py))?;
                kwargs.set_item("output_attentions", true)?;
                kwargs.set_item("output_hidden_states", true)?;
                kwargs.set_item("return_dict", true)?;
                kwargs
            }))?;
            output = output_next.unbind();
            past = output.bind(py).getattr("past_key_values")?.unbind();
        }

        let predicted_list = PyList::new(py, predicted)?;
        let cat_kwargs = PyDict::new(py);
        cat_kwargs.set_item("dim", 1)?;
        let predicted_tokens = torch.call_method("cat", (predicted_list,), Some(&cat_kwargs))?;
        Ok(predicted_tokens.into())
    }

    /// Run turbo inference (faster variant).
    pub fn inference_turbo(
        &self,
        py: Python<'_>,
        t3_cond: &T3Cond,
        text_tokens: &Bound<'_, PyAny>,
        options: &T3InferenceOptions,
    ) -> PyResult<Py<PyAny>> {
        let torch = py.import("torch")?;
        let torch_fn = py.import("torch.nn.functional")?;
        let hp = self.inner.getattr(py, "hp")?;
        let start_speech: i64 = hp.getattr(py, "start_speech_token")?.extract(py)?;
        let stop_speech: i64 = hp.getattr(py, "stop_speech_token")?.extract(py)?;

        let logits_mod = py.import("transformers.generation.logits_process")?;
        let logits_list = logits_mod.getattr("LogitsProcessorList")?.call0()?;
        if options.temperature > 0.0 && (options.temperature - 1.0).abs() > f32::EPSILON {
            let temp = logits_mod
                .getattr("TemperatureLogitsWarper")?
                .call1((options.temperature,))?;
            let _ = logits_list.call_method1("append", (temp,));
        }
        if options.top_p < 1.0 {
            let top_p = logits_mod
                .getattr("TopPLogitsWarper")?
                .call1((options.top_p,))?;
            let _ = logits_list.call_method1("append", (top_p,));
        }
        if options.repetition_penalty != 1.0 {
            let rep = logits_mod
                .getattr("RepetitionPenaltyLogitsProcessor")?
                .call1((options.repetition_penalty,))?;
            let _ = logits_list.call_method1("append", (rep,));
        }

        let first_col = text_tokens.call_method1("narrow", (1, 0, 1))?;
        let speech_start_token = torch
            .call_method1("ones_like", (first_col,))?
            .call_method1("__mul__", (start_speech,))?;

        let (embeds, _) = self.prepare_input_embeds(
            py,
            t3_cond,
            text_tokens,
            &speech_start_token,
            0.0,
        )?;
        let embeds = embeds.bind(py);

        let tfmr = self.inner.getattr(py, "tfmr")?;
        let speech_emb = self.inner.getattr(py, "speech_emb")?;
        let speech_head = self.inner.getattr(py, "speech_head")?;

        let llm_outputs = tfmr.bind(py).call((), Some(&{
            let kwargs = PyDict::new(py);
            kwargs.set_item("inputs_embeds", embeds)?;
            kwargs.set_item("use_cache", true)?;
            kwargs
        }))?;
        let hidden_states = llm_outputs.get_item(0)?;
        let mut past_key_values = llm_outputs.getattr("past_key_values")?.unbind();

        let last_idx = Self::tensor_size(py, &hidden_states, 1)? - 1;
        let speech_hidden = hidden_states
            .call_method1("select", (1, last_idx))?
            .call_method1("unsqueeze", (1,))?;
        let speech_logits = speech_head.bind(py).call1((speech_hidden,))?;
        let last_logits = speech_logits.call_method1("select", (1, 0))?;
        let processed_logits = logits_list.call1((speech_start_token.clone(), last_logits))?;
        let probs = torch_fn.call_method("softmax", (processed_logits, -1), None)?;
        let mut next_speech_token = torch.call_method1("multinomial", (probs, 1))?.unbind();

        let mut generated: Vec<Py<PyAny>> = vec![next_speech_token.clone_ref(py)];
        let mut current = next_speech_token.clone_ref(py);

        let max_gen_len = options.max_new_tokens.map(|v| v as i64).unwrap_or(1000);
        for _ in 0..max_gen_len {
            let current_embed = speech_emb.bind(py).call1((current.bind(py),))?;
            let llm_outputs = tfmr.bind(py).call((), Some(&{
                let kwargs = PyDict::new(py);
                kwargs.set_item("inputs_embeds", current_embed)?;
                kwargs.set_item("past_key_values", past_key_values.bind(py))?;
                kwargs.set_item("use_cache", true)?;
                kwargs
            }))?;
            let hidden_states = llm_outputs.get_item(0)?;
            past_key_values = llm_outputs.getattr("past_key_values")?.unbind();
            let speech_logits = speech_head.bind(py).call1((hidden_states,))?;
            let last_logits = speech_logits.call_method1("select", (1, 0))?;

            let input_ids = torch.call_method("cat", (PyList::new(py, &generated)?,), Some(&{
                let kwargs = PyDict::new(py);
                kwargs.set_item("dim", 1)?;
                kwargs
            }))?;
            let processed_logits = logits_list.call1((input_ids.clone(), last_logits))?;
            let all_inf = torch
                .call_method1(
                    "all",
                    (processed_logits.call_method1("__eq__", (-f64::INFINITY,))?,),
                )?
                .extract::<bool>()?;
            if all_inf {
                break;
            }

            let probs = torch_fn.call_method("softmax", (processed_logits, -1), None)?;
            next_speech_token = torch.call_method1("multinomial", (probs, 1))?.unbind();
            generated.push(next_speech_token.clone_ref(py));
            current = next_speech_token.clone_ref(py);

            let all_stop = torch
                .call_method1("all", (next_speech_token.bind(py).call_method1("__eq__", (stop_speech,))?,))?
                .extract::<bool>()?;
            if all_stop {
                break;
            }
        }

        let all_tokens = torch.call_method("cat", (PyList::new(py, &generated)?,), Some(&{
            let kwargs = PyDict::new(py);
            kwargs.set_item("dim", 1)?;
            kwargs
        }))?;

        let len = Self::tensor_size(py, &all_tokens, 1)?;
        if len > 0 {
            let last = all_tokens
                .call_method1("select", (0, 0))?
                .call_method1("select", (0, len - 1))?
                .call_method0("item")?;
            let last_val: i64 = last.extract()?;
            if last_val == stop_speech {
                let trimmed = all_tokens.call_method1("narrow", (1, 0, len - 1))?;
                return Ok(trimmed.into());
            }
        }

        Ok(all_tokens.into())
    }
}
