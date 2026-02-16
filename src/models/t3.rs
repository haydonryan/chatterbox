//! Rust wrapper for the Python T3 model module.
//!
//! T3 is a transformer-based text-to-speech token model that converts
//! text tokens into speech tokens using a LLaMA backbone.

use pyo3::prelude::*;
use pyo3::types::PyDict;

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
        let kwargs = PyDict::new(py);
        kwargs.set_item("t3_cond", t3_cond.as_py())?;
        kwargs.set_item("text_tokens", text_tokens)?;
        kwargs.set_item("temperature", options.temperature)?;
        kwargs.set_item("top_p", options.top_p)?;
        kwargs.set_item("min_p", options.min_p)?;
        kwargs.set_item("repetition_penalty", options.repetition_penalty)?;
        kwargs.set_item("cfg_weight", options.cfg_weight)?;
        kwargs.set_item("num_return_sequences", options.num_return_sequences)?;
        kwargs.set_item("stop_on_eos", options.stop_on_eos)?;
        kwargs.set_item("do_sample", options.do_sample)?;

        if let Some(max_tokens) = options.max_new_tokens {
            kwargs.set_item("max_new_tokens", max_tokens)?;
        }

        let result = self.inner.call_method(py, "inference", (), Some(&kwargs))?;
        Ok(result)
    }

    /// Run turbo inference (faster variant).
    pub fn inference_turbo(
        &self,
        py: Python<'_>,
        t3_cond: &T3Cond,
        text_tokens: &Bound<'_, PyAny>,
        options: &T3InferenceOptions,
    ) -> PyResult<Py<PyAny>> {
        let kwargs = PyDict::new(py);
        kwargs.set_item("t3_cond", t3_cond.as_py())?;
        kwargs.set_item("text_tokens", text_tokens)?;
        kwargs.set_item("temperature", options.temperature)?;
        kwargs.set_item("top_p", options.top_p)?;
        kwargs.set_item("min_p", options.min_p)?;
        kwargs.set_item("repetition_penalty", options.repetition_penalty)?;
        kwargs.set_item("cfg_weight", options.cfg_weight)?;

        if let Some(max_tokens) = options.max_new_tokens {
            kwargs.set_item("max_new_tokens", max_tokens)?;
        }

        let result = self.inner.call_method(py, "inference_turbo", (), Some(&kwargs))?;
        Ok(result)
    }
}
