//! Rust wrapper for the Python S3Gen model module.
//!
//! S3Gen (S3Token2Wav) converts speech tokens to audio waveforms using
//! conditional flow matching (CFM) and HiFiGAN vocoder.

use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Output sample rate for S3Gen (24kHz).
pub const S3GEN_SR: u32 = 24000;

/// Input/conditioning sample rate (16kHz).
pub const S3_SR: u32 = 16000;

/// Reference embeddings from S3Gen.embed_ref().
///
/// Contains pre-computed speaker and prompt information for inference.
pub struct S3GenRefDict {
    inner: Py<PyAny>,
}

impl S3GenRefDict {
    /// Create from a Python dict object.
    pub fn from_py(obj: Py<PyAny>) -> Self {
        Self { inner: obj }
    }

    /// Get the underlying Python dict.
    pub fn as_py(&self) -> &Py<PyAny> {
        &self.inner
    }

    /// Get the prompt tokens tensor.
    pub fn prompt_token(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let token = self.inner.call_method1(py, "__getitem__", ("prompt_token",))?;
        Ok(token)
    }

    /// Get the prompt token lengths.
    pub fn prompt_token_len(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let len = self.inner.call_method1(py, "__getitem__", ("prompt_token_len",))?;
        Ok(len)
    }

    /// Get the prompt mel-spectrogram features.
    pub fn prompt_feat(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let feat = self.inner.call_method1(py, "__getitem__", ("prompt_feat",))?;
        Ok(feat)
    }

    /// Get the speaker embedding (xvector).
    pub fn embedding(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let emb = self.inner.call_method1(py, "__getitem__", ("embedding",))?;
        Ok(emb)
    }
}

/// Options for S3Gen inference.
#[derive(Debug, Clone)]
pub struct S3GenInferenceOptions {
    /// Number of CFM ODE solver steps (default: 10 for standard, 2 for meanflow).
    pub n_cfm_timesteps: Option<u32>,
    /// Whether to drop invalid tokens before inference.
    pub drop_invalid_tokens: bool,
}

impl Default for S3GenInferenceOptions {
    fn default() -> Self {
        Self {
            n_cfm_timesteps: None, // Use model default
            drop_invalid_tokens: true,
        }
    }
}

impl S3GenInferenceOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn n_cfm_timesteps(mut self, value: u32) -> Self {
        self.n_cfm_timesteps = Some(value);
        self
    }

    pub fn drop_invalid_tokens(mut self, value: bool) -> Self {
        self.drop_invalid_tokens = value;
        self
    }
}

/// Wrapper for the Python S3Gen (S3Token2Wav) model.
///
/// S3Gen converts speech tokens to audio waveforms using:
/// - Conditional flow matching (CFM) for token-to-mel conversion
/// - HiFiGAN vocoder for mel-to-waveform synthesis
/// - Speaker embedding via CAMPPlus xvector encoder
pub struct S3Gen {
    inner: Py<PyAny>,
}

impl S3Gen {
    /// Create a new S3Gen instance with default settings.
    pub fn new(py: Python<'_>) -> PyResult<Self> {
        let s3gen_mod = py.import("chatterbox.models.s3gen")?;
        let s3gen_class = s3gen_mod.getattr("S3Gen")?;
        let instance = s3gen_class.call0()?;

        Ok(Self {
            inner: instance.into(),
        })
    }

    /// Create a new S3Gen instance with meanflow mode.
    pub fn with_meanflow(py: Python<'_>, meanflow: bool) -> PyResult<Self> {
        let s3gen_mod = py.import("chatterbox.models.s3gen")?;
        let s3gen_class = s3gen_mod.getattr("S3Gen")?;

        let kwargs = PyDict::new(py);
        kwargs.set_item("meanflow", meanflow)?;

        let instance = s3gen_class.call((), Some(&kwargs))?;

        Ok(Self {
            inner: instance.into(),
        })
    }

    /// Get the underlying Python object.
    pub fn as_py(&self) -> &Py<PyAny> {
        &self.inner
    }

    /// Load state dict from a tensor dict.
    pub fn load_state_dict(
        &self,
        py: Python<'_>,
        state_dict: &Bound<'_, PyAny>,
        strict: bool,
    ) -> PyResult<()> {
        let kwargs = PyDict::new(py);
        kwargs.set_item("strict", strict)?;
        self.inner
            .call_method(py, "load_state_dict", (state_dict,), Some(&kwargs))?;
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

    /// Get the model's device.
    pub fn device(&self, py: Python<'_>) -> PyResult<String> {
        let device = self.inner.getattr(py, "device")?;
        let device_str: String = device.call_method0(py, "__str__")?.extract(py)?;
        Ok(device_str)
    }

    /// Extract reference embeddings from audio.
    ///
    /// # Arguments
    /// * `ref_wav` - Reference waveform as numpy array or tensor
    /// * `ref_sr` - Sample rate of reference audio
    /// * `device` - Device to process on (default: "auto")
    ///
    /// # Returns
    /// S3GenRefDict containing speaker embedding and prompt tokens
    pub fn embed_ref(
        &self,
        py: Python<'_>,
        ref_wav: &Bound<'_, PyAny>,
        ref_sr: u32,
        device: Option<&str>,
    ) -> PyResult<S3GenRefDict> {
        let kwargs = PyDict::new(py);
        kwargs.set_item("ref_sr", ref_sr)?;
        if let Some(dev) = device {
            kwargs.set_item("device", dev)?;
        }

        let result = self
            .inner
            .call_method(py, "embed_ref", (ref_wav,), Some(&kwargs))?;
        Ok(S3GenRefDict::from_py(result))
    }

    /// Run inference to generate audio from speech tokens.
    ///
    /// # Arguments
    /// * `speech_tokens` - Input speech tokens from T3 model
    /// * `ref_dict` - Pre-computed reference embeddings from embed_ref()
    /// * `options` - Inference options
    ///
    /// # Returns
    /// Tuple of (output_wavs, output_sources) tensors
    pub fn inference(
        &self,
        py: Python<'_>,
        speech_tokens: &Bound<'_, PyAny>,
        ref_dict: &S3GenRefDict,
        options: &S3GenInferenceOptions,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let kwargs = PyDict::new(py);
        kwargs.set_item("speech_tokens", speech_tokens)?;
        kwargs.set_item("ref_dict", ref_dict.as_py())?;
        kwargs.set_item("drop_invalid_tokens", options.drop_invalid_tokens)?;

        if let Some(timesteps) = options.n_cfm_timesteps {
            kwargs.set_item("n_cfm_timesteps", timesteps)?;
        }

        let result = self.inner.call_method(py, "inference", (), Some(&kwargs))?;

        // Result is a tuple (output_wavs, output_sources)
        let wavs = result.call_method1(py, "__getitem__", (0,))?;
        let sources = result.call_method1(py, "__getitem__", (1,))?;

        Ok((wavs, sources))
    }

    /// Run flow inference to generate mel-spectrograms (no vocoder).
    ///
    /// # Arguments
    /// * `speech_tokens` - Input speech tokens
    /// * `ref_dict` - Pre-computed reference embeddings
    /// * `finalize` - Whether streaming is finished
    /// * `n_cfm_timesteps` - Number of ODE solver steps
    ///
    /// # Returns
    /// Mel-spectrogram tensor
    pub fn flow_inference(
        &self,
        py: Python<'_>,
        speech_tokens: &Bound<'_, PyAny>,
        ref_dict: &S3GenRefDict,
        finalize: bool,
        n_cfm_timesteps: Option<u32>,
    ) -> PyResult<Py<PyAny>> {
        let kwargs = PyDict::new(py);
        kwargs.set_item("speech_tokens", speech_tokens)?;
        kwargs.set_item("ref_dict", ref_dict.as_py())?;
        kwargs.set_item("finalize", finalize)?;

        if let Some(timesteps) = n_cfm_timesteps {
            kwargs.set_item("n_cfm_timesteps", timesteps)?;
        }

        let result = self
            .inner
            .call_method(py, "flow_inference", (), Some(&kwargs))?;
        Ok(result)
    }

    /// Run HiFiGAN vocoder inference (mel-to-waveform).
    ///
    /// # Arguments
    /// * `speech_feat` - Mel-spectrogram tensor
    /// * `cache_source` - Optional HiFiGAN cache
    ///
    /// # Returns
    /// Waveform tensor
    pub fn hift_inference(
        &self,
        py: Python<'_>,
        speech_feat: &Bound<'_, PyAny>,
        cache_source: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let result = if let Some(cache) = cache_source {
            self.inner.call_method1(py, "hift_inference", (speech_feat, cache))?
        } else {
            let none = py.None();
            self.inner.call_method1(py, "hift_inference", (speech_feat, none))?
        };
        Ok(result)
    }

    /// Get the internal S3Tokenizer.
    pub fn tokenizer(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let tokenizer = self.inner.getattr(py, "tokenizer")?;
        Ok(tokenizer)
    }
}
