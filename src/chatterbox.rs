use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict};
use std::path::Path;

use crate::error::{ChatterboxError, Result};

// Sample rate constants from Python
pub const S3GEN_SR: u32 = 24000; // Output sample rate
pub const S3_SR: u32 = 16000; // Input sample rate for conditioning

/// Compute device for model inference
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Cuda,
    Mps,
}

impl Device {
    /// Convert to Python device string
    pub fn as_str(&self) -> &'static str {
        match self {
            Device::Cpu => "cpu",
            Device::Cuda => "cuda",
            Device::Mps => "mps",
        }
    }

    /// Detect the best available device
    pub fn detect(py: Python<'_>) -> Result<Self> {
        let torch = py.import("torch")?;

        // Check CUDA
        if torch
            .getattr("cuda")?
            .call_method0("is_available")?
            .extract::<bool>()?
        {
            return Ok(Device::Cuda);
        }

        // Check MPS (Apple Silicon)
        let mps_available: bool = torch
            .getattr("backends")?
            .getattr("mps")?
            .call_method0("is_available")?
            .extract()?;
        if mps_available {
            return Ok(Device::Mps);
        }

        Ok(Device::Cpu)
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Wrapper for Python Conditionals dataclass
///
/// Holds conditioning information for T3 and S3Gen models:
/// - T3: speaker embedding, speech tokens, emotion
/// - S3Gen: prompt tokens, features, embedding
pub struct Conditionals {
    inner: Py<PyAny>,
}

impl Conditionals {
    /// Create from an existing Python Conditionals object
    pub fn from_py(obj: Py<PyAny>) -> Self {
        Self { inner: obj }
    }

    /// Get the underlying Python object
    pub fn as_py(&self) -> &Py<PyAny> {
        &self.inner
    }

    /// Move to a different device
    pub fn to_device(&self, py: Python<'_>, device: Device) -> Result<Self> {
        let moved = self.inner.call_method1(py, "to", (device.as_str(),))?;
        Ok(Self { inner: moved })
    }

    /// Save conditionals to a file
    pub fn save(&self, py: Python<'_>, path: &Path) -> Result<()> {
        let path_str = path.to_string_lossy();
        self.inner
            .call_method1(py, "save", (path_str.as_ref(),))?;
        Ok(())
    }

    /// Load conditionals from a file
    pub fn load(py: Python<'_>, path: &Path, device: Option<Device>) -> Result<Self> {
        let tts_module = py.import("chatterbox.tts")?;
        let conditionals_class = tts_module.getattr("Conditionals")?;

        let path_str = path.to_string_lossy();
        let map_location = device.map(|d| d.as_str()).unwrap_or("cpu");

        let obj = conditionals_class.call_method1("load", (path_str.as_ref(), map_location))?;
        Ok(Self { inner: obj.into() })
    }
}

/// Options for audio generation
///
/// Use the builder pattern for ergonomic construction:
/// ```rust
/// let options = GenerateOptions::new()
///     .temperature(0.9)
///     .cfg_weight(0.7);
/// ```
#[derive(Debug, Clone)]
pub struct GenerateOptions {
    pub repetition_penalty: f32,
    pub min_p: f32,
    pub top_p: f32,
    pub audio_prompt_path: Option<String>,
    pub exaggeration: f32,
    pub cfg_weight: f32,
    pub temperature: f32,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            repetition_penalty: 1.2,
            min_p: 0.05,
            top_p: 1.0,
            audio_prompt_path: None,
            exaggeration: 0.5,
            cfg_weight: 0.5,
            temperature: 0.8,
        }
    }
}

impl GenerateOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn repetition_penalty(mut self, value: f32) -> Self {
        self.repetition_penalty = value;
        self
    }

    pub fn min_p(mut self, value: f32) -> Self {
        self.min_p = value;
        self
    }

    pub fn top_p(mut self, value: f32) -> Self {
        self.top_p = value;
        self
    }

    pub fn audio_prompt_path<P: AsRef<str>>(mut self, path: P) -> Self {
        self.audio_prompt_path = Some(path.as_ref().to_string());
        self
    }

    pub fn exaggeration(mut self, value: f32) -> Self {
        self.exaggeration = value;
        self
    }

    pub fn cfg_weight(mut self, value: f32) -> Self {
        self.cfg_weight = value;
        self
    }

    pub fn temperature(mut self, value: f32) -> Self {
        self.temperature = value;
        self
    }
}

/// Generated audio output wrapper
///
/// Holds a PyTorch tensor containing the generated audio waveform
pub struct AudioOutput {
    tensor: Py<PyAny>,
    sample_rate: u32,
}

impl AudioOutput {
    /// Create from a PyTorch tensor
    pub fn from_tensor(tensor: Py<PyAny>, sample_rate: u32) -> Self {
        Self {
            tensor,
            sample_rate,
        }
    }

    /// Get the sample rate
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Get the underlying PyTorch tensor
    pub fn as_tensor(&self) -> &Py<PyAny> {
        &self.tensor
    }

    /// Get audio shape as (channels, samples)
    pub fn shape(&self, py: Python<'_>) -> Result<(usize, usize)> {
        let shape = self.tensor.getattr(py, "shape")?;
        let dims: Vec<usize> = shape.extract(py)?;
        if dims.len() == 2 {
            Ok((dims[0], dims[1]))
        } else if dims.len() == 1 {
            Ok((1, dims[0]))
        } else {
            Err(ChatterboxError::Python(
                pyo3::exceptions::PyValueError::new_err("Unexpected tensor shape").into(),
            ))
        }
    }

    /// Convert to f32 vec (copies data from Python)
    pub fn to_vec(&self, py: Python<'_>) -> Result<Vec<f32>> {
        let cpu_tensor = self.tensor.call_method0(py, "cpu")?;
        let numpy = cpu_tensor.call_method0(py, "numpy")?;
        let flat = numpy.call_method0(py, "flatten")?;
        let data: Vec<f32> = flat.extract(py)?;
        Ok(data)
    }

    /// Save audio to WAV file using torchaudio
    pub fn save_wav(&self, py: Python<'_>, path: &Path) -> Result<()> {
        let torchaudio = py.import("torchaudio")?;
        let path_str = path.to_string_lossy();
        torchaudio.call_method1("save", (path_str.as_ref(), &self.tensor, self.sample_rate))?;
        Ok(())
    }
}

/// Main ChatterboxTTS wrapper
///
/// Provides text-to-speech synthesis using the ChatterboxTTS Python model.
pub struct ChatterboxTTS {
    /// The underlying Python ChatterboxTTS instance
    inner: Py<PyAny>,
    /// Device the model is loaded on
    device: Device,
}

impl ChatterboxTTS {
    // Class constants (from Python)
    /// Encoder conditioning length in samples (6 seconds at S3_SR)
    pub const ENC_COND_LEN: u32 = 6 * S3_SR;
    /// Decoder conditioning length in samples (10 seconds at S3GEN_SR)
    pub const DEC_COND_LEN: u32 = 10 * S3GEN_SR;

    /// Load model from pretrained weights on HuggingFace Hub
    pub fn from_pretrained(py: Python<'_>, device: Device) -> Result<Self> {
        let tts_module = py.import("chatterbox.tts")?;
        let tts_class = tts_module.getattr("ChatterboxTTS")?;

        let kwargs = [("device", device.as_str())].into_py_dict(py)?;
        let instance = tts_class.call_method("from_pretrained", (), Some(&kwargs))?;

        Ok(Self {
            inner: instance.into(),
            device,
        })
    }

    /// Load model from a local checkpoint directory
    pub fn from_local(py: Python<'_>, ckpt_dir: &Path, device: Device) -> Result<Self> {
        let tts_module = py.import("chatterbox.tts")?;
        let tts_class = tts_module.getattr("ChatterboxTTS")?;

        let ckpt_str = ckpt_dir.to_string_lossy();
        let instance =
            tts_class.call_method1("from_local", (ckpt_str.as_ref(), device.as_str()))?;

        Ok(Self {
            inner: instance.into(),
            device,
        })
    }

    /// Get the device this model is loaded on
    pub fn device(&self) -> Device {
        self.device
    }

    /// Get the sample rate of generated audio (24000 Hz)
    pub fn sample_rate(&self, py: Python<'_>) -> Result<u32> {
        let sr = self.inner.getattr(py, "sr")?;
        Ok(sr.extract(py)?)
    }

    /// Get current conditionals if set
    pub fn conditionals(&self, py: Python<'_>) -> Result<Option<Conditionals>> {
        let conds = self.inner.getattr(py, "conds")?;
        if conds.is_none(py) {
            Ok(None)
        } else {
            Ok(Some(Conditionals::from_py(conds)))
        }
    }

    /// Set conditionals directly
    pub fn set_conditionals(&self, py: Python<'_>, conds: &Conditionals) -> Result<()> {
        self.inner.setattr(py, "conds", conds.as_py())?;
        Ok(())
    }

    /// Prepare voice conditionals from a reference audio file
    ///
    /// This extracts speaker characteristics from the reference audio
    /// for voice cloning during generation.
    ///
    /// # Arguments
    /// * `wav_path` - Path to a WAV file with the target voice
    /// * `exaggeration` - Emotion exaggeration factor (0.0-1.0, default 0.5)
    pub fn prepare_conditionals(
        &self,
        py: Python<'_>,
        wav_path: &Path,
        exaggeration: Option<f32>,
    ) -> Result<()> {
        let wav_str = wav_path.to_string_lossy();
        let exag = exaggeration.unwrap_or(0.5);

        self.inner
            .call_method1(py, "prepare_conditionals", (wav_str.as_ref(), exag))?;
        Ok(())
    }

    /// Generate speech audio from text
    ///
    /// # Arguments
    /// * `text` - The text to synthesize
    /// * `options` - Generation options (temperature, cfg_weight, etc.)
    ///
    /// # Returns
    /// An `AudioOutput` containing the generated waveform tensor
    pub fn generate(&self, py: Python<'_>, text: &str, options: GenerateOptions) -> Result<AudioOutput> {
        let kwargs = PyDict::new(py);
        kwargs.set_item("repetition_penalty", options.repetition_penalty)?;
        kwargs.set_item("min_p", options.min_p)?;
        kwargs.set_item("top_p", options.top_p)?;
        kwargs.set_item("exaggeration", options.exaggeration)?;
        kwargs.set_item("cfg_weight", options.cfg_weight)?;
        kwargs.set_item("temperature", options.temperature)?;

        if let Some(ref path) = options.audio_prompt_path {
            kwargs.set_item("audio_prompt_path", path)?;
        }

        let wav = self.inner.call_method(py, "generate", (text,), Some(&kwargs))?;

        let sr = self.sample_rate(py)?;
        Ok(AudioOutput::from_tensor(wav, sr))
    }

    /// Simple generation with default options
    pub fn generate_simple(&self, py: Python<'_>, text: &str) -> Result<AudioOutput> {
        self.generate(py, text, GenerateOptions::default())
    }

    /// Generate with voice cloning from a reference audio file
    pub fn generate_with_voice(
        &self,
        py: Python<'_>,
        text: &str,
        voice_path: &Path,
        options: Option<GenerateOptions>,
    ) -> Result<AudioOutput> {
        let opts = options
            .unwrap_or_default()
            .audio_prompt_path(voice_path.to_string_lossy());
        self.generate(py, text, opts)
    }
}

/// Utility function to print device and environment info
pub fn print_device_info(py: Python<'_>) -> Result<()> {
    let torch = py.import("torch")?;

    let device = Device::detect(py)?;
    println!("Device: {}", device);

    let torch_version: String = torch.getattr("__version__")?.extract()?;
    println!("PyTorch version: {}", torch_version);

    let cuda_version = torch.getattr("version")?.getattr("cuda")?;
    println!("CUDA version: {}", cuda_version);

    let cuda_available: bool = torch
        .getattr("cuda")?
        .call_method0("is_available")?
        .extract()?;
    println!("CUDA available: {}", cuda_available);

    if cuda_available {
        let device_count: i32 = torch
            .getattr("cuda")?
            .call_method0("device_count")?
            .extract()?;
        println!("CUDA device count: {}", device_count);

        if device_count > 0 {
            let device_name: String = torch
                .getattr("cuda")?
                .call_method1("get_device_name", (0,))?
                .extract()?;
            println!("CUDA device name: {}", device_name);
        }
    }

    Ok(())
}
