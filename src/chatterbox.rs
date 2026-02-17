use hf_hub::api::sync::Api;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySlice};
use std::path::{Path, PathBuf};

use crate::error::{ChatterboxError, Result};
use crate::models::s3gen::SPEECH_VOCAB_SIZE;
use crate::models::{EnTokenizer, S3Gen, S3GenInferenceOptions, S3GenRef, T3, T3Cond, VoiceEncoder};
use tch::{Device as TchDevice, IndexOp, Kind as TchKind, Tensor as TchTensor};

// Sample rate constants from Python
pub const S3GEN_SR: u32 = 24000; // Output sample rate
pub const S3_SR: u32 = 16000; // Input sample rate for conditioning

/// Quick cleanup func for punctuation from LLMs or containing chars not seen often in the dataset.
pub fn punc_norm(text: &str) -> String {
    if text.is_empty() {
        return "You need to add some text for me to talk.".to_string();
    }

    let mut text = text.to_string();

    // Capitalise first letter
    if let Some(first) = text.chars().next() {
        if first.is_lowercase() {
            let mut out = String::new();
            out.extend(first.to_uppercase());
            out.push_str(&text[first.len_utf8()..]);
            text = out;
        }
    }

    // Remove multiple space chars
    text = text.split_whitespace().collect::<Vec<_>>().join(" ");

    // Replace uncommon/LLM punctuation
    let punc_to_replace = [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        ("“", "\""),
        ("”", "\""),
        ("‘", "'"),
        ("’", "'"),
    ];
    for (old, new) in punc_to_replace {
        text = text.replace(old, new);
    }

    // Add full stop if no ending punc
    text = text.trim_end_matches(' ').to_string();
    let sentence_enders = [".", "!", "?", "-", ","];
    if !sentence_enders.iter().any(|p| text.ends_with(p)) {
        text.push('.');
    }

    text
}

// HuggingFace repo ID for model weights
const REPO_ID: &str = "ResembleAI/chatterbox";

// Model files to download
const MODEL_FILES: &[&str] = &[
    "ve.safetensors",
    "t3_cfg.safetensors",
    "s3gen.safetensors",
    "tokenizer.json",
    "conds.pt",
];

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

    pub fn to_tch_device(&self) -> TchDevice {
        match self {
            Device::Cpu => TchDevice::Cpu,
            Device::Cuda => TchDevice::Cuda(0),
            Device::Mps => TchDevice::Mps,
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

/// Conditioning information for T3 and S3Gen models.
///
/// Mirrors the Python Conditionals dataclass:
/// - T3: speaker embedding, speech tokens, emotion
/// - S3Gen: prompt tokens, features, embedding
pub struct Conditionals {
    t3: T3Cond,
    gen_dict: Py<PyAny>,
}

impl Conditionals {
    /// Create new conditionals from T3Cond and S3Gen reference dict.
    pub fn new(t3: T3Cond, gen_dict: Py<PyAny>) -> Self {
        Self { t3, gen_dict }
    }

    /// Create from an existing Python object with `t3` and `gen` attributes.
    pub fn from_py(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        let t3 = obj.getattr("t3")?;
        let gen_dict = obj.getattr("gen")?;
        Ok(Self {
            t3: T3Cond::from_py(t3.unbind()),
            gen_dict: gen_dict.unbind(),
        })
    }

    /// Access the T3 conditionals.
    pub fn t3(&self) -> &T3Cond {
        &self.t3
    }

    /// Access the S3Gen reference dict.
    pub fn gen_dict(&self) -> &Py<PyAny> {
        &self.gen_dict
    }

    /// Convert to a Python object with `t3` and `gen` attributes.
    pub fn to_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let types = py.import("types")?;
        let ns_class = types.getattr("SimpleNamespace")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("t3", self.t3.as_py())?;
        kwargs.set_item("gen", self.gen_dict.bind(py))?;
        let ns = ns_class.call((), Some(&kwargs))?;
        Ok(ns.into())
    }

    /// Move conditionals to a different device.
    pub fn to_device(&self, py: Python<'_>, device: Device) -> Result<Self> {
        let device_str = device.as_str();
        let t3 = self.t3.to_device(py, device_str)?;

        let torch = py.import("torch")?;
        let gen_dict = self
            .gen_dict
            .bind(py)
            .cast::<PyDict>()
            .map_err(|e| ChatterboxError::Python(pyo3::exceptions::PyTypeError::new_err(e.to_string())))?;
        let moved_dict = PyDict::new(py);
        for (key, value) in gen_dict.iter() {
            let is_tensor: bool = torch
                .getattr("is_tensor")?
                .call1((value.clone(),))?
                .extract()?;
            if is_tensor {
                let moved = value.call_method1("to", (device_str,))?;
                moved_dict.set_item(key, moved)?;
            } else {
                moved_dict.set_item(key, value)?;
            }
        }

        Ok(Self {
            t3,
            gen_dict: moved_dict.into(),
        })
    }

    /// Save conditionals to a file.
    pub fn save(&self, py: Python<'_>, path: &Path) -> Result<()> {
        let torch = py.import("torch")?;
        let path_str = path.to_string_lossy();

        let arg_dict = PyDict::new(py);
        let t3_dict = self.t3.as_py().call_method0(py, "__dict__")?;
        arg_dict.set_item("t3", t3_dict)?;
        arg_dict.set_item("gen", self.gen_dict.bind(py))?;

        torch.call_method1("save", (arg_dict, path_str.as_ref()))?;
        Ok(())
    }

    /// Load conditionals from a file.
    pub fn load(py: Python<'_>, path: &Path, device: Option<Device>) -> Result<Self> {
        let torch = py.import("torch")?;
        let cond_enc_mod = py.import("chatterbox.models.t3.modules.cond_enc")?;
        let t3_cond_class = cond_enc_mod.getattr("T3Cond")?;

        let path_str = path.to_string_lossy();
        let map_location = device.map(|d| d.as_str()).unwrap_or("cpu");
        let kwargs = PyDict::new(py);
        kwargs.set_item("map_location", map_location)?;
        kwargs.set_item("weights_only", true)?;

        let loaded = torch.call_method("load", (path_str.as_ref(),), Some(&kwargs))?;
        let loaded = loaded
            .cast::<PyDict>()
            .map_err(|e| ChatterboxError::Python(pyo3::exceptions::PyTypeError::new_err(e.to_string())))?;
        let t3_item = loaded.get_item("t3")?.ok_or_else(|| {
            ChatterboxError::Python(pyo3::exceptions::PyKeyError::new_err("t3"))
        })?;
        let t3_kwargs = t3_item
            .cast::<PyDict>()
            .map_err(|e| ChatterboxError::Python(pyo3::exceptions::PyTypeError::new_err(e.to_string())))?;
        let gen_item = loaded.get_item("gen")?.ok_or_else(|| {
            ChatterboxError::Python(pyo3::exceptions::PyKeyError::new_err("gen"))
        })?;
        let gen_dict = gen_item.unbind();

        let t3_cond = t3_cond_class.call((), Some(t3_kwargs))?;
        Ok(Self {
            t3: T3Cond::from_py(t3_cond.into()),
            gen_dict,
        })
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
    /// Tokenizer used for text preprocessing
    tokenizer: EnTokenizer,
    /// Rust-native S3Gen inference
    s3gen_rs: S3Gen,
    /// Rust-native VoiceEncoder
    voice_encoder: VoiceEncoder,
}

impl ChatterboxTTS {
    // Class constants (from Python)
    /// Encoder conditioning length in samples (6 seconds at S3_SR)
    pub const ENC_COND_LEN: u32 = 6 * S3_SR;
    /// Decoder conditioning length in samples (10 seconds at S3GEN_SR)
    pub const DEC_COND_LEN: u32 = 10 * S3GEN_SR;

    //////////////////////////////////////////////////////////////////////////
    /// Load model from pretrained weights on HuggingFace Hub
    //////////////////////////////////////////////////////////////////////////
    pub fn from_pretrained(py: Python<'_>, device: Device) -> Result<Self> {
        let torch = py.import("torch")?;

        // Check MPS availability and fall back to CPU if needed
        let actual_device = if device == Device::Mps {
            let mps_available: bool = torch
                .getattr("backends")?
                .getattr("mps")?
                .call_method0("is_available")?
                .extract()?;

            if !mps_available {
                let mps_built: bool = torch
                    .getattr("backends")?
                    .getattr("mps")?
                    .call_method0("is_built")?
                    .extract()?;

                if !mps_built {
                    println!("MPS not available because the current PyTorch install was not built with MPS enabled.");
                } else {
                    println!("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine.");
                }
                Device::Cpu
            } else {
                device
            }
        } else {
            device
        };

        // Download model files from HuggingFace Hub using Rust hf-hub crate
        let huggingface_api = Api::new().map_err(|e| {
            ChatterboxError::Python(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to create HuggingFace API: {}",
                e
            )))
        })?;
        let repo = huggingface_api.model(REPO_ID.to_string());

        let mut ckpt_dir: Option<PathBuf> = None;
        for fpath in MODEL_FILES {
            println!("[DEBUG] Downloading: {}", fpath);
            let local_path = repo.get(fpath).map_err(|e| {
                ChatterboxError::Python(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to download {}: {}",
                    fpath, e
                )))
            })?;
            println!("[DEBUG] Downloaded to: {}", local_path.display());
            // All files go to the same directory, so we just need the parent of any file
            if ckpt_dir.is_none() {
                ckpt_dir = local_path.parent().map(|p| p.to_path_buf());
            }
        }

        let ckpt_dir = ckpt_dir.expect("MODEL_FILES should not be empty");
        println!("[DEBUG] Model directory: {}", ckpt_dir.display());

        // Load from local directory
        Self::from_local(py, &ckpt_dir, actual_device)
    }

    //////////////////////////////////////////////////////////////////////////
    /// Load model from a local checkpoint directory
    //////////////////////////////////////////////////////////////////////////
    pub fn from_local(py: Python<'_>, ckpt_dir: &Path, device: Device) -> Result<Self> {
        let torch = py.import("torch")?;
        let safetensors = py.import("safetensors.torch")?;
        let load_file = safetensors.getattr("load_file")?;

        // Import model classes
        let s3gen_mod = py.import("chatterbox.models.s3gen")?;
        let s3gen_class = s3gen_mod.getattr("S3Gen")?;

        let device_str = device.as_str();

        // Always load to CPU first for non-CUDA devices to handle CUDA-saved models
        if device == Device::Cpu || device == Device::Mps {
            let _ = torch.call_method1("device", ("cpu",))?;
        }

        // Load VoiceEncoder using Rust implementation
        println!("[DEBUG] Loading VoiceEncoder (Rust)...");
        let ve_path = ckpt_dir.join("ve.safetensors");
        let voice_encoder = VoiceEncoder::from_safetensors(&ve_path, device.to_tch_device())?;

        // Load T3 using Rust wrapper
        println!("[DEBUG] Loading T3...");
        let t3 = T3::new(py)?;
        let t3_path = ckpt_dir.join("t3_cfg.safetensors");
        let t3_state = load_file.call1((t3_path.to_string_lossy().as_ref(),))?;
        // Check if "model" key exists and extract it
        let t3_state = if t3_state.call_method0("keys")?.call_method1("__contains__", ("model",))?.extract::<bool>()? {
            t3_state.get_item("model")?.get_item(0)?
        } else {
            t3_state
        };
        t3.load_state_dict(py, &t3_state)?;
        t3.to_device(py, device_str)?;
        t3.eval(py)?;

        println!("[DEBUG] Loading S3Gen...");
        let s3gen = s3gen_class.call0()?;
        let s3gen_path = ckpt_dir.join("s3gen.safetensors");
        let s3gen_state = load_file.call1((s3gen_path.to_string_lossy().as_ref(),))?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("strict", false)?;
        s3gen.call_method("load_state_dict", (s3gen_state,), Some(&kwargs))?;
        s3gen.call_method1("to", (device_str,))?;
        s3gen.call_method0("eval")?;

        let s3gen_rs = S3Gen::from_safetensors(&s3gen_path, device.to_tch_device(), false)?;

        println!("[DEBUG] Loading EnTokenizer...");
        let tokenizer_path = ckpt_dir.join("tokenizer.json");
        let tokenizer = EnTokenizer::new(py, &tokenizer_path)?;

        // Load conditionals if conds.pt exists
        let conds_path = ckpt_dir.join("conds.pt");
        let conds = if conds_path.exists() {
            println!("[DEBUG] Loading Conditionals from conds.pt...");
            let loaded_conds = Conditionals::load(py, &conds_path, Some(device))?;
            let loaded_conds = loaded_conds.to_device(py, device)?;
            loaded_conds.to_py(py)?
        } else {
            py.None()
        };

        println!("[DEBUG] Creating ChatterboxTTS instance...");
        let types = py.import("types")?;
        let ns_class = types.getattr("SimpleNamespace")?;
        let ns_kwargs = PyDict::new(py);
        ns_kwargs.set_item("sr", S3GEN_SR)?;
        ns_kwargs.set_item("t3", t3.as_py())?;
        ns_kwargs.set_item("s3gen", s3gen)?;
        ns_kwargs.set_item("ve", py.None())?;
        ns_kwargs.set_item("tokenizer", tokenizer.as_py())?;
        ns_kwargs.set_item("device", device_str)?;
        ns_kwargs.set_item("conds", conds)?;
        let instance = ns_class.call((), Some(&ns_kwargs))?;

        Ok(Self {
            inner: instance.into(),
            device,
            tokenizer,
            s3gen_rs,
            voice_encoder,
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
            Ok(Some(Conditionals::from_py(&conds.bind(py))?))
        }
    }

    /// Set conditionals directly
    pub fn set_conditionals(&self, py: Python<'_>, conds: &Conditionals) -> Result<()> {
        let conds_obj = conds.to_py(py)?;
        self.inner.setattr(py, "conds", conds_obj)?;
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

        let librosa = py.import("librosa")?;
        let torch = py.import("torch")?;

        let device_str = self.device.as_str();
        let t3_obj = self.inner.getattr(py, "t3")?;
        let t3 = t3_obj.bind(py);
        // Load reference wav at 24kHz
        let load_kwargs = PyDict::new(py);
        load_kwargs.set_item("sr", S3GEN_SR)?;
        let wav_tuple = librosa.call_method("load", (wav_str.as_ref(),), Some(&load_kwargs))?;
        let s3gen_ref_wav_np = wav_tuple.get_item(0)?;
        let s3gen_ref_wav = torch.call_method1("from_numpy", (s3gen_ref_wav_np.clone(),))?;

        // Truncate decoder conditioning audio
        let dec_slice = PySlice::new(py, 0, Self::DEC_COND_LEN as isize, 1);
        let s3gen_ref_wav = s3gen_ref_wav.get_item(dec_slice)?;

        let s3gen_ref_tch = py_tensor_to_tch(py, &s3gen_ref_wav)?;
        let s3gen_ref = self.s3gen_rs.embed_ref(&s3gen_ref_tch, S3GEN_SR as i64)?;
        let s3gen_ref_dict = s3gen_ref_to_py(py, &s3gen_ref)?;

        // Speech cond prompt tokens (optional)
        let plen: u32 = t3
            .getattr("hp")?
            .getattr("speech_cond_prompt_len")?
            .extract()?;

        let t3_cond_prompt_tokens: Option<Py<PyAny>> = if plen > 0 {
            let ref_16k = self.s3gen_rs.resample_to_16k(&s3gen_ref_tch)?;
            let ref_16k = if ref_16k.dim() == 1 {
                ref_16k.unsqueeze(0)
            } else {
                ref_16k
            };
            let enc_len = Self::ENC_COND_LEN as i64;
            let ref_16k = if ref_16k.size()[1] > enc_len {
                ref_16k.narrow(1, 0, enc_len)
            } else {
                ref_16k
            };
            let (tokens, _lens) = self.s3gen_rs.tokenizer().forward(&ref_16k, Some(plen as i64))?;
            let tokens_py = tch_tensor_to_py_with_dtype(py, &tokens, "int64")?;
            let tokens_py = tokens_py.bind(py).call_method1("to", (device_str,))?;
            Some(tokens_py.unbind())
        } else {
            None
        };

        // Voice-encoder speaker embedding (Rust)
        let ref_16k = self.s3gen_rs.resample_to_16k(&s3gen_ref_tch)?;
        let ve_embeds = self.voice_encoder.embeds_from_wavs(
            &[ref_16k],
            S3_SR as i64,
            false,
            Some(20.0),
            Some(32),
        )?;
        let ve_embed = ve_embeds.mean_dim(&[0_i64][..], true, TchKind::Float);
        let ve_embed = tch_tensor_to_py_with_dtype(py, &ve_embed, "float32")?;
        let ve_embed = ve_embed.bind(py).call_method1("to", (device_str,))?;

        let t3_tokens_bound = t3_cond_prompt_tokens.as_ref().map(|t| t.bind(py));
        let t3_cond = match t3_tokens_bound {
            Some(ref tokens) => T3Cond::new(py, &ve_embed, Some(tokens), Some(exag))?,
            None => T3Cond::new(py, &ve_embed, None, Some(exag))?,
        };
        let t3_cond = t3_cond.to_device(py, device_str)?;

        let conds = Conditionals::new(t3_cond, s3gen_ref_dict);
        let conds_obj = conds.to_py(py)?;
        self.inner.setattr(py, "conds", conds_obj)?;
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
        let device_str = self.device.as_str();
        let torch = py.import("torch")?.unbind();
        let torch_fn = py.import("torch.nn.functional")?.unbind();
        if let Some(ref path) = options.audio_prompt_path {
            self.prepare_conditionals(py, Path::new(path), Some(options.exaggeration))?;
        } else {
            let conds = self.inner.getattr(py, "conds")?;
            if conds.is_none(py) {
                return Err(ChatterboxError::ConditionalsNotPrepared);
            }
        }

        let conds = self.inner.getattr(py, "conds")?;
        let t3_cond = conds.getattr(py, "t3")?;

        // Update exaggeration if needed
        let emotion_adv = t3_cond.getattr(py, "emotion_adv")?;
        let current_exag: f32 = emotion_adv
            .bind(py)
            .get_item((0, 0, 0))?
            .extract()?;
        if options.exaggeration != current_exag {
            let speaker_emb = t3_cond.getattr(py, "speaker_emb")?;
            let cond_prompt_speech_tokens = t3_cond.getattr(py, "cond_prompt_speech_tokens")?;
            let cond_tokens = if cond_prompt_speech_tokens.is_none(py) {
                None
            } else {
                Some(cond_prompt_speech_tokens.bind(py))
            };

            let speaker_emb = speaker_emb.bind(py);
            let new_t3_cond = match cond_tokens {
                Some(ref tokens) => T3Cond::new(py, &speaker_emb, Some(tokens), Some(options.exaggeration))?,
                None => T3Cond::new(py, &speaker_emb, None, Some(options.exaggeration))?,
            };
            let new_t3_cond = new_t3_cond.to_device(py, device_str)?;
            conds.setattr(py, "t3", new_t3_cond.as_py())?;
        }

        // Norm and tokenize text
        let text = punc_norm(text);
        let mut text_tokens = self
            .tokenizer
            .text_to_tokens(py, text.as_str())?
            .call_method1(py, "to", (device_str,))?;

        if options.cfg_weight > 0.0 {
            let cat_inputs = PyList::new(py, &[text_tokens.clone_ref(py), text_tokens.clone_ref(py)])?;
            let cat_kwargs = PyDict::new(py);
            cat_kwargs.set_item("dim", 0)?;
            text_tokens = torch.call_method(py, "cat", (cat_inputs,), Some(&cat_kwargs))?;
        }

        let t3 = self.inner.getattr(py, "t3")?;
        let hp = t3.getattr(py, "hp")?;
        let sot: i64 = hp.getattr(py, "start_text_token")?.extract(py)?;
        let eot: i64 = hp.getattr(py, "stop_text_token")?.extract(py)?;

        let pad_kwargs = PyDict::new(py);
        pad_kwargs.set_item("value", sot)?;
        text_tokens = torch_fn.call_method(py, "pad", (text_tokens, (1, 0)), Some(&pad_kwargs))?;

        let pad_kwargs = PyDict::new(py);
        pad_kwargs.set_item("value", eot)?;
        text_tokens = torch_fn.call_method(py, "pad", (text_tokens, (0, 1)), Some(&pad_kwargs))?;

        let inference_mode = torch.call_method0(py, "inference_mode")?;
        inference_mode.call_method0(py, "__enter__")?;
        let speech_tokens_result = (|| -> PyResult<Py<PyAny>> {
            let t3_cond = conds.getattr(py, "t3")?;
            let t3_kwargs = PyDict::new(py);
            t3_kwargs.set_item("t3_cond", t3_cond)?;
            t3_kwargs.set_item("text_tokens", text_tokens)?;
            t3_kwargs.set_item("max_new_tokens", 1000)?;
            t3_kwargs.set_item("temperature", options.temperature)?;
            t3_kwargs.set_item("cfg_weight", options.cfg_weight)?;
            t3_kwargs.set_item("repetition_penalty", options.repetition_penalty)?;
            t3_kwargs.set_item("min_p", options.min_p)?;
            t3_kwargs.set_item("top_p", options.top_p)?;

            let speech_tokens = t3.call_method(py, "inference", (), Some(&t3_kwargs))?;
            Ok(speech_tokens.bind(py).get_item(0)?.unbind())
        })();
        inference_mode.call_method1(py, "__exit__", (py.None(), py.None(), py.None()))?;
        let speech_tokens = speech_tokens_result?;

        let gen_any = conds.getattr(py, "gen")?;
        let gen_dict = gen_any.bind(py);
        let ref_dict = self.py_ref_to_s3gen(py, &gen_dict)?;
        let tokens_tch = py_tensor_to_tch(py, &speech_tokens.bind(py))?;
        let tokens_tch = drop_invalid_s3_tokens(&tokens_tch)?;
        let wav_tch = self
            .s3gen_rs
            .inference(&tokens_tch, &ref_dict, &S3GenInferenceOptions::default())?
            .0;
        let wav = tch_tensor_to_py(py, &wav_tch)?;

        let sr = self.sample_rate(py)?;
        Ok(AudioOutput::from_tensor(wav, sr))
    }

    fn py_ref_to_s3gen(&self, py: Python<'_>, gen_dict: &Bound<'_, PyAny>) -> Result<S3GenRef> {
        let prompt_token = gen_dict.get_item("prompt_token")?;
        let prompt_token_len = gen_dict.get_item("prompt_token_len")?;
        let prompt_feat = gen_dict.get_item("prompt_feat")?;
        let embedding = gen_dict.get_item("embedding")?;

        let prompt_token = py_tensor_to_tch(py, &prompt_token)?;
        let prompt_token_len = py_tensor_to_tch(py, &prompt_token_len)?;
        let prompt_feat = py_tensor_to_tch(py, &prompt_feat)?;
        let embedding = py_tensor_to_tch(py, &embedding)?;

        Ok(S3GenRef::new(
            prompt_token,
            prompt_token_len,
            prompt_feat,
            None,
            embedding,
        ))
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

fn py_tensor_to_tch(py: Python<'_>, tensor: &Bound<'_, PyAny>) -> Result<TchTensor> {
    let tensor = tensor.call_method0("detach")?.call_method0("cpu")?;
    let numpy = tensor.call_method0("numpy")?;
    let dtype: String = numpy.getattr("dtype")?.getattr("name")?.extract()?;
    let shape: Vec<i64> = numpy.getattr("shape")?.extract()?;
    let flat = numpy.call_method0("ravel")?;

    let tch = if dtype.starts_with("int") || dtype.starts_with("uint") {
        let data: Vec<i64> = flat.extract()?;
        TchTensor::from_slice(&data).reshape(&shape)
    } else {
        let data: Vec<f32> = flat.extract()?;
        TchTensor::from_slice(&data).reshape(&shape)
    };
    Ok(tch)
}

fn tch_tensor_to_py(py: Python<'_>, tensor: &TchTensor) -> Result<Py<PyAny>> {
    let tensor = tensor.to_device(TchDevice::Cpu).to_kind(TchKind::Float);
    let shape: Vec<i64> = tensor.size();
    let flat = tensor.contiguous().view([-1]);
    let data: Vec<f32> = Vec::<f32>::try_from(&flat).map_err(|e| {
        ChatterboxError::TensorConversion(format!("tch tensor -> vec failed: {e}"))
    })?;

    let numpy = py.import("numpy")?;
    let array = numpy.call_method1("array", (data,))?;
    let array = array.call_method1("reshape", (shape,))?;
    let torch = py.import("torch")?;
    let out = torch.call_method1("from_numpy", (array,))?;
    Ok(out.into())
}

fn tch_tensor_to_py_with_dtype(
    py: Python<'_>,
    tensor: &TchTensor,
    dtype_name: &str,
) -> Result<Py<PyAny>> {
    let torch = py.import("torch")?;
    let out = tch_tensor_to_py(py, tensor)?;
    let dtype = torch.getattr(dtype_name)?;
    let cast = out.bind(py).call_method1("to", (dtype,))?;
    Ok(cast.unbind())
}

fn s3gen_ref_to_py(py: Python<'_>, ref_dict: &S3GenRef) -> Result<Py<PyAny>> {
    let dict = PyDict::new(py);
    let prompt_token = tch_tensor_to_py_with_dtype(py, &ref_dict.prompt_token, "int64")?;
    let prompt_token_len = tch_tensor_to_py_with_dtype(py, &ref_dict.prompt_token_len, "int64")?;
    let prompt_feat = tch_tensor_to_py(py, &ref_dict.prompt_feat)?;
    let embedding = tch_tensor_to_py(py, &ref_dict.embedding)?;

    dict.set_item("prompt_token", prompt_token)?;
    dict.set_item("prompt_token_len", prompt_token_len)?;
    dict.set_item("prompt_feat", prompt_feat)?;
    dict.set_item("prompt_feat_len", py.None())?;
    dict.set_item("embedding", embedding)?;
    Ok(dict.into())
}

fn drop_invalid_s3_tokens(tokens: &TchTensor) -> Result<TchTensor> {
    let mut x = if tokens.dim() == 2 && tokens.size()[0] == 1 {
        tokens.squeeze_dim(0)
    } else {
        tokens.shallow_clone()
    };
    if x.dim() != 1 {
        return Err(ChatterboxError::ShapeMismatch(
            "drop_invalid_s3_tokens expects [T] or [1, T] tokens".to_string(),
        ));
    }

    let sos = SPEECH_VOCAB_SIZE;
    let eos = SPEECH_VOCAB_SIZE + 1;
    let mut start = 0_i64;
    let mut end = x.size()[0];

    let sos_mask = x.eq(sos);
    if sos_mask.any().int64_value(&[]) != 0 {
        let idx = sos_mask.nonzero();
        start = idx.i((0, 0)).int64_value(&[]) + 1;
    }
    let eos_mask = x.eq(eos);
    if eos_mask.any().int64_value(&[]) != 0 {
        let idx = eos_mask.nonzero();
        end = idx.i((0, 0)).int64_value(&[]);
    }

    let len = end.saturating_sub(start);
    if len == 0 {
        return Ok(TchTensor::zeros([0], (x.kind(), x.device())));
    }
    x = x.narrow(0, start, len);

    let mask = x.lt(SPEECH_VOCAB_SIZE);
    Ok(x.masked_select(&mask))
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
