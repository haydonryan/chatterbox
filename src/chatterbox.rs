use hf_hub::api::sync::Api;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySlice};
use std::path::{Path, PathBuf};

use crate::error::{ChatterboxError, Result};
use crate::models::{T3, T3Cond, VoiceEncoder};

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
        let tokenizers_mod = py.import("chatterbox.models.tokenizers")?;
        let tts_mod = py.import("chatterbox.tts")?;

        let s3gen_class = s3gen_mod.getattr("S3Gen")?;
        let en_tokenizer_class = tokenizers_mod.getattr("EnTokenizer")?;
        let conditionals_class = tts_mod.getattr("Conditionals")?;
        let chatterbox_class = tts_mod.getattr("ChatterboxTTS")?;

        let device_str = device.as_str();

        // Always load to CPU first for non-CUDA devices to handle CUDA-saved models
        let map_location = if device == Device::Cpu || device == Device::Mps {
            Some(torch.call_method1("device", ("cpu",))?)
        } else {
            None
        };

        // Load VoiceEncoder using Rust wrapper
        println!("[DEBUG] Loading VoiceEncoder...");
        let ve = VoiceEncoder::new(py)?;
        let ve_path = ckpt_dir.join("ve.safetensors");
        let ve_state = load_file.call1((ve_path.to_string_lossy().as_ref(),))?;
        ve.load_state_dict(py, &ve_state)?;
        ve.to_device(py, device_str)?;
        ve.eval(py)?;

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

        println!("[DEBUG] Loading EnTokenizer...");
        let tokenizer_path = ckpt_dir.join("tokenizer.json");
        let tokenizer = en_tokenizer_class.call1((tokenizer_path.to_string_lossy().as_ref(),))?;

        // Load conditionals if conds.pt exists
        let conds_path = ckpt_dir.join("conds.pt");
        let conds = if conds_path.exists() {
            println!("[DEBUG] Loading Conditionals from conds.pt...");
            let conds_path_str = conds_path.to_string_lossy();
            let loaded_conds = if let Some(ref map_loc) = map_location {
                let kwargs = PyDict::new(py);
                kwargs.set_item("map_location", map_loc)?;
                conditionals_class.call_method("load", (conds_path_str.as_ref(),), Some(&kwargs))?
            } else {
                conditionals_class.call_method1("load", (conds_path_str.as_ref(),))?
            };
            loaded_conds.call_method1("to", (device_str,))?
        } else {
            py.None().into_bound(py)
        };

        println!("[DEBUG] Creating ChatterboxTTS instance...");
        // Create the ChatterboxTTS instance directly
        // Pass the inner Python objects from Rust wrappers
        let instance = chatterbox_class.call1((t3.as_py(), s3gen, ve.as_py(), tokenizer, device_str, conds))?;

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

        let librosa = py.import("librosa")?;
        let torch = py.import("torch")?;
        let tts_mod = py.import("chatterbox.tts")?;
        let conditionals_class = tts_mod.getattr("Conditionals")?;

        let device_str = self.device.as_str();
        let s3gen_obj = self.inner.getattr(py, "s3gen")?;
        let s3gen = s3gen_obj.bind(py);
        let t3_obj = self.inner.getattr(py, "t3")?;
        let t3 = t3_obj.bind(py);
        let ve_obj = self.inner.getattr(py, "ve")?;
        let ve = ve_obj.bind(py);

        // Load reference wav at 24kHz
        let load_kwargs = PyDict::new(py);
        load_kwargs.set_item("sr", S3GEN_SR)?;
        let wav_tuple = librosa.call_method("load", (wav_str.as_ref(),), Some(&load_kwargs))?;
        let s3gen_ref_wav = wav_tuple.get_item(0)?;

        // Resample to 16kHz for voice encoder
        let resample_kwargs = PyDict::new(py);
        resample_kwargs.set_item("orig_sr", S3GEN_SR)?;
        resample_kwargs.set_item("target_sr", S3_SR)?;
        let ref_16k_wav =
            librosa.call_method("resample", (s3gen_ref_wav.clone(),), Some(&resample_kwargs))?;

        // Truncate decoder conditioning audio
        let dec_slice = PySlice::new(py, 0, Self::DEC_COND_LEN as isize, 1);
        let s3gen_ref_wav = s3gen_ref_wav.get_item(dec_slice)?;

        // S3Gen reference embeddings
        let embed_kwargs = PyDict::new(py);
        embed_kwargs.set_item("ref_sr", S3GEN_SR)?;
        embed_kwargs.set_item("device", device_str)?;
        let s3gen_ref_dict = s3gen.call_method("embed_ref", (s3gen_ref_wav,), Some(&embed_kwargs))?;

        // Speech cond prompt tokens (optional)
        let plen: u32 = t3
            .getattr("hp")?
            .getattr("speech_cond_prompt_len")?
            .extract()?;

        let t3_cond_prompt_tokens: Option<Py<PyAny>> = if plen > 0 {
            let enc_slice = PySlice::new(py, 0, Self::ENC_COND_LEN as isize, 1);
            let ref_16k_slice = ref_16k_wav.get_item(enc_slice)?;
            let ref_list = PyList::new(py, &[ref_16k_slice])?;

            let tokzr = s3gen.getattr("tokenizer")?;
            let forward_kwargs = PyDict::new(py);
            forward_kwargs.set_item("max_len", plen)?;
            let forward_out = tokzr.call_method("forward", (ref_list,), Some(&forward_kwargs))?;
            let tokens = forward_out.get_item(0)?;

            let tokens = torch.call_method1("atleast_2d", (tokens,))?;
            let tokens = tokens.call_method1("to", (device_str,))?;
            Some(tokens.unbind())
        } else {
            None
        };

        // Voice-encoder speaker embedding
        let wavs_list = PyList::new(py, &[ref_16k_wav])?;
        let ve_kwargs = PyDict::new(py);
        ve_kwargs.set_item("sample_rate", S3_SR)?;
        ve_kwargs.set_item("as_spk", false)?;
        let ve_embeds = ve.call_method("embeds_from_wavs", (wavs_list,), Some(&ve_kwargs))?;

        let ve_embed = torch.call_method1("from_numpy", (ve_embeds,))?;
        let mean_kwargs = PyDict::new(py);
        mean_kwargs.set_item("dim", 0)?;
        mean_kwargs.set_item("keepdim", true)?;
        let ve_embed = ve_embed.call_method("mean", (), Some(&mean_kwargs))?;
        let ve_embed = ve_embed.call_method1("to", (device_str,))?;

        let t3_tokens_bound = t3_cond_prompt_tokens.as_ref().map(|t| t.bind(py));
        let t3_cond = match t3_tokens_bound {
            Some(ref tokens) => T3Cond::new(py, &ve_embed, Some(tokens), Some(exag))?,
            None => T3Cond::new(py, &ve_embed, None, Some(exag))?,
        };
        let t3_cond = t3_cond.to_device(py, device_str)?;

        let conds = conditionals_class.call1((t3_cond.as_py(), s3gen_ref_dict))?;
        self.inner.setattr(py, "conds", conds)?;
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
        let s3tokenizer_mod = py.import("chatterbox.models.s3tokenizer")?.unbind();
        let drop_invalid_tokens = s3tokenizer_mod.getattr(py, "drop_invalid_tokens")?;

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
        let tokenizer = self.inner.getattr(py, "tokenizer")?;
        let mut text_tokens = tokenizer
            .call_method1(py, "text_to_tokens", (text.as_str(),))?
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
        let mut speech_tokens = speech_tokens_result?;

        speech_tokens = drop_invalid_tokens.call1(py, (speech_tokens,))?;
        let mask = speech_tokens.call_method1(py, "__lt__", (6561,))?;
        speech_tokens = speech_tokens.call_method1(py, "__getitem__", (mask,))?;
        speech_tokens = speech_tokens.call_method1(py, "to", (device_str,))?;

        let s3gen = self.inner.getattr(py, "s3gen")?;
        let s3gen_kwargs = PyDict::new(py);
        s3gen_kwargs.set_item("speech_tokens", speech_tokens)?;
        s3gen_kwargs.set_item("ref_dict", conds.getattr(py, "gen")?)?;

        let s3gen_out = s3gen.call_method(py, "inference", (), Some(&s3gen_kwargs))?;
        let mut wav = s3gen_out.bind(py).get_item(0)?.unbind();
        wav = wav.call_method1(py, "squeeze", (0,))?;
        wav = wav.call_method0(py, "detach")?;
        wav = wav.call_method0(py, "cpu")?;
        let wav = wav.call_method1(py, "unsqueeze", (0,))?;

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
