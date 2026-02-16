//! Rust wrapper for the Python VoiceEncoder module.
//!
//! The VoiceEncoder is an LSTM-based neural network that extracts speaker embeddings
//! from audio waveforms or mel-spectrograms.

use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Configuration for the VoiceEncoder.
///
/// Mirrors the Python VoiceEncConfig dataclass.
#[derive(Debug, Clone)]
pub struct VoiceEncConfig {
    pub sample_rate: u32,
    pub preemphasis: f32,
    pub n_fft: u32,
    pub hop_size: u32,
    pub win_size: u32,
    pub num_mels: u32,
    pub fmin: u32,
    pub fmax: u32,
    pub mel_power: f32,
    pub ve_hidden_size: u32,
    pub speaker_embed_size: u32,
    pub ve_partial_frames: u32,
    pub normalized_mels: bool,
    pub ve_final_relu: bool,
    pub stft_magnitude_min: f32,
}

impl Default for VoiceEncConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            preemphasis: 0.0,
            n_fft: 400,
            hop_size: 160,
            win_size: 400,
            num_mels: 40,
            fmin: 0,
            fmax: 8000,
            mel_power: 2.0,
            ve_hidden_size: 256,
            speaker_embed_size: 256,
            ve_partial_frames: 160,
            normalized_mels: false,
            ve_final_relu: true,
            stft_magnitude_min: 1e-4,
        }
    }
}

/// Wrapper for the Python VoiceEncoder neural network.
///
/// The VoiceEncoder extracts 256-dimensional speaker embeddings from audio.
/// It uses a 3-layer LSTM followed by a linear projection and L2 normalization.
pub struct VoiceEncoder {
    inner: Py<PyAny>,
}

impl VoiceEncoder {
    /// Create a new VoiceEncoder instance.
    pub fn new(py: Python<'_>) -> PyResult<Self> {
        let ve_mod = py.import("chatterbox.models.voice_encoder")?;
        let ve_class = ve_mod.getattr("VoiceEncoder")?;
        let instance = ve_class.call0()?;

        Ok(Self {
            inner: instance.into(),
        })
    }

    /// Create a new VoiceEncoder with custom configuration.
    pub fn with_config(py: Python<'_>, _config: &VoiceEncConfig) -> PyResult<Self> {
        // For now, use default config from Python side
        // Custom config would require creating Python VoiceEncConfig object
        Self::new(py)
    }

    /// Get the underlying Python object.
    pub fn as_py(&self) -> &Py<PyAny> {
        &self.inner
    }

    /// Load state dict from a file path.
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

    /// Extract embeddings from mel-spectrograms.
    ///
    /// # Arguments
    /// * `mels` - Mel-spectrograms as a tensor or list of arrays
    /// * `mel_lens` - Optional lengths of each mel-spectrogram
    /// * `as_spk` - If true, return speaker embedding (average of utterance embeddings)
    /// * `batch_size` - Optional batch size for processing
    ///
    /// # Returns
    /// Embeddings as a numpy array (256-dimensional, L2-normalized)
    pub fn embeds_from_mels(
        &self,
        py: Python<'_>,
        mels: &Bound<'_, PyAny>,
        mel_lens: Option<&Bound<'_, PyAny>>,
        as_spk: bool,
        batch_size: Option<u32>,
    ) -> PyResult<Py<PyAny>> {
        let kwargs = PyDict::new(py);
        kwargs.set_item("as_spk", as_spk)?;
        if let Some(lens) = mel_lens {
            kwargs.set_item("mel_lens", lens)?;
        }
        if let Some(bs) = batch_size {
            kwargs.set_item("batch_size", bs)?;
        }

        let result = self.inner.call_method(py, "embeds_from_mels", (mels,), Some(&kwargs))?;
        Ok(result)
    }

    /// Extract embeddings from audio waveforms.
    ///
    /// # Arguments
    /// * `wavs` - List of audio waveforms as numpy arrays
    /// * `sample_rate` - Sample rate of the audio (will be resampled to 16kHz if different)
    /// * `as_spk` - If true, return speaker embedding (average of utterance embeddings)
    /// * `trim_top_db` - dB threshold for silence trimming (None to disable)
    ///
    /// # Returns
    /// Embeddings as a numpy array (256-dimensional, L2-normalized)
    pub fn embeds_from_wavs(
        &self,
        py: Python<'_>,
        wavs: &Bound<'_, PyAny>,
        sample_rate: u32,
        as_spk: bool,
        trim_top_db: Option<f32>,
    ) -> PyResult<Py<PyAny>> {
        let kwargs = PyDict::new(py);
        kwargs.set_item("sample_rate", sample_rate)?;
        kwargs.set_item("as_spk", as_spk)?;
        if let Some(db) = trim_top_db {
            kwargs.set_item("trim_top_db", db)?;
        }

        let result = self.inner.call_method(py, "embeds_from_wavs", (wavs,), Some(&kwargs))?;
        Ok(result)
    }

    /// Compute cosine similarity between two sets of embeddings.
    ///
    /// # Arguments
    /// * `embeds_x` - First set of embeddings
    /// * `embeds_y` - Second set of embeddings
    ///
    /// # Returns
    /// Similarity score(s) in range [-1, 1]
    pub fn voice_similarity(
        py: Python<'_>,
        embeds_x: &Bound<'_, PyAny>,
        embeds_y: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let ve_mod = py.import("chatterbox.models.voice_encoder")?;
        let ve_class = ve_mod.getattr("VoiceEncoder")?;
        let result = ve_class.call_method1("voice_similarity", (embeds_x, embeds_y))?;
        Ok(result.into())
    }

    /// Average utterance embeddings into a single speaker embedding.
    ///
    /// # Arguments
    /// * `utt_embeds` - Utterance embeddings to average
    ///
    /// # Returns
    /// Speaker embedding (256-dimensional, L2-normalized)
    pub fn utt_to_spk_embed(
        py: Python<'_>,
        utt_embeds: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let ve_mod = py.import("chatterbox.models.voice_encoder")?;
        let ve_class = ve_mod.getattr("VoiceEncoder")?;
        let result = ve_class.call_method1("utt_to_spk_embed", (utt_embeds,))?;
        Ok(result.into())
    }
}

/// Compute mel-spectrogram from audio waveform.
///
/// This wraps the Python melspectrogram function.
pub fn melspectrogram(
    py: Python<'_>,
    wav: &Bound<'_, PyAny>,
    pad: bool,
) -> PyResult<Py<PyAny>> {
    let melspec_mod = py.import("chatterbox.models.voice_encoder.melspec")?;
    let config_mod = py.import("chatterbox.models.voice_encoder.config")?;

    let hp = config_mod.getattr("VoiceEncConfig")?.call0()?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("pad", pad)?;

    let result = melspec_mod.call_method("melspectrogram", (wav, hp), Some(&kwargs))?;
    Ok(result.into())
}
