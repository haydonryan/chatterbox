//! Rust implementation scaffold for S3Gen (S3Token2Wav).
//!
//! This module is the Rust-native replacement for `src/chatterbox/models/s3gen`.
//! It loads weights from `s3gen.safetensors` and will progressively mirror the
//! Python implementation (flow + HiFiGAN + tokenizer + speaker encoder).

pub mod utils;
pub mod weights;
pub mod tokenizer;
pub mod speaker_encoder;
pub mod hifigan;
pub mod flow;
use crate::models::s3gen::utils::mel::mel_spectrogram;

use std::path::Path;
use tch::{Device, IndexOp, Kind, Tensor};

use crate::error::{ChatterboxError, Result};

/// Output sample rate for S3Gen (24kHz).
pub const S3GEN_SR: u32 = 24000;

/// Input/conditioning sample rate (16kHz).
pub const S3_SR: u32 = 16000;

/// Speech tokenizer vocabulary size.
pub const SPEECH_VOCAB_SIZE: i64 = 6561;

/// Reference embeddings from S3Gen.embed_ref().
///
/// Contains pre-computed speaker and prompt information for inference.
#[derive(Debug)]
pub struct S3GenRef {
    pub prompt_token: Tensor,
    pub prompt_token_len: Tensor,
    pub prompt_feat: Tensor,
    pub prompt_feat_len: Option<Tensor>,
    pub embedding: Tensor,
}

/// Backwards-compatible alias for the previous Python-backed type name.
pub type S3GenRefDict = S3GenRef;

impl S3GenRef {
    pub fn new(
        prompt_token: Tensor,
        prompt_token_len: Tensor,
        prompt_feat: Tensor,
        prompt_feat_len: Option<Tensor>,
        embedding: Tensor,
    ) -> Self {
        Self {
            prompt_token,
            prompt_token_len,
            prompt_feat,
            prompt_feat_len,
            embedding,
        }
    }

    pub fn to_device(&self, device: Device) -> Self {
        Self {
            prompt_token: self.prompt_token.to_device(device),
            prompt_token_len: self.prompt_token_len.to_device(device),
            prompt_feat: self.prompt_feat.to_device(device),
            prompt_feat_len: self.prompt_feat_len.as_ref().map(|t| t.to_device(device)),
            embedding: self.embedding.to_device(device),
        }
    }
}

/// Options for S3Gen inference.
#[derive(Debug, Clone)]
pub struct S3GenInferenceOptions {
    /// Number of CFM ODE solver steps (default: 10 for standard, 2 for meanflow).
    pub n_cfm_timesteps: Option<i64>,
    /// Whether to drop invalid tokens before inference.
    pub drop_invalid_tokens: bool,
}

impl Default for S3GenInferenceOptions {
    fn default() -> Self {
        Self {
            n_cfm_timesteps: None,
            drop_invalid_tokens: true,
        }
    }
}

impl S3GenInferenceOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn n_cfm_timesteps(mut self, value: i64) -> Self {
        self.n_cfm_timesteps = Some(value);
        self
    }

    pub fn drop_invalid_tokens(mut self, value: bool) -> Self {
        self.drop_invalid_tokens = value;
        self
    }
}

/// Rust-native S3Gen model.
///
/// This struct loads and owns all S3Gen weights. The forward path
/// will be implemented in Rust (no Python).
pub struct S3Gen {
    weights: weights::Weights,
    device: Device,
    meanflow: bool,
    dtype: Kind,
    tokenizer: tokenizer::S3Tokenizer,
    speaker_encoder: speaker_encoder::CampPlus,
    hifigan: hifigan::HiFTGenerator,
    flow: flow::CausalMaskedDiffWithXvec,
}

impl S3Gen {
    /// Load S3Gen from a `s3gen.safetensors` checkpoint.
    pub fn from_safetensors<P: AsRef<Path>>(
        path: P,
        device: Device,
        meanflow: bool,
    ) -> Result<Self> {
        let weights = weights::Weights::load(path, device)?;
        let tokenizer = tokenizer::S3Tokenizer::from_weights(&weights, device)?;
        let speaker_encoder = speaker_encoder::CampPlus::from_weights(&weights)?;
        let hifigan = hifigan::HiFTGenerator::from_weights(&weights)?;
        let flow = flow::CausalMaskedDiffWithXvec::from_weights(&weights)?;
        Ok(Self {
            weights,
            device,
            meanflow,
            dtype: Kind::Float,
            tokenizer,
            speaker_encoder,
            hifigan,
            flow,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn tokenizer(&self) -> &tokenizer::S3Tokenizer {
        &self.tokenizer
    }

    pub fn speaker_encoder(&self) -> &speaker_encoder::CampPlus {
        &self.speaker_encoder
    }

    pub fn resample_to_16k(&self, wav: &Tensor) -> Result<Tensor> {
        resample_linear(wav, S3GEN_SR as i64, S3_SR as i64)
    }

    pub fn dtype(&self) -> Kind {
        self.dtype
    }

    pub fn set_dtype(&mut self, dtype: Kind) {
        self.dtype = dtype;
    }

    /// Compute reference embeddings (prompt tokens, mel, speaker embedding).
    pub fn embed_ref(&self, ref_wav: &Tensor, ref_sr: i64) -> Result<S3GenRef> {
        let mut ref_wav = if ref_wav.dim() == 1 {
            ref_wav.unsqueeze(0)
        } else {
            ref_wav.shallow_clone()
        };

        if ref_wav.size()[1] > 10 * ref_sr {
            eprintln!("[WARN] S3Gen received ref longer than 10s ({} samples at {}Hz).", ref_wav.size()[1], ref_sr);
        }

        ref_wav = ref_wav.to_device(self.device).to_kind(self.dtype);

        let ref_wav_24 = if ref_sr != S3GEN_SR as i64 {
            resample_linear(&ref_wav, ref_sr, S3GEN_SR as i64)?
        } else {
            ref_wav.shallow_clone()
        };

        let ref_mels_24 = mel_spectrogram(
            &ref_wav_24,
            1920,
            80,
            S3GEN_SR as i64,
            480,
            1920,
            0.0,
            8000.0,
            false,
        )
        .transpose(1, 2)
        .to_kind(self.dtype);

        let ref_wav_16 = if ref_sr != S3_SR as i64 {
            resample_linear(&ref_wav, ref_sr, S3_SR as i64)?
        } else {
            ref_wav.shallow_clone()
        };

        let embedding = self.speaker_encoder.inference(&[ref_wav_16.shallow_clone()])?;

        let (mut ref_tokens, mut ref_token_lens) =
            self.tokenizer.forward(&ref_wav_16.to_kind(Kind::Float), None)?;

        if ref_mels_24.size()[1] != 2 * ref_tokens.size()[1] {
            eprintln!(
                "[WARN] Reference mel length is not equal to 2 * reference token length."
            );
            let keep = ref_mels_24.size()[1] / 2;
            ref_tokens = ref_tokens.narrow(1, 0, keep);
            if ref_token_lens.numel() > 0 {
                ref_token_lens.i(0).fill_(keep);
            }
        }

        Ok(S3GenRef::new(
            ref_tokens,
            ref_token_lens,
            ref_mels_24,
            None,
            embedding,
        ))
    }

    /// Run S3Gen inference (token -> waveform).
    ///
    /// NOTE: full Rust implementation is in-progress.
    pub fn inference(
        &self,
        _speech_tokens: &Tensor,
        _ref_dict: &S3GenRef,
        _options: &S3GenInferenceOptions,
    ) -> Result<(Tensor, Tensor)> {
        let ref_dict = _ref_dict.to_device(self.device);
        let mut speech_tokens = _speech_tokens.to_device(self.device);
        if _options.drop_invalid_tokens {
            let mask = speech_tokens.lt(SPEECH_VOCAB_SIZE);
            speech_tokens = speech_tokens.masked_select(&mask);
        }

        if speech_tokens.dim() == 1 {
            speech_tokens = speech_tokens.unsqueeze(0);
        }

        let n_timesteps = _options.n_cfm_timesteps;
        let mel = self.flow_inference(&speech_tokens, &ref_dict, true, n_timesteps)?;
        if std::env::var("S3GEN_DEBUG").is_ok() {
            debug_stats("flow_mel", &mel);
        }
        let wav = self.hift_inference(&mel)?;
        if std::env::var("S3GEN_DEBUG").is_ok() {
            debug_stats("hifigan_wav", &wav);
        }
        Ok((wav, Tensor::zeros([1, 1, 0], (Kind::Float, self.device))))
    }

    /// Run flow inference (token -> mel).
    ///
    /// NOTE: full Rust implementation is in-progress.
    pub fn flow_inference(
        &self,
        _speech_tokens: &Tensor,
        _ref_dict: &S3GenRef,
        _finalize: bool,
        _n_cfm_timesteps: Option<i64>,
    ) -> Result<Tensor> {
        let ref_dict = _ref_dict.to_device(self.device);
        let tokens = if _speech_tokens.dim() == 1 {
            _speech_tokens.to_device(self.device).unsqueeze(0)
        } else {
            _speech_tokens.to_device(self.device)
        };
        let token_len = Tensor::from_slice(&[tokens.size()[1]]).to_device(self.device);
        let prompt_token = ref_dict.prompt_token.to_device(self.device);
        let prompt_token_len = ref_dict.prompt_token_len.to_device(self.device);
        let prompt_feat = ref_dict.prompt_feat.to_device(self.device);
        let embedding = ref_dict.embedding.to_device(self.device);
        let (mel, _cache) = self.flow.inference(
            &tokens,
            &token_len,
            &prompt_token,
            &prompt_token_len,
            &prompt_feat,
            &embedding,
            _finalize,
            _n_cfm_timesteps,
        )?;
        Ok(mel)
    }

    /// Run HiFiGAN vocoder inference (mel -> waveform).
    ///
    /// NOTE: full Rust implementation is in-progress.
    pub fn hift_inference(&self, _speech_feat: &Tensor) -> Result<Tensor> {
        let (wav, _source) = self.hifigan.inference(_speech_feat, &Tensor::zeros([1, 1, 0], (Kind::Float, self.device)));
        Ok(wav)
    }

    /// Access raw weights (for module implementations).
    pub fn weights(&self) -> &weights::Weights {
        &self.weights
    }
}

fn resample_linear(wav: &Tensor, src_sr: i64, dst_sr: i64) -> Result<Tensor> {
    if src_sr == dst_sr {
        return Ok(wav.shallow_clone());
    }
    let squeeze = wav.dim() == 1;
    let wav = if squeeze {
        wav.unsqueeze(0)
    } else {
        wav.shallow_clone()
    };
    if wav.dim() != 2 {
        return Err(ChatterboxError::ShapeMismatch(
            "resample_linear expects [T] or [B, T] waveform".to_string(),
        ));
    }
    let scale = dst_sr as f64 / src_sr as f64;
    let wav = wav.unsqueeze(1);
    let out = wav.upsample_linear1d_vec(None::<&[i64]>, false, &[scale]);
    let out = out.squeeze_dim(1);
    Ok(if squeeze { out.squeeze_dim(0) } else { out })
}

fn debug_stats(name: &str, tensor: &Tensor) {
    let t = tensor.to_device(Device::Cpu).to_kind(Kind::Float);
    let mean = t.mean(Kind::Float).double_value(&[]);
    let min = t.min().double_value(&[]);
    let max = t.max().double_value(&[]);
    let shape = t.size();
    eprintln!(
        "[S3GEN_DEBUG] {} shape={:?} mean={:.6} min={:.6} max={:.6}",
        name, shape, mean, min, max
    );
}
