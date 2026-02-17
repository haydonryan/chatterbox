//! Rust implementation scaffold for S3Gen (S3Token2Wav).
//!
//! This module is the Rust-native replacement for `src/chatterbox/models/s3gen`.
//! It loads weights from `s3gen.safetensors` and will progressively mirror the
//! Python implementation (flow + HiFiGAN + tokenizer + speaker encoder).

pub mod utils;
pub mod weights;
pub mod tokenizer;
pub mod speaker_encoder;

use std::path::Path;
use tch::{Device, Kind, Tensor};

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
        Ok(Self {
            weights,
            device,
            meanflow,
            dtype: Kind::Float,
            tokenizer,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn tokenizer(&self) -> &tokenizer::S3Tokenizer {
        &self.tokenizer
    }

    pub fn dtype(&self) -> Kind {
        self.dtype
    }

    pub fn set_dtype(&mut self, dtype: Kind) {
        self.dtype = dtype;
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
        Err(ChatterboxError::NotImplemented(
            "S3Gen Rust inference is not implemented yet",
        ))
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
        Err(ChatterboxError::NotImplemented(
            "S3Gen Rust flow inference is not implemented yet",
        ))
    }

    /// Run HiFiGAN vocoder inference (mel -> waveform).
    ///
    /// NOTE: full Rust implementation is in-progress.
    pub fn hift_inference(&self, _speech_feat: &Tensor) -> Result<Tensor> {
        Err(ChatterboxError::NotImplemented(
            "S3Gen Rust HiFiGAN inference is not implemented yet",
        ))
    }

    /// Access raw weights (for module implementations).
    pub fn weights(&self) -> &weights::Weights {
        &self.weights
    }
}
