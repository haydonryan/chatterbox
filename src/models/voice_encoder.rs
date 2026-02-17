//! Rust implementation of the VoiceEncoder.
//!
//! This replaces the Python VoiceEncoder and provides speaker embeddings
//! directly from waveforms or mel spectrograms.

use std::path::Path;

use tch::{Device, IndexOp, Kind, Tensor};

use crate::error::{ChatterboxError, Result};
use crate::models::s3gen::utils::mel::mel_filterbank;
use crate::models::s3gen::weights::Weights;

/// Configuration for the VoiceEncoder.
#[derive(Debug, Clone)]
pub struct VoiceEncConfig {
    pub sample_rate: i64,
    pub preemphasis: f64,
    pub n_fft: i64,
    pub hop_size: i64,
    pub win_size: i64,
    pub num_mels: i64,
    pub fmin: f64,
    pub fmax: f64,
    pub mel_power: f64,
    pub ve_hidden_size: i64,
    pub speaker_embed_size: i64,
    pub ve_partial_frames: i64,
    pub normalized_mels: bool,
    pub ve_final_relu: bool,
    pub stft_magnitude_min: f64,
}

impl Default for VoiceEncConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            preemphasis: 0.0,
            n_fft: 400,
            hop_size: 160,
            win_size: 400,
            num_mels: 40,
            fmin: 0.0,
            fmax: 8000.0,
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

pub struct VoiceEncoder {
    config: VoiceEncConfig,
    device: Device,
    dtype: Kind,
    lstm_weights: Vec<Tensor>,
    proj_weight: Tensor,
    proj_bias: Tensor,
    similarity_weight: Option<Tensor>,
    similarity_bias: Option<Tensor>,
}

impl VoiceEncoder {
    pub fn from_safetensors<P: AsRef<Path>>(path: P, device: Device) -> Result<Self> {
        let weights = Weights::load(path, device)?;
        let prefix = find_prefix(&weights, &["", "model", "voice_encoder", "ve"])?;

        let (lstm_weights, hidden, dtype) = load_lstm_weights(&weights, &prefix)?;

        let proj_weight = weights.get(&prefixed(&prefix, "proj.weight"))?;
        let proj_bias = weights.get(&prefixed(&prefix, "proj.bias"))?;

        let similarity_weight = {
            let key = prefixed(&prefix, "similarity_weight");
            if weights.contains(&key) {
                Some(weights.get(&key)?)
            } else {
                None
            }
        };
        let similarity_bias = {
            let key = prefixed(&prefix, "similarity_bias");
            if weights.contains(&key) {
                Some(weights.get(&key)?)
            } else {
                None
            }
        };

        let config = VoiceEncConfig {
            ve_hidden_size: hidden,
            ..VoiceEncConfig::default()
        };

        Ok(Self {
            config,
            device,
            dtype,
            lstm_weights,
            proj_weight,
            proj_bias,
            similarity_weight,
            similarity_bias,
        })
    }

    pub fn config(&self) -> &VoiceEncConfig {
        &self.config
    }

    /// Compute speaker embeddings from a list of waveforms.
    pub fn embeds_from_wavs(
        &self,
        wavs: &[Tensor],
        sample_rate: i64,
        as_spk: bool,
        trim_top_db: Option<f64>,
        batch_size: Option<i64>,
    ) -> Result<Tensor> {
        if wavs.is_empty() {
            return Err(ChatterboxError::ShapeMismatch(
                "VoiceEncoder requires at least one waveform".to_string(),
            ));
        }

        let mut mels: Vec<Tensor> = Vec::with_capacity(wavs.len());
        for wav in wavs {
            let mut wav = if wav.dim() == 1 {
                wav.unsqueeze(0)
            } else {
                wav.shallow_clone()
            };
            if wav.dim() != 2 || wav.size()[0] != 1 {
                return Err(ChatterboxError::ShapeMismatch(
                    "VoiceEncoder expects [T] or [1, T] waveform".to_string(),
                ));
            }
            wav = wav.to_device(self.device).to_kind(self.dtype);
            let wav = if sample_rate != self.config.sample_rate {
                resample_linear(&wav, sample_rate, self.config.sample_rate)?
            } else {
                wav
            };
            let wav = if let Some(db) = trim_top_db {
                trim_silence(&wav, db)?
            } else {
                wav
            };
            let mel = melspectrogram(&wav, &self.config, true);
            let mel = mel.transpose(0, 1); // (M, T) -> (T, M)
            mels.push(mel);
        }

        let mel_lens: Vec<i64> = mels.iter().map(|m| m.size()[0]).collect();
        let packed = pack_mels(&mels, self.device)?;
        self.embeds_from_mels(&packed, Some(&mel_lens), as_spk, batch_size)
    }

    /// Compute speaker embeddings from mel spectrograms.
    pub fn embeds_from_mels(
        &self,
        mels: &Tensor,
        mel_lens: Option<&[i64]>,
        as_spk: bool,
        batch_size: Option<i64>,
    ) -> Result<Tensor> {
        let mels = mels.to_device(self.device).to_kind(self.dtype);
        let batch = mels.size()[0];
        let mel_lens: Vec<i64> = if let Some(lens) = mel_lens {
            lens.to_vec()
        } else {
            vec![mels.size()[1]; batch as usize]
        };

        let utt_embeds = self.inference(&mels, &mel_lens, 0.5, Some(1.3), 0.8, batch_size)?;
        if as_spk {
            Ok(Self::utt_to_spk_embed(&utt_embeds))
        } else {
            Ok(utt_embeds)
        }
    }

    /// Average utterance embeddings into a single speaker embedding.
    pub fn utt_to_spk_embed(utt_embeds: &Tensor) -> Tensor {
        let mean = utt_embeds.mean_dim(&[0_i64][..], false, Kind::Float);
        l2_normalize(&mean)
    }

    fn inference(
        &self,
        mels: &Tensor,
        mel_lens: &[i64],
        overlap: f64,
        rate: Option<f64>,
        min_coverage: f64,
        batch_size: Option<i64>,
    ) -> Result<Tensor> {
        let batch = mels.size()[0];
        let frame_step = get_frame_step(overlap, rate, &self.config)?;

        let mut n_partials: Vec<i64> = Vec::with_capacity(batch as usize);
        let mut target_lens: Vec<i64> = Vec::with_capacity(batch as usize);
        for &len in mel_lens {
            let (n_wins, target) = get_num_wins(len, frame_step, min_coverage, &self.config);
            n_partials.push(n_wins);
            target_lens.push(target);
        }

        let max_target = target_lens.iter().copied().max().unwrap_or(0);
        let mut mels = mels.shallow_clone();
        if max_target > mels.size()[1] {
            let pad = Tensor::zeros(
                [batch, max_target - mels.size()[1], self.config.num_mels],
                (self.dtype, self.device),
            );
            mels = Tensor::cat(&[mels, pad], 1);
        }

        let mut partials: Vec<Tensor> = Vec::new();
        for b in 0..batch {
            let mel = mels.i((b, .., ..));
            let n = n_partials[b as usize];
            for i in 0..n {
                let start = i * frame_step;
                let part = mel.narrow(0, start, self.config.ve_partial_frames);
                partials.push(part);
            }
        }
        let partials = Tensor::stack(&partials, 0);

        let total = partials.size()[0];
        let chunk = batch_size.unwrap_or(total).max(1);
        let mut embeds: Vec<Tensor> = Vec::new();
        let mut offset = 0;
        while offset < total {
            let len = (total - offset).min(chunk);
            let part = partials.narrow(0, offset, len);
            let emb = self.forward_partials(&part);
            embeds.push(emb);
            offset += len;
        }
        let partial_embeds = Tensor::cat(&embeds, 0);

        let mut utt_embeds: Vec<Tensor> = Vec::with_capacity(batch as usize);
        let mut cursor = 0;
        for n in n_partials {
            let slice = partial_embeds.narrow(0, cursor, n);
            let mean = slice.mean_dim(&[0_i64][..], false, Kind::Float);
            utt_embeds.push(l2_normalize(&mean));
            cursor += n;
        }

        Ok(Tensor::stack(&utt_embeds, 0))
    }

    fn forward_partials(&self, mels: &Tensor) -> Tensor {
        let input = mels.to_device(self.device).to_kind(self.dtype);
        let batch = input.size()[0];
        let h0 = Tensor::zeros(
            [3, batch, self.config.ve_hidden_size],
            (self.dtype, self.device),
        );
        let c0 = Tensor::zeros(
            [3, batch, self.config.ve_hidden_size],
            (self.dtype, self.device),
        );

        let params: Vec<&Tensor> = self.lstm_weights.iter().collect();
        let (_out, h, _c) = input.lstm(
            &[&h0, &c0],
            &params,
            true,
            3,
            0.0,
            false,
            false,
            true,
        );
        let h_last = h.i((2, .., ..));
        let mut proj = h_last.matmul(&self.proj_weight.transpose(0, 1)) + &self.proj_bias;
        if self.config.ve_final_relu {
            proj = proj.relu();
        }
        l2_normalize(&proj)
    }
}

fn find_prefix(weights: &Weights, candidates: &[&str]) -> Result<String> {
    for cand in candidates {
        let key = prefixed(cand, "lstm.weight_ih_l0");
        if weights.contains(&key) {
            return Ok(cand.to_string());
        }
    }
    Err(ChatterboxError::MissingWeight(
        "lstm.weight_ih_l0".to_string(),
    ))
}

fn prefixed(prefix: &str, key: &str) -> String {
    if prefix.is_empty() {
        key.to_string()
    } else {
        format!("{prefix}.{key}")
    }
}

fn load_lstm_weights(weights: &Weights, prefix: &str) -> Result<(Vec<Tensor>, i64, Kind)> {
    let mut flat = Vec::with_capacity(12);
    let mut hidden = 0;
    let mut dtype = Kind::Float;
    for layer in 0..3 {
        let w_ih = weights.get(&prefixed(prefix, &format!("lstm.weight_ih_l{layer}")))?;
        let w_hh = weights.get(&prefixed(prefix, &format!("lstm.weight_hh_l{layer}")))?;
        hidden = w_ih.size()[0] / 4;
        dtype = w_ih.kind();

        let bias_ih_key = prefixed(prefix, &format!("lstm.bias_ih_l{layer}"));
        let bias_hh_key = prefixed(prefix, &format!("lstm.bias_hh_l{layer}"));
        let bias_ih = if weights.contains(&bias_ih_key) {
            weights.get(&bias_ih_key)?
        } else {
            Tensor::zeros([4 * hidden], (dtype, w_ih.device()))
        };
        let bias_hh = if weights.contains(&bias_hh_key) {
            weights.get(&bias_hh_key)?
        } else {
            Tensor::zeros([4 * hidden], (dtype, w_ih.device()))
        };

        flat.push(w_ih);
        flat.push(w_hh);
        flat.push(bias_ih);
        flat.push(bias_hh);
    }
    Ok((flat, hidden, dtype))
}

fn pack_mels(mels: &[Tensor], device: Device) -> Result<Tensor> {
    let batch = mels.len() as i64;
    let max_len = mels.iter().map(|m| m.size()[0]).max().unwrap_or(0);
    let n_mels = mels
        .first()
        .map(|m| m.size()[1])
        .unwrap_or(0);
    let kind = mels.first().map(|m| m.kind()).unwrap_or(Kind::Float);
    let mut out = Tensor::zeros([batch, max_len, n_mels], (kind, device));
    for (idx, mel) in mels.iter().enumerate() {
        let len = mel.size()[0];
        out.i((idx as i64, 0..len, ..)).copy_(mel);
    }
    Ok(out)
}

fn get_num_wins(n_frames: i64, step: i64, min_coverage: f64, cfg: &VoiceEncConfig) -> (i64, i64) {
    let win_size = cfg.ve_partial_frames;
    let tmp = (n_frames - win_size + step).max(0);
    let n_wins = tmp / step;
    let remainder = tmp % step;
    let mut n_wins = n_wins;
    if n_wins == 0 || (remainder + (win_size - step)) as f64 / win_size as f64 >= min_coverage {
        n_wins += 1;
    }
    let target_n = win_size + step * (n_wins - 1);
    (n_wins, target_n)
}

fn get_frame_step(overlap: f64, rate: Option<f64>, cfg: &VoiceEncConfig) -> Result<i64> {
    if !(0.0..1.0).contains(&overlap) {
        return Err(ChatterboxError::ShapeMismatch(
            "overlap must be in [0, 1)".to_string(),
        ));
    }
    let step = if let Some(rate) = rate {
        ((cfg.sample_rate as f64 / rate) / cfg.ve_partial_frames as f64).round() as i64
    } else {
        (cfg.ve_partial_frames as f64 * (1.0 - overlap)).round() as i64
    };
    if step <= 0 || step > cfg.ve_partial_frames {
        return Err(ChatterboxError::ShapeMismatch(
            "invalid frame_step for VoiceEncoder".to_string(),
        ));
    }
    Ok(step)
}

fn melspectrogram(wav: &Tensor, cfg: &VoiceEncConfig, pad: bool) -> Tensor {
    let mut y = if wav.dim() == 1 {
        wav.unsqueeze(0)
    } else {
        wav.shallow_clone()
    };
    let device = y.device();

    if cfg.preemphasis > 0.0 {
        let coeff = cfg.preemphasis;
        let first = y.i((.., 0..1));
        let rest = y.i((.., 1..));
        let prev = y.i((.., 0..y.size()[1] - 1));
        let rest = rest - prev * coeff;
        y = Tensor::cat(&[first, rest], 1);
    }

    let window = Tensor::hann_window(cfg.win_size, (Kind::Float, device));
    let spec = y.stft_center(
        cfg.n_fft,
        Some(cfg.hop_size),
        Some(cfg.win_size),
        Some(&window),
        pad,
        "reflect",
        false,
        true,
        true,
        false,
    );
    let spec = spec.view_as_real();
    let mut mag = spec
        .pow_tensor_scalar(2.0)
        .sum_dim_intlist([-1].as_slice(), false, Kind::Float)
        .sqrt();

    if (cfg.mel_power - 1.0).abs() > f64::EPSILON {
        mag = mag.pow_tensor_scalar(cfg.mel_power);
    }

    let mel_basis = mel_filterbank(
        cfg.n_fft,
        cfg.num_mels,
        cfg.sample_rate,
        cfg.fmin,
        cfg.fmax,
        device,
    );
    let mel = mel_basis.unsqueeze(0).matmul(&mag);
    mel.squeeze_dim(0)
}

fn l2_normalize(x: &Tensor) -> Tensor {
    let denom = x
        .pow_tensor_scalar(2.0)
        .sum_dim_intlist([-1].as_slice(), true, Kind::Float)
        .sqrt()
        .clamp_min(1e-12);
    x / denom
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

fn trim_silence(wav: &Tensor, top_db: f64) -> Result<Tensor> {
    let wav = if wav.dim() == 1 { wav } else { &wav.i((0, ..)) };
    let abs = wav.abs();
    let max = abs.max().double_value(&[]);
    if max <= 0.0 {
        return Ok(wav.shallow_clone());
    }
    let threshold = max * 10f64.powf(-top_db / 20.0);
    let mask = abs.ge(threshold);
    let idx = mask.nonzero();
    if idx.numel() == 0 {
        return Ok(wav.shallow_clone());
    }
    let start = idx.i((0, 0)).int64_value(&[]);
    let end = idx.i((idx.size()[0] - 1, 0)).int64_value(&[]) + 1;
    Ok(wav.narrow(0, start, end - start))
}
