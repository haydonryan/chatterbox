use tch::{Device, IndexOp, Kind, Tensor};

use crate::error::{ChatterboxError, Result};
use crate::models::s3gen::weights::Weights;

const N_MELS: i64 = 128;
const N_AUDIO_STATE: i64 = 1280;
const N_AUDIO_HEAD: i64 = 20;
const N_AUDIO_LAYER: usize = 6;
const N_FFT: i64 = 400;
const HOP_SIZE: i64 = 160;
const MAX_FRAMES: i64 = 3000;

pub struct S3Tokenizer {
    mel_filters: Tensor,
    window: Tensor,
    encoder: AudioEncoderV2,
    quantizer: FSQVectorQuantization,
    device: Device,
}

impl S3Tokenizer {
    pub fn from_weights(weights: &Weights, device: Device) -> Result<Self> {
        let mel_filters = weights.get("tokenizer._mel_filters")?;
        let window = weights.get("tokenizer.window")?;
        let encoder = AudioEncoderV2::from_weights(weights)?;
        let quantizer = FSQVectorQuantization::from_weights(weights)?;

        Ok(Self {
            mel_filters,
            window,
            encoder,
            quantizer,
            device,
        })
    }

    pub fn forward(&self, wavs: &Tensor, max_len: Option<i64>) -> Result<(Tensor, Tensor)> {
        let wavs = if wavs.dim() == 1 {
            wavs.unsqueeze(0)
        } else {
            wavs.shallow_clone()
        };
        let batch = wavs.size()[0];
        let mut mels: Vec<Tensor> = Vec::with_capacity(batch as usize);
        for idx in 0..batch {
            let wav = wavs.get(idx);
            let mut mel = self.log_mel_spectrogram(&wav, 0);
            if let Some(max_len) = max_len {
                let max_frames = max_len * 4;
                let mel_frames = mel.size()[2];
                let take = mel_frames.min(max_frames);
                mel = mel.narrow(2, 0, take);
            }
            mels.push(mel.squeeze_dim(0));
        }

        let (padded, mel_lens) = pad_mels(&mels);
        if mel_lens.gt(MAX_FRAMES).any().int64_value(&[]) != 0 {
            return Err(ChatterboxError::NotImplemented(
                "S3Tokenizer long-audio path not implemented",
            ));
        }

        let (codes, code_len) = self.encoder.quantize(&padded, &mel_lens, &self.quantizer)?;
        Ok((codes, code_len))
    }

    fn log_mel_spectrogram(&self, audio: &Tensor, padding: i64) -> Tensor {
        let mut audio = if audio.dim() == 1 {
            audio.unsqueeze(0)
        } else {
            audio.shallow_clone()
        };
        audio = audio.to_device(self.device);
        if padding > 0 {
            audio = audio.pad(&[0, padding], "constant", Some(0.0));
        }
        let window = self.window.to_device(self.device);
        let stft = audio.stft_center(
            N_FFT,
            Some(HOP_SIZE),
            Some(N_FFT),
            Some(&window),
            true,
            "reflect",
            false,
            true,
            true,
            false,
        );
        let frames = stft.size()[2].saturating_sub(1);
        let stft = stft.narrow(2, 0, frames);
        let magnitudes = stft.abs().pow_tensor_scalar(2.0);
        let mel_filters = self.mel_filters.to_device(self.device);
        let mel_spec = mel_filters.unsqueeze(0).matmul(&magnitudes);
        let log_spec = mel_spec.clamp_min(1e-10).log10();
        let log_spec = log_spec.maximum(&(log_spec.max() - 8.0));
        (log_spec + 4.0) / 4.0
    }
}

fn pad_mels(samples: &[Tensor]) -> (Tensor, Tensor) {
    let batch = samples.len() as i64;
    let mut lengths: Vec<i64> = Vec::with_capacity(samples.len());
    for sample in samples {
        lengths.push(sample.size()[1]);
    }
    let max_len = *lengths.iter().max().unwrap_or(&0);
    let n_mels = samples.get(0).map(|s| s.size()[0]).unwrap_or(N_MELS);
    let device = samples.get(0).map(|s| s.device()).unwrap_or(Device::Cpu);

    let mut padded = Tensor::zeros([batch, n_mels, max_len], (Kind::Float, device));
    for (idx, sample) in samples.iter().enumerate() {
        let len = lengths[idx];
        padded.i((idx as i64, .., 0..len)).copy_(sample);
    }

    let lengths = Tensor::from_slice(&lengths)
        .to_device(device)
        .to_kind(Kind::Int64);
    (padded, lengths)
}

struct AudioEncoderV2 {
    conv1: Conv1d,
    conv2: Conv1d,
    blocks: Vec<ResidualAttentionBlock>,
    rotary_cos: Tensor,
    rotary_sin: Tensor,
}

impl AudioEncoderV2 {
    fn from_weights(weights: &Weights) -> Result<Self> {
        let conv1 = Conv1d::new(
            weights.get("tokenizer.encoder.conv1.weight")?,
            Some(weights.get("tokenizer.encoder.conv1.bias")?),
            2,
            1,
            1,
            1,
        );
        let conv2 = Conv1d::new(
            weights.get("tokenizer.encoder.conv2.weight")?,
            Some(weights.get("tokenizer.encoder.conv2.bias")?),
            2,
            1,
            1,
            1,
        );
        let mut blocks = Vec::with_capacity(N_AUDIO_LAYER);
        for i in 0..N_AUDIO_LAYER {
            blocks.push(ResidualAttentionBlock::from_weights(weights, i)?);
        }
        let (rotary_cos, rotary_sin) = precompute_rotary(64, 2048, Device::Cpu);
        Ok(Self {
            conv1,
            conv2,
            blocks,
            rotary_cos,
            rotary_sin,
        })
    }

    fn quantize(
        &self,
        mel: &Tensor,
        mel_len: &Tensor,
        quantizer: &FSQVectorQuantization,
    ) -> Result<(Tensor, Tensor)> {
        let (hidden, code_len) = self.forward(mel, mel_len)?;
        let code = quantizer.encode(&hidden);
        Ok((code, code_len))
    }

    fn forward(&self, x: &Tensor, x_len: &Tensor) -> Result<(Tensor, Tensor)> {
        let mut x = x.shallow_clone();
        let mut x_len = x_len.shallow_clone();
        let t = x.size()[2];
        let mask = make_non_pad_mask(&x_len, t).unsqueeze(1);
        x = x * &mask;
        x = self.conv1.forward(&x).gelu("none");
        x_len = conv_out_length(&x_len, 3, 2, 1, 1);
        let x_slen = conv_out_length_scalar(t, 3, 2, 1, 1);
        let mask = make_non_pad_mask(&x_len, x_slen).unsqueeze(1);
        x = x * &mask;
        x = self.conv2.forward(&x).gelu("none");
        x_len = conv_out_length(&x_len, 3, 2, 1, 1);
        let x_slen = conv_out_length_scalar(x_slen, 3, 2, 1, 1);
        let mask = make_non_pad_mask(&x_len, x_slen).unsqueeze(1);
        x = x.permute(&[0, 2, 1]); // (B, T, C)
        let mask_pad = mask.transpose(1, 2);
        let mask_bias = mask_to_bias(&mask, x.kind());
        let seq_len = x.size()[1];
        let cos = self.rotary_cos.i((0..seq_len, ..)).to_device(x.device());
        let sin = self.rotary_sin.i((0..seq_len, ..)).to_device(x.device());

        let mut h = x;
        for block in &self.blocks {
            h = block.forward(&h, &mask_bias.unsqueeze(1), &mask_pad, &cos, &sin)?;
        }
        Ok((h, x_len))
    }
}

struct ResidualAttentionBlock {
    attn: FSMNMultiHeadAttention,
    attn_ln: LayerNorm,
    mlp: Mlp,
    mlp_ln: LayerNorm,
}

impl ResidualAttentionBlock {
    fn from_weights(weights: &Weights, idx: usize) -> Result<Self> {
        let prefix = format!("tokenizer.encoder.blocks.{}", idx);
        Ok(Self {
            attn: FSMNMultiHeadAttention::from_weights(weights, &prefix)?,
            attn_ln: LayerNorm::from_weights(weights, &format!("{}.attn_ln", prefix))?,
            mlp: Mlp::from_weights(weights, &format!("{}.mlp", prefix))?,
            mlp_ln: LayerNorm::from_weights(weights, &format!("{}.mlp_ln", prefix))?,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        mask: &Tensor,
        mask_pad: &Tensor,
        rotary_cos: &Tensor,
        rotary_sin: &Tensor,
    ) -> Result<Tensor> {
        let attn_in = self.attn_ln.forward(x);
        let (attn_out, _) = self.attn.forward(&attn_in, mask, mask_pad, rotary_cos, rotary_sin)?;
        let mut h = x + attn_out;
        let mlp_out = self.mlp.forward(&self.mlp_ln.forward(&h));
        h = h + mlp_out;
        Ok(h)
    }
}

struct FSMNMultiHeadAttention {
    n_head: i64,
    query: Linear,
    key: Linear,
    value: Linear,
    out: Linear,
    fsmn_weight: Tensor,
    left_padding: i64,
    right_padding: i64,
}

impl FSMNMultiHeadAttention {
    fn from_weights(weights: &Weights, prefix: &str) -> Result<Self> {
        let q = Linear::from_weights(weights, &format!("{}.attn.query", prefix), true)?;
        let k = Linear::from_weights(weights, &format!("{}.attn.key", prefix), false)?;
        let v = Linear::from_weights(weights, &format!("{}.attn.value", prefix), true)?;
        let out = Linear::from_weights(weights, &format!("{}.attn.out", prefix), true)?;
        let fsmn_weight = weights.get(&format!("{}.attn.fsmn_block.weight", prefix))?;
        let kernel_size = fsmn_weight.size()[2];
        let left_padding = (kernel_size - 1) / 2;
        let right_padding = kernel_size - 1 - left_padding;
        Ok(Self {
            n_head: N_AUDIO_HEAD,
            query: q,
            key: k,
            value: v,
            out,
            fsmn_weight,
            left_padding,
            right_padding,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        mask: &Tensor,
        mask_pad: &Tensor,
        rotary_cos: &Tensor,
        rotary_sin: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let q = self.query.forward(x);
        let k = self.key.forward(x);
        let v = self.value.forward(x);
        let (out, qk, fsm_memory) =
            self.qkv_attention(&q, &k, &v, mask, mask_pad, rotary_cos, rotary_sin)?;
        let out = out + fsm_memory;
        Ok((self.out.forward(&out), qk))
    }

    fn qkv_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: &Tensor,
        mask_pad: &Tensor,
        rotary_cos: &Tensor,
        rotary_sin: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let d = q.size()[2];
        let head_dim = d / self.n_head;
        let scale = (head_dim as f64).powf(-0.25);
        let q = q.view([q.size()[0], q.size()[1], self.n_head, head_dim]);
        let k = k.view([k.size()[0], k.size()[1], self.n_head, head_dim]);
        let v = v.view([v.size()[0], v.size()[1], self.n_head, head_dim]);

        let (q, k) = apply_rotary(&q, &k, rotary_cos, rotary_sin);
        let fsm_memory = self.forward_fsmn(&v, mask_pad);

        let q = q.permute(&[0, 2, 1, 3]) * scale;
        let v = v.permute(&[0, 2, 1, 3]);
        let k = k.permute(&[0, 2, 3, 1]) * scale;
        let mut qk = q.matmul(&k);
        qk = qk + mask;
        let qk_f = qk.to_kind(Kind::Float);
        let w = qk_f.softmax(-1, Kind::Float).to_kind(q.kind());
        let out = w.matmul(&v).permute(&[0, 2, 1, 3]).flatten(2, -1);
        Ok((out, qk_f, fsm_memory))
    }

    fn forward_fsmn(&self, inputs: &Tensor, mask: &Tensor) -> Tensor {
        let b = inputs.size()[0];
        let t = inputs.size()[1];
        let inputs = inputs.view([b, t, -1]);
        let mut x = inputs.shallow_clone();
        let mask = mask.to_kind(x.kind());
        if mask.size()[2] > 0 {
            x = x * &mask;
        }
        x = x.transpose(1, 2);
        x = x.pad(&[self.left_padding, self.right_padding], "constant", Some(0.0));
        let groups = x.size()[1];
        let weight = self.fsmn_weight.to_kind(x.kind());
        let x = x.conv1d(&weight, Option::<&Tensor>::None, 1, 0, 1, groups);
        let x = x.transpose(1, 2) + inputs;
        x * mask
    }
}

struct FSQVectorQuantization {
    project_down_w: Tensor,
    project_down_b: Tensor,
}

impl FSQVectorQuantization {
    fn from_weights(weights: &Weights) -> Result<Self> {
        Ok(Self {
            project_down_w: weights.get("tokenizer.quantizer._codebook.project_down.weight")?,
            project_down_b: weights.get("tokenizer.quantizer._codebook.project_down.bias")?,
        })
    }

    fn encode(&self, x: &Tensor) -> Tensor {
        let x_shape = x.size();
        let x_flat = x.view([-1, x_shape[2]]);
        let h = x_flat.linear(
            &self.project_down_w.to_kind(x.kind()),
            Some(self.project_down_b.to_kind(x.kind())),
        );
        let h = h.tanh() * 0.9990000128746033;
        let h = h.round() + 1.0;
        let powers = Tensor::arange(8, (Kind::Float, x.device()));
        let powers = (powers * 3.0_f64.ln()).exp();
        let mu = (h * powers.unsqueeze(0)).sum_dim_intlist([1].as_slice(), false, Kind::Float);
        mu.view([x_shape[0], x_shape[1]]).to_kind(Kind::Int64)
    }
}

struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

impl Mlp {
    fn from_weights(weights: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            fc1: Linear::from_weights(weights, &format!("{}.0", prefix), true)?,
            fc2: Linear::from_weights(weights, &format!("{}.2", prefix), true)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        self.fc2.forward(&self.fc1.forward(x).gelu("none"))
    }
}

struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    fn from_weights(weights: &Weights, prefix: &str, bias: bool) -> Result<Self> {
        let weight = weights.get(&format!("{}.weight", prefix))?;
        let bias = if bias {
            Some(weights.get(&format!("{}.bias", prefix))?)
        } else {
            None
        };
        Ok(Self { weight, bias })
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let weight = self.weight.to_kind(x.kind());
        let bias = self.bias.as_ref().map(|b| b.to_kind(x.kind()));
        x.linear(&weight, bias.as_ref())
    }
}

struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl LayerNorm {
    fn from_weights(weights: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            weight: weights.get(&format!("{}.weight", prefix))?,
            bias: weights.get(&format!("{}.bias", prefix))?,
            eps: 1e-5,
        })
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let weight = self.weight.to_kind(Kind::Float);
        let bias = self.bias.to_kind(Kind::Float);
        let y = x
            .to_kind(Kind::Float)
            .layer_norm(&[x.size()[2]], Some(&weight), Some(&bias), self.eps, false);
        y.to_kind(x.kind())
    }
}

struct Conv1d {
    weight: Tensor,
    bias: Option<Tensor>,
    stride: i64,
    padding: i64,
    dilation: i64,
    groups: i64,
}

impl Conv1d {
    fn new(
        weight: Tensor,
        bias: Option<Tensor>,
        stride: i64,
        padding: i64,
        dilation: i64,
        groups: i64,
    ) -> Self {
        Self {
            weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let weight = self.weight.to_kind(x.kind());
        let bias = self.bias.as_ref().map(|b| b.to_kind(x.kind()));
        x.conv1d(&weight, bias.as_ref(), self.stride, self.padding, self.dilation, self.groups)
    }
}

fn conv_out_length(x_len: &Tensor, kernel: i64, stride: i64, pad: i64, dilation: i64) -> Tensor {
    (x_len + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1
}

fn conv_out_length_scalar(x_len: i64, kernel: i64, stride: i64, pad: i64, dilation: i64) -> i64 {
    (x_len + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1
}

fn make_non_pad_mask(lengths: &Tensor, max_len: i64) -> Tensor {
    let lengths = lengths.to_kind(Kind::Int64);
    let batch = lengths.size()[0];
    let seq_range = Tensor::arange(max_len, (Kind::Int64, lengths.device()));
    let seq_range_expand = seq_range.unsqueeze(0).expand([batch, max_len], true);
    let seq_length_expand = lengths.unsqueeze(-1);
    seq_range_expand.lt_tensor(&seq_length_expand)
}

fn mask_to_bias(mask: &Tensor, dtype: Kind) -> Tensor {
    let mask = mask.to_kind(dtype);
    (Tensor::ones_like(&mask) - mask) * -1.0e10
}

fn precompute_rotary(dim: i64, end: i64, device: Device) -> (Tensor, Tensor) {
    let exponents =
        Tensor::arange_start_step(0, dim, 2, (Kind::Float, device)) / (dim as f64);
    let inv_timescales = (-exponents * 10000f64.ln()).exp();
    let t = Tensor::arange(end, (Kind::Float, device));
    let freqs = t.unsqueeze(1).matmul(&inv_timescales.unsqueeze(0));
    let cos = freqs.cos();
    let sin = freqs.sin();
    let cos = Tensor::cat(&[cos.shallow_clone(), cos], 1);
    let sin = Tensor::cat(&[sin.shallow_clone(), sin], 1);
    (cos, sin)
}

fn apply_rotary(q: &Tensor, k: &Tensor, cos: &Tensor, sin: &Tensor) -> (Tensor, Tensor) {
    let cos = cos.unsqueeze(0).unsqueeze(2).to_kind(q.kind());
    let sin = sin.unsqueeze(0).unsqueeze(2).to_kind(q.kind());
    let d = q.size()[3];
    let half = d / 2;
    let q_l = q.narrow(3, 0, half);
    let q_r = q.narrow(3, half, half);
    let k_l = k.narrow(3, 0, half);
    let k_r = k.narrow(3, half, half);
    let q_rot = Tensor::cat(&[q_r.neg(), q_l], 3);
    let k_rot = Tensor::cat(&[k_r.neg(), k_l], 3);
    let q = q * &cos + q_rot * &sin;
    let k = k * &cos + k_rot * &sin;
    (q, k)
}
