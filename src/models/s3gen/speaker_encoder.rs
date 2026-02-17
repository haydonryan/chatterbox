use tch::{Device, IndexOp, Kind, Tensor};

use crate::error::{ChatterboxError, Result};
use crate::models::s3gen::weights::Weights;

const SAMPLE_RATE: i64 = 16_000;
const NUM_MEL_BINS: i64 = 80;

pub struct CampPlus {
    head: Fcm,
    xvector: XVector,
    output_level: OutputLevel,
}

#[derive(Clone, Copy)]
enum OutputLevel {
    Segment,
    #[allow(dead_code)]
    Frame,
}

impl CampPlus {
    pub fn from_weights(weights: &Weights) -> Result<Self> {
        let prefix = find_prefix(weights, &["speaker_encoder", "s3gen.speaker_encoder"])?;
        let head = Fcm::from_weights(weights, &format!("{prefix}.head"))?;
        let xvector = XVector::from_weights(weights, &format!("{prefix}.xvector"))?;
        Ok(Self {
            head,
            xvector,
            output_level: OutputLevel::Segment,
        })
    }

    pub fn inference(&self, audio_list: &[Tensor]) -> Result<Tensor> {
        if audio_list.is_empty() {
            return Err(ChatterboxError::ShapeMismatch(
                "CampPlus inference requires at least one audio tensor".to_string(),
            ));
        }
        let device = audio_list[0].device();
        let mut feats: Vec<Tensor> = Vec::with_capacity(audio_list.len());
        for audio in audio_list {
            let audio = if audio.dim() == 1 {
                audio.unsqueeze(0)
            } else {
                audio.shallow_clone()
            };
            let fbank = kaldi_fbank(&audio, NUM_MEL_BINS)?;
            let mean = fbank.mean_dim(&[0_i64][..], true, Kind::Float);
            let fbank = fbank - mean;
            feats.push(fbank);
        }
        let feats = pad_list(&feats);
        let out = self.forward(&feats.to_device(device));
        Ok(out)
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let mut x = x.permute(&[0, 2, 1]); // (B, T, F) -> (B, F, T)
        x = self.head.forward(&x);
        x = self.xvector.forward(&x);
        match self.output_level {
            OutputLevel::Segment => x,
            OutputLevel::Frame => x.transpose(1, 2),
        }
    }
}

fn find_prefix(weights: &Weights, candidates: &[&str]) -> Result<String> {
    for cand in candidates {
        let key = format!("{cand}.head.conv1.weight");
        if weights.contains(&key) {
            return Ok(cand.to_string());
        }
    }
    Err(ChatterboxError::MissingWeight(
        "speaker_encoder.head.conv1.weight".to_string(),
    ))
}

fn pad_list(xs: &[Tensor]) -> Tensor {
    let batch = xs.len() as i64;
    let max_len = xs.iter().map(|t| t.size()[0]).max().unwrap_or(0);
    let feat_dim = xs.get(0).map(|t| t.size()[1]).unwrap_or(NUM_MEL_BINS);
    let device = xs.get(0).map(|t| t.device()).unwrap_or(Device::Cpu);
    let mut pad = Tensor::zeros([batch, max_len, feat_dim], (Kind::Float, device));
    for (idx, x) in xs.iter().enumerate() {
        let len = x.size()[0];
        pad.i((idx as i64, 0..len, ..)).copy_(x);
    }
    pad
}

fn kaldi_fbank(waveform: &Tensor, num_mel_bins: i64) -> Result<Tensor> {
    let waveform = if waveform.dim() == 2 {
        waveform.i((0, ..))
    } else {
        waveform.shallow_clone()
    };

    let device = waveform.device();
    let kind = waveform.kind();
    let window_shift = (SAMPLE_RATE as f64 * 10.0 / 1000.0) as i64; // 10ms
    let window_size = (SAMPLE_RATE as f64 * 25.0 / 1000.0) as i64; // 25ms
    if waveform.size()[0] < window_size {
        return Ok(Tensor::zeros([0, num_mel_bins], (kind, device)));
    }
    let padded_window_size = next_power_of_two(window_size);

    let mut frames = waveform.unfold(0, window_size, window_shift);
    if frames.numel() == 0 {
        return Ok(Tensor::zeros([0, num_mel_bins], (kind, device)));
    }

    // remove DC offset
    let mean = frames.mean_dim(&[1_i64][..], true, Kind::Float);
    frames = &frames - &mean;

    // preemphasis
    let preemph = 0.97;
    if preemph != 0.0 {
        let offset = frames
            .unsqueeze(0)
            .pad(&[1, 0], "replicate", None)
            .squeeze_dim(0);
        let prev = offset.i((.., 0..window_size));
        frames = frames - prev * preemph;
    }

    // povey window
    let window = Tensor::hann_window_periodic(window_size, false, (kind, device)).pow_tensor_scalar(0.85);
    frames = frames * window.unsqueeze(0);

    if padded_window_size != window_size {
        let pad_right = padded_window_size - window_size;
        frames = frames
            .unsqueeze(0)
            .pad(&[0, pad_right], "constant", Some(0.0))
            .squeeze_dim(0);
    }

    let spectrum = frames
        .fft_rfft(Some(padded_window_size), -1, "backward")
        .abs()
        .pow_tensor_scalar(2.0);

    let mel = mel_filterbank_kaldi(
        num_mel_bins,
        padded_window_size,
        SAMPLE_RATE as f64,
        20.0,
        0.0,
        100.0,
        -500.0,
        1.0,
        kind,
        device,
    );
    let mel = mel.pad(&[0, 1], "constant", Some(0.0));
    let mel = spectrum.matmul(&mel.transpose(0, 1));
    let eps = Tensor::from(1.0e-10).to_device(device).to_kind(kind);
    Ok(mel.maximum(&eps).log())
}

fn next_power_of_two(mut n: i64) -> i64 {
    if n <= 0 {
        return 1;
    }
    n -= 1;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n += 1;
    n
}

fn mel_filterbank_kaldi(
    num_bins: i64,
    window_length_padded: i64,
    sample_freq: f64,
    low_freq: f64,
    high_freq: f64,
    vtln_low: f64,
    vtln_high: f64,
    vtln_warp: f64,
    kind: Kind,
    device: Device,
) -> Tensor {
    let num_fft_bins = window_length_padded / 2;
    let nyquist = 0.5 * sample_freq;
    let high_freq = if high_freq <= 0.0 { high_freq + nyquist } else { high_freq };
    let fft_bin_width = sample_freq / window_length_padded as f64;
    let mel_low = mel_scale_scalar(low_freq);
    let mel_high = mel_scale_scalar(high_freq);
    let mel_delta = (mel_high - mel_low) / (num_bins + 1) as f64;

    let vtln_high = if vtln_high < 0.0 { vtln_high + nyquist } else { vtln_high };
    let bin = Tensor::arange(num_bins, (kind, device)).unsqueeze(1);
    let left_mel = mel_low + &bin * mel_delta;
    let center_mel = mel_low + (&bin + 1.0) * mel_delta;
    let right_mel = mel_low + (&bin + 2.0) * mel_delta;

    let (left_mel, center_mel, right_mel) = if (vtln_warp - 1.0).abs() > f64::EPSILON {
        (
            vtln_warp_mel_freq(vtln_low, vtln_high, low_freq, high_freq, vtln_warp, &left_mel),
            vtln_warp_mel_freq(vtln_low, vtln_high, low_freq, high_freq, vtln_warp, &center_mel),
            vtln_warp_mel_freq(vtln_low, vtln_high, low_freq, high_freq, vtln_warp, &right_mel),
        )
    } else {
        (left_mel, center_mel, right_mel)
    };

    let mel = mel_scale(&(&Tensor::arange(num_fft_bins, (kind, device)) * fft_bin_width)).unsqueeze(0);
    let up_slope = (&mel - &left_mel) / (&center_mel - &left_mel);
    let down_slope = (&right_mel - &mel) / (&right_mel - &center_mel);
    if (vtln_warp - 1.0).abs() <= f64::EPSILON {
        let bins = up_slope.minimum(&down_slope);
        bins.maximum(&Tensor::zeros_like(&bins))
    } else {
        let mut bins = Tensor::zeros_like(&up_slope);
        let up_idx = mel.gt_tensor(&left_mel).logical_and(&mel.le_tensor(&center_mel));
        let down_idx = mel.gt_tensor(&center_mel).logical_and(&mel.lt_tensor(&right_mel));
        bins = bins.where_self(&up_idx, &up_slope);
        bins = bins.where_self(&down_idx, &down_slope);
        bins
    }
}

fn mel_scale_scalar(freq: f64) -> f64 {
    1127.0 * (1.0 + freq / 700.0).ln()
}

fn mel_scale(freq: &Tensor) -> Tensor {
    (freq / 700.0 + 1.0).log() * 1127.0
}

fn inverse_mel_scale(mel: &Tensor) -> Tensor {
    ((mel / 1127.0).exp() - 1.0) * 700.0
}

fn vtln_warp_mel_freq(
    vtln_low: f64,
    vtln_high: f64,
    low_freq: f64,
    high_freq: f64,
    vtln_warp: f64,
    freq: &Tensor,
) -> Tensor {
    let low_mel = mel_scale_scalar(low_freq);
    let high_mel = mel_scale_scalar(high_freq);
    let vtln_low = mel_scale_scalar(vtln_low);
    let vtln_high = mel_scale_scalar(vtln_high);
    let scale = (vtln_high - vtln_low) / (high_mel - low_mel);
    let warped = (freq - low_mel) * scale + vtln_low;
    warped
}

struct BatchNorm1d {
    weight: Option<Tensor>,
    bias: Option<Tensor>,
    running_mean: Tensor,
    running_var: Tensor,
    eps: f64,
}

impl BatchNorm1d {
    fn from_weights(weights: &Weights, prefix: &str, affine: bool) -> Result<Self> {
        let weight = if affine {
            weights.get(&format!("{prefix}.weight")).ok()
        } else {
            None
        };
        let bias = if affine {
            weights.get(&format!("{prefix}.bias")).ok()
        } else {
            None
        };
        let running_mean = weights.get(&format!("{prefix}.running_mean"))?;
        let running_var = weights.get(&format!("{prefix}.running_var"))?;
        Ok(Self {
            weight,
            bias,
            running_mean,
            running_var,
            eps: 1e-5,
        })
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let device = x.device();
        let kind = x.kind();
        let mut y = x - self.running_mean.to_device(device).to_kind(kind).view([1, -1, 1]);
        y = y / (self.running_var.to_device(device).to_kind(kind).view([1, -1, 1]) + self.eps).sqrt();
        if let Some(weight) = &self.weight {
            y = y * weight.to_device(device).to_kind(kind).view([1, -1, 1]);
        }
        if let Some(bias) = &self.bias {
            y = y + bias.to_device(device).to_kind(kind).view([1, -1, 1]);
        }
        y
    }

    fn forward_2d(&self, x: &Tensor) -> Tensor {
        let device = x.device();
        let kind = x.kind();
        let mut y = x - self.running_mean.to_device(device).to_kind(kind).view([1, -1]);
        y = y / (self.running_var.to_device(device).to_kind(kind).view([1, -1]) + self.eps).sqrt();
        if let Some(weight) = &self.weight {
            y = y * weight.to_device(device).to_kind(kind).view([1, -1]);
        }
        if let Some(bias) = &self.bias {
            y = y + bias.to_device(device).to_kind(kind).view([1, -1]);
        }
        y
    }
}

struct BatchNorm2d {
    weight: Tensor,
    bias: Tensor,
    running_mean: Tensor,
    running_var: Tensor,
    eps: f64,
}

impl BatchNorm2d {
    fn from_weights(weights: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            weight: weights.get(&format!("{prefix}.weight"))?,
            bias: weights.get(&format!("{prefix}.bias"))?,
            running_mean: weights.get(&format!("{prefix}.running_mean"))?,
            running_var: weights.get(&format!("{prefix}.running_var"))?,
            eps: 1e-5,
        })
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let device = x.device();
        let kind = x.kind();
        let mean = self.running_mean.to_device(device).to_kind(kind).view([1, -1, 1, 1]);
        let var = self.running_var.to_device(device).to_kind(kind).view([1, -1, 1, 1]);
        let mut y = (x - mean) / (var + self.eps).sqrt();
        y = y * self.weight.to_device(device).to_kind(kind).view([1, -1, 1, 1]);
        y = y + self.bias.to_device(device).to_kind(kind).view([1, -1, 1, 1]);
        y
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
    fn from_weights(
        weights: &Weights,
        prefix: &str,
        stride: i64,
        padding: i64,
        dilation: i64,
        bias: bool,
    ) -> Result<Self> {
        let weight = weights.get(&format!("{prefix}.weight"))?;
        let bias = if bias {
            weights.get(&format!("{prefix}.bias")).ok()
        } else {
            None
        };
        Ok(Self {
            weight,
            bias,
            stride,
            padding,
            dilation,
            groups: 1,
        })
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let weight = self.weight.to_kind(x.kind());
        let bias = self.bias.as_ref().map(|b| b.to_kind(x.kind()));
        x.conv1d(
            &weight,
            bias.as_ref(),
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
    }
}

struct Conv2d {
    weight: Tensor,
    bias: Option<Tensor>,
    stride: (i64, i64),
    padding: (i64, i64),
    dilation: (i64, i64),
    groups: i64,
}

impl Conv2d {
    fn from_weights(
        weights: &Weights,
        prefix: &str,
        stride: (i64, i64),
        padding: (i64, i64),
        dilation: (i64, i64),
        bias: bool,
    ) -> Result<Self> {
        let weight = weights.get(&format!("{prefix}.weight"))?;
        let bias = if bias {
            weights.get(&format!("{prefix}.bias")).ok()
        } else {
            None
        };
        Ok(Self {
            weight,
            bias,
            stride,
            padding,
            dilation,
            groups: 1,
        })
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let weight = self.weight.to_kind(x.kind());
        let bias = self.bias.as_ref().map(|b| b.to_kind(x.kind()));
        x.conv2d(
            &weight,
            bias.as_ref(),
            &[self.stride.0, self.stride.1],
            &[self.padding.0, self.padding.1],
            &[self.dilation.0, self.dilation.1],
            self.groups,
        )
    }
}

struct BasicResBlock {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    conv2: Conv2d,
    bn2: BatchNorm2d,
    shortcut: Option<(Conv2d, BatchNorm2d)>,
}

impl BasicResBlock {
    fn from_weights(weights: &Weights, prefix: &str, stride: i64, in_planes: i64, planes: i64) -> Result<Self> {
        let conv1 = Conv2d::from_weights(weights, &format!("{prefix}.conv1"), (stride, 1), (1, 1), (1, 1), false)?;
        let bn1 = BatchNorm2d::from_weights(weights, &format!("{prefix}.bn1"))?;
        let conv2 = Conv2d::from_weights(weights, &format!("{prefix}.conv2"), (1, 1), (1, 1), (1, 1), false)?;
        let bn2 = BatchNorm2d::from_weights(weights, &format!("{prefix}.bn2"))?;

        let shortcut = if stride != 1 || in_planes != planes {
            let conv = Conv2d::from_weights(
                weights,
                &format!("{prefix}.shortcut.0"),
                (stride, 1),
                (0, 0),
                (1, 1),
                false,
            )?;
            let bn = BatchNorm2d::from_weights(weights, &format!("{prefix}.shortcut.1"))?;
            Some((conv, bn))
        } else {
            None
        };

        Ok(Self {
            conv1,
            bn1,
            conv2,
            bn2,
            shortcut,
        })
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let mut out = self.bn1.forward(&self.conv1.forward(x)).relu();
        out = self.bn2.forward(&self.conv2.forward(&out));
        let shortcut = match &self.shortcut {
            Some((conv, bn)) => bn.forward(&conv.forward(x)),
            None => x.shallow_clone(),
        };
        (out + shortcut).relu()
    }
}

struct Fcm {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    layer1: Vec<BasicResBlock>,
    layer2: Vec<BasicResBlock>,
    conv2: Conv2d,
    bn2: BatchNorm2d,
}

impl Fcm {
    fn from_weights(weights: &Weights, prefix: &str) -> Result<Self> {
        let conv1 = Conv2d::from_weights(weights, &format!("{prefix}.conv1"), (1, 1), (1, 1), (1, 1), false)?;
        let bn1 = BatchNorm2d::from_weights(weights, &format!("{prefix}.bn1"))?;

        let mut layer1 = Vec::with_capacity(2);
        let mut in_planes = 32;
        for i in 0..2 {
            let stride = if i == 0 { 2 } else { 1 };
            let block = BasicResBlock::from_weights(
                weights,
                &format!("{prefix}.layer1.{i}"),
                stride,
                in_planes,
                32,
            )?;
            in_planes = 32;
            layer1.push(block);
        }

        let mut layer2 = Vec::with_capacity(2);
        for i in 0..2 {
            let stride = if i == 0 { 2 } else { 1 };
            let block = BasicResBlock::from_weights(
                weights,
                &format!("{prefix}.layer2.{i}"),
                stride,
                in_planes,
                32,
            )?;
            in_planes = 32;
            layer2.push(block);
        }

        let conv2 = Conv2d::from_weights(weights, &format!("{prefix}.conv2"), (2, 1), (1, 1), (1, 1), false)?;
        let bn2 = BatchNorm2d::from_weights(weights, &format!("{prefix}.bn2"))?;

        Ok(Self {
            conv1,
            bn1,
            layer1,
            layer2,
            conv2,
            bn2,
        })
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let mut out = x.unsqueeze(1);
        out = self.bn1.forward(&self.conv1.forward(&out)).relu();
        for block in &self.layer1 {
            out = block.forward(&out);
        }
        for block in &self.layer2 {
            out = block.forward(&out);
        }
        out = self.bn2.forward(&self.conv2.forward(&out)).relu();
        let shape = out.size();
        out.reshape([shape[0], shape[1] * shape[2], shape[3]])
    }
}

enum NonLinearOp {
    BatchNorm(BatchNorm1d),
    Relu,
    PRelu(Tensor),
}

struct NonLinear1d {
    ops: Vec<NonLinearOp>,
}

impl NonLinear1d {
    fn from_weights(weights: &Weights, prefix: &str, config_str: &str, channels: i64) -> Result<Self> {
        let mut ops = Vec::new();
        for name in config_str.split('-') {
            match name {
                "batchnorm" => ops.push(NonLinearOp::BatchNorm(BatchNorm1d::from_weights(
                    weights,
                    &format!("{prefix}.batchnorm"),
                    true,
                )?)),
                "batchnorm_" => ops.push(NonLinearOp::BatchNorm(BatchNorm1d::from_weights(
                    weights,
                    &format!("{prefix}.batchnorm"),
                    false,
                )?)),
                "relu" => ops.push(NonLinearOp::Relu),
                "prelu" => {
                    let w = weights.get(&format!("{prefix}.prelu.weight")).unwrap_or_else(|_| {
                        Tensor::ones([channels], (Kind::Float, Device::Cpu))
                    });
                    ops.push(NonLinearOp::PRelu(w))
                }
                _ => {
                    return Err(ChatterboxError::NotImplemented("Unsupported nonlinear op"))
                }
            }
        }
        Ok(Self { ops })
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let mut out = x.shallow_clone();
        for op in &self.ops {
            out = match op {
                NonLinearOp::BatchNorm(bn) => {
                    if out.dim() == 2 {
                        bn.forward_2d(&out)
                    } else {
                        bn.forward(&out)
                    }
                }
                NonLinearOp::Relu => out.relu(),
                NonLinearOp::PRelu(w) => {
                    let w = w.to_device(out.device()).to_kind(out.kind());
                    out.prelu(&w)
                }
            };
        }
        out
    }
}

struct TDNNLayer {
    linear: Conv1d,
    nonlinear: NonLinear1d,
}

impl TDNNLayer {
    fn from_weights(weights: &Weights, prefix: &str, channels: i64) -> Result<Self> {
        Ok(Self {
            linear: Conv1d::from_weights(weights, &format!("{prefix}.linear"), 2, 2, 1, false)?,
            nonlinear: NonLinear1d::from_weights(weights, &format!("{prefix}.nonlinear"), "batchnorm-relu", channels)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        self.nonlinear.forward(&self.linear.forward(x))
    }
}

struct CAMLayer {
    linear_local: Conv1d,
    linear1: Conv1d,
    linear2: Conv1d,
}

impl CAMLayer {
    fn from_weights(weights: &Weights, prefix: &str, padding: i64, dilation: i64) -> Result<Self> {
        Ok(Self {
            linear_local: Conv1d::from_weights(
                weights,
                &format!("{prefix}.linear_local"),
                1,
                padding,
                dilation,
                false,
            )?,
            linear1: Conv1d::from_weights(weights, &format!("{prefix}.linear1"), 1, 0, 1, true)?,
            linear2: Conv1d::from_weights(weights, &format!("{prefix}.linear2"), 1, 0, 1, true)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let y = self.linear_local.forward(x);
        let context = x.mean_dim(&[-1_i64][..], true, Kind::Float) + seg_pooling(x, 100);
        let context = self.linear1.forward(&context).relu();
        let m = self.linear2.forward(&context).sigmoid();
        y * m
    }
}

fn seg_pooling(x: &Tensor, seg_len: i64) -> Tensor {
    let seg = x.avg_pool1d(seg_len, seg_len, 0, true, true);
    let shape = seg.size();
    let seg = seg
        .unsqueeze(-1)
        .expand(&[shape[0], shape[1], shape[2], seg_len], true)
        .reshape([shape[0], shape[1], -1]);
    seg.i((.., .., 0..x.size()[2]))
}

struct CAMDenseTDNNLayer {
    nonlinear1: NonLinear1d,
    linear1: Conv1d,
    nonlinear2: NonLinear1d,
    cam_layer: CAMLayer,
}

impl CAMDenseTDNNLayer {
    fn from_weights(
        weights: &Weights,
        prefix: &str,
        in_channels: i64,
        bn_channels: i64,
        kernel_size: i64,
        dilation: i64,
    ) -> Result<Self> {
        let padding = (kernel_size - 1) / 2 * dilation;
        Ok(Self {
            nonlinear1: NonLinear1d::from_weights(weights, &format!("{prefix}.nonlinear1"), "batchnorm-relu", in_channels)?,
            linear1: Conv1d::from_weights(weights, &format!("{prefix}.linear1"), 1, 0, 1, false)?,
            nonlinear2: NonLinear1d::from_weights(weights, &format!("{prefix}.nonlinear2"), "batchnorm-relu", bn_channels)?,
            cam_layer: CAMLayer::from_weights(weights, &format!("{prefix}.cam_layer"), padding, dilation)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let x = self.linear1.forward(&self.nonlinear1.forward(x));
        self.cam_layer.forward(&self.nonlinear2.forward(&x))
    }
}

struct CAMDenseTDNNBlock {
    layers: Vec<CAMDenseTDNNLayer>,
}

impl CAMDenseTDNNBlock {
    fn from_weights(
        weights: &Weights,
        prefix: &str,
        num_layers: i64,
        in_channels: i64,
        out_channels: i64,
        bn_channels: i64,
        kernel_size: i64,
        dilation: i64,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(num_layers as usize);
        for i in 0..num_layers {
            let layer = CAMDenseTDNNLayer::from_weights(
                weights,
                &format!("{prefix}.tdnnd{}", i + 1),
                in_channels + i * out_channels,
                bn_channels,
                kernel_size,
                dilation,
            )?;
            layers.push(layer);
        }
        Ok(Self { layers })
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let mut h = x.shallow_clone();
        for layer in &self.layers {
            let out = layer.forward(&h);
            h = Tensor::cat(&[h, out], 1);
        }
        h
    }
}

struct TransitLayer {
    nonlinear: NonLinear1d,
    linear: Conv1d,
}

impl TransitLayer {
    fn from_weights(weights: &Weights, prefix: &str, in_channels: i64) -> Result<Self> {
        Ok(Self {
            nonlinear: NonLinear1d::from_weights(weights, &format!("{prefix}.nonlinear"), "batchnorm-relu", in_channels)?,
            linear: Conv1d::from_weights(weights, &format!("{prefix}.linear"), 1, 0, 1, false)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        self.linear.forward(&self.nonlinear.forward(x))
    }
}

struct DenseLayer {
    linear: Conv1d,
    nonlinear: NonLinear1d,
}

impl DenseLayer {
    fn from_weights(weights: &Weights, prefix: &str, in_channels: i64) -> Result<Self> {
        Ok(Self {
            linear: Conv1d::from_weights(weights, &format!("{prefix}.linear"), 1, 0, 1, false)?,
            nonlinear: NonLinear1d::from_weights(weights, &format!("{prefix}.nonlinear"), "batchnorm_", in_channels)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let x = if x.dim() == 2 {
            self.linear.forward(&x.unsqueeze(-1)).squeeze_dim(-1)
        } else {
            self.linear.forward(x)
        };
        self.nonlinear.forward(&x)
    }
}

struct StatsPool;

impl StatsPool {
    fn forward(&self, x: &Tensor) -> Tensor {
        let mean = x.mean_dim(&[-1_i64][..], false, Kind::Float);
        let std = x.std_dim(&[-1_i64][..], true, false);
        Tensor::cat(&[mean, std], -1)
    }
}

struct XVector {
    tdnn: TDNNLayer,
    blocks: Vec<CAMDenseTDNNBlock>,
    transits: Vec<TransitLayer>,
    out_nonlinear: NonLinear1d,
    stats: StatsPool,
    dense: DenseLayer,
}

impl XVector {
    fn from_weights(weights: &Weights, prefix: &str) -> Result<Self> {
        let tdnn = TDNNLayer::from_weights(weights, &format!("{prefix}.tdnn"), 128)?;

        let mut blocks = Vec::new();
        let mut transits = Vec::new();

        let growth_rate = 32;
        let bn_size = 4;
        let bn_channels = bn_size * growth_rate;
        let mut channels = 128;

        let configs = [(12, 1), (24, 2), (16, 2)];
        for (idx, (num_layers, dilation)) in configs.iter().enumerate() {
            let block = CAMDenseTDNNBlock::from_weights(
                weights,
                &format!("{prefix}.block{}", idx + 1),
                *num_layers as i64,
                channels,
                growth_rate,
                bn_channels,
                3,
                *dilation as i64,
            )?;
            blocks.push(block);
            channels += (*num_layers as i64) * growth_rate;

            let transit = TransitLayer::from_weights(
                weights,
                &format!("{prefix}.transit{}", idx + 1),
                channels,
            )?;
            transits.push(transit);
            channels /= 2;
        }

        let out_nonlinear = NonLinear1d::from_weights(weights, &format!("{prefix}.out_nonlinear"), "batchnorm-relu", channels)?;
        let stats = StatsPool;
        let dense = DenseLayer::from_weights(weights, &format!("{prefix}.dense"), channels * 2)?;

        Ok(Self {
            tdnn,
            blocks,
            transits,
            out_nonlinear,
            stats,
            dense,
        })
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let mut h = self.tdnn.forward(x);
        for (block, transit) in self.blocks.iter().zip(self.transits.iter()) {
            h = block.forward(&h);
            h = transit.forward(&h);
        }
        h = self.out_nonlinear.forward(&h);
        let h = self.stats.forward(&h);
        self.dense.forward(&h)
    }
}
