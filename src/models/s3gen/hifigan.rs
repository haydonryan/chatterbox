use std::f64::consts::PI;

use tch::{Device, IndexOp, Kind, Tensor};

use crate::error::Result;
use crate::models::s3gen::weights::Weights;

pub struct HiFTGenerator {
    conv_pre: Conv1dWN,
    ups: Vec<ConvTranspose1dWN>,
    source_downs: Vec<Conv1d>,
    source_resblocks: Vec<ResBlock>,
    resblocks: Vec<ResBlock>,
    conv_post: Conv1dWN,
    m_source: SourceModuleHnNSF,
    f0_predictor: ConvRNNF0Predictor,
    f0_upsample_scale: i64,
    num_upsamples: usize,
    num_kernels: usize,
    istft_n_fft: i64,
    istft_hop: i64,
    lrelu_slope: f64,
    audio_limit: f64,
    stft_window: Tensor,
}

impl HiFTGenerator {
    pub fn from_weights(weights: &Weights) -> Result<Self> {
        let conv_pre = Conv1dWN::from_weights(weights, "mel2wav.conv_pre", 1, 3, 1)?;

        let upsample_rates = [8, 5, 3];
        let upsample_kernel_sizes = [16, 11, 7];
        let num_upsamples = upsample_rates.len();

        let mut ups = Vec::with_capacity(num_upsamples);
        for (idx, (&u, &k)) in upsample_rates.iter().zip(upsample_kernel_sizes.iter()).enumerate() {
            let prefix = format!("mel2wav.ups.{idx}");
            let padding = (k - u) / 2;
            ups.push(ConvTranspose1dWN::from_weights(
                weights,
                &prefix,
                u as i64,
                padding as i64,
                1,
            )?);
        }

        let source_resblock_kernel_sizes = [7, 7, 11];
        let source_resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]];
        let mut source_downs = Vec::with_capacity(source_resblock_kernel_sizes.len());
        let mut source_resblocks = Vec::with_capacity(source_resblock_kernel_sizes.len());

        let downsample_rates = [1, 3, 5];
        let mut downsample_cum = vec![1];
        for r in downsample_rates.iter().take(downsample_rates.len() - 1) {
            let last = *downsample_cum.last().unwrap();
            downsample_cum.push(last * r);
        }
        let mut downsample_cum_rev = downsample_cum.clone();
        downsample_cum_rev.reverse();

        let istft_n_fft = 16;
        let istft_hop = 4;

        for i in 0..source_resblock_kernel_sizes.len() {
            let u = downsample_cum_rev[i];
            let k = source_resblock_kernel_sizes[i];
            let d = &source_resblock_dilation_sizes[i];
            let out_ch = 512 / (2_i64.pow((i + 1) as u32));
            let prefix = format!("mel2wav.source_downs.{i}");
            let conv = if u == 1 {
                Conv1d::from_weights(weights, &prefix, 1, 0, 1, 1)?
            } else {
                Conv1d::from_weights(weights, &prefix, u as i64 * 2, (u as i64) / 2, u as i64, 1)?
            };
            source_downs.push(conv);

            let res_prefix = format!("mel2wav.source_resblocks.{i}");
            source_resblocks.push(ResBlock::from_weights(
                weights,
                &res_prefix,
                out_ch,
                k as i64,
                d,
            )?);
        }

        let resblock_kernel_sizes = [3, 7, 11];
        let resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]];
        let num_kernels = resblock_kernel_sizes.len();
        let mut resblocks = Vec::with_capacity(num_upsamples * num_kernels);
        for i in 0..num_upsamples {
            let ch = 512 / (2_i64.pow((i + 1) as u32));
            for (j, (&k, d)) in resblock_kernel_sizes
                .iter()
                .zip(resblock_dilation_sizes.iter())
                .enumerate()
            {
                let idx = i * num_kernels + j;
                let prefix = format!("mel2wav.resblocks.{idx}");
                resblocks.push(ResBlock::from_weights(weights, &prefix, ch, k as i64, d)?);
            }
        }

        let conv_post = Conv1dWN::from_weights(weights, "mel2wav.conv_post", 1, 3, 1)?;
        let m_source = SourceModuleHnNSF::from_weights(weights)?;
        let f0_predictor = ConvRNNF0Predictor::from_weights(weights)?;
        let f0_upsample_scale = upsample_rates.iter().product::<i32>() as i64 * istft_hop;

        let stft_window = Tensor::hann_window(istft_n_fft, (Kind::Float, Device::Cpu));

        Ok(Self {
            conv_pre,
            ups,
            source_downs,
            source_resblocks,
            resblocks,
            conv_post,
            m_source,
            f0_predictor,
            f0_upsample_scale,
            num_upsamples,
            num_kernels,
            istft_n_fft,
            istft_hop,
            lrelu_slope: 0.1,
            audio_limit: 0.99,
            stft_window,
        })
    }

    fn stft(&self, x: &Tensor) -> (Tensor, Tensor) {
        let spec = x.stft_center(
            self.istft_n_fft,
            Some(self.istft_hop),
            Some(self.istft_n_fft),
            Some(&self.stft_window.to_device(x.device())),
            false,
            "reflect",
            false,
            true,
            true,
            false,
        );
        let spec = spec.view_as_real();
        let real = spec.i((.., .., .., 0));
        let imag = spec.i((.., .., .., 1));
        (real, imag)
    }

    fn istft(&self, magnitude: &Tensor, phase: &Tensor) -> Tensor {
        let magnitude = magnitude.clamp_max(1e2);
        let device = magnitude.device();
        let real = &magnitude * phase.cos();
        let imag = &magnitude * phase.sin();
        let complex = Tensor::complex(&real, &imag);
        complex.istft(
            self.istft_n_fft,
            Some(self.istft_hop),
            Some(self.istft_n_fft),
            Some(&self.stft_window.to_device(device)),
            true,
            false,
            true,
            None,
            false,
        )
    }

    fn decode(&self, x: &Tensor, s: &Tensor) -> Tensor {
        let (s_real, s_imag) = self.stft(&s.squeeze_dim(1));
        let s_stft = Tensor::cat(&[s_real, s_imag], 1);

        let mut h = self.conv_pre.forward(x);
        for i in 0..self.num_upsamples {
            h = leaky_relu(&h, self.lrelu_slope);
            h = self.ups[i].forward(&h);
            if i == self.num_upsamples - 1 {
                h = h.pad(&[1, 0], "reflect", None);
            }
            let mut si = self.source_downs[i].forward(&s_stft);
            si = self.source_resblocks[i].forward(&si);
            let ht = h.size()[2];
            let sit = si.size()[2];
            if ht != sit {
                let t = ht.min(sit);
                if ht != t {
                    h = h.narrow(2, 0, t);
                }
                if sit != t {
                    si = si.narrow(2, 0, t);
                }
            }
            h = h + si;

            let mut xs: Option<Tensor> = None;
            for j in 0..self.num_kernels {
                let idx = i * self.num_kernels + j;
                let out = self.resblocks[idx].forward(&h);
                xs = Some(match xs {
                    Some(prev) => prev + out,
                    None => out,
                });
            }
            h = xs.unwrap() / self.num_kernels as f64;
        }
        h = h.leaky_relu();
        h = self.conv_post.forward(&h);
        let mag = h.i((.., 0..(self.istft_n_fft / 2 + 1), ..)).exp();
        let phase = h.i((.., (self.istft_n_fft / 2 + 1).., ..)).sin();
        let y = self.istft(&mag, &phase).clamp(-self.audio_limit, self.audio_limit);
        y
    }

    pub fn inference(&self, speech_feat: &Tensor, cache_source: &Tensor) -> (Tensor, Tensor) {
        let f0 = self.f0_predictor.forward(speech_feat);
        let scale = [self.f0_upsample_scale as f64];
        let mut s = f0
            .unsqueeze(1)
            .upsample_nearest1d_vec(None::<&[i64]>, &scale)
            .transpose(1, 2);
        let (sine, _noise, _uv) = self.m_source.forward(&s);
        s = sine.transpose(1, 2);
        if cache_source.size()[2] != 0 {
            s.i((.., .., 0..cache_source.size()[2]))
                .copy_(cache_source);
        }
        let wav = self.decode(speech_feat, &s);
        (wav, s)
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
        kernel_size: i64,
        padding: i64,
        stride: i64,
        dilation: i64,
    ) -> Result<Self> {
        let weight = weights.get(&format!("{prefix}.weight"))?;
        let bias = weights.get(&format!("{prefix}.bias")).ok();
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
        x.conv1d(&weight, bias.as_ref(), self.stride, self.padding, self.dilation, self.groups)
    }
}

struct Conv1dWN {
    weight: Tensor,
    bias: Option<Tensor>,
    stride: i64,
    padding: i64,
    dilation: i64,
    groups: i64,
}

impl Conv1dWN {
    fn from_weights(
        weights: &Weights,
        prefix: &str,
        stride: i64,
        padding: i64,
        dilation: i64,
    ) -> Result<Self> {
        let g = weights.get(&format!("{prefix}.parametrizations.weight.original0"))?;
        let v = weights.get(&format!("{prefix}.parametrizations.weight.original1"))?;
        let weight = weight_norm(&v, &g, 0);
        let bias = weights.get(&format!("{prefix}.bias")).ok();
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
        x.conv1d(&weight, bias.as_ref(), self.stride, self.padding, self.dilation, self.groups)
    }
}

struct ConvTranspose1dWN {
    weight: Tensor,
    bias: Option<Tensor>,
    stride: i64,
    padding: i64,
    dilation: i64,
    groups: i64,
}

impl ConvTranspose1dWN {
    fn from_weights(
        weights: &Weights,
        prefix: &str,
        stride: i64,
        padding: i64,
        dilation: i64,
    ) -> Result<Self> {
        let g = weights.get(&format!("{prefix}.parametrizations.weight.original0"))?;
        let v = weights.get(&format!("{prefix}.parametrizations.weight.original1"))?;
        let weight = weight_norm(&v, &g, 0);
        let bias = weights.get(&format!("{prefix}.bias")).ok();
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
        x.conv_transpose1d(&weight, bias.as_ref(), self.stride, self.padding, 0, self.groups, self.dilation)
    }
}

struct Snake {
    alpha: Tensor,
}

impl Snake {
    fn from_weights(weights: &Weights, prefix: &str) -> Result<Self> {
        let alpha = weights.get(&format!("{prefix}.alpha"))?;
        Ok(Self { alpha })
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let alpha = self.alpha.to_kind(x.kind()).unsqueeze(0).unsqueeze(-1);
        let inv = 1.0 / (&alpha + 1e-9);
        x + inv * (x * &alpha).sin().pow_tensor_scalar(2.0)
    }
}

struct ResBlock {
    convs1: Vec<Conv1dWN>,
    convs2: Vec<Conv1dWN>,
    activations1: Vec<Snake>,
    activations2: Vec<Snake>,
}

impl ResBlock {
    fn from_weights(
        weights: &Weights,
        prefix: &str,
        _channels: i64,
        kernel_size: i64,
        dilations: &[i64],
    ) -> Result<Self> {
        let mut convs1 = Vec::with_capacity(dilations.len());
        let mut convs2 = Vec::with_capacity(dilations.len());
        let mut activations1 = Vec::with_capacity(dilations.len());
        let mut activations2 = Vec::with_capacity(dilations.len());
        for (idx, &dilation) in dilations.iter().enumerate() {
            let conv1_prefix = format!("{prefix}.convs1.{idx}");
            let conv2_prefix = format!("{prefix}.convs2.{idx}");
            let act1_prefix = format!("{prefix}.activations1.{idx}");
            let act2_prefix = format!("{prefix}.activations2.{idx}");
            let padding1 = get_padding(kernel_size, dilation);
            let padding2 = get_padding(kernel_size, 1);
            convs1.push(Conv1dWN::from_weights(weights, &conv1_prefix, 1, padding1, dilation)?);
            convs2.push(Conv1dWN::from_weights(weights, &conv2_prefix, 1, padding2, 1)?);
            activations1.push(Snake::from_weights(weights, &act1_prefix)?);
            activations2.push(Snake::from_weights(weights, &act2_prefix)?);
        }
        Ok(Self {
            convs1,
            convs2,
            activations1,
            activations2,
        })
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let mut h = x.shallow_clone();
        for idx in 0..self.convs1.len() {
            let mut xt = self.activations1[idx].forward(&h);
            xt = self.convs1[idx].forward(&xt);
            xt = self.activations2[idx].forward(&xt);
            xt = self.convs2[idx].forward(&xt);
            h = xt + h;
        }
        h
    }
}

struct SineGen {
    sampling_rate: f64,
    harmonic_num: i64,
    sine_amp: f64,
    noise_std: f64,
    voiced_threshold: f64,
}

impl SineGen {
    fn new(sampling_rate: f64, harmonic_num: i64, sine_amp: f64, noise_std: f64, voiced_threshold: f64) -> Self {
        Self {
            sampling_rate,
            harmonic_num,
            sine_amp,
            noise_std,
            voiced_threshold,
        }
    }

    fn f02uv(&self, f0: &Tensor) -> Tensor {
        f0.gt(self.voiced_threshold).to_kind(Kind::Float)
    }

    fn forward(&self, f0: &Tensor) -> (Tensor, Tensor, Tensor) {
        let device = f0.device();
        let b = f0.size()[0];
        let t = f0.size()[2];
        let mut f_mat = Tensor::zeros([b, self.harmonic_num + 1, t], (Kind::Float, device));
        for i in 0..=self.harmonic_num {
            let scale = (i + 1) as f64 / self.sampling_rate;
            f_mat.i((.., i, ..)).copy_(&(f0.squeeze_dim(1) * scale));
        }
        let cum = f_mat.cumsum(-1, Kind::Float);
        let theta = (&cum - cum.floor()) * (2.0 * PI);
        let phase = Tensor::rand([b, self.harmonic_num + 1, 1], (Kind::Float, device)) * (2.0 * PI) - PI;
        let phase = phase;
        let _ = phase.i((.., 0, ..)).fill_(0.0);
        let sine = (theta + phase).sin() * self.sine_amp;
        let uv = self.f02uv(f0);
        let noise_amp = &uv * self.noise_std + (1.0 - &uv) * (self.sine_amp / 3.0);
        let noise = noise_amp * Tensor::randn_like(&sine);
        let sine = sine * &uv + &noise;
        (sine, uv, noise)
    }
}

struct SourceModuleHnNSF {
    sine_gen: SineGen,
    sine_amp: f64,
    linear_w: Tensor,
    linear_b: Tensor,
}

impl SourceModuleHnNSF {
    fn from_weights(weights: &Weights) -> Result<Self> {
        let sine_amp = 0.1;
        let linear_w = weights.get("mel2wav.m_source.l_linear.weight")?;
        let linear_b = weights.get("mel2wav.m_source.l_linear.bias")?;
        Ok(Self {
            sine_gen: SineGen::new(24000.0, 8, sine_amp, 0.003, 10.0),
            sine_amp,
            linear_w,
            linear_b,
        })
    }

    fn forward(&self, x: &Tensor) -> (Tensor, Tensor, Tensor) {
        let (sine, uv, _noise) = self.sine_gen.forward(&x.transpose(1, 2));
        let sine = sine.transpose(1, 2);
        let uv = uv.transpose(1, 2);
        let sine_merge = sine
            .linear(&self.linear_w, Some(&self.linear_b))
            .tanh();
        let noise = Tensor::randn_like(&uv) * (self.sine_amp / 3.0);
        (sine_merge, noise, uv)
    }
}

struct ConvRNNF0Predictor {
    convs: Vec<Conv1dWN>,
    classifier_w: Tensor,
    classifier_b: Tensor,
}

impl ConvRNNF0Predictor {
    fn from_weights(weights: &Weights) -> Result<Self> {
        let mut convs = Vec::with_capacity(5);
        for idx in [0, 2, 4, 6, 8] {
            let prefix = format!("mel2wav.f0_predictor.condnet.{idx}");
            convs.push(Conv1dWN::from_weights(weights, &prefix, 1, 1, 1)?);
        }
        let classifier_w = weights.get("mel2wav.f0_predictor.classifier.weight")?;
        let classifier_b = weights.get("mel2wav.f0_predictor.classifier.bias")?;
        Ok(Self {
            convs,
            classifier_w,
            classifier_b,
        })
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let mut h = x.shallow_clone();
        for conv in &self.convs {
            h = conv.forward(&h).elu();
        }
        let h = h.transpose(1, 2);
        h.linear(&self.classifier_w, Some(&self.classifier_b)).abs().squeeze_dim(-1)
    }
}

fn get_padding(kernel_size: i64, dilation: i64) -> i64 {
    (kernel_size * dilation - dilation) / 2
}

fn leaky_relu(x: &Tensor, slope: f64) -> Tensor {
    let relu = x.relu();
    &relu + (x - &relu) * slope
}

fn weight_norm(v: &Tensor, g: &Tensor, dim: i64) -> Tensor {
    let mut dims: Vec<i64> = (0..v.dim()).map(|d| d as i64).collect();
    dims.retain(|&d| d != dim);
    let v_norm = v.pow_tensor_scalar(2.0).sum_dim_intlist(&dims, true, Kind::Float).sqrt();
    v * (g / v_norm)
}
