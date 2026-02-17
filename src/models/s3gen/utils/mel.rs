use tch::{Device, Kind, Tensor};

fn hz_to_mel(hz: f64) -> f64 {
    1127.0 * (1.0 + hz / 700.0).ln()
}

fn mel_to_hz(mel: f64) -> f64 {
    700.0 * (mel / 1127.0).exp() - 700.0
}

fn mel_filterbank(
    n_fft: i64,
    n_mels: i64,
    sample_rate: i64,
    fmin: f64,
    fmax: f64,
    device: Device,
) -> Tensor {
    let n_fft_bins = n_fft / 2 + 1;
    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);
    let mel_step = (mel_max - mel_min) / (n_mels + 1) as f64;

    let mut mel_points = Vec::with_capacity((n_mels + 2) as usize);
    for i in 0..(n_mels + 2) {
        mel_points.push(mel_min + mel_step * i as f64);
    }
    let hz_points: Vec<f64> = mel_points.into_iter().map(mel_to_hz).collect();
    let bin_points: Vec<i64> = hz_points
        .iter()
        .map(|hz| ((n_fft + 1) as f64 * hz / sample_rate as f64).floor() as i64)
        .collect();

    let mut fb = vec![0f32; (n_mels * n_fft_bins) as usize];
    for m in 0..n_mels {
        let f_m_left = bin_points[m as usize].clamp(0, n_fft_bins - 1);
        let f_m = bin_points[(m + 1) as usize].clamp(0, n_fft_bins - 1);
        let f_m_right = bin_points[(m + 2) as usize].clamp(0, n_fft_bins - 1);

        if f_m_left == f_m || f_m == f_m_right {
            continue;
        }

        for k in f_m_left..f_m {
            let val = (k - f_m_left) as f32 / (f_m - f_m_left) as f32;
            fb[(m * n_fft_bins + k) as usize] = val;
        }
        for k in f_m..f_m_right {
            let val = (f_m_right - k) as f32 / (f_m_right - f_m) as f32;
            fb[(m * n_fft_bins + k) as usize] = val;
        }

        // Slaney-style normalization
        let enorm = 2.0 / (hz_points[(m + 2) as usize] - hz_points[m as usize]);
        let row_offset = (m * n_fft_bins) as usize;
        for k in 0..n_fft_bins {
            fb[row_offset + k as usize] *= enorm as f32;
        }
    }

    Tensor::from_slice(&fb)
        .reshape([n_mels, n_fft_bins])
        .to_device(device)
}

fn dynamic_range_compression(x: &Tensor, clip_val: f64) -> Tensor {
    x.clamp_min(clip_val).log()
}

fn spectral_normalize(x: &Tensor) -> Tensor {
    dynamic_range_compression(x, 1e-5)
}

/// Matcha-style mel-spectrogram extraction used by S3Gen.
pub fn mel_spectrogram(
    y: &Tensor,
    n_fft: i64,
    num_mels: i64,
    sampling_rate: i64,
    hop_size: i64,
    win_size: i64,
    fmin: f64,
    fmax: f64,
    center: bool,
) -> Tensor {
    let mut y = if y.dim() == 1 { y.unsqueeze(0) } else { y.shallow_clone() };
    let device = y.device();
    let window = Tensor::hann_window(win_size, (Kind::Float, device));

    let pad = (n_fft - hop_size) / 2;
    y = y
        .unsqueeze(1)
        .pad(&[pad, pad], "reflect", None)
        .squeeze_dim(1);

    let spec = y.stft_center(
        n_fft,
        Some(hop_size),
        Some(win_size),
        Some(&window),
        center,
        "reflect",
        false,
        true,
        true,
        false,
    );

    let spec = spec.view_as_real();
    let mag = spec
        .pow_tensor_scalar(2.0)
        .sum_dim_intlist([-1].as_slice(), false, Kind::Float)
        .sqrt();
    let mel_basis = mel_filterbank(n_fft, num_mels, sampling_rate, fmin, fmax, device);
    let mel = mel_basis.unsqueeze(0).matmul(&mag);
    spectral_normalize(&mel)
}
