use std::path::Path;

use crate::error::{ChatterboxError, Result};

pub fn load_wav_mono(path: &Path) -> Result<(Vec<f32>, u32)> {
    let data = std::fs::read(path)?;
    if data.len() < 12 || &data[0..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        return Err(ChatterboxError::ShapeMismatch(
            "unsupported audio format (expected RIFF/WAVE)".to_string(),
        ));
    }

    let mut fmt: Option<WavFmt> = None;
    let mut data_chunk: Option<(usize, usize)> = None;
    let mut offset = 12;
    while offset + 8 <= data.len() {
        let id = &data[offset..offset + 4];
        let size = read_u32(&data, offset + 4)? as usize;
        offset += 8;
        if offset + size > data.len() {
            return Err(ChatterboxError::ShapeMismatch(
                "corrupted WAV chunk size".to_string(),
            ));
        }
        match id {
            b"fmt " => {
                fmt = Some(parse_fmt(&data[offset..offset + size])?);
            }
            b"data" => {
                data_chunk = Some((offset, size));
            }
            _ => {}
        }
        offset += size + (size % 2);
        if fmt.is_some() && data_chunk.is_some() {
            break;
        }
    }

    let fmt = fmt.ok_or_else(|| {
        ChatterboxError::ShapeMismatch("missing WAV fmt chunk".to_string())
    })?;
    let (data_off, data_len) = data_chunk.ok_or_else(|| {
        ChatterboxError::ShapeMismatch("missing WAV data chunk".to_string())
    })?;

    let bytes_per_sample = (fmt.bits_per_sample as usize + 7) / 8;
    if bytes_per_sample == 0 || fmt.channels == 0 {
        return Err(ChatterboxError::ShapeMismatch(
            "invalid WAV format".to_string(),
        ));
    }
    let frame_size = bytes_per_sample * fmt.channels as usize;
    if data_len < frame_size {
        return Err(ChatterboxError::ShapeMismatch(
            "WAV data chunk too small".to_string(),
        ));
    }
    let frames = data_len / frame_size;
    let mut out = Vec::with_capacity(frames);

    let mut pos = data_off;
    for _ in 0..frames {
        let mut acc = 0.0f64;
        for _ in 0..fmt.channels {
            let sample = match fmt.audio_format {
                1 => read_pcm_sample(&data, pos, fmt.bits_per_sample)?,
                3 => read_float_sample(&data, pos, fmt.bits_per_sample)?,
                _ => {
                    return Err(ChatterboxError::ShapeMismatch(
                        "unsupported WAV audio format".to_string(),
                    ))
                }
            };
            acc += sample as f64;
            pos += bytes_per_sample;
        }
        out.push((acc / fmt.channels as f64) as f32);
    }

    Ok((out, fmt.sample_rate))
}

pub fn resample_linear(samples: &[f32], src_sr: u32, dst_sr: u32) -> Vec<f32> {
    if src_sr == dst_sr || samples.is_empty() {
        return samples.to_vec();
    }
    let scale = dst_sr as f64 / src_sr as f64;
    let out_len = ((samples.len() as f64) * scale).round().max(1.0) as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let pos = i as f64 / scale;
        let idx = pos.floor() as usize;
        let frac = pos - idx as f64;
        let s0 = samples.get(idx).copied().unwrap_or(0.0);
        let s1 = samples.get(idx + 1).copied().unwrap_or(s0);
        out.push((s0 as f64 + (s1 as f64 - s0 as f64) * frac) as f32);
    }
    out
}

pub fn write_wav_f32(path: &Path, samples: &[f32], sample_rate: u32, channels: u16) -> Result<()> {
    if channels == 0 {
        return Err(ChatterboxError::ShapeMismatch(
            "cannot write WAV with zero channels".to_string(),
        ));
    }
    let bytes_per_sample = 4u16;
    let block_align = channels * bytes_per_sample;
    let byte_rate = sample_rate as u32 * block_align as u32;
    let data_bytes = (samples.len() * 4) as u32;
    let riff_size = 4 + (8 + 16) + (8 + data_bytes);

    let mut buf = Vec::with_capacity((riff_size + 8) as usize);
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&riff_size.to_le_bytes());
    buf.extend_from_slice(b"WAVE");

    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes());
    buf.extend_from_slice(&3u16.to_le_bytes()); // IEEE float
    buf.extend_from_slice(&channels.to_le_bytes());
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    buf.extend_from_slice(&block_align.to_le_bytes());
    buf.extend_from_slice(&(bytes_per_sample * 8).to_le_bytes());

    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_bytes.to_le_bytes());
    for sample in samples {
        buf.extend_from_slice(&sample.to_le_bytes());
    }

    std::fs::write(path, buf)?;
    Ok(())
}

pub fn write_wav_pcm16(path: &Path, samples: &[f32], sample_rate: u32, channels: u16) -> Result<()> {
    if channels == 0 {
        return Err(ChatterboxError::ShapeMismatch(
            "cannot write WAV with zero channels".to_string(),
        ));
    }
    let bytes_per_sample = 2u16;
    let block_align = channels * bytes_per_sample;
    let byte_rate = sample_rate as u32 * block_align as u32;
    let data_bytes = (samples.len() * 2) as u32;
    let riff_size = 4 + (8 + 16) + (8 + data_bytes);

    let mut buf = Vec::with_capacity((riff_size + 8) as usize);
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&riff_size.to_le_bytes());
    buf.extend_from_slice(b"WAVE");

    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes());
    buf.extend_from_slice(&1u16.to_le_bytes()); // PCM
    buf.extend_from_slice(&channels.to_le_bytes());
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    buf.extend_from_slice(&block_align.to_le_bytes());
    buf.extend_from_slice(&(bytes_per_sample * 8).to_le_bytes());

    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_bytes.to_le_bytes());
    for sample in samples {
        let clamped = sample.max(-1.0).min(1.0);
        let scaled = if clamped >= 0.0 {
            (clamped * 32767.0).round() as i16
        } else {
            (clamped * 32768.0).round() as i16
        };
        buf.extend_from_slice(&scaled.to_le_bytes());
    }

    std::fs::write(path, buf)?;
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct WavFmt {
    audio_format: u16,
    channels: u16,
    sample_rate: u32,
    bits_per_sample: u16,
}

fn parse_fmt(chunk: &[u8]) -> Result<WavFmt> {
    if chunk.len() < 16 {
        return Err(ChatterboxError::ShapeMismatch(
            "invalid WAV fmt chunk".to_string(),
        ));
    }
    let audio_format = read_u16(chunk, 0)?;
    let channels = read_u16(chunk, 2)?;
    let sample_rate = read_u32(chunk, 4)?;
    let bits_per_sample = read_u16(chunk, 14)?;
    Ok(WavFmt {
        audio_format,
        channels,
        sample_rate,
        bits_per_sample,
    })
}

fn read_u16(data: &[u8], offset: usize) -> Result<u16> {
    if offset + 2 > data.len() {
        return Err(ChatterboxError::ShapeMismatch(
            "unexpected end of WAV data".to_string(),
        ));
    }
    Ok(u16::from_le_bytes([data[offset], data[offset + 1]]))
}

fn read_u32(data: &[u8], offset: usize) -> Result<u32> {
    if offset + 4 > data.len() {
        return Err(ChatterboxError::ShapeMismatch(
            "unexpected end of WAV data".to_string(),
        ));
    }
    Ok(u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]))
}

fn read_pcm_sample(data: &[u8], offset: usize, bits: u16) -> Result<f32> {
    match bits {
        8 => {
            if offset + 1 > data.len() {
                return Err(ChatterboxError::ShapeMismatch(
                    "unexpected end of WAV data".to_string(),
                ));
            }
            let v = data[offset] as f32;
            Ok((v - 128.0) / 128.0)
        }
        16 => {
            let v = read_i16(data, offset)? as f32;
            Ok(v / 32768.0)
        }
        24 => {
            if offset + 3 > data.len() {
                return Err(ChatterboxError::ShapeMismatch(
                    "unexpected end of WAV data".to_string(),
                ));
            }
            let b0 = data[offset] as u32;
            let b1 = data[offset + 1] as u32;
            let b2 = data[offset + 2] as u32;
            let mut v = (b2 << 16) | (b1 << 8) | b0;
            if v & 0x800000 != 0 {
                v |= 0xFF000000;
            }
            let signed = v as i32;
            Ok(signed as f32 / 8_388_608.0)
        }
        32 => {
            let v = read_i32(data, offset)? as f32;
            Ok(v / 2_147_483_648.0)
        }
        _ => Err(ChatterboxError::ShapeMismatch(
            "unsupported PCM bit depth".to_string(),
        )),
    }
}

fn read_float_sample(data: &[u8], offset: usize, bits: u16) -> Result<f32> {
    match bits {
        32 => {
            if offset + 4 > data.len() {
                return Err(ChatterboxError::ShapeMismatch(
                    "unexpected end of WAV data".to_string(),
                ));
            }
            Ok(f32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]))
        }
        _ => Err(ChatterboxError::ShapeMismatch(
            "unsupported float WAV bit depth".to_string(),
        )),
    }
}

fn read_i16(data: &[u8], offset: usize) -> Result<i16> {
    if offset + 2 > data.len() {
        return Err(ChatterboxError::ShapeMismatch(
            "unexpected end of WAV data".to_string(),
        ));
    }
    Ok(i16::from_le_bytes([data[offset], data[offset + 1]]))
}

fn read_i32(data: &[u8], offset: usize) -> Result<i32> {
    if offset + 4 > data.len() {
        return Err(ChatterboxError::ShapeMismatch(
            "unexpected end of WAV data".to_string(),
        ));
    }
    Ok(i32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]))
}
