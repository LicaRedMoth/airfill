//! Reading and writing audio through ffmpeg: any input it decodes, FLAC or WAV out.

use crate::rng::Rng;
use std::io::{Read, Write};
use std::process::{Command, Stdio};

/// Sample format of the file written.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Bits {
    Int16,
    Int24,
    Float32,
}

/// Channels of float samples from any file ffmpeg reads, resampled with soxr to `rate`.
pub fn decode(path: &str, rate: u32, channels: usize) -> Result<Vec<Vec<f32>>, String> {
    let output = Command::new("ffmpeg")
        .args(["-v", "error", "-nostdin", "-i", path, "-ac", &channels.to_string()])
        .args(["-af", &format!("aresample={rate}:resampler=soxr:precision=28")])
        .args(["-f", "f32le", "-"])
        .output()
        .map_err(|e| format!("cannot run ffmpeg: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "ffmpeg could not read {path}: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(deinterleave(&output.stdout, channels))
}

/// Sample rate and channel count of the first audio stream of a file, from ffprobe.
pub fn probe(path: &str) -> Result<(u32, usize), String> {
    let output = Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "a:0", "-show_entries", "stream=sample_rate,channels"])
        .args(["-of", "default=nw=1:nk=1", path])
        .output()
        .map_err(|e| format!("cannot run ffprobe: {e}"))?;
    let text = String::from_utf8_lossy(&output.stdout);
    let mut lines = text.lines();
    let rate = lines.next().and_then(|l| l.trim().parse().ok());
    let channels = lines.next().and_then(|l| l.trim().parse().ok());
    match (rate, channels) {
        (Some(r), Some(c)) => Ok((r, c)),
        _ => Err(format!("no audio stream in {path}")),
    }
}

/// Whether `out` is a WAV, and an error for float samples in a FLAC.
pub fn is_wav(out: &str, bits: Bits) -> Result<bool, String> {
    let wav = out.to_lowercase().ends_with(".wav");
    if !wav && bits == Bits::Float32 {
        return Err("FLAC has no float samples; write a .wav for --bits 32f".to_string());
    }
    Ok(wav)
}

/// FLAC or WAV (by the extension of `out`) with the tags of `tags_from`, and its cover for FLAC.
/// Integer formats get TPDF dither of one step.
pub fn encode(
    channels: &[Vec<f32>],
    rate: u32,
    bits: Bits,
    flac_level: u32,
    tags_from: &str,
    out: &str,
) -> Result<(), String> {
    let wav = is_wav(out, bits)?;

    let mut rng = Rng::new(0);
    let mut pcm = Vec::with_capacity(channels[0].len() * channels.len() * 4);
    for i in 0..channels[0].len() {
        for ch in channels {
            match bits {
                Bits::Int16 => {
                    let dither = rng.next_f32() - rng.next_f32();
                    let value = (ch[i] * 32767.0 + dither).round().clamp(-32768.0, 32767.0) as i16;
                    pcm.extend_from_slice(&value.to_le_bytes());
                }
                Bits::Int24 => {
                    let dither = (rng.next_f32() - rng.next_f32()) as f64;
                    let value =
                        (ch[i] as f64 * 8388607.0 + dither).round().clamp(-8388608.0, 8388607.0) as i32;
                    pcm.extend_from_slice(&value.to_le_bytes()[..3]);
                }
                Bits::Float32 => pcm.extend_from_slice(&ch[i].to_le_bytes()),
            }
        }
    }

    let (raw, codec) = match (bits, wav) {
        (Bits::Int16, true) => ("s16le", "pcm_s16le"),
        (Bits::Int24, true) => ("s24le", "pcm_s24le"),
        (Bits::Float32, _) => ("f32le", "pcm_f32le"),
        (Bits::Int16, false) => ("s16le", "flac"),
        (Bits::Int24, false) => ("s24le", "flac"),
    };
    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-v", "error", "-nostdin", "-y", "-i", tags_from])
        .args(["-f", raw, "-ar", &rate.to_string(), "-ac", &channels.len().to_string(), "-i", "pipe:0"])
        .args(["-map", "1:a", "-map_metadata", "0", "-c:a", codec]);
    // a WAV holds one stream only, so the cover stays out
    if !wav {
        cmd.args(["-map", "0:v:disp:attached_pic?", "-c:v", "copy"])
            .args(["-compression_level", &flac_level.to_string()]);
    }

    let mut child =
        cmd.arg(out).stdin(Stdio::piped()).spawn().map_err(|e| format!("cannot run ffmpeg: {e}"))?;
    child
        .stdin
        .take()
        .expect("piped stdin")
        .write_all(&pcm)
        .map_err(|e| format!("writing to ffmpeg: {e}"))?;
    let status = child.wait().map_err(|e| format!("ffmpeg: {e}"))?;
    if !status.success() {
        return Err(format!("ffmpeg could not write {out}"));
    }
    Ok(())
}

/// Stereo float samples from stdin (f32le, interleaved), for the benchmark.
pub fn read_raw() -> Result<Vec<Vec<f32>>, String> {
    let mut bytes = Vec::new();
    std::io::stdin().read_to_end(&mut bytes).map_err(|e| format!("reading stdin: {e}"))?;
    Ok(deinterleave(&bytes, 2))
}

pub fn write_raw(channels: &[Vec<f32>]) -> Result<(), String> {
    let mut bytes = Vec::with_capacity(channels[0].len() * 8);
    for i in 0..channels[0].len() {
        for ch in channels {
            bytes.extend_from_slice(&ch[i].to_le_bytes());
        }
    }
    std::io::stdout().write_all(&bytes).map_err(|e| format!("writing stdout: {e}"))
}

pub fn change_level(channels: &mut [Vec<f32>], db: f64) {
    let scale = 10f64.powf(db / 20.0) as f32;
    for v in channels.iter_mut().flatten() {
        *v *= scale;
    }
}

pub fn peak_db(channels: &[Vec<f32>]) -> f64 {
    let peak = channels.iter().flatten().fold(0.0f32, |m, v| m.max(v.abs()));
    20.0 * (peak as f64 + 1e-12).log10()
}

fn deinterleave(bytes: &[u8], channels: usize) -> Vec<Vec<f32>> {
    let samples: Vec<f32> = bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
    let mut out = vec![Vec::with_capacity(samples.len() / channels); channels];
    for frame in samples.chunks_exact(channels) {
        for (ch, v) in out.iter_mut().zip(frame) {
            ch.push(*v);
        }
    }
    out
}
