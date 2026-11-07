//! The airfill command: decode with ffmpeg, fill, write FLAC or WAV.

use airfill::audio::{self, Bits};
use airfill::engine::{self, Settings};
use clap::Parser;
use std::io::{IsTerminal, Write};

/// Fill the highs a lossy codec cut off (MP3, AAC, Opus...), following the cut over time.
/// Writes FLAC or WAV (by the extension) with the tags of the input. Needs ffmpeg.
#[derive(Parser)]
#[command(version)]
struct Args {
    /// Any file ffmpeg reads
    input: Option<String>,
    /// The .flac or .wav file to write
    output: Option<String>,
    /// Bits per sample of the output: 16, 24, or 32f (32-bit float, WAV only)
    #[arg(long, default_value = "16", value_parser = parse_bits)]
    bits: Bits,
    /// FLAC compression level, 0 (fastest) to 12 (smallest); the sound is the same
    #[arg(long, default_value_t = 0, value_parser = clap::value_parser!(u32).range(0..=12))]
    flac_level: u32,
    /// Sample rate of the output (default: that of the input, at least 44100)
    #[arg(long)]
    rate: Option<u32>,
    /// Highest frequency the fill reaches, Hz (default: 20000, or the top of the band at 48 kHz
    /// and up; tuned and measured for 20000 only)
    #[arg(long)]
    top: Option<f64>,
    /// Level change of the input in dB (default: lowered only as much as needed not to clip)
    #[arg(long, allow_negative_numbers = true)]
    gain: Option<f64>,
    /// Extra slope of the fill above the cut, dB/oct: - darker, + brighter
    #[arg(long, default_value_t = 0.0, allow_negative_numbers = true)]
    tilt: f64,
    /// Side of the fill raised (+) or lowered (-) by this many dB, power kept
    #[arg(long, default_value_t = 0.0, allow_negative_numbers = true)]
    fill_width: f64,
    /// Fill the side up from its own cut when the codec cut it lower than the mid, but not under
    /// this share of the mid cut; 1 = off
    #[arg(long, default_value_t = 0.6)]
    side_min: f64,
    /// How much fine structure of the band under the cut the fill copies, 0 (smooth) to 1
    #[arg(long, default_value_t = 0.0)]
    texture: f64,
    /// Fill spectral holes deeper than this many dB under the cut with noise, 0 = off
    #[arg(long, default_value_t = 15.0)]
    holes: f64,
    /// How much the fill follows the level of the band under the cut, 0..1
    #[arg(long, default_value_t = 0.7)]
    follow: f64,
    /// For cuts under 15 kHz the fill falls by at most this many dB per second, 0 = no limit
    #[arg(long, default_value_t = 80.0)]
    fall: f64,
    /// Read stereo f32le from stdin and write it to stdout (needs --rate); for the benchmark
    #[arg(long)]
    raw: bool,
}

fn parse_bits(text: &str) -> Result<Bits, String> {
    match text {
        "16" => Ok(Bits::Int16),
        "24" => Ok(Bits::Int24),
        "32f" => Ok(Bits::Float32),
        _ => Err("use 16, 24 or 32f".to_string()),
    }
}

fn settings(a: &Args) -> Settings {
    Settings {
        top_hz: a.top.unwrap_or(0.0),
        tilt_db_oct: a.tilt,
        fill_width_db: a.fill_width,
        side_min: a.side_min,
        texture: a.texture,
        holes_db: a.holes,
        follow: a.follow,
        fall_db_s: a.fall,
        ..Settings::default()
    }
}

fn run(a: Args) -> Result<(), String> {
    let s = settings(&a);
    if a.raw {
        let rate = a.rate.ok_or("--raw needs --rate")?;
        let x = audio::read_raw()?;
        let (y, report) = engine::airfill(&x, rate as f64, &s);
        eprintln!("{}", summary(&report, &x, &y));
        return audio::write_raw(&y);
    }

    let input = a.input.as_deref().ok_or("give an input file and an output file")?;
    let output = a.output.as_deref().ok_or("give an output file")?;
    audio::is_wav(output, a.bits)?;

    // progress goes to stderr and only to a terminal, so pipes and logs stay clean
    let terminal = std::io::stderr().is_terminal();
    let status = |text: &str| {
        if terminal {
            eprint!("\r{text:w$}\r", w = BAR + 6);
            let _ = std::io::stderr().flush();
        }
    };

    // mono stays mono; more than two channels are mixed down to stereo
    status("reading...");
    let (native, channels) = audio::probe(input)?;
    let rate = a.rate.unwrap_or(native.max(44100));
    let mut x = audio::decode(input, rate, channels.min(2))?;
    if let Some(gain) = a.gain {
        audio::change_level(&mut x, gain);
    }

    let (mut y, report) = engine::airfill_with_progress(&x, rate as f64, &s, &mut |done| {
        status(&format!("{} {:3.0}%", bar(done), 100.0 * done));
    });
    status("");
    println!("{}", summary(&report, &x, &y));

    // integer samples clip over 0 dBFS; leave 0.3 dB for the dither and the player
    let peak = audio::peak_db(&y);
    if a.bits != Bits::Float32 && peak > -0.3 {
        match a.gain {
            None => {
                audio::change_level(&mut y, -0.3 - peak);
                println!("lowered by {:.2} dB so the file does not clip", peak + 0.3);
            }
            Some(gain) if peak > 0.0 => {
                let fix = gain - ((peak + 0.3) * 10.0).ceil() / 10.0;
                eprintln!("warning: output peaks at {peak:+.2} dBFS and clips; --gain {fix:.1} would not");
            }
            Some(_) => {}
        }
    }
    status("writing...");
    let written = audio::encode(&y, rate, a.bits, a.flac_level, input, output);
    status("");
    written
}

const BAR: usize = 28;

/// A bar of BAR cells in half-cell steps: heavy line done, a half-heavy cell, light line to go.
fn bar(done: f64) -> String {
    let halves = (done.clamp(0.0, 1.0) * BAR as f64 * 2.0) as usize;
    let full = halves / 2;
    let mut out = "\u{2501}".repeat(full);
    if full < BAR {
        out.push(if halves % 2 == 1 { '\u{257e}' } else { '\u{2500}' });
        out.push_str(&"\u{2500}".repeat(BAR - 1 - full));
    }
    out
}

fn summary(report: &engine::Report, x: &[Vec<f32>], y: &[Vec<f32>]) -> String {
    match report.cut_median_hz {
        Some(cut) => format!(
            "filled {:.0}% of frames, median cut {:.0} Hz, peak {:+.2} dBFS (input {:+.2})",
            100.0 * report.filled_share,
            cut,
            audio::peak_db(y),
            audio::peak_db(x)
        ),
        None => "no codec cut found, nothing filled".to_string(),
    }
}

fn main() {
    if let Err(e) = run(Args::parse()) {
        eprintln!("airfill: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bar_fills_in_half_cells() {
        assert_eq!(bar(0.0), "\u{2500}".repeat(BAR));
        // 15 half cells: 7 full ones and a half
        assert_eq!(bar(15.5 / 56.0), format!("{}\u{257e}{}", "\u{2501}".repeat(7), "\u{2500}".repeat(20)));
        assert_eq!(bar(0.5), format!("{}{}", "\u{2501}".repeat(14), "\u{2500}".repeat(14)));
        assert_eq!(bar(1.0), "\u{2501}".repeat(BAR));
    }
}
