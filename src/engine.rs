//! Per STFT frame (2048 points, hop 512): find the codec cut, fit a line to the spectrum under it,
//! and add above the cut only the power missing under that line, with phases from under the cut.

use crate::filters::{mean_1d, mean_freq, mean_time, median, median_1d, percentile_rank, rank_2d};
use crate::grid::Grid;
use crate::rng::Rng;
use crate::stft::{Stft, BINS, HOP, NFFT};
use realfft::num_complex::Complex32;

#[derive(Clone, Debug)]
pub struct Settings {
    /// The fill fades out over the 1 kHz around this; 0 = 20 kHz, or the top of the band for rates
    /// of 48 kHz and up. The defaults were tuned for 20 kHz.
    pub top_hz: f64,
    /// Level of the whole fill, dB; a fill with no texture came out this much too dark on dev.
    pub level_db: f64,
    /// Extra slope above the cut, dB per octave, 0 at the cut.
    pub tilt_db_oct: f64,
    /// Seconds the fitted slope is smoothed over; synth sweeps made 3 s swing.
    pub slope_s: f64,
    /// The bands the slope and anchor are measured in follow the median cut of this many seconds.
    pub ref_s: f64,
    pub fit_lo: f64,
    pub fit_hi: f64,
    pub edge: f64,
    /// Limits of the fitted slope, dB/oct; at low cuts the band it is fitted in holds the codec
    /// roll-off, and -24 left the fill of quiet intros 9 dB too dark at the top.
    pub slope_min: f64,
    pub slope_max: f64,
    /// A second slope from fit_hi to near_hi of the cut pulls the slope down when it is steeper.
    pub near_hi: f64,
    pub near_min: f64,
    pub near_weight: f64,
    /// Under fall_below_hz the anchor falls by at most this many dB per second (0 = no limit).
    pub fall_db_s: f64,
    pub fall_below_hz: f64,
    /// dB the anchor is lowered per unit of anchor-band flatness over its usual value.
    pub flat_comp_db: f64,
    /// How much the fill follows the anchor around its usual level (1 = fully).
    pub follow: f64,
    /// A side cut at least side_gap_hz under the mid cut, and above side_min of it, is filled up
    /// to the mid cut (1 = off).
    pub side_min: f64,
    pub side_gap_hz: f64,
    /// Level of that side fill against the line fitted to the side, dB.
    pub side_level_db: f64,
    /// Side of the fill raised (+) or lowered (-) by this many dB, its power kept.
    pub fill_width_db: f64,
    /// How much fine structure of the source the fill keeps (0..1), fading out as the structure
    /// stays from frame to frame (tones, codec birdies): gone at texture_stay of correlation.
    pub texture: f64,
    pub texture_stay: f64,
    /// Spectral holes under the cut (holes_db under the holes_pct percentile of 9 bins x 5 frames,
    /// from holes_from of the cut up) get noise at holes_level_db, per channel (0 = off).
    pub holes_db: f64,
    pub holes_from: f64,
    pub holes_level_db: f64,
    pub holes_pct: f64,
    pub min_edge_db: f64,
    /// A cut under this level (dB per STFT bin) is the noise floor, not a codec.
    pub floor_db: f64,
    pub hold_s: f64,
    pub search_lo_hz: f64,
    /// Lowest cut followed in quiet passages, once the track has shown a codec cut.
    pub follow_lo_hz: f64,
    /// Long input is done in blocks of chunk_s with context_s on each side, crossfaded over fade_s.
    pub chunk_s: f64,
    pub context_s: f64,
    pub fade_s: f64,
    /// Extra level (dB) on the line at these frequencies, counted from the cut (dev corpus medians).
    pub rolloff: Vec<(f64, f64)>,
    /// Above flat_from..flat_to times the cut the line comes from flat_s seconds rather than the
    /// frame, so the invented top keeps its level but not the slow swings of the music.
    pub flat_from: f64,
    pub flat_to: f64,
    pub flat_s: f64,
    /// Collect per-frame decisions into `Report::trace` (off by default; costs a second line fit).
    pub trace: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            top_hz: 0.0,
            level_db: 0.7,
            tilt_db_oct: 0.0,
            slope_s: 10.0,
            ref_s: 10.0,
            fit_lo: 0.55,
            fit_hi: 0.80,
            edge: 0.10,
            slope_min: -15.0,
            slope_max: -3.0,
            near_hi: 0.95,
            near_min: -48.0,
            near_weight: 0.5,
            fall_db_s: 80.0,
            fall_below_hz: 15000.0,
            flat_comp_db: 9.9,
            follow: 0.7,
            side_min: 0.6,
            side_gap_hz: 500.0,
            side_level_db: -3.0,
            fill_width_db: 0.0,
            texture: 0.0,
            texture_stay: 0.5,
            holes_db: 15.0,
            holes_from: 0.4,
            holes_level_db: -6.0,
            holes_pct: 75.0,
            min_edge_db: 15.0,
            floor_db: -125.0,
            hold_s: 1.0,
            search_lo_hz: 8000.0,
            follow_lo_hz: 3000.0,
            chunk_s: 60.0,
            context_s: 15.0,
            fade_s: 1.0,
            rolloff: vec![(15100.0, 0.0), (16000.0, -2.0), (16950.0, -3.2), (17960.0, -5.6), (19030.0, -6.4)],
            flat_from: 1.4,
            flat_to: 2.0,
            flat_s: 60.0,
            trace: false,
        }
    }
}

pub struct Report {
    pub filled_share: f64,
    pub cut_median_hz: Option<f64>,
    pub trace: Vec<TraceFrame>,
}

/// One STFT frame of the decisions, for `--log`. Times are seconds from the start of the input.
#[derive(Clone, Debug)]
pub struct TraceFrame {
    pub block: usize,
    pub in_core: bool,
    pub time: f64,
    pub cut_hz: f64,
    pub reference_hz: f64,
    pub strength: f64,
    pub slope_db_oct: f64,
    pub anchor_db: f64,
    pub slope_long_db_oct: f64,
    pub anchor_long_db: f64,
    pub flat_corr_db: f64,
    pub line_8k_db: f64,
    pub line_12k_db: f64,
    pub line_16k_db: f64,
    pub side_cut_hz: f64,
    pub side_strength: f64,
}

/// Round with halves to the even neighbour, the rule the frame counts were tuned with.
fn round_even(x: f64) -> i64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 && (r as i64) % 2 != 0 {
        (r - x.signum()) as i64
    } else {
        r as i64
    }
}

fn frames_of(seconds: f64, sr: f64) -> usize {
    round_even(seconds * sr / HOP as f64).max(1) as usize
}

fn odd(n: usize) -> usize {
    n | 1
}

fn db(v: f64) -> f64 {
    10.0 * v.log10()
}

/// The highest frequency the fill reaches, Hz.
fn top_hz(s: &Settings, sr: f64) -> f64 {
    if s.top_hz > 0.0 {
        s.top_hz
    } else if sr >= 48000.0 {
        sr / 2.0
    } else {
        20000.0
    }
}

fn interp(x: f64, xs: &[f64], ys: &[f64]) -> f64 {
    if x <= xs[0] {
        return ys[0];
    }
    for i in 1..xs.len() {
        if x <= xs[i] {
            let w = (x - xs[i - 1]) / (xs[i] - xs[i - 1]);
            return ys[i - 1] + w * (ys[i] - ys[i - 1]);
        }
    }
    ys[ys.len() - 1]
}

/// Rows lo..hi of a grid as a grid of their own (filters then reflect at the edges of the band).
fn rows(g: &Grid<f32>, lo: usize, hi: usize) -> Grid<f32> {
    let mut out = Grid::new(hi - lo, g.frames, 0.0f32);
    for t in 0..g.frames {
        out.frame_mut(t).copy_from_slice(&g.frame(t)[lo..hi]);
    }
    out
}

fn put_rows(g: &mut Grid<f32>, lo: usize, part: &Grid<f32>) {
    for t in 0..g.frames {
        g.frame_mut(t)[lo..lo + part.bins].copy_from_slice(part.frame(t));
    }
}

/// Frequency and size (dB) of the largest drop between lo and hi Hz in every frame.
fn sharpest(drop: &Grid<f64>, f: &[f64], lo: f64, hi: f64) -> (Vec<f64>, Vec<f64>) {
    let mut cut = Vec::with_capacity(drop.frames);
    let mut size = Vec::with_capacity(drop.frames);
    for t in 0..drop.frames {
        let mut best = 0;
        let mut best_value = f64::NEG_INFINITY;
        for (k, fk) in f.iter().enumerate() {
            let v = if *fk >= lo && *fk <= hi { drop.at(k, t) } else { f64::NEG_INFINITY };
            if v > best_value {
                best = k;
                best_value = v;
            }
        }
        cut.push(f[best]);
        size.push(best_value);
    }
    (cut, size)
}

/// Cut frequency (Hz) and fill strength (0..1) of every frame, from the mean power of the channels.
fn cut_track(p: &Grid<f32>, fr: &[f64], sr: f64, s: &Settings) -> (Vec<f64>, Vec<f64>) {
    let df = fr[1];
    let mut d = p.map(|v| db(v as f64 + 1e-20) as f32);
    let lo = (((s.follow_lo_hz - 600.0) / df) as isize).max(0) as usize;
    let filtered = crate::filters::median_2d(&rows(&d, lo, d.bins), 9, 5);
    put_rows(&mut d, lo, &filtered);

    // level of the 250 Hz just under each bin minus the 250 Hz just over it
    let w = round_even(250.0 / df).max(2) as usize;
    let ks: Vec<usize> = ((w + 2)..(fr.len() - w - 2)).collect();
    let mut drop = Grid::new(ks.len(), d.frames, 0.0f64);
    let mut c = vec![0.0f64; d.bins + 1];
    for t in 0..d.frames {
        for (i, v) in d.frame(t).iter().enumerate() {
            c[i + 1] = c[i] + *v as f64;
        }
        for (j, &k) in ks.iter().enumerate() {
            let below = (c[k - 2] - c[k - 2 - w]) / w as f64;
            let above = (c[k + 2 + w] - c[k + 2]) / w as f64;
            let v = if below < s.floor_db { f64::NEG_INFINITY } else { below - above };
            drop.set(j, t, v);
        }
    }
    let f: Vec<f64> = ks.iter().map(|&k| fr[k]).collect();

    let top = top_hz(s, sr).min(sr / 2.0 - 1000.0);
    let (mut cut, mut edge) = sharpest(&drop, &f, s.search_lo_hz, top);
    let mut has_cut: Vec<bool> = edge.iter().map(|e| *e >= s.min_edge_db).collect();

    // in quiet passages the codec lowers its cut: follow it down to half its usual frequency
    if has_cut.iter().any(|h| *h) {
        let found: Vec<f64> = cut.iter().zip(&has_cut).filter(|(_, h)| **h).map(|(c, _)| *c).collect();
        let low = s.follow_lo_hz.max(0.5 * median(&found));
        let (low_cut, low_edge) = sharpest(&drop, &f, low, s.search_lo_hz);
        for t in 0..cut.len() {
            if !has_cut[t] && low_edge[t] >= s.min_edge_db {
                cut[t] = low_cut[t];
                edge[t] = low_edge[t];
                has_cut[t] = true;
            }
        }
    }

    let frames = frames_of(s.hold_s, sr);
    let idx: Vec<usize> = (0..cut.len()).filter(|t| has_cut[*t]).collect();
    if !idx.is_empty() {
        // where there is no cut, carry the nearest real one, then smooth over time
        let xs: Vec<f64> = idx.iter().map(|t| *t as f64).collect();
        let ys: Vec<f64> = idx.iter().map(|t| cut[*t]).collect();
        let carried: Vec<f64> = (0..cut.len()).map(|t| interp(t as f64, &xs, &ys)).collect();
        cut = median_1d(&carried, odd(frames));
    }
    let marks: Vec<f64> = has_cut.iter().map(|h| if *h { 1.0 } else { 0.0 }).collect();
    let strength: Vec<f64> = mean_1d(&marks, frames).iter().map(|v| (v * 2.0).clamp(0.0, 1.0)).collect();
    (cut, strength)
}

/// Least-squares slope (dB/oct) of ds between lo and hi times ref and under limit, per frame.
fn fit_slope(
    ds: &Grid<f32>,
    logf: &[f64],
    fr: &[f64],
    reference: &[f64],
    limit: &[f64],
    lo: f64,
    hi: f64,
) -> Vec<f64> {
    let mut out = Vec::with_capacity(ds.frames);
    for t in 0..ds.frames {
        let (mut n, mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for b in 0..ds.bins {
            let f = fr[b];
            if f >= lo * reference[t] && f < hi * reference[t] && f < limit[t] {
                let (x, y) = (logf[b], ds.at(b, t) as f64);
                n += 1.0;
                sx += x;
                sy += y;
                sxx += x * x;
                sxy += x * y;
            }
        }
        out.push((n * sxy - sx * sy) / (n * sxx - sx * sx));
    }
    out
}

struct Line {
    slope: Vec<f64>,
    anchor: Vec<f64>,
    anchor_f: Vec<f64>,
    slope_long: Vec<f64>,
    anchor_long: Vec<f64>,
    reference: Vec<f64>,
    flat_corr: Vec<f64>,
}

/// Slope from a time-smoothed spectrum, anchor level and frequency from the frame itself.
fn line_params(p: &Grid<f32>, fr: &[f64], sr: f64, cut: &[f64], s: &Settings) -> Line {
    let df = fr[1];
    let span = odd(frames_of(s.slope_s, sr));
    let ref_span = odd(frames_of(s.ref_s, sr));

    // bands follow the usual cut, so they stay put when the codec drops its cut for a moment
    let mut reference = median_1d(cut, ref_span);
    for t in 0..cut.len() {
        if cut[t] < 0.85 * reference[t] {
            reference[t] = cut[t];
        }
    }
    let limit: Vec<f64> = cut.iter().map(|c| 0.97 * c).collect();
    let pi = mean_time(p, 3);

    let mut ps = mean_time(p, frames_of(s.hold_s, sr));
    let r_min = reference.iter().cloned().fold(f64::INFINITY, f64::min);
    let r_max = reference.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let lo = ((s.fit_lo * r_min / df) as isize - 25).max(0) as usize;
    let hi = ((s.fit_hi.max(s.near_hi) * r_max / df) as usize + 25).min(fr.len());
    for t in 0..ps.frames {
        let column: Vec<f64> = ps.frame(t)[lo..hi].iter().map(|v| *v as f64).collect();
        for (o, v) in ps.frame_mut(t)[lo..hi].iter_mut().zip(median_1d(&column, 41)) {
            *o = v as f32;
        }
    }

    let ds = ps.map(|v| db(v as f64 + 1e-20) as f32);
    let di = pi.map(|v| db(v as f64 + 1e-20) as f32);
    let logf: Vec<f64> = fr.iter().map(|f| f.max(1.0).log2()).collect();

    // median against single odd frames, then a mean so the slope glides instead of switching
    let mut slope = fit_slope(&ds, &logf, fr, &reference, &limit, s.fit_lo, s.fit_hi);
    slope = mean_1d(&median_1d(&slope, span), span);
    for v in slope.iter_mut() {
        *v = v.clamp(s.slope_min, s.slope_max);
    }
    if s.near_hi > s.fit_hi {
        let near = fit_slope(&ds, &logf, fr, &reference, &limit, s.fit_hi, s.near_hi);
        let near = mean_1d(&median_1d(&near, span), span);
        for (v, n) in slope.iter_mut().zip(near) {
            *v += s.near_weight * (n.max(s.near_min) - *v).min(0.0);
        }
    }

    let frames = cut.len();
    let mut anchor = vec![-200.0; frames];
    let mut flat = vec![1.0; frames];
    for t in 0..frames {
        let mut levels = Vec::new();
        let mut powers = 0.0;
        for (b, &f) in fr.iter().enumerate() {
            if f >= (s.fit_hi - s.edge) * reference[t] && f < s.fit_hi * reference[t] && f < limit[t] {
                levels.push(di.at(b, t) as f64);
                powers += pi.at(b, t) as f64;
            }
        }
        if !levels.is_empty() {
            anchor[t] = median(&levels);
            let gm = levels.iter().sum::<f64>() / levels.len() as f64;
            let am = db(powers / levels.len() as f64 + 1e-20);
            flat[t] = 10f64.powf((gm - am) / 10.0);
        }
    }
    let anchor_f: Vec<f64> = reference.iter().map(|r| (s.fit_hi - s.edge / 2.0) * r).collect();

    // the anchor rises at once but falls slowly for low cuts: a fill that followed every syllable
    // of a voice under the cut sounded like radio static
    if s.fall_db_s > 0.0 && median(cut) < s.fall_below_hz {
        let step = s.fall_db_s * HOP as f64 / sr;
        for t in 1..frames {
            anchor[t] = anchor[t].max(anchor[t - 1] - step);
        }
    }
    let plain = anchor.clone();

    // a noisy anchor band (sibilants, shakers) lifted the fill more than real highs rise
    let mut flat_corr = vec![0.0; frames];
    if s.flat_comp_db > 0.0 {
        let flat = mean_1d(&flat, frames_of(0.5, sr));
        let usual = median_1d(&flat, ref_span);
        for t in 0..frames {
            flat_corr[t] = -s.flat_comp_db * (flat[t] - usual[t]);
            anchor[t] += flat_corr[t];
        }
    }

    // real highs rise less than the band under the cut when the music gets louder there
    if s.follow < 1.0 {
        let usual = median_1d(&anchor, ref_span);
        for t in 0..frames {
            anchor[t] = usual[t] + s.follow * (anchor[t] - usual[t]);
        }
    }

    // both of the above change how the fill swings, not how loud it is
    if s.follow < 1.0 || s.flat_comp_db > 0.0 {
        let win = frames_of(s.ref_s, sr);
        let before = mean_1d(&plain.iter().map(|a| 10f64.powf(a / 10.0)).collect::<Vec<_>>(), win);
        let after = mean_1d(&anchor.iter().map(|a| 10f64.powf(a / 10.0)).collect::<Vec<_>>(), win);
        for t in 0..frames {
            anchor[t] += db(before[t].max(1e-30) / after[t].max(1e-30));
        }
    }
    let (slope_long, anchor_long) = if s.flat_s > 0.0 {
        let span = odd(frames_of(s.flat_s, sr));
        (mean_1d(&slope, span), mean_1d(&anchor, span))
    } else {
        (slope.clone(), anchor.clone())
    };
    Line { slope, anchor, anchor_f, slope_long, anchor_long, reference, flat_corr }
}

/// Line level at `f` for frame `t`, before the tilt and the bend: the part the trace reports.
fn line_level(line: &Line, f: f64, cut: f64, t: usize, s: &Settings) -> f64 {
    let w = if s.flat_to > s.flat_from {
        let r = ((f / cut - s.flat_from) / (s.flat_to - s.flat_from)).clamp(0.0, 1.0);
        0.5 - 0.5 * (std::f64::consts::PI * r).cos()
    } else {
        0.0
    };
    let anchor = line.anchor[t] + w * (line.anchor_long[t] - line.anchor[t]);
    let slope = line.slope[t] + w * (line.slope_long[t] - line.slope[t]);
    anchor + slope * (f.max(1.0).log2() - line.anchor_f[t].log2())
}

fn pearson(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len() as f64;
    let ma = a.iter().map(|v| *v as f64).sum::<f64>() / n;
    let mb = b.iter().map(|v| *v as f64).sum::<f64>() / n;
    let (mut ab, mut aa, mut bb) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (*x as f64 - ma, *y as f64 - mb);
        ab += x * y;
        aa += x * x;
        bb += y * y;
    }
    if aa == 0.0 || bb == 0.0 {
        0.0
    } else {
        ab / (aa * bb).sqrt()
    }
}

/// New STFT bins to add to one channel.
fn fill_channel(
    z: &Grid<Complex32>,
    fr: &[f64],
    sr: f64,
    cut: &[f64],
    strength: &[f64],
    s: &Settings,
) -> Grid<Complex32> {
    let df = fr[1];
    let p = z.map(|v| v.norm_sqr());
    let have = mean_freq(&mean_time(&p, 3), 5);
    let smooth = mean_freq(&p, 9);
    let mut dev = p.clone();
    for (d, sm) in dev.data.iter_mut().zip(&smooth.data) {
        *d = (*d / (*sm + 1e-20)).clamp(1e-4, 1e4).ln();
    }
    let line = line_params(&p, fr, sr, cut, s);

    let logf: Vec<f64> = fr.iter().map(|f| f.max(1.0).log2()).collect();
    let (xs, ys): (Vec<f64>, Vec<f64>) = s.rolloff.iter().cloned().unzip();
    let extra: Vec<f64> = fr.iter().map(|f| if xs.is_empty() { 0.0 } else { interp(*f, &xs, &ys) }).collect();
    let reach = top_hz(s, sr);
    let top: Vec<f64> = fr
        .iter()
        .map(|f| {
            let r = ((reach + 500.0 - f) / 1000.0).clamp(0.0, 1.0);
            0.5 - 0.5 * (std::f64::consts::PI * r).cos()
        })
        .collect();

    let mut new = Grid::new(z.bins, z.frames, Complex32::new(0.0, 0.0));
    for t in 0..z.frames {
        if strength[t] <= 0.0 {
            continue;
        }
        let b0 = round_even(s.fit_hi * cut[t] / df) as usize;
        let width = (cut[t] / 3.0 / df) as usize / 4 * 4;
        let b1 = (fr.len() - 1).min(((reach + 500.0) / df) as usize);
        if width < 4 || b1 <= b0 {
            continue;
        }
        let cut_bin = (fr.len() - 1).min((cut[t] / df) as usize);
        let log_cut = cut[t].log2();

        let mut want = Vec::with_capacity(b1 - b0 + 1);
        for b in b0..=b1 {
            // the bend counts from the cut, so the fill starts level with what is under it
            let bend = (extra[b] - extra[cut_bin]).min(0.0);
            let tilt = s.tilt_db_oct * (logf[b] - log_cut).max(0.0);
            let level = line_level(&line, fr[b], cut[t], t, s) + s.level_db + bend + tilt;
            let missing = (10f64.powf(level / 10.0) - have.at(b, t) as f64).max(0.0);
            want.push(missing * strength[t] * top[b] * top[b]);
        }

        // part of the fine structure of the source goes up, unless it stays from frame to frame
        // (tones, codec birdies); 4 hops back the frames no longer overlap
        if s.texture > 0.0 {
            let stay = if t >= 4 {
                pearson(&dev.frame(t)[b0 - width..b0], &dev.frame(t - 4)[b0 - width..b0])
            } else {
                0.0
            };
            let power = s.texture * (1.0 - stay / s.texture_stay).clamp(0.0, 1.0);
            let mut tex = Vec::with_capacity(want.len());
            for b in b0..=b1 {
                let src = b0 - width + (b - b0) % width;
                let ratio = (p.at(src, t) as f64 / (smooth.at(src, t) as f64 + 1e-20)).clamp(0.01, 100.0);
                tex.push(ratio.powf(power));
            }
            let mean = tex.iter().sum::<f64>() / tex.len() as f64 + 1e-20;
            for (w, x) in want.iter_mut().zip(tex) {
                *w *= x / mean;
            }
        }

        for (i, b) in (b0..=b1).enumerate() {
            let src = b0 - width + (b - b0) % width;
            let source = z.at(src, t);
            let phase = source / (source.norm() + 1e-12);
            new.set(b, t, phase * want[i].sqrt() as f32);
        }
    }
    new
}

/// Noise for the spectral holes a low-bitrate codec leaves under its cut, per channel.
fn noise_fill(
    z: &[Grid<Complex32>],
    fr: &[f64],
    cut: &[f64],
    strength: &[f64],
    s: &Settings,
) -> Vec<Grid<Complex32>> {
    let df = fr[1];
    let found: Vec<f64> = cut.iter().zip(strength).filter(|(_, st)| **st > 0.0).map(|(c, _)| *c).collect();
    let usual = median(&found);
    let lo = ((s.holes_from * usual / df) as isize - 5).max(0) as usize;
    let hi = ((usual / df) as usize + 5).min(fr.len());
    let rank = percentile_rank(45, s.holes_pct);

    let mut rng = Rng::new(3);
    let mut out = Vec::new();
    for zc in z {
        let level = zc.map(|v| db(v.norm_sqr() as f64 + 1e-20) as f32);
        let band = rows(&level, lo, hi);
        let local = rank_2d(&band, 9, 5, rank);

        let mut add = Grid::new(zc.bins, zc.frames, Complex32::new(0.0, 0.0));
        for t in 0..zc.frames {
            for j in 0..band.bins {
                let f = fr[lo + j];
                // the generator runs for every bin of the band,
                // so the noise does not depend on where the holes are
                let angle = 2.0 * std::f32::consts::PI * rng.next_f32();
                let inside = strength[t] > 0.0 && f >= s.holes_from * cut[t] && f < 0.97 * cut[t];
                if inside && band.at(j, t) < local.at(j, t) - s.holes_db as f32 {
                    let amp = 10f32.powf((local.at(j, t) + s.holes_level_db as f32) / 20.0);
                    add.set(lo + j, t, Complex32::from_polar(amp, angle));
                }
            }
        }
        out.push(add);
    }
    out
}

/// Stereo fill with its side scaled by gain (per frame), the power of the fill kept.
fn side_gain(new: &[Grid<Complex32>], gain: &[f64], span: usize) -> Vec<Grid<Complex32>> {
    let (frames, bins) = (new[0].frames, new[0].bins);
    let mut pm = vec![0.0; frames];
    let mut ps = vec![0.0; frames];
    for t in 0..frames {
        for b in 0..bins {
            let (l, r) = (new[0].at(b, t), new[1].at(b, t));
            pm[t] += ((l + r) * 0.5).norm_sqr() as f64;
            ps[t] += ((l - r) * 0.5).norm_sqr() as f64;
        }
    }
    let pm: Vec<f64> = mean_1d(&pm, span).iter().map(|v| v.max(0.0)).collect();
    let ps: Vec<f64> = mean_1d(&ps, span).iter().map(|v| v.max(0.0)).collect();

    let mut left = new[0].clone();
    let mut right = new[1].clone();
    for t in 0..frames {
        let keep = ((pm[t] + ps[t]) / (pm[t] + gain[t] * gain[t] * ps[t] + 1e-30)).sqrt() as f32;
        let g = gain[t] as f32;
        for b in 0..bins {
            let (l, r) = (new[0].at(b, t), new[1].at(b, t));
            let mid = (l + r) * 0.5 * keep;
            let side = (l - r) * 0.5 * g * keep;
            left.set(b, t, mid + side);
            right.set(b, t, mid - side);
        }
    }
    vec![left, right]
}

/// The signal to add to one block (channels of samples), and its report.
/// Joint stereo often cuts the side lower than the mid: fill the side from its own cut up to
/// the mid cut, where the stereo fill takes over.
fn side_gap(
    z: &[Grid<Complex32>],
    fr: &[f64],
    sr: f64,
    cut: &[f64],
    s: &Settings,
) -> (Grid<Complex32>, Vec<f64>, Vec<f64>) {
    let mut side = z[0].clone();
    for (v, r) in side.data.iter_mut().zip(&z[1].data) {
        *v = (*v - r) * 0.5;
    }
    let (side_cut, mut strength) = cut_track(&side.map(|v| v.norm_sqr()), fr, sr, s);

    // a side far under the mid is a near-mono mix, not the codec
    for t in 0..cut.len() {
        if side_cut[t] > cut[t] - s.side_gap_hz || side_cut[t] < s.side_min * cut[t] {
            strength[t] = 0.0;
        }
    }
    let quieter = Settings { level_db: s.level_db + s.side_level_db, ..s.clone() };
    let mut fill = fill_channel(&side, fr, sr, &side_cut, &strength, &quieter);

    // fade in over 80-90% of the side cut, so the start leaves no line; over the mid cut the
    // stereo fill has a side of its own
    let df = fr[1];
    for t in 0..cut.len() {
        let hi = ((cut[t] / df) as usize).min(fill.bins);
        let frame = fill.frame_mut(t);
        for (b, v) in frame.iter_mut().enumerate() {
            let fade = ((b as f64 * df / side_cut[t] - 0.8) / 0.1).clamp(0.0, 1.0);
            if b >= hi {
                *v = Complex32::new(0.0, 0.0);
            } else {
                *v *= fade as f32;
            }
        }
    }
    (fill, side_cut, strength)
}

/// Fill one block; `done` gets the share of the block finished, at stages weighed by their time.
fn fill_block(
    x: &[Vec<f32>],
    sr: f64,
    s: &Settings,
    start_sample: usize,
    done: &mut dyn FnMut(f64),
) -> (Vec<Vec<f32>>, Report) {
    let samples = x[0].len();
    let fr: Vec<f64> = (0..BINS).map(|b| b as f64 * sr / NFFT as f64).collect();
    let mut stft = Stft::new();
    let z: Vec<Grid<Complex32>> = x.iter().map(|ch| stft.forward(ch)).collect();
    done(0.02);

    let mut p = Grid::new(BINS, z[0].frames, 0.0f32);
    for zc in &z {
        for (o, v) in p.data.iter_mut().zip(&zc.data) {
            *o += v.norm_sqr() / z.len() as f32;
        }
    }
    let (cut, strength) = cut_track(&p, &fr, sr, s);
    done(0.27);

    let line = if s.trace { Some(line_params(&z[0].map(|v| v.norm_sqr()), &fr, sr, &cut, s)) } else { None };
    let filled: Vec<f64> = cut.iter().zip(&strength).filter(|(_, st)| **st > 0.0).map(|(c, _)| *c).collect();
    let mut report = Report {
        filled_share: filled.len() as f64 / cut.len() as f64,
        cut_median_hz: if filled.is_empty() { None } else { Some(median(&filled)) },
        trace: Vec::new(),
    };
    if let Some(line) = &line {
        report.trace = (0..cut.len())
            .map(|t| TraceFrame {
                block: 0,
                in_core: true,
                time: (start_sample + t * HOP) as f64 / sr,
                cut_hz: cut[t],
                reference_hz: line.reference[t],
                strength: strength[t],
                slope_db_oct: line.slope[t],
                anchor_db: line.anchor[t],
                slope_long_db_oct: line.slope_long[t],
                anchor_long_db: line.anchor_long[t],
                flat_corr_db: line.flat_corr[t],
                line_8k_db: line_level(line, 8000.0, cut[t], t, s),
                line_12k_db: line_level(line, 12000.0, cut[t], t, s),
                line_16k_db: line_level(line, 16000.0, cut[t], t, s),
                side_cut_hz: f64::NAN,
                side_strength: f64::NAN,
            })
            .collect();
    }
    if filled.is_empty() {
        done(1.0);
        return (vec![vec![0.0; samples]; x.len()], report);
    }

    // one channel per core
    let mut new: Vec<Grid<Complex32>> = std::thread::scope(|scope| {
        let mut jobs = Vec::new();
        for zc in &z {
            let (fr, cut, strength) = (&fr, &cut, &strength);
            jobs.push(scope.spawn(move || fill_channel(zc, fr, sr, cut, strength, s)));
        }
        jobs.into_iter().map(|j| j.join().expect("fill thread")).collect()
    });
    done(0.51);
    if s.holes_db > 0.0 {
        let holes = noise_fill(&z, &fr, &cut, &strength, s);
        for (n, h) in new.iter_mut().zip(holes) {
            for (a, b) in n.data.iter_mut().zip(h.data) {
                *a += b;
            }
        }
    }

    if x.len() == 2 && s.fill_width_db != 0.0 {
        let gain = vec![10f64.powf(s.fill_width_db / 20.0); cut.len()];
        new = side_gain(&new, &gain, frames_of(s.slope_s, sr));
    }
    done(0.68);
    if x.len() == 2 && s.side_min < 1.0 {
        let (side, side_cut, side_strength) = side_gap(&z, &fr, sr, &cut, s);
        for (i, v) in side.data.iter().enumerate() {
            new[0].data[i] += v;
            new[1].data[i] -= v;
        }
        if s.trace {
            for (row, (&sc, &ss)) in report.trace.iter_mut().zip(side_cut.iter().zip(&side_strength)) {
                row.side_cut_hz = sc;
                row.side_strength = ss;
            }
        }
    }

    done(0.9);
    let added = new.iter().map(|n| stft.inverse(n, samples)).collect();
    done(1.0);
    (added, report)
}

/// The input with the highs filled. Long input goes in blocks with context, crossfaded where
/// blocks meet, so memory stays the same for any length.
pub fn airfill(x: &[Vec<f32>], sr: f64, s: &Settings) -> (Vec<Vec<f32>>, Report) {
    airfill_with_progress(x, sr, s, &mut |_| {})
}

/// Core and context of an internal block, rounded to whole STFT hops: blocks then start on the
/// same frame grid, so the fill keeps its phase across a seam and the crossfade cannot cancel it.
fn block_layout(sr: f64, chunk_s: f64, context_s: f64) -> (usize, usize) {
    let hops = |seconds: f64| (seconds * sr / HOP as f64).round().max(0.0) as usize;
    (hops(chunk_s).max(1) * HOP, hops(context_s) * HOP)
}

/// The same, calling `progress` with the share done as the work goes.
pub fn airfill_with_progress(
    x: &[Vec<f32>],
    sr: f64,
    s: &Settings,
    progress: &mut dyn FnMut(f64),
) -> (Vec<Vec<f32>>, Report) {
    let n = x[0].len();
    let (core, ctx) = block_layout(sr, s.chunk_s, s.context_s);
    let fade = s.fade_s * sr;
    progress(0.0);
    if n <= core + 2 * ctx {
        let (added, report) = fill_block(x, sr, s, 0, progress);
        let out = x.iter().zip(added).map(|(a, b)| a.iter().zip(b).map(|(u, v)| u + v).collect()).collect();
        return (out, report);
    }

    let mut added = vec![vec![0.0f32; n]; x.len()];
    let mut weight = vec![0.0f32; n];
    let (mut filled, mut cuts) = (0.0, Vec::new());
    let mut trace = Vec::new();
    let mut block_index = 0;
    let mut start = 0;
    while start < n {
        let end = (start + core).min(n);
        let (a, b) = (start.saturating_sub(ctx), (end + ctx).min(n));
        let block: Vec<Vec<f32>> = x.iter().map(|ch| ch[a..b].to_vec()).collect();
        let share = |f: f64| (start as f64 + f * (end - start) as f64) / n as f64;
        let (part, mut report) = fill_block(&block, sr, s, a, &mut |f| progress(share(f)));
        for row in &mut report.trace {
            row.block = block_index;
            row.in_core = row.time >= start as f64 / sr && row.time < end as f64 / sr;
        }
        trace.extend(report.trace);
        block_index += 1;

        // 1 over the core, linear ramps of fade samples centred on its inner edges
        for i in a..b {
            let mut w = 1.0f64;
            if start > 0 {
                w = w.min(((i as f64 - start as f64 + fade / 2.0) / fade).clamp(0.0, 1.0));
            }
            if end < n {
                w = w.min(((end as f64 + fade / 2.0 - i as f64) / fade).clamp(0.0, 1.0));
            }
            for (ch, p) in added.iter_mut().zip(&part) {
                ch[i] += p[i - a] * w as f32;
            }
            weight[i] += w as f32;
        }

        filled += report.filled_share * (end - start) as f64;
        if let Some(c) = report.cut_median_hz {
            cuts.push(c);
        }
        start = end;
    }

    let out = x
        .iter()
        .zip(added)
        .map(|(ch, add)| ch.iter().zip(add).zip(&weight).map(|((u, v), w)| u + v / w.max(1e-6)).collect())
        .collect();
    let report = Report {
        filled_share: filled / n as f64,
        cut_median_hz: if cuts.is_empty() { None } else { Some(median(&cuts)) },
        trace,
    };
    (out, report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounds_half_to_even() {
        assert_eq!(round_even(2.5), 2);
        assert_eq!(round_even(3.5), 4);
        assert_eq!(round_even(937.5), 938);
        assert_eq!(round_even(86.13), 86);
    }

    #[test]
    fn blocks_start_on_the_stft_hop_grid() {
        for sr in [22050.0, 44100.0, 48000.0, 96000.0] {
            let (core, ctx) = block_layout(sr, 60.0, 15.0);
            assert_eq!(core % HOP, 0, "core at {sr}");
            assert_eq!(ctx % HOP, 0, "context at {sr}");
        }
    }

    #[test]
    fn trace_has_one_row_per_frame() {
        let sr = 48000.0;
        let n = 24000;
        let x = vec![vec![0.01f32; n], vec![-0.01f32; n]];
        let s = Settings { trace: true, ..Settings::default() };
        let (_, report) = airfill(&x, sr, &s);
        assert_eq!(report.trace.len(), Stft::frames_for(n));
        assert!(report.trace.windows(2).all(|w| w[0].time < w[1].time));
    }
}
