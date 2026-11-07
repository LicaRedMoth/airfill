//! STFT and inverse with the conventions of scipy.signal.stft / istft: periodic Hann window,
//! half a window of zeros on each side, the end padded to whole hops, "spectrum" scaling.

use crate::grid::Grid;
use realfft::num_complex::Complex32;
use realfft::RealFftPlanner;

pub const NFFT: usize = 2048;
pub const HOP: usize = 512;
pub const BINS: usize = NFFT / 2 + 1;

pub struct Stft {
    window: Vec<f32>,
    window_sum: f32,
    planner: RealFftPlanner<f32>,
}

impl Default for Stft {
    fn default() -> Self {
        Self::new()
    }
}

impl Stft {
    pub fn new() -> Self {
        let window: Vec<f32> = (0..NFFT)
            .map(|n| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / NFFT as f64).cos()) as f32)
            .collect();
        let window_sum = window.iter().sum();
        Stft { window, window_sum, planner: RealFftPlanner::new() }
    }

    pub fn frames_for(samples: usize) -> usize {
        let padded = samples + NFFT;
        let extra = (HOP - (padded - NFFT) % HOP) % HOP;
        (padded + extra - NFFT) / HOP + 1
    }

    pub fn forward(&mut self, x: &[f32]) -> Grid<Complex32> {
        let frames = Self::frames_for(x.len());
        let half = NFFT / 2;
        let fft = self.planner.plan_fft_forward(NFFT);
        let mut input = fft.make_input_vec();
        let mut output = fft.make_output_vec();
        let mut out = Grid::new(BINS, frames, Complex32::new(0.0, 0.0));

        for t in 0..frames {
            for (n, slot) in input.iter_mut().enumerate() {
                // position in the signal; the first half window is the zero padding
                let i = (t * HOP + n) as isize - half as isize;
                let sample = if i >= 0 && (i as usize) < x.len() { x[i as usize] } else { 0.0 };
                *slot = sample * self.window[n];
            }
            fft.process(&mut input, &mut output).expect("fft sizes are fixed");
            for (o, v) in out.frame_mut(t).iter_mut().zip(output.iter()) {
                *o = *v / self.window_sum;
            }
        }
        out
    }

    /// The signal of `samples` samples whose STFT is z (overlap-add, window-squared normalised).
    pub fn inverse(&mut self, z: &Grid<Complex32>, samples: usize) -> Vec<f32> {
        let half = NFFT / 2;
        let length = (z.frames - 1) * HOP + NFFT;
        let ifft = self.planner.plan_fft_inverse(NFFT);
        let mut spectrum = ifft.make_input_vec();
        let mut frame = ifft.make_output_vec();
        let mut sum = vec![0.0f32; length];
        let mut norm = vec![0.0f32; length];

        for t in 0..z.frames {
            spectrum.copy_from_slice(z.frame(t));
            // a real signal has no imaginary part at DC and Nyquist; scipy ignores it there too
            spectrum[0].im = 0.0;
            spectrum[BINS - 1].im = 0.0;
            ifft.process(&mut spectrum, &mut frame).expect("fft sizes are fixed");
            for n in 0..NFFT {
                let value = frame[n] / NFFT as f32 * self.window_sum;
                sum[t * HOP + n] += value * self.window[n];
                norm[t * HOP + n] += self.window[n] * self.window[n];
            }
        }

        let mut out = Vec::with_capacity(samples);
        for i in half..(half + samples).min(length) {
            out.push(if norm[i] > 1e-10 { sum[i] / norm[i] } else { 0.0 });
        }
        out.resize(samples, 0.0);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_gives_the_signal_back() {
        let x: Vec<f32> = (0..10000).map(|i| ((i as f32) * 0.013).sin() * 0.5).collect();
        let mut stft = Stft::new();
        let z = stft.forward(&x);
        let y = stft.inverse(&z, x.len());
        let worst = x.iter().zip(&y).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(worst < 1e-5, "worst error {worst}");
    }

    #[test]
    fn frame_count_matches_scipy() {
        // len(scipy.signal.stft(np.zeros(n), nperseg=2048, noverlap=1536)[1])
        assert_eq!(Stft::frames_for(10000), 21);
        assert_eq!(Stft::frames_for(44100), 88);
    }

    #[test]
    fn values_match_scipy() {
        // scipy.signal.stft(sin(0.013 n) / 2, nperseg=2048, noverlap=1536)[2][0, 3]
        let x: Vec<f32> = (0..10000).map(|i| ((i as f32) * 0.013).sin() * 0.5).collect();
        let z = Stft::new().forward(&x);
        assert!((z.at(0, 3).re - -0.001_351_756).abs() < 1e-6);
    }
}
