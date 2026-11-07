#![no_main]
//! Any bytes become a stereo signal at one of the usual sample rates; the engine must not panic
//! and must keep the length.

use airfill::engine::{airfill, Settings};
use libfuzzer_sys::fuzz_target;

const RATES: [f64; 6] = [8000.0, 16000.0, 22050.0, 44100.0, 48000.0, 96000.0];

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }
    let rate = RATES[data[0] as usize % RATES.len()];

    let mut left = Vec::new();
    let mut right = Vec::new();
    for pair in data[1..].chunks_exact(8) {
        left.push(f32::from_le_bytes([pair[0], pair[1], pair[2], pair[3]]));
        right.push(f32::from_le_bytes([pair[4], pair[5], pair[6], pair[7]]));
    }

    let x = [left, right];
    let (y, _) = airfill(&x, rate, &Settings::default());
    assert_eq!(y.len(), 2);
    assert_eq!(y[0].len(), x[0].len());

    // what a decoder can give (finite, within +-16) must come out finite
    if x.iter().flatten().all(|v| v.is_finite() && v.abs() <= 16.0) {
        assert!(y.iter().flatten().all(|v| v.is_finite()), "non-finite output");
    }
});
