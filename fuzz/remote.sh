#!/bin/bash
# Runs the tests, Miri and the fuzzer on a fresh Linux machine (Kaggle, Colab, a VM).
# Only the code and the synthetic seeds go there; no music.
set -euo pipefail

hours="${1:-6}"

if ! command -v ffmpeg >/dev/null 2>&1; then
    apt-get update -qq
    apt-get install -y -qq ffmpeg
fi
if ! command -v cargo >/dev/null 2>&1; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain nightly
fi
# shellcheck disable=SC1091
source "$HOME/.cargo/env"
rustup toolchain install nightly --profile minimal --component miri
rustup default nightly
command -v cargo-fuzz >/dev/null 2>&1 || cargo install cargo-fuzz --locked

cd "$(cd "$(dirname "$0")/.." && pwd)"
cargo test --release
cargo +nightly miri test
bash fuzz/make_seeds.sh
cargo +nightly fuzz run engine fuzz/corpus/engine -- \
    -max_total_time=$((hours * 3600)) -max_len=600000 -rss_limit_mb=3000
