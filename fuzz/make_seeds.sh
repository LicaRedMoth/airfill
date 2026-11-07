#!/bin/bash
# Starting inputs for the fuzzer: a rate byte (index into RATES of the target), then stereo f32le.
# Noise cut like an MP3 at 11 and 16 kHz, silence, a sine over 0 dBFS and noise at 8 kHz.
set -e
cd "$(dirname "$0")"
mkdir -p corpus/engine
out=corpus/engine

for cut in 11000 16000; do
    lowpass="lowpass=f=$cut:p=2"
    { printf '\x03'; ffmpeg -v error -f lavfi -i "anoisesrc=c=pink:r=44100:d=1.5:seed=1,$lowpass,$lowpass,$lowpass" \
        -ac 2 -f f32le -; } > "$out/noise_$cut"
done

{ printf '\x04'; head -c 400000 /dev/zero; } > "$out/silence_48k"
{ printf '\x03'; ffmpeg -v error -f lavfi -i "sine=f=440:r=44100:d=1,volume=8" -ac 2 -f f32le -; } > "$out/loud_sine"
{ printf '\x00'; ffmpeg -v error -f lavfi -i "anoisesrc=r=8000:d=1:seed=2" -ac 2 -f f32le -; } > "$out/noise_8k"
