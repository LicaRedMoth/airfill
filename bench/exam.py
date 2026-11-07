#!/usr/bin/env python3
"""One full track (any file ffmpeg reads) -> MP3 -> plain decode and airfill, both judged against it
per 60 s block. Writes the MP3, both FLACs and spectrograms to OUTDIR.

  python3 exam.py ORIGINAL OUTDIR [--kbps 64] [--top 20000] [--tag NAME] [airfill flags, e.g. --holes 0]
"""
import os
import argparse
import subprocess
import numpy as np
import soundfile as sf
from degrade import encode_mp3, decode, align
from metrics import find_cutoff, compare
from systems import run_airfill

COLUMNS = ['lsd_hf', 'env_err', 'env_corr', 'bias', 'seam', 'stereo', 'stereo_lf', 'width_jit', 'width_med']
BLOCK_S = 60


def read_file(path, sr):
    with open(path, 'rb') as f:
        return decode(f.read(), sr).astype(np.float32)


def write_flac(y, sr, tags_from, out):
    """16-bit FLAC (TPDF dither, compression level 0) with the tags of tags_from."""
    rng = np.random.default_rng(0)
    tpdf = rng.random(y.shape, dtype=np.float32) - rng.random(y.shape, dtype=np.float32)
    pcm = np.clip(np.round(y * 32767 + tpdf), -32768, 32767).astype('<i2')
    cmd = ['ffmpeg', '-v', 'error', '-nostdin', '-y', '-i', tags_from,
           '-f', 's16le', '-ar', str(sr), '-ac', '2', '-i', 'pipe:0',
           '-map', '1:a', '-map_metadata', '0', '-c:a', 'flac', '-compression_level', '0', out]
    subprocess.run(cmd, input=pcm.tobytes(), check=True)


def spectrogram(path, png):
    cmd = ['ffmpeg', '-v', 'error', '-nostdin', '-y', '-i', path, '-lavfi',
           'aresample=resampler=soxr,showspectrumpic=s=1600x700:legend=1:scale=log:mode=combined', png]
    subprocess.run(cmd, check=True)


def blocks(ref, versions, sr, top):
    """Metrics of every version against ref, block by block; prints a table, returns medians."""
    n = BLOCK_S * sr
    rows = {label: [] for label, _ in versions}
    print(f"{'block':>9s} {'cut':>6s} {'version':10s} " + ' '.join(f'{c:>9s}' for c in COLUMNS))
    for a in range(0, len(ref) - sr * 10, n):
        r = ref[a:a + n]
        cut = find_cutoff(r, versions[0][1][a:a + n], sr)
        for label, y in versions:
            m = compare(r, y[a:a + n], sr, cut, top=top)
            rows[label].append(m)
            print(f'{a // sr:4d}-{min(len(ref), a + n) // sr:<4d} {cut / 1000:5.1f}k {label:10s} '
                  + ' '.join(f'{m[c]:+9.2f}' for c in COLUMNS))
    return {label: {c: np.nanmedian([m[c] for m in ms]) for c in COLUMNS} for label, ms in rows.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('original')
    ap.add_argument('outdir')
    ap.add_argument('--kbps', type=int, default=64)
    ap.add_argument('--top', type=float, default=20000, help='judge only up to this frequency')
    ap.add_argument('--tag', default='', help='added to the name of the airfill output')
    ap.add_argument('--metrics-only', action='store_true', help='only judge the files of an earlier run')
    a, airfill_args = ap.parse_known_args()
    os.makedirs(a.outdir, exist_ok=True)
    name = os.path.splitext(os.path.basename(a.original))[0]
    base = os.path.join(a.outdir, f'{name}_{a.kbps}')

    sr = sf.info(a.original).samplerate
    ref = read_file(a.original, sr)
    plain_path = base + '_plain.flac'
    fill_path = base + f'_airfill{a.tag}.flac'

    if a.metrics_only:
        # judge files an earlier run wrote; they already line up with the original
        deg = sf.read(plain_path, dtype='float32')[0][:len(ref)]
        filled = sf.read(fill_path, dtype='float32')[0][:len(ref)]
    else:
        data = encode_mp3(ref, sr, a.kbps)
        with open(base + '.mp3', 'wb') as f:
            f.write(data)
        deg, shift = align(ref, decode(data, sr).astype(np.float32))
        print(f'original {sr} Hz, {len(ref) / sr:.0f} s; mp3 {len(data)} bytes, codec delay {shift} samples')

        filled, (share, cut) = run_airfill(deg, sr, airfill_args)
        filled = filled.astype(np.float32)
        cut_text = f'{cut:.0f} Hz' if cut else 'none'
        peak, peak_ref = 20 * np.log10(np.abs(filled).max()), 20 * np.log10(np.abs(ref).max())
        print(f'airfill: filled {share:.0%} of frames, median cut {cut_text}, '
              f'peak {peak:+.2f} dBFS (original {peak_ref:+.2f})\n')

        # the delay is already removed, so every file lines up with the original sample for sample
        write_flac(deg, sr, a.original, plain_path)
        write_flac(filled, sr, a.original, fill_path)

    medians = blocks(ref, [('plain', deg), ('airfill', filled)], sr, a.top)
    print(f"\n{'median':>21s} " + ' '.join(f'{c:>9s}' for c in COLUMNS))
    for label, m in medians.items():
        print(f'{label:>21s} ' + ' '.join(f'{m[c]:+9.2f}' for c in COLUMNS))

    if a.metrics_only:
        return
    spectrogram(a.original, base + '_original.png')
    spectrogram(plain_path, os.path.splitext(plain_path)[0] + '.png')
    spectrogram(fill_path, os.path.splitext(fill_path)[0] + '.png')


if __name__ == '__main__':
    main()
