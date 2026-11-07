#!/usr/bin/env python3
"""Which FLACs are really full band: the 1 kHz band levels of 30 s from a few tracks per album,
with the hard shelf of an MP3 transcode flagged. One CSV line per track.

  python3 scan_sources.py MUSIC_DIR > scan.csv
"""
import os
import sys
import glob
import json
import random
import subprocess
import numpy as np
from scipy.signal import welch

ROOT = sys.argv[1] if len(sys.argv) > 1 else '.'
PER_ALBUM = 2
SECONDS = 30


def probe(path):
    out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_entries',
                          'stream=sample_rate,bits_per_raw_sample,channels:format=duration',
                          '-of', 'json', path], capture_output=True, text=True).stdout
    info = json.loads(out)
    s = info['streams'][0]
    return int(s['sample_rate']), int(s.get('bits_per_raw_sample') or 0), int(s['channels']), \
        float(info['format']['duration'])


def excerpt(path, start, seconds, sr):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-ss', str(start), '-t', str(seconds), '-i', path,
                          '-f', 'f32le', '-ac', '2', '-ar', str(sr), '-'],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, '<f4').reshape(-1, 2).astype('float64')


def band_levels(x, sr):
    fr, p = welch(x.mean(1), sr, nperseg=8192)
    ref = p[(fr >= 8000) & (fr < 9000)].mean() + 1e-30
    top = min(22, sr // 2000)
    return [10 * np.log10(p[(fr >= k * 1000) & (fr < (k + 1) * 1000)].mean() / ref + 1e-30)
            for k in range(8, top)]


def shelf(levels):
    # a drop of 10+ dB between neighbouring 1 kHz bands from 13 kHz up, flat after it
    for i in range(5, len(levels) - 2):
        step = levels[i] - levels[i - 1]
        floor = levels[i:]
        if step < -10 and max(floor) - min(floor) < 6:
            return 8 + i
    return None


def main():
    rng = random.Random(7)
    albums = sorted(d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d)))
    print('path,rate,bits,shelf_khz,above19_db,duration')
    for album in albums:
        tracks = sorted(glob.glob(os.path.join(ROOT, album, '**', '*.flac'), recursive=True))
        for path in rng.sample(tracks, min(PER_ALBUM, len(tracks))):
            try:
                sr, bits, ch, dur = probe(path)
                if dur < SECONDS * 2:
                    continue
                x = excerpt(path, dur / 2 - SECONDS / 2, SECONDS, sr)
                lv = band_levels(x, sr)
                cut = shelf(lv)
                above = lv[11] if len(lv) > 11 else float('nan')
                print(f'"{path}",{sr},{bits},{cut or ""},{above:.1f},{dur:.0f}', flush=True)
            except Exception as e:
                print(f'# skip {path}: {e}', file=sys.stderr, flush=True)


if __name__ == '__main__':
    main()
