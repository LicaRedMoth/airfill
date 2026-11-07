#!/usr/bin/env python3
"""The corpus from scan.csv: 40 s of one full-band track per artist (none with a cut of its own), its MP3s
and their codec delays; half 'dev' for tuning, half 'test' only for the final numbers.

  python3 make_corpus.py MUSIC_DIR      (the folder scan.csv was made from; albums named Artist_Album)
"""
import os
import sys
import csv
import random
import soundfile as sf
from scan_sources import excerpt
from degrade import encode_mp3, decode, align
from ref_check import has_shelf

HERE = os.path.dirname(os.path.abspath(__file__))
MUSIC = sys.argv[1] if len(sys.argv) > 1 else '.'
ITEMS = 30
SECONDS = 40
BITRATES = [128, 96, 64]


def artist(album):
    parts = album.split('_')
    return '_'.join(parts[:2]) if parts[0] == 'The' else parts[0]


def candidates(rows):
    good = [r for r in rows if not r['shelf_khz'] and float(r['above19_db']) > -50
            and r['rate'] in ('44100', '48000')]
    rng = random.Random(11)
    rng.shuffle(good)
    return good


def main():
    rows = list(csv.DictReader(open(os.path.join(HERE, 'scan.csv'))))
    os.makedirs(os.path.join(HERE, 'corpus', 'ref'), exist_ok=True)
    os.makedirs(os.path.join(HERE, 'corpus', 'mp3'), exist_ok=True)

    out, seen = [], set()
    for r in candidates(rows):
        who = artist(os.path.relpath(r['path'], MUSIC).split('/')[0])
        if who in seen or len(out) >= ITEMS:
            continue
        sr, dur = int(r['rate']), float(r['duration'])
        x = excerpt(r['path'], dur / 2 - SECONDS / 2, SECONDS, sr)
        shelf, step = has_shelf(x, sr)
        if shelf:
            print('skip, own cut', f'{step:.0f} dB:', os.path.relpath(r['path'], MUSIC), flush=True)
            continue
        seen.add(who)

        name = f'{len(out):02d}'
        sf.write(os.path.join(HERE, 'corpus', 'ref', name + '.flac'), x, sr, subtype='PCM_24')
        row = dict(item=name, split='dev' if len(out) % 2 == 0 else 'test', rate=sr, path=r['path'])
        for kbps in BITRATES:
            data = encode_mp3(x, sr, kbps)
            with open(os.path.join(HERE, 'corpus', 'mp3', f'{name}_{kbps}.mp3'), 'wb') as f:
                f.write(data)
            _, shift = align(x, decode(data, sr))
            row[f'shift_{kbps}'] = shift
        out.append(row)
        print(name, row['split'], os.path.relpath(r['path'], MUSIC), flush=True)

    with open(os.path.join(HERE, 'corpus', 'list.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)


if __name__ == '__main__':
    main()
