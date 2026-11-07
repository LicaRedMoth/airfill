#!/usr/bin/env python3
"""Run contestants over the corpus and write results.csv plus a summary.

  python3 run_bench.py                      every system, dev split, every bitrate
  python3 run_bench.py --split test         the held-out half
  python3 run_bench.py --systems none,airfill --kbps 128
  AIRFILL_ARGS="--holes 0" python3 run_bench.py --systems airfill --out results_noholes.csv
"""
import os
import csv
import sys
import time
import argparse
import warnings
import numpy as np
import soundfile as sf
from degrade import decode
from metrics import find_cutoff, compare
from systems import SYSTEMS
from ref_check import too_quiet

HERE = os.path.dirname(os.path.abspath(__file__))
COLUMNS = ['lsd_hf', 'lsd_edge', 'lsd_lf', 'env_err', 'env_corr', 'bias', 'seam',
           'stereo', 'stereo_lf', 'width_lf', 'width_jit', 'width_med']


def lossy(item, kbps, sr, shift, n):
    with open(os.path.join(HERE, 'corpus', 'mp3', f'{item}_{kbps}.mp3'), 'rb') as f:
        y = decode(f.read(), sr)[shift:]
    if len(y) < n:
        y = np.vstack([y, np.zeros((n - len(y), 2))])
    return y[:n]


def main():
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='dev')
    ap.add_argument('--systems', default=','.join(SYSTEMS))
    ap.add_argument('--kbps', default='128,96,64')
    ap.add_argument('--out', default='results.csv')
    a = ap.parse_args()
    names = a.systems.split(',')
    rates = [int(k) for k in a.kbps.split(',')]

    listed = csv.DictReader(open(os.path.join(HERE, 'corpus', 'list.csv')))
    items = [r for r in listed if r['split'] == a.split]
    rows = []
    for r in items:
        ref, sr = sf.read(os.path.join(HERE, 'corpus', 'ref', r['item'] + '.flac'), dtype='float64')
        quiet, hf_level = too_quiet(ref, sr)
        for kbps in rates:
            deg = lossy(r['item'], kbps, sr, int(r[f'shift_{kbps}']), len(ref))
            cut = find_cutoff(ref, deg, sr)
            for name in names:
                t0 = time.time()
                y = SYSTEMS[name](deg, sr)
                m = compare(ref, y, sr, cut)
                m.update(item=r['item'], kbps=kbps, system=name, cutoff=cut, seconds=time.time() - t0,
                         hf_level=hf_level, quiet=int(quiet))
                rows.append(m)
                print(f"{r['item']} {kbps:3d} {name:8s} cut {cut / 1000:5.2f}k  " +
                      '  '.join(f'{c} {m[c]:+6.2f}' for c in COLUMNS), flush=True)

    path = os.path.join(HERE, a.out)
    with open(path, 'w', newline='') as f:
        fields = ['item', 'kbps', 'system', 'cutoff', 'seconds', 'hf_level', 'quiet'] + COLUMNS
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    summary([r for r in rows if not r['quiet']], names, rates, 'items with real highs')
    summary([r for r in rows if r['quiet']], names, rates, 'quiet items (highs at the noise floor)')


def summary(rows, names, rates, title):
    items = len({r['item'] for r in rows})
    print(f'\n{title}, {items} items: medians '
          '(lsd/env/stereo: lower is better; env_corr: higher; bias/seam: near 0)')
    print(f"{'kbps':>4} {'system':8s} " + ' '.join(f'{c:>9s}' for c in COLUMNS))
    for kbps in rates:
        for name in names:
            sel = [r for r in rows if r['kbps'] == kbps and r['system'] == name]
            if not sel:
                continue
            vals = [np.nanmedian([r[c] for r in sel]) for c in COLUMNS]
            print(f'{kbps:4d} {name:8s} ' + ' '.join(f'{v:+9.2f}' for v in vals))


if __name__ == '__main__':
    main()
