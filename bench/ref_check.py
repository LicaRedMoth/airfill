#!/usr/bin/env python3
"""Do the references themselves have a cut? Runs the cut detector of airfill on each ref.

  python3 ref_check.py
"""
import os
import csv
import numpy as np
import soundfile as sf
from scipy.signal import stft
from systems import run_airfill

HERE = os.path.dirname(os.path.abspath(__file__))
NFFT, HOP = 2048, 512


def long_term_db(x, sr):
    """Mean power per STFT bin over the whole excerpt, in dB, and the bin frequencies."""
    fr, _, z = stft(x.T, sr, nperseg=NFFT, noverlap=NFFT - HOP)
    return fr, 10 * np.log10((np.abs(z) ** 2).mean(axis=(0, 2)) + 1e-20)


def has_shelf(x, sr, found=None):
    """The excerpt itself has a codec-like cut under 19 kHz: found in half the frames, long-term step 12+ dB.
    `found` is the (share, cut) airfill reported for x, when it is known already."""
    share, cut = found or run_airfill(x, sr)[1]
    if cut is None or share < 0.5:
        return False, 0.0
    fr, lt = long_term_db(x, sr)
    step = lt[(fr >= cut - 1500) & (fr < cut - 300)].mean() - lt[(fr >= cut + 300) & (fr < cut + 1500)].mean()
    return bool(step >= 12 and cut < 19000), float(step)


def too_quiet(x, sr, floor_db=-100.0):
    """Nothing to judge above 11 kHz: mean power per STFT bin there is at the noise floor."""
    fr, _, z = stft(x.T, sr, nperseg=NFFT, noverlap=NFFT - HOP)
    k = (fr >= 11000) & (fr < 16000)
    level = 10 * np.log10((np.abs(z[:, k]) ** 2).mean() + 1e-30)
    return bool(level < floor_db), float(level)


def main():
    for r in csv.DictReader(open(os.path.join(HERE, 'corpus', 'list.csv'))):
        x, sr = sf.read(os.path.join(HERE, 'corpus', 'ref', r['item'] + '.flac'), dtype='float64')
        _, (share, cut) = run_airfill(x, sr)
        shelf, step = has_shelf(x, sr, (share, cut))
        quiet, level = too_quiet(x, sr)

        # long-term level per 2 kHz band, relative to 2-4 kHz
        fr, lt = long_term_db(x, sr)
        base = lt[(fr >= 2000) & (fr < 4000)].mean()
        levels = []
        for lo in range(10000, 22000, 2000):
            k = (fr >= lo) & (fr < lo + 2000)
            levels.append(lt[k].mean() - base if k.any() else np.nan)

        cut_text = f'{cut / 1000:5.2f}k' if cut else '    -'
        print(f"{r['item']} {r['split']:4s} shelf {shelf:d} step {step:5.1f} hf {level:6.1f} "
              f"cut-frames {share:4.0%} cut {cut_text} | " + ' '.join(f'{v:+6.1f}' for v in levels))


if __name__ == '__main__':
    main()
