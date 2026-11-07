"""How close a processed lossy copy is to the lossless reference, in magnitudes only (made-up highs
can not match the original phase). The measures are explained in the README."""
import numpy as np
from scipy.signal import stft
from scipy.ndimage import uniform_filter1d

NFFT, HOP = 2048, 512
FLOOR_DB = -100.0


def spectra(x, sr):
    """Power spectrogram per channel: (channels, bins, frames), and the bin frequencies."""
    fr, _, z = stft(x.T, sr, nperseg=NFFT, noverlap=NFFT - HOP)
    return fr, z


def to_db(p, ref_max):
    return 10 * np.log10(np.maximum(p, ref_max * 10 ** (FLOOR_DB / 10)))


def find_cutoff(ref, deg, sr):
    """Lowest frequency over 8 kHz from which the copy stays 10+ dB under the reference in 90% of the
    250 Hz bands above (LAME leaves islands); bands empty in the reference itself do not count."""
    fr, zr = spectra(ref, sr)
    _, zd = spectra(deg, sr)
    pr = (np.abs(zr) ** 2).mean(axis=(0, 2))
    pd = (np.abs(zd) ** 2).mean(axis=(0, 2))
    edges = np.arange(8000, min(sr / 2 - 500, 21000), 250)

    level, level_deg = [], []
    for f in edges:
        k = (fr >= f) & (fr < f + 250)
        level.append(10 * np.log10(pr[k].sum() + 1e-30))
        level_deg.append(10 * np.log10(pd[k].sum() + 1e-30))
    level = np.array(level)
    gap = level - np.array(level_deg)

    real = level > level.max() - 35
    for i in range(len(gap)):
        if real[i] and gap[i] > 10 and (gap[i:][real[i:]] > 10).mean() >= 0.9:
            return float(edges[i])
    return float(edges[-1])


def bands(lo, hi):
    """1/6-octave bands from lo to hi; the last one is cut at hi if it is at least 1/12 oct wide."""
    out = []
    f = lo
    while f < hi:
        g = min(f * 2 ** (1 / 6), hi)
        if np.log2(g / f) >= 1 / 12:
            out.append((f, g))
        f = g
    return out


def compare(ref, y, sr, cutoff, top=20000):
    fr, zr = spectra(ref, sr)
    _, zy = spectra(y, sr)
    pr, py = np.abs(zr) ** 2, np.abs(zy) ** 2
    pmax = pr.max()
    top = min(top, sr / 2 - 500)
    hf = (fr >= cutoff) & (fr < top)
    lf = (fr >= 1000) & (fr < 0.7 * cutoff)
    edge = (fr >= 0.7 * cutoff) & (fr < cutoff)

    # frames where the reference has something above the cutoff
    band_energy = pr[:, hf, :].sum(axis=(0, 1))
    live = band_energy > band_energy.max() * 1e-6

    def lsd(mask):
        d = to_db(pr[:, mask][:, :, live], pmax) - to_db(py[:, mask][:, :, live], pmax)
        return float(np.sqrt((d ** 2).mean(axis=1)).mean())

    out = dict(lsd_hf=lsd(hf), lsd_edge=lsd(edge), lsd_lf=lsd(lf))

    # 1/6-octave band levels over time
    errs, corrs = [], []
    for lo, hi in bands(cutoff, top):
        k = (fr >= lo) & (fr < hi)
        er = to_db(pr[:, k].sum(axis=(0, 1))[live], pmax)
        ey = to_db(py[:, k].sum(axis=(0, 1))[live], pmax)
        errs.append(np.sqrt(((er - ey) ** 2).mean()))
        if er.std() > 0.5 and ey.std() > 0.5:
            corrs.append(np.corrcoef(er, ey)[0, 1])
    out['env_err'] = float(np.mean(errs))
    out['env_corr'] = float(np.mean(corrs)) if corrs else float('nan')

    # long-term spectra: overall bias and the step at the cutoff
    lr = to_db(pr.mean(axis=(0, 2)), pmax)
    ly = to_db(py.mean(axis=(0, 2)), pmax)
    out['bias'] = float((ly[hf] - lr[hf]).mean())

    def step(levels):
        above = levels[(fr >= cutoff) & (fr < cutoff + 600)].mean()
        below = levels[(fr >= cutoff - 1200) & (fr < cutoff - 200)].mean()
        return above - below

    out['seam'] = float(step(ly) - step(lr))

    # stereo above the cutoff (the fill), and under it from 1.5 kHz (what the codec kept)
    out['stereo'], _ = stereo_error(zr, zy, fr, cutoff, top)
    out['stereo_lf'], out['width_lf'] = stereo_error(zr, zy, fr, 1500, cutoff)
    out['width_jit'], out['width_med'] = width_jitter(zr, zy, fr, 0.55 * cutoff, cutoff)
    return out


def corr_lr(z):
    a, b = z[0], z[1]
    den = np.sqrt((np.abs(a) ** 2).sum() * (np.abs(b) ** 2).sum())
    if den == 0:
        return None
    return float(np.real((a * np.conj(b)).sum()) / den)


def stereo_error(zr, zy, fr, lo, hi):
    """Mean error, over 1/6-octave bands from lo to hi, of the L/R correlation and of the
    side-to-mid level in dB, processed against reference."""
    corr_err, width_err = [], []
    for a, b in bands(lo, hi):
        k = (fr >= a) & (fr < b)
        cr = corr_lr(zr[:, k])
        cy = corr_lr(zy[:, k])
        if cr is None or cy is None:
            continue
        corr_err.append(abs(cr - cy))
        width_err.append(abs(side_to_mid(zr[:, k]) - side_to_mid(zy[:, k])))
    if not corr_err:
        return float('nan'), float('nan')
    return float(np.mean(corr_err)), float(np.mean(width_err))


def side_to_mid(z):
    side = (np.abs(z[0] - z[1]) ** 2).sum()
    mid = (np.abs(z[0] + z[1]) ** 2).sum()
    return 10 * np.log10((side + 1e-30) / (mid + 1e-30))


def width_jitter(zr, zy, fr, lo, hi, block=4):
    """RMS and median, over blocks of `block` frames and 1/3-octave bands from lo to hi, of the
    side-to-mid level (dB) of y minus that of the reference. Silent blocks do not count."""
    diffs = []
    f = lo
    while f * 2 ** (1 / 3) <= hi:
        k = (fr >= f) & (fr < f * 2 ** (1 / 3))
        tracks = []
        for z in (zr, zy):
            mid = uniform_filter1d((np.abs(z[0, k] + z[1, k]) ** 2).sum(axis=0), block)[::block]
            side = uniform_filter1d((np.abs(z[0, k] - z[1, k]) ** 2).sum(axis=0), block)[::block]
            tracks.append((10 * np.log10((side + 1e-20) / (mid + 1e-20)), 10 * np.log10(mid + 1e-20)))
        live = tracks[0][1] > tracks[0][1].max() - 30
        diffs.append((tracks[1][0] - tracks[0][0])[live])
        f *= 2 ** (1 / 3)
    if not diffs:
        return float('nan'), float('nan')
    d = np.concatenate(diffs)
    return float(np.sqrt((d ** 2).mean())), float(np.median(d))
