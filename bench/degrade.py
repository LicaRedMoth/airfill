"""Make the lossy copy of a reference: LAME CBR through pipes, decoded back to float,
then shifted so it lines up with the reference sample for sample."""
import subprocess
import numpy as np


def encode_mp3(x, sr, kbps):
    """MP3 bytes of x (float, stereo) at kbps CBR, joint stereo, LAME defaults otherwise."""
    pcm = np.clip(np.round(x * 32767), -32768, 32767).astype('<i2').tobytes()
    cmd = ['lame', '--quiet', '-r', '-s', f'{sr / 1000:g}', '--bitwidth', '16', '--signed',
           '--little-endian', '-m', 'j', '-b', str(kbps), '--cbr', '-', '-']
    return subprocess.run(cmd, input=pcm, capture_output=True, check=True).stdout


def decode(data, sr):
    """Float samples of an encoded stream, at sr, stereo."""
    cmd = ['ffmpeg', '-v', 'error', '-i', 'pipe:0', '-ac', '2',
           '-af', f'aresample={sr}:resampler=soxr:precision=28', '-f', 'f32le', '-']
    raw = subprocess.run(cmd, input=data, capture_output=True, check=True).stdout
    return np.frombuffer(raw, '<f4').reshape(-1, 2).astype('float64')


def align(ref, y, max_shift=4000):
    """Shift y so it matches ref (codec delay), cut or pad to the length of ref."""
    n = min(len(ref), len(y), 48000 * 10)
    a, b = ref[:n].mean(1), y[:n].mean(1)
    size = 1 << int(np.ceil(np.log2(2 * n)))
    c = np.fft.irfft(np.fft.rfft(b, size) * np.conj(np.fft.rfft(a, size)), size)
    lags = np.r_[np.arange(0, max_shift + 1), np.arange(-max_shift, 0)]
    best = lags[np.argmax(c[lags])]

    if best >= 0:
        y = y[best:]
    else:
        y = np.vstack([np.zeros((-best, 2)), y])
    if len(y) < len(ref):
        y = np.vstack([y, np.zeros((len(ref) - len(y), 2))])
    return y[:len(ref)], int(best)


def lossy_copy(ref, sr, kbps):
    data = encode_mp3(ref, sr, kbps)
    y, shift = align(ref, decode(data, sr))
    return y, data, shift
