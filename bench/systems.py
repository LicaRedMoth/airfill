"""The contestants. Each takes the lossy copy (float, samples x 2) and its rate, returns audio."""
import os
import re
import shlex
import subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BINARY = os.environ.get('AIRFILL', os.path.join(os.path.dirname(HERE), 'target', 'release', 'airfill'))


def run_airfill(x, sr, args=()):
    """The airfill binary on x; returns the output and (share of frames filled, median cut in Hz or None)."""
    done = subprocess.run([BINARY, '--raw', '--rate', str(sr), *args], input=x.astype('<f4').tobytes(),
                          capture_output=True, check=True)
    y = np.frombuffer(done.stdout, '<f4').reshape(-1, 2).astype('float64')

    report = done.stderr.decode()
    found = re.search(r'filled (\d+)% of frames, median cut (\d+) Hz', report)
    if found is None:
        return y, (0.0, None)
    return y, (int(found.group(1)) / 100, float(found.group(2)))


def none(x, sr):
    return x


def exciter(x, sr):
    # ffmpeg's harmonic exciter with its defaults, the usual way to brighten a dull file
    pcm = x.astype('<f4').tobytes()
    cmd = ['ffmpeg', '-v', 'error', '-f', 'f32le', '-ar', str(sr), '-ac', '2', '-i', 'pipe:0',
           '-af', 'aexciter', '-f', 'f32le', '-']
    raw = subprocess.run(cmd, input=pcm, capture_output=True, check=True).stdout
    return np.frombuffer(raw, '<f4').reshape(-1, 2).astype('float64')[:len(x)]


def airfill(x, sr):
    # extra flags for a run, e.g. AIRFILL_ARGS="--holes 0"
    y, _ = run_airfill(x, sr, shlex.split(os.environ.get('AIRFILL_ARGS', '')))
    return y


SYSTEMS = dict(none=none, exciter=exciter, airfill=airfill)
