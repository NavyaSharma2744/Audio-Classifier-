"""Log-spectrograms for every block of audio that survived anonymisation.

The cache holds each sampled 0.213 s block as a small (15 sub-window, 40 band)
log-spectrogram. `summary_features` collapses that to the mean and standard
deviation per band, which is what the gradient boosting model consumes; keeping
the spectrogram lets a model that cares about onset shape use it instead.
"""
import json
import wave
from pathlib import Path

import numpy as np

ROOT = Path("data/VITALISE data light 2.3")
CACHE = Path("data/audio_features.npz")

BLOCK = 32 * 320       # samples per PANNs prediction, 0.213 s
WINDOW = 1024
HOP = 640
SAMPLE_RATE = 48000
BANDS = 40
FMIN, FMAX = 50, 14000
BLOCK_STRIDE = 5       # keep every 5th block
ERASED_THRESHOLD = 0.99  # a block is anonymised if this many samples are exactly zero

TARGETS = ["Television", "Music", "Silence", "Dog", "Cat", "Door", "Vehicle"]
LABEL_THRESHOLD = 0.2


def band_matrix():
    """Maps rfft bins onto BANDS log-spaced frequency bands."""
    freqs = np.fft.rfftfreq(WINDOW, 1 / SAMPLE_RATE)
    edges = np.geomspace(FMIN, FMAX, BANDS + 1)
    bins = np.zeros((BANDS, len(freqs)), dtype=np.float32)
    for band, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        inside = (freqs >= low) & (freqs < high)
        if inside.any():
            bins[band, inside] = 1 / inside.sum()
    return bins


def block_spectrograms(blocks, bands):
    """(n_blocks, BLOCK) samples -> (n_blocks, n_windows, BANDS) log-spectrograms."""
    starts = np.arange(0, BLOCK - WINDOW + 1, HOP)
    windows = blocks[:, starts[:, None] + np.arange(WINDOW)]  # (n, n_windows, WINDOW)
    power = np.abs(np.fft.rfft(windows * np.hanning(WINDOW), axis=-1)) ** 2
    return np.log(power @ bands.T + 1e-10)


def summary_features(spectrograms):
    """Collapse the time axis to a mean and standard deviation per band."""
    spectrograms = spectrograms.astype(np.float32)
    return np.concatenate([spectrograms.mean(axis=1), spectrograms.std(axis=1)], axis=1)


def label_probabilities(json_path, n_blocks):
    """PANNs probability per target class, 0.0 where the class missed the stored top 7."""
    data = json.loads(json_path.read_text())
    frames = [item["predictions"] for item in data if "predictions" in item][:n_blocks]
    return np.array([
        [next((p["prob"] for p in frame if p["class"] == name), 0.0) for name in TARGETS]
        for frame in frames
    ], dtype=np.float32)


def build():
    bands = band_matrix()
    features, targets, days = [], [], []

    for wav_path in sorted((ROOT / "Audio/14").glob("*.WAV")):
        stem = wav_path.name.replace("_modified.WAV", "")
        with wave.open(str(wav_path), "rb") as wav:
            raw = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")

        n_blocks = len(raw) // BLOCK
        blocks = raw[: n_blocks * BLOCK].reshape(n_blocks, BLOCK)
        label = label_probabilities(ROOT / "Annotations/14" / f"{stem}_light.json", n_blocks)

        n_blocks = len(label)
        blocks = blocks[:n_blocks]
        erased = (blocks == 0).mean(axis=1) > ERASED_THRESHOLD
        keep = np.flatnonzero(~erased)[::BLOCK_STRIDE]

        spectrogram = block_spectrograms(blocks[keep].astype(np.float32) / 32768.0, bands)
        features.append(spectrogram.astype(np.float16))
        targets.append(label[keep])
        days.append(np.full(len(keep), stem[:8]))
        print(f"{stem}  kept {len(keep):,} of {n_blocks:,} blocks")

    np.savez_compressed(
        CACHE,
        spectrograms=np.concatenate(features),
        label_probs=np.concatenate(targets).astype(np.float16),
        day=np.concatenate(days),
        target_names=np.array(TARGETS),
    )
    print(f"\nwrote {CACHE}")


def load():
    if not CACHE.exists():
        build()
    return np.load(CACHE, allow_pickle=False)


def check():
    """A pure tone must land in the band containing its frequency, and nowhere else."""
    bands = band_matrix()
    edges = np.geomspace(FMIN, FMAX, BANDS + 1)
    time = np.arange(BLOCK) / SAMPLE_RATE

    for frequency in (200, 1000, 5000):
        tone = np.sin(2 * np.pi * frequency * time)[None, :].astype(np.float32)
        loudest = summary_features(block_spectrograms(tone, bands))[0, :BANDS].argmax()
        assert edges[loudest] <= frequency < edges[loudest + 1], (frequency, loudest)

    silence = block_spectrograms(np.zeros((1, BLOCK), dtype=np.float32), bands)
    assert silence.max() < np.log(1e-9), "silence should hit the log floor"
    print("checks passed")


if __name__ == "__main__":
    import sys

    check() if "check" in sys.argv else build()
