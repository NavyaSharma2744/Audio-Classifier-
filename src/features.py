"""Audio in, numbers out. Nothing else in the project opens a WAV.

A recording is cut into blocks of 0.213 s -- the same rate the dataset's
annotations use -- and each block becomes a mel spectrogram summarised by the
mean and standard deviation of every band: what it sounds like, and how much
it moved.

The mel settings come from the dataset's own metadata, so the model sees the
same representation PANNs saw when it produced the labels we train on.
"""
import wave

import librosa
import numpy as np

SAMPLE_RATE = 48000
BLOCK = 10240          # samples per block: 0.2133 s, matching the annotations
WINDOW = 1024          # metadata: window_size
HOP = 320              # metadata: hop_size -- 32 frames per block
MELS = 64              # metadata: mel_bins
FMIN, FMAX = 50, 14000  # metadata: fmin, fmax

# Zero-fraction per block is sharply bimodal: real audio sits near 0.01, erased
# audio near 0.99 (the anonymisation leaves a little dither, so it is not exactly
# 1.0). Anything in between is safe; 0.99 is NOT -- it splits the erased peak.
ERASED_THRESHOLD = 0.5


def mel_filterbank():
    """Maps FFT bins onto mel bands. Mel because pitch is perceived logarithmically."""
    return librosa.filters.mel(sr=SAMPLE_RATE, n_fft=WINDOW, n_mels=MELS,
                               fmin=FMIN, fmax=FMAX)


def read_blocks(wav_path):
    """A WAV as (n_blocks, BLOCK) of int16 samples."""
    with wave.open(str(wav_path), "rb") as wav:
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
    n_blocks = len(samples) // BLOCK
    return samples[: n_blocks * BLOCK].reshape(n_blocks, BLOCK)


def is_erased(blocks):
    """True where anonymisation replaced the audio with zeros."""
    return (blocks == 0).mean(axis=1) > ERASED_THRESHOLD


def spectrogram(blocks, bands=None):
    """(n_blocks, BLOCK) int16 -> (n_blocks, 29 windows, MELS) log-mel."""
    bands = mel_filterbank() if bands is None else bands
    signal = blocks.astype(np.float32) / 32768.0
    starts = np.arange(0, BLOCK - WINDOW + 1, HOP)
    windows = signal[:, starts[:, None] + np.arange(WINDOW)]
    power = np.abs(np.fft.rfft(windows * np.hanning(WINDOW), axis=-1)) ** 2
    return np.log(power @ bands.T + 1e-10)       # log again: loudness is multiplicative


def summarise(spectrograms):
    """Collapse the time axis: what it sounds like, and whether it holds still."""
    spectrograms = spectrograms.astype(np.float32)
    return np.concatenate([spectrograms.mean(axis=1), spectrograms.std(axis=1)], axis=1)


def describe(blocks, bands=None):
    """(n_blocks, BLOCK) int16 -> (n_blocks, 2 * MELS) features."""
    return summarise(spectrogram(blocks, bands))


def check():
    """A pure tone must peak at the mel band nearest its own frequency.

    Tolerance allows one FFT bin either side: at 48 kHz a 1024-point window
    resolves 47 Hz, which is wider than a mel band down in the bass.
    """
    bands = mel_filterbank()
    # Filter centres are the interior points of an (MELS + 2) mel grid.
    centres = librosa.mel_frequencies(n_mels=MELS + 2, fmin=FMIN, fmax=FMAX)[1:-1]
    resolution = SAMPLE_RATE / WINDOW
    time = np.arange(BLOCK) / SAMPLE_RATE

    for frequency in (200, 1000, 5000):
        tone = (np.sin(2 * np.pi * frequency * time) * 20000).astype(np.int16)
        peak = centres[describe(tone[None, :], bands)[0, :MELS].argmax()]
        assert abs(peak - frequency) < max(2 * resolution, 0.1 * frequency), (frequency, peak)

    silence = describe(np.zeros((1, BLOCK), dtype=np.int16), bands)
    assert silence[0, :MELS].max() < np.log(1e-9), "silence should hit the log floor"
    print("features.py: checks passed")


if __name__ == "__main__":
    check()
