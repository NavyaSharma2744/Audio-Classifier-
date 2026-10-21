"""Do the WAVs still contain the speech the JSON says is there?

The dataset paper says speech was replaced with noise at 1e-10 relative
amplitude. If true, every frame the annotations call Speech should be
digitally silent in the audio.
"""
import json
import wave
from pathlib import Path

import numpy as np

BLOCK = 32 * 320  # samples between predictions: frames_for_analysis * hop_size
SPEECH_THRESHOLD = 0.2  # the threshold the dataset authors removed speech at

ROOT = Path("data/VITALISE data light 2.3")
STEM = "20231116_180000"


def speech_probs(json_path):
    data = json.loads(json_path.read_text())
    return np.array([
        next((p["prob"] for p in item["predictions"] if p["class"] == "Speech"), 0.0)
        for item in data if "predictions" in item
    ])


def block_rms(wav_path, n_blocks):
    with wave.open(str(wav_path), "rb") as wav:
        raw = wav.readframes(n_blocks * BLOCK)
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    blocks = samples[: len(samples) // BLOCK * BLOCK].reshape(-1, BLOCK)
    return np.sqrt((blocks ** 2).mean(axis=1))


def main():
    probs = speech_probs(ROOT / "Annotations/14" / f"{STEM}_light.json")
    rms = block_rms(ROOT / "Audio/14" / f"{STEM}_modified.WAV", len(probs))
    probs = probs[: len(rms)]

    speech = probs >= SPEECH_THRESHOLD
    silent = rms == 0.0

    print(f"blocks:                 {len(rms):,}")
    print(f"Speech >= {SPEECH_THRESHOLD}:          {speech.mean():6.1%}")
    print(f"digitally silent:       {silent.mean():6.1%}")
    print(f"silent | Speech:        {silent[speech].mean():6.1%}")
    print(f"silent | no Speech:     {silent[~speech].mean():6.1%}")
    print(f"median RMS, Speech:     {np.median(rms[speech]):.2e}")
    print(f"median RMS, no Speech:  {np.median(rms[~speech]):.2e}")


if __name__ == "__main__":
    main()
