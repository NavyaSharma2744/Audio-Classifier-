"""Turn 376 MB of per-frame JSON into a small cached array.

Output: one row per 15-second bin, holding the mean PANNs probability of each
class in the vocabulary, plus the recording it came from and its wall-clock time.
"""
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path("data/VITALISE data light 2.3")
CACHE = Path("data/binned.npz")

FRAME_SECONDS = 32 * 320 / 48000  # 0.2133 s between PANNs predictions
BIN_SECONDS = 15
FRAMES_PER_BIN = round(BIN_SECONDS / FRAME_SECONDS)
VOCAB_SIZE = 40


def recordings():
    for path in sorted((ROOT / "Annotations/14").glob("*_light.json")):
        stamp = datetime.strptime(path.name[:15], "%Y%m%d_%H%M%S")
        yield stamp, path


def frame_predictions(path):
    """Each frame as a {class: prob} dict."""
    return [
        {p["class"]: p["prob"] for p in item["predictions"]}
        for item in json.loads(path.read_text())
        if "predictions" in item
    ]


def build_vocabulary():
    counts = Counter()
    for _, path in recordings():
        for frame in frame_predictions(path):
            counts.update(frame)
    return [name for name, _ in counts.most_common(VOCAB_SIZE)]


def bin_recording(frames, vocab):
    """Mean probability per class over each 15-second bin."""
    probs = np.array([[frame.get(name, 0.0) for name in vocab] for frame in frames])
    n_bins = len(probs) // FRAMES_PER_BIN
    return probs[: n_bins * FRAMES_PER_BIN].reshape(n_bins, FRAMES_PER_BIN, -1).mean(axis=1)


def build():
    vocab = build_vocabulary()
    bins, recording_ids, seconds_of_day = [], [], []

    for index, (stamp, path) in enumerate(recordings()):
        binned = bin_recording(frame_predictions(path), vocab)
        start = stamp.hour * 3600 + stamp.minute * 60
        bins.append(binned)
        recording_ids.append(np.full(len(binned), index))
        seconds_of_day.append(start + np.arange(len(binned)) * BIN_SECONDS)
        print(f"{path.name}  {len(binned)} bins")

    np.savez_compressed(
        CACHE,
        probs=np.concatenate(bins).astype(np.float32),
        recording=np.concatenate(recording_ids),
        second_of_day=np.concatenate(seconds_of_day),
        date=np.array([stamp.strftime("%Y-%m-%d") for stamp, _ in recordings()]),
        vocab=np.array(vocab),
    )
    print(f"\nwrote {CACHE}")


def load():
    if not CACHE.exists():
        build()
    return np.load(CACHE, allow_pickle=False)


if __name__ == "__main__":
    build()
