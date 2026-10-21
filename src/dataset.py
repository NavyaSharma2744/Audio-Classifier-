"""Build the training set: every usable block of audio, described and labelled.

    python src/dataset.py            build the cache
    python src/dataset.py csv        also export it as CSV for inspection
    python src/dataset.py describe   what ended up in it

One row is 0.213 s of audio: 64 mel means, 64 mel standard deviations, one
true/false per class, and the day it came from. Erased blocks are dropped --
there is nothing to hear in them -- and only every 5th survivor is kept,
because blocks a second apart are near duplicates of each other.

Labels come from PANNs, not from people. Speech is not among them: half its
frames ARE the erasure, so a Speech target teaches the model that silence
means speech.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

import features
from table import table

ROOT = Path("data/Raw_data")
RECORDER = "14"
CACHE = Path("data/training_set.npz")
CSV_PATH = Path("data/training_set.csv")

CLASSES = ["Television", "Music", "Silence", "Dog", "Cat", "Door", "Vehicle"]
PRESENT_ABOVE = 0.2    # the threshold the dataset's own anonymisation used
BLOCK_STRIDE = 5       # keep every 5th surviving block


def read_labels(json_path, n_blocks):
    """Is each class present, per block, according to PANNs."""
    frames = [item["predictions"] for item in json.loads(json_path.read_text())
              if "predictions" in item][:n_blocks]
    return np.array([[any(p["class"] == name and p["prob"] >= PRESENT_ABOVE
                          for p in frame) for name in CLASSES] for frame in frames])


def build():
    bands = features.mel_filterbank()
    all_features, all_labels, all_days = [], [], []

    for wav_path in sorted((ROOT / "Audio" / RECORDER).glob("*.WAV")):
        stem = wav_path.name.replace("_modified.WAV", "")
        blocks = features.read_blocks(wav_path)
        labels = read_labels(ROOT / "Annotations" / RECORDER / f"{stem}_light.json",
                             len(blocks))
        blocks = blocks[: len(labels)]

        keep = np.flatnonzero(~features.is_erased(blocks))[::BLOCK_STRIDE]
        all_features.append(features.spectrogram(blocks[keep], bands).astype(np.float16))
        all_labels.append(labels[keep])
        all_days.append(np.full(len(keep), stem[:8]))
        print(f"{stem}  {len(keep):,} usable blocks of {len(blocks):,}")

    np.savez_compressed(CACHE,
                        spectrograms=np.concatenate(all_features),
                        labels=np.concatenate(all_labels),
                        day=np.concatenate(all_days))
    print(f"\nwrote {CACHE}")


def load_spectrograms():
    """(n, 29 windows, 64 mel bands) -- what a convolutional model wants."""
    if not CACHE.exists():
        build()
    data = np.load(CACHE, allow_pickle=False)
    return data["spectrograms"], data["labels"], data["day"]


def load():
    """The same rows flattened to 128 summary features -- what the tree models want."""
    spectrograms, labels, days = load_spectrograms()
    return features.summarise(spectrograms), labels, days


def column_names():
    return ([f"mel_mean_{band:02d}" for band in range(features.MELS)]
            + [f"mel_std_{band:02d}" for band in range(features.MELS)]
            + [f"is_{name.lower()}" for name in CLASSES]
            + ["day"])


def to_csv():
    """One row per block. Large and slow to read -- the .npz is the working format."""
    X, y, days = load()
    with CSV_PATH.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(column_names())
        for row, labels, day in zip(X, y, days):
            writer.writerow([f"{value:.4f}" for value in row]
                            + [int(label) for label in labels] + [day])
    print(f"wrote {CSV_PATH}  {CSV_PATH.stat().st_size / 1e6:.0f} MB, "
          f"{len(X):,} rows x {len(column_names())} columns")


def describe():
    X, y, days = load()
    seconds = features.BLOCK / features.SAMPLE_RATE
    recorded = 16851 * 28
    hours = lambda blocks: f"{blocks * seconds / 3600:.1f} h"

    print(f"one row = {seconds * 1000:.0f} ms of audio -> {X.shape[1]} features "
          f"+ {len(CLASSES)} labels, {X.nbytes / 1e6:.0f} MB\n")

    table(["stage", "blocks", "audio", "share"],
          [[label, f"{count:,}", hours(count), f"{count / recorded:.0%}"]
           for label, count in [("recorded", recorded),
                                ("survived erasure", len(X) * BLOCK_STRIDE),
                                (f"kept (every {BLOCK_STRIDE}th)", len(X))]])

    print()
    positives = lambda rows: [f"{int(y[rows, c].sum()):,}" for c in range(len(CLASSES))]
    table(["day", "rows", *CLASSES],
          [[day, f"{(days == day).sum():,}", *positives(days == day)]
           for day in np.unique(days)]
          + [["TOTAL", f"{len(X):,}", *positives(slice(None))]])

    print("\nshare of each class falling on its single busiest day:")
    busiest = lambda c: max(y[days == day, c].sum() for day in np.unique(days))
    table(["class", "on one day", ""],
          [[name, "none at all" if not y[:, c].sum() else f"{busiest(c) / y[:, c].sum():.0%}",
            "<-- one day carries it" if y[:, c].sum() and busiest(c) / y[:, c].sum() > 0.6 else ""]
           for c, name in enumerate(CLASSES)])


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "build"
    {"build": build, "csv": to_csv, "describe": describe}[command]()
