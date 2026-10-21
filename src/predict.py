"""Point it at a WAV, get the probability of each class over time.

    python src/predict.py <file.WAV> [seconds_per_row]

Blocks the anonymisation erased are excluded from the averages and reported
separately: silence is not evidence of absence.
"""
import csv
import sys
from pathlib import Path

import joblib
import numpy as np

import features
from table import table
from train import MODEL

REPORTS = Path("reports")
DEFAULT_SECONDS_PER_ROW = 60


def predict(wav_path):
    """Per-block class probabilities, and which blocks had no audio left."""
    tagger = joblib.load(MODEL)
    blocks = features.read_blocks(wav_path)
    described = features.describe(blocks)
    probabilities = np.column_stack(
        [model.predict_proba(described)[:, 1] for model in tagger["models"]])
    return tagger["classes"], probabilities, features.is_erased(blocks)


def timeline(probabilities, erased, blocks_per_row):
    """Average the blocks that still have audio into one row per time slice."""
    for start in range(0, len(probabilities), blocks_per_row):
        window = slice(start, start + blocks_per_row)
        heard = ~erased[window]
        yield (start * features.BLOCK / features.SAMPLE_RATE,
               1 - heard.mean(),
               probabilities[window][heard].mean(axis=0) if heard.any() else None)


def clock(second):
    return f"{int(second) // 60}:{int(second) % 60:02d}"


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)

    wav_path = Path(sys.argv[1])
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_SECONDS_PER_ROW
    blocks_per_row = max(1, round(seconds * features.SAMPLE_RATE / features.BLOCK))

    classes, probabilities, erased = predict(wav_path)
    rows = list(timeline(probabilities, erased, blocks_per_row))
    duration = len(probabilities) * features.BLOCK / features.SAMPLE_RATE

    print(f"{wav_path.name}   {duration / 60:.1f} min   {erased.mean():.0%} erased\n")
    table(["class", "whole file"],
          [[name, f"{probabilities[~erased][:, index].mean():.3f}"]
           for index, name in enumerate(classes)])

    print()
    table(["time", "erased", *classes],
          [[clock(second), f"{gone:.0%}",
            *(["no audio left"] + [""] * (len(classes) - 1) if means is None
              else [f"{value:.3f}" for value in means])]
           for second, gone, means in rows])

    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / f"{wav_path.stem}_predictions.csv"
    with out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["second", "erased_fraction", *classes])
        for second, gone, means in rows:
            writer.writerow([round(second, 1), round(gone, 3)]
                            + ([] if means is None else [round(float(v), 4) for v in means]))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
