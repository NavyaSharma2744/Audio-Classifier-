"""
Analyze Day 14 of the Sounds of Home dataset.

Writes the analysis results to a Markdown report rather than
printing the results to the terminal.
"""

import json
import wave
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np


BLOCK = 32 * 320  # Samples between predictions
THRESHOLD = 0.2
SAMPLED_BLOCKS = 400  # Blocks sampled per file

ROOT = Path("data/Raw_data")
REPORT_PATH = Path("Dataset_Analysis.md")

DATASET_NAME = "Sounds of Home"
DATASET_URL = "https://www.cvssp.org/data/ai4s/sounds_of_home/"


def erased_fraction(wav_path):
    """Estimate the fraction of sampled blocks that are almost entirely zero."""

    with wave.open(str(wav_path), "rb") as wav:
        n_blocks = wav.getnframes() // BLOCK

        if n_blocks == 0:
            return 0.0

        sample_count = min(SAMPLED_BLOCKS, n_blocks)
        zeros = []

        for start in np.linspace(
            0, n_blocks - 1, sample_count, dtype=int
        ):
            wav.setpos(int(start * BLOCK))
            block = np.frombuffer(
                wav.readframes(BLOCK), dtype="<i2"
            )

            if len(block) == 0:
                continue

            zeros.append((block == 0).mean() > 0.99)

    return float(np.mean(zeros)) if zeros else 0.0


def class_counts(json_path):
    """Count frames in which each predicted class exceeds the threshold."""

    data = json.loads(json_path.read_text(encoding="utf-8"))
    counts = Counter()
    frames = 0

    for item in data:
        if "predictions" not in item:
            continue

        frames += 1

        counts.update(
            prediction["class"]
            for prediction in item["predictions"]
            if prediction["prob"] >= THRESHOLD
        )

    return counts, frames


def main():
    totals = Counter()
    total_frames = 0
    file_results = []
    warnings = []

    audio_dir = ROOT / "Audio/14"
    annotation_dir = ROOT / "Annotations/14"

    wav_files = sorted(audio_dir.glob("*.WAV"))

    if not wav_files:
        raise FileNotFoundError(
            f"No WAV files found in {audio_dir.resolve()}"
        )

    for wav_path in wav_files:
        stem = wav_path.name.replace("_modified.WAV", "")
        json_path = annotation_dir / f"{stem}_light.json"

        if not json_path.exists():
            warnings.append(
                f"Missing annotation file for `{wav_path.name}`: "
                f"`{json_path}`"
            )
            continue

        counts, frames = class_counts(json_path)
        erased = erased_fraction(wav_path)

        totals.update(counts)
        total_frames += frames

        top_classes = ", ".join(
            f"{name} ({count:,})"
            for name, count in counts.most_common(4)
        )

        file_results.append({
            "filename": stem,
            "erased": erased,
            "frames": frames,
            "top_classes": top_classes or "No classes above threshold",
        })

    total_hours = total_frames * BLOCK / 48000 / 3600

    # Build the Markdown report.
    lines = [
        f"# Analysis Results: {DATASET_NAME}",
        "",
        "**Subset:** Day 14 only",
        "",
        f"**Dataset source:** [Sounds of Home]({DATASET_URL})",
        "",
        f"**Analysis generated:** {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}",
        "",
        "## Results from the Latest Run",
        "",
        "### 1. Analysis Settings",
        "",
        f"- **Dataset directory:** `{ROOT}`",
        f"- **Audio files discovered:** {len(wav_files)}",
        f"- **Prediction probability threshold:** {THRESHOLD}",
        f"- **Audio block size:** {BLOCK:,} samples",
        f"- **Blocks sampled per audio file:** Up to {SAMPLED_BLOCKS}",
        "",
        "### 2. Per-File Results",
        "",
        "| Recording | Estimated erased blocks | Prediction frames | Top 4 classes above threshold |",
        "|---|---:|---:|---|",
    ]

    for result in file_results:
        lines.append(
            f"| {result['filename']} "
            f"| {result['erased']:.1%} "
            f"| {result['frames']:,} "
            f"| {result['top_classes']} |"
        )

    lines.extend([
        "",
        "### 3. Overall Summary",
        "",
        f"- **Recordings analyzed successfully:** {len(file_results)}",
        f"- **Total prediction frames:** {total_frames:,}",
        f"- **Estimated analyzed duration:** {total_hours:.2f} hours",
        "",
        "### 4. Most Frequently Predicted Sound Classes",
        "",
        "The table below counts prediction-frame occurrences where a class "
        f"has a probability of at least {THRESHOLD}. These are counts, "
        "not unique sound events.",
        "",
        "| Rank | Sound class | Frames above threshold | Percentage of all prediction frames |",
        "|---:|---|---:|---:|",
    ])

    for rank, (name, count) in enumerate(totals.most_common(30), start=1):
        percentage = count / total_frames if total_frames else 0.0

        lines.append(
            f"| {rank} | {name} | {count:,} | {percentage:.1%} |"
        )

    lines.extend([
        "",
        "### 5. Notes and Limitations",
        "",
        "- **Erased audio estimate:** A sampled block is classified as erased "
        "when more than 99% of its samples are exactly zero. The reported "
        "percentage is the fraction of sampled blocks meeting this condition, "
        "not the exact percentage of audio duration erased.",
        "- **Prediction threshold:** Only class predictions with probability "
        f"at least {THRESHOLD} are counted.",
        "- **Frame counts are not event counts:** A sound lasting several "
        "seconds may contribute many prediction frames.",
        "- **Model predictions are not ground truth:** The reported classes "
        "are model-generated annotations and may include false positives "
        "or miss real sounds.",
        "- **Duration estimate:** Total hours are calculated from the number "
        "of prediction frames and the configured block size. Treat this as "
        "an approximate estimate of analyzed duration.",
    ])

    if warnings:
        lines.extend([
            "",
            "### 6. Warnings",
            "",
        ])

        for warning in warnings:
            lines.append(f"- {warning}")

    lines.extend([
        "",
        "---",
        "",
        "*End of report.*",
        "",
    ])

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print(f"Analysis complete. Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()