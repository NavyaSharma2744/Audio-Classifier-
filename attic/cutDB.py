from pathlib import Path
import random

random.seed(42)  # reproducible

audio_root = Path("../../../Downloads/VITALISE data light 2.3/Audio")
ann_root = Path("../../../Downloads/VITALISE data light 2.3/Annotations")

KEEP_FRACTION = 0.25

for audio_folder in audio_root.iterdir():
    if not audio_folder.is_dir():
        continue

    ann_folder = ann_root / audio_folder.name
    if not ann_folder.exists():
        continue

    wavs = sorted(audio_folder.glob("*.WAV"))

    prefixes = [
        wav.name.replace("_modified.WAV", "")
        for wav in wavs
    ]

    keep = set(random.sample(prefixes,
                             int(len(prefixes) * KEEP_FRACTION)))

    deleted = 0

    for prefix in prefixes:
        if prefix in keep:
            continue

        wav = audio_folder / f"{prefix}_modified.WAV"
        js = ann_folder / f"{prefix}_light.json"

        if wav.exists():
            wav.unlink()

        if js.exists():
            js.unlink()

        deleted += 1

    print(f"{audio_folder.name}: kept {len(keep)}, deleted {deleted}")