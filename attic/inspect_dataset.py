import json
import wave
from pathlib import Path


# --------------------------------------------------
# Paths
# --------------------------------------------------

audio_path = Path(
    "data/VITALISE data light 2.3/Audio/14/20231116_180000_modified.WAV"
)

json_path = Path(
    "data/VITALISE data light 2.3/Annotations/14/20231116_180000_light.json"
)


# --------------------------------------------------
# AUDIO
# --------------------------------------------------

with wave.open(str(audio_path), "rb") as wav:
    channels = wav.getnchannels()
    sample_rate = wav.getframerate()
    sample_width = wav.getsampwidth()
    num_samples = wav.getnframes()

duration = num_samples / sample_rate


print("=== AUDIO ===")
print(f"File:         {audio_path.name}")
print(f"Channels:     {channels}")
print(f"Sample rate:  {sample_rate:,} Hz")
print(f"Sample width: {sample_width} bytes")
print(f"Samples:      {num_samples:,}")
print(f"Duration:     {duration:.2f} seconds")
print(f"Duration:     {duration / 60:.2f} minutes")


# --------------------------------------------------
# JSON
# --------------------------------------------------

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)


# Separate predictions from metadata
predictions = [
    item for item in data
    if "predictions" in item
]

metadata = [
    item for item in data
    if "metadata" in item
][0]


print()
print("=== JSON ===")
print(f"Total JSON objects:  {len(data):,}")
print(f"Prediction frames:  {len(predictions):,}")


# --------------------------------------------------
# TIMING
# --------------------------------------------------

first = predictions[0]
last = predictions[-1]

print()
print("=== TIMING ===")
print(f"First prediction: {first['time']:.4f} sec")
print(f"Last prediction:  {last['time']:.4f} sec")
print(f"Last frame index: {last['frame_index']}")
print(f"Last sample index:{last['sample_index']}")


# --------------------------------------------------
# METADATA
# --------------------------------------------------

print()
print("=== MODEL METADATA ===")

for key, value in metadata["metadata"].items():
    print(f"{key:<22} {value}")


# --------------------------------------------------
# FIRST PREDICTION
# --------------------------------------------------

print()
print("=== FIRST PREDICTION ===")

for prediction in first["predictions"]:
    print(
        f"{prediction['class']:<30}"
        f"{prediction['prob']:.4f}"
    )