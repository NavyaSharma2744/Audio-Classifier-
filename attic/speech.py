import json
import matplotlib.pyplot as plt


JSON_PATH = (
    "data/VITALISE data light 2.3/"
    "Annotations/14/20231116_180000_light.json"
)


# Load predictions
with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)


# Extract Speech scores
times = []
scores = []

for item in data:

    if "predictions" not in item:
        continue

    times.append(item["time"])

    score = 0.0

    for prediction in item["predictions"]:
        if prediction["class"] == "Speech":
            score = prediction["prob"]
            break

    scores.append(score)


# Plot
plt.figure(figsize=(14, 5))

plt.plot(times, scores)

plt.xlabel("Time (seconds)")
plt.ylabel("Speech score")
plt.title("Speech predictions — 20231116_180000")

plt.ylim(0, 1)
plt.grid(True)

plt.tight_layout()
plt.show()