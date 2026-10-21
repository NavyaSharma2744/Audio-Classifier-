
import json

from pathlib import Path

prediction_class = []
for json_path in Path("data/VITALISE data light 2.3/Annotations/14/").glob("*.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(type(data))

    for item in data:

    # Skip metadata object
        if "predictions" not in item:
            continue

        for i in item["predictions"]:
            if i["class"] not in prediction_class:
                prediction_class.append(i["class"])

print(prediction_class)

print(f"Total number of unique classes: {len(prediction_class)}")