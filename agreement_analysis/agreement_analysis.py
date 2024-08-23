import json
import glob
from pathlib import Path

import pandas as pd

# import all json files in the directory with glob

chunk = "0"

json_files = glob.glob("*.json")
json_data = {}
for json_file in json_files:
    with open(json_file, "r") as f:
        data = json.load(f)
    annotations = data["annotations"][chunk]
    filename = Path(json_file).stem
    json_data[filename] = {
        k: v["answer"] for k, v in annotations.items() if isinstance(v, dict)
    }

    # filter value equal to "n/a"
    json_data[filename] = {k: v for k, v in json_data[filename].items() if v != "n/a"}

    if "seunggoo" in json_file:
        print(json_data[filename])

    print(f"Loaded {len(json_data[filename])} annotations from {json_file}")

df = pd.DataFrame(json_data)

# remove rows with NaNs
df = df.dropna()
print(df)


all_good = df == "all_good"
all_good["sum"] = all_good.sum(axis=1)

print(f"Agreement report ({len(df)} songs)\n")


for annotator in df.columns:
    per = (all_good[annotator]).sum() / len(all_good)
    print(f"Percentage of accepted {annotator}: {per:.2%}")

    values = df[annotator].value_counts()
    print(values.to_string(), "\n")

print("---")

per_fullagr = (all_good["sum"] == 3).sum() / len(all_good)
per_majoagr = (all_good["sum"] >= 2).sum() / len(all_good)

print(f"Percentage of OK songs with full agreement: {per_fullagr:.2%}")
print(f"Percentage of OK songs wth majority agreement: {per_majoagr:.2%}")

print("---")

reasons = [
    "not_emotionally_conveying",
    "bad_audio",
    "other_reasons",
]
for reason in reasons:
    refused = df == reason
    refused["sum"] = refused.sum(axis=1)
    per_fullagr = (refused["sum"] == 3).sum() / len(refused)
    per_majoagr = (refused["sum"] >= 2).sum() / len(refused)

    print(
        f"Percentage of songs with rejection reason `{reason}` with full agreement: {per_fullagr:.2%}"
    )
    print(
        f"Percentage of songs with rejection reason `{reason}` with majority agreement: {per_majoagr:.2%}"
    )

print("---")
