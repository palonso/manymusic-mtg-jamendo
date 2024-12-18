import json
from pathlib import Path

import pandas as pd

# Load chunk info
preselection_data_file = Path("data", "candidates.tsv")
preselection_data = pd.read_csv(preselection_data_file, sep="\t")

chunks = preselection_data["chunk_id"].unique()

# Load all annotations
ann_dir = Path("agreement_analysis", "annotations")
ann_files = ann_dir.rglob("*.json")


# Generate chunk data
data = {}
for ann_file in ann_files:
    with open(ann_file, "r") as f:
        ann_data = json.load(f)
        anns = ann_data["annotations"]

        user_id = str(ann_file.parent.name)

        for chunk_id, chunk_anns in anns.items():
            ok = True
            for i, value in enumerate(list(chunk_anns.values())):
                if not value:
                    ok = False

                    break

            if not ok:
                # Skip the case in which first value is already empty.
                # This means that the user clink on the chunk option (an initialized it)
                # but did not click on any track.
                if i != 0:
                    print(f"Chunk {chunk_id} annotated by {user_id} not OK!")
                    print(f"Value number {i}:", list(chunk_anns.values())[i])

                continue

            else:
                print(f"Chunk {chunk_id} annotated by {user_id} OK!")

            if chunk_id not in data:
                # Data structure init
                data[chunk_id] = dict()
                data[chunk_id]["user_ids"] = []
                data[chunk_id]["track_ids"] = list(chunk_anns.keys())
                data[chunk_id]["annotations"] = dict()

            else:
                # Chack that track ids are the same
                assert data[chunk_id]["track_ids"] == list(chunk_anns.keys())

            data[chunk_id]["user_ids"].append(user_id)
            data[chunk_id]["annotations"][user_id] = [
                ann["answer"] for ann in chunk_anns.values()
            ]


# Create method to get chunk analysis

# Promediate results across all chunks
