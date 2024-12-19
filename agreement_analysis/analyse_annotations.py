import json
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# Load chunk info
preselection_data_file = Path("data", "candidates.tsv")
preselection_data = pd.read_csv(preselection_data_file, sep="\t")

chunks = preselection_data["chunk_id"].unique()

# Load all annotations
ann_dir = Path("agreement_analysis", "annotations")
ann_files = ann_dir.rglob("*.json")


def prune_incomplete_chunks(data: dict) -> dict:
    """Remove chunks that are not annotated by 3 users"""

    for chunk_id in list(data.keys()):
        n_annotators = len(data[chunk_id]["user_ids"])
        if n_annotators < 3:
            print(f"discarding chunk {chunk_id} with {n_annotators} annotators")
            del data[chunk_id]

    return data


def compute_full_agreement(data: dict) -> list[str]:
    agreements = []
    for tidx in range(len(data["track_ids"])):
        answers = [data["annotations"][uid][tidx] for uid in data["user_ids"]]

        # check if all values are the same
        if len(set(answers)) == 1:
            agreements.append(answers[0])
        else:
            agreements.append("disagreement")

    return agreements


print("Loading annotation data")

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

print("\n\nPruining incomplete chunks")
data = prune_incomplete_chunks(data)

print(f"Continuing with {len(data)} chunks: {list(data.keys())}")

# Promediate results across all chunks

# for now take one chunk
chunk_id = "0"
chunk_data = data[chunk_id]

c_fa = compute_full_agreement(chunk_data)
# ax = sns.histplot(x=c_fa).set_title(f"Full agreement for chunk {chunk_id}")
# plt.show()

answer = []
annotator = []
for uid in chunk_data["user_ids"]:
    answer.extend(chunk_data["annotations"][uid])
    annotator.extend([uid] * len(chunk_data["track_ids"]))

# trim annotator name for clarity
annotator = [a.split("-")[0] for a in annotator]
answer = [a.replace("_", " ") for a in answer]

df = pd.DataFrame({"answer": answer, "annotator": annotator})

# print unique annotators

sns.histplot(
    data=df,
    x="annotator",
    hue="answer",
    multiple="dodge",
    # log_scale=(0, 2),
).set_title(f"Full agreement for chunk {chunk_id}")
plt.savefig("annotator_answer.png")
