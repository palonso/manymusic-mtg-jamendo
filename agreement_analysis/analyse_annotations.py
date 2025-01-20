import json
from pathlib import Path
from typing import List, Tuple
from collections import Counter

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
        if n_annotators > 3:
            print(f"WARNING chunk {chunk_id} with {n_annotators} annotators")
            for uid in data[chunk_id]["user_ids"]:
                print("\t", uid)
            print("\n")

    return data


def parse_answer(answer: str) -> str:
    answer = answer.replace("_", " ")
    answer = answer.replace("all good", "good")
    answer = answer.replace("not emotionally conveying", "not emo.")
    answer = answer.replace("other reasons", "other")
    answer = answer.replace("copyrighted content", "copyright")
    return answer


def compute_full_agreement(data: dict) -> Tuple[List[str], int]:
    agreements = []
    n_good = 0
    for tidx in range(len(data["track_ids"])):
        answers = [data["annotations"][uid][tidx] for uid in data["user_ids"]]

        # check if all values are the same
        if len(set(answers)) == 1:
            parsed_anwser = parse_answer(answers[0])
            agreements.append(parsed_anwser)
            if answers[0] == "all_good":
                n_good += 1
        else:
            agreements.append("disagree")

    return agreements, n_good


def compute_maj_agreement(data: dict) -> Tuple[List[str], int]:
    agreements = []
    n_good = 0
    for tidx in range(len(data["track_ids"])):
        answers = [data["annotations"][uid][tidx] for uid in data["user_ids"]]

        # check if only one of the annotators disagrees
        if len(set(answers)) <= 2:
            parsed_anwser = parse_answer(answers[0])
            agreements.append(parsed_anwser)
            if answers[0] == "all_good":
                n_good += 1
        else:
            agreements.append("disagree")

    return agreements, n_good


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

            if user_id == "373cf81a-bb35-46f7-8a21-b2a890174a01" and chunk_id in (
                "3",
                "7",
                "8",
                "11",
                "12",
            ):
                continue

            if (
                user_id == "e80b0144-bceb-4c21-9200-d5252513f23f_chunk_6"
                and chunk_id in ("7", "8")
            ):
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


# sort chunk ids
chunk_ids = [int(i) for i in data.keys()]
chunk_ids.sort()
chunk_ids = [str(i) for i in chunk_ids]

print(f"Continuing with {len(chunk_ids)} chunks: {chunk_ids}")


# Export tsv with all the good files with the fields: track_id, chunk_id, user_1_id, user_2_id, user_3_id
tsv_data = []


sns.set_style("darkgrid")
sns.set(font_scale=0.85)
y_max = 200

f, ax = plt.subplots(len(data.keys()), 3, figsize=(10, 10))


for i, chunk_id in enumerate(chunk_ids):
    chunk_data = data[chunk_id]

    c_fa, n_good = compute_full_agreement(chunk_data)
    keys, values = zip(*Counter(c_fa).most_common())

    for tid, answer in zip(chunk_data["track_ids"], c_fa):
        if answer == "good":
            tsv_data.append(
                {
                    "track_id": tid,
                    "chunk_id": chunk_id,
                    "user_1_id": chunk_data["user_ids"][0].split("-")[0],
                    "user_2_id": chunk_data["user_ids"][1].split("-")[0],
                    "user_3_id": chunk_data["user_ids"][2].split("-")[0],
                }
            )

    total_good_fa += n_good
    total_tracks += len(c_fa)

    # % of good and bad
    good_per = 100 * n_good / len(c_fa)
    sns.barplot(
        x=keys,
        y=values,
        order=keys,
        ax=ax[i, 0],
    ).set_title(
        f"Chunk {chunk_id} full agreement\n Good: {n_good}/{len(c_fa)} ({good_per:.1f}%)"
    )
    ax[i, 0].set_ylim(0, y_max)
    ax[i, 0].set_xticklabels(ax[i, 0].get_xticklabels(), rotation=90)

    c_fa, n_good = compute_maj_agreement(chunk_data)
    keys, values = zip(*Counter(c_fa).most_common())

    # % of good and bad
    good_per = 100 * n_good / len(c_fa)

    sns.barplot(
        x=keys,
        y=values,
        order=keys,
        ax=ax[i, 1],
    ).set_title(
        f"Chunk {chunk_id} majority agreement\n Good: {n_good}/{len(c_fa)} ({good_per:.1f}%)"
    )
    ax[i, 1].set_ylim(0, y_max)
    ax[i, 1].set_xticklabels(ax[i, 1].get_xticklabels(), rotation=90)

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
        ax=ax[i, 2],
        # log_scale=(0, 2),
    ).set_title(f"Chunk {chunk_id} annotator answers\n")
    ax[i, 2].set_ylim(0, y_max)

# TODO Promediate results across all chunks

plt.subplots_adjust(left=0.1, bottom=0.12, right=0.9, top=0.9, wspace=0.4, hspace=0.7)
plt.savefig("agreement_analysis.png")


# print all good results
print(
    f"Total good full agreement: {total_good_fa}/{total_tracks} ({100 * total_good_fa / total_tracks:.1f}%)"
)
print(
    f"Total good majority agreement: {total_good_maj}/{total_tracks} ({100 * total_good_maj / total_tracks:.1f}%)"
)

tsv_df = pd.DataFrame(tsv_data)
tsv_df.to_csv("data/full_agreement_tracks.tsv", sep="\t", index=False)
