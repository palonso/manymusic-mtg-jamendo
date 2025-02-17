from pathlib import Path
import json
import sys

import pandas as pd

sys.path.append("mtg-jamendo-dataset/scripts/")
import commons

genre_threshold = 0.1

data_dir = Path("data/")

# Load Discogsd metadata
data = pd.read_csv(data_dir / "mtg-jamendo-predictions.tsv", sep="\t", index_col=0)

data.index = pd.Index(map(lambda x: int(x.split("/")[1]), data.index))

data = data.filter(like="genre_discogs400-discogs-effnet-1")
data = data.groupby(lambda x: x.split("---")[1], axis=1).max()

genres_blacklist = set(["Non-Music", "Stage & Screen", "Children's"])

# Remove blacklisted columns
data = data.drop(columns=genres_blacklist)


# Replace each genre with a boolean column indicating if the genre is above the threshold
for genre in data.columns:
    data[genre] = data[genre] > genre_threshold

# Load mtg_jamendo metadata

print("Loading mtg-jamendo metadata...")
mtg_jamendo_file = "mtg-jamendo-dataset/data/autotagging.tsv"
tracks, _, _ = commons.read_file(mtg_jamendo_file)

# Load data/clean_tids.json, containing the list of ids after filtering
print("Loading clean_tids...")
with open(data_dir / "clean_tids.json") as f:
    clean_ids = json.load(f)

print("clean ids sample", clean_ids[:5])

# Load data/candidates.tsv, containing the list of ids after sampling
print("Loading candidates...")
candidates = pd.read_csv(data_dir / "candidates.tsv", sep="\t", index_col=0)
candidates_ids = candidates.index.tolist()

print("candidates sample", candidates_ids[:5])

# Load data with majority agreement in data/majority_agreement_tracks.tsv
print("Loading majority_agreement_tracks...")
majority_agreement_tracks = pd.read_csv(
    data_dir / "majority_agreement_tracks.tsv", sep="\t", index_col=0
)

ma_ids = majority_agreement_tracks.index.tolist()
print("majority agreement sample", ma_ids[:5])

# Load data with majority agreement in data/majority_agreement_tracks.tsv
print("Loading full_agreement_tracks...")
full_agreeement_tracks = pd.read_csv(
    data_dir / "full_agreement_tracks.tsv", sep="\t", index_col=0
)
fa_ids = full_agreeement_tracks.index.tolist()
print("full agreement sample", fa_ids[:5])

# 1. Leverage generate_candidates and do the same for the whole MTG-Jamendo
# 2. Check if in clean ids, if so set passed filters.
# 3. Check if in candidates, if so set sampled
# 4. Check final files to set all_good_* and ann_*

# columns:
# tid, <genres>, passed_filters, sampled, all_good_fa, all_good_ma, ann_1, ann_2, and_4

data["filters_ok"] = data.index.isin(clean_ids)
data["kmeans_ok"] = data.index.isin(candidates_ids)
data["majority_agreement_ok"] = data.index.isin(ma_ids)
data["full_agreement_ok"] = data.index.isin(fa_ids)

data.to_csv(data_dir / "manymusic-genre-subset-ids.tsv", sep="\t")


for colu in data.columns:
    print(data[colu].value_counts())
