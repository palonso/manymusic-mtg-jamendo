import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.image import imread

from utils import load_av_time_data, smooth_data, plot_av, play

sys.path.append("mtg-jamendo-dataset/scripts/")
import commons

data_dir = Path("data/")

tracks_per_page = 5


@st.cache_data
def load_data():
    """Load and prepare ground truth in the streamlit cache."""

    data_models = pd.read_csv(
        data_dir / "mtg-jamendo-predictions.tsv", sep="\t", index_col=0
    )
    data_av = pd.read_pickle(data_dir / "mtg-jamendo-predictions-av.pk")
    data_algos = pd.read_pickle(data_dir / "mtg-jamendo-predictions-algos.pk")

    data = pd.concat([data_models, data_av, data_algos], axis=1)
    data.index = pd.Index(map(lambda x: int(x.split("/")[1]), data.index))

    mtg_jamendo_file = "mtg-jamendo-dataset/data/autotagging.tsv"
    tracks, _, _ = commons.read_file(mtg_jamendo_file)
    return data, tracks


def get_top_tags(tids: list, n_most_common: int = 5):
    """Get the top tags for a list of tids."""
    tags = []
    for tid in tids:
        track = tracks[tid]
        tags += [t.split("---")[1] for t in track["tags"]]

    return Counter(tags).most_common(n_most_common)


data, tracks = load_data()
tids_init = set(tracks.keys())
tids_clean = tids_init

cluster_params = (
    "clustering_genre_thres_0.1_n_samples_200_smoothing_5_decimate_5_norm_none"
)

st.write("## Loading list of candidates")
with open(f"data/{cluster_params}/candidates.json", "r") as f:
    data = json.load(f)

data_av_clean, tids_clean = load_av_time_data(tids_clean, tracks)
print(f"Kept {len(data_av_clean)} samples")

data_av_smooth = smooth_data(data_av_clean)

# Count tracks
n_tracks = 0
genres_with_clusters = []
for k, v in data.items():
    if len(v) > 1:
        genres_with_clusters.append(k)
    for k2, v2 in v.items():
        n_tracks += len(v2)

st.write(f"Loaded `{n_tracks}` tracks")
st.write(f"Genres with multiple clusters: {'; '.join(genres_with_clusters)}")

choices_1 = data.keys()
choice_1 = st.selectbox("Select a genre", choices_1)

data_genre = data[choice_1]

choices_2 = data_genre.keys()
choice_2 = st.selectbox("Select a type of data", choices_2)

ids = data_genre[choice_2]

cluster_img_file = data_dir / cluster_params / f"{choice_1}_av_scatter.png"
if cluster_img_file.exists():
    cluster_img = imread(cluster_img_file)
    st.image(cluster_img, use_column_width=True)

st.write(f"### Cluster `{choice_2}` kernel:")

cluster_data_file = data_dir / cluster_params / f"kmeans_centers_{choice_1}.npy"
cluster_data = np.load(cluster_data_file)
cluster_idx = int(choice_2.split("_")[-1])
plot_av(cluster_data[cluster_idx])


st.write(f"`{len(ids)}` tracks on this cluster. Most common tags:")
st.dataframe(get_top_tags(ids))


if "choice_1" not in st.session_state:
    st.session_state.choice_1 = choice_1
    st.session_state.choice_2 = choice_2
    st.session_state.page = 0
    st.session_state.n_pages = math.ceil(len(ids) / tracks_per_page)

if st.session_state.choice_1 != choice_1 or st.session_state.choice_2 != choice_2:
    st.session_state.page = 0
    st.session_state.n_pages = math.ceil(len(ids) / tracks_per_page)

    st.session_state.choice_1 = choice_1
    st.session_state.choice_2 = choice_2


def next_page():
    st.session_state.page += 1
    if st.session_state.page >= st.session_state.n_pages - 1:
        st.session_state.page = st.session_state.n_pages - 1


def prev_page():
    st.session_state.page -= 1
    if st.session_state.page < 0:
        st.session_state.page = 0


st.write(f"Page {st.session_state.page + 1}/{st.session_state.n_pages}")


ids_show = ids[
    st.session_state.page * tracks_per_page : (st.session_state.page + 1)
    * tracks_per_page
]

for tid in ids_show:
    play(tid, tracks)
    plot_av(data_av_smooth[tid])


col1, _, _, col2 = st.columns(4)
col2.button("Next page ➡️", on_click=next_page)
col1.button("⬅️  Previous page", on_click=prev_page)
