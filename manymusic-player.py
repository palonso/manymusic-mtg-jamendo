import json
import math
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append("mtg-jamendo-dataset/scripts/")
import commons

data_dir = Path("data/")


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


def audio_url(trackid):
    """Return the Jamendo URL for a given trackid."""


def play(tid: str):
    """Play a track and print tags from its tid."""
    jamendo_url = audio_url(tid)
    track = tracks[tid]
    tags = [t.split("---")[1] for t in track["tags"]]

    st.write("---")
    st.write(f"**Track {tid}** - tags: {', '.join(tags)}")
    st.audio(jamendo_url, format="audio/mp3", start_time=0)


data, tracks = load_data()
tids_init = set(tracks.keys())
tids_clean = tids_init

st.write("## Loading  list of candidates")
with open("data/candidates.json", "r") as f:
    data = json.load(f)


choices_1 = ["arousal", "valence", "genres"]
choice_1 = st.selectbox("Select a cluster", choices_1)


if choice_1 in ("arousal", "valence"):
    choices_2 = data[choice_1].keys()
    choice_2 = st.selectbox("Select a sub-cluster", choices_2)
    ids = data[choice_1][choice_2]
elif choice_1 == "genres":
    choices_2 = set(data.keys()) - set(["arousal", "valence"])
    choice_2 = st.selectbox("Select a sub-cluster", choices_2)
    ids = data[choice_2]
else:
    raise NotImplementedError("choose arousal, valence or genres")

tracks_per_page = 5

if "page" not in st.session_state:
    st.session_state.page = 0
    st.session_state.n_pages = math.ceil(len(ids) / tracks_per_page)


def next_page():
    st.session_state.page += 1
    if st.session_state.page >= st.session_state.n_pages - 1:
        st.session_state.page = st.session_state.n_pages - 1


def prev_page():
    st.session_state.page -= 1
    if st.session_state.page < 0:
        st.session_state.page = 0


st.write(f"Page {st.session_state.page + 1}/{st.session_state.n_pages}")

col1, _, _, col2 = st.columns(4)
col2.button("Next page ->", on_click=next_page)
col1.button("<- Previous page", on_click=prev_page)

ids_show = ids[
    st.session_state.page
    * tracks_per_page : (st.session_state.page + 1)
    * tracks_per_page
]

for id in ids_show:
    play(id)
