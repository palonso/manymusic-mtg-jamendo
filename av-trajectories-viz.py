import cmathimport
import pickle as pk
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import seaborn as sns
import streamlit as st
from scipy.ndimage import gaussian_filter1d


aspects = ("arousal", "valence")
traject_types = ("ascending", "descending", "peaks")

sys.path.append("mtg-jamendo-dataset/scripts/")
import commons

data_dir = Path("data/")

# load algorithm data
av_predictions_dir = (
    data_dir / "predictions" / "emomusic-msd-musicnn-2" / "emomusic-msd-musicnn-2"
)


def audio_url(trackid):
    return f"https://mp3d.jamendo.com/?trackid={trackid}&format=mp32#t=0,120"


st.write(
    """
# ManyMusic Dataset
## Interactive Visualization of MTG Jamendo Dataset
"""
)


@st.cache_data
def load_data():
    # load and prepare groundtruth
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


data, tracks = load_data()
tids_init = set(tracks.keys())
tids_clean = tids_init


@st.cache_data
def load_av_time_data():
    data_av_time = dict()
    pbar_av_time = st.progress(0.0, text="Loading AV predictions")
    tids_clean_list = list(tids_clean)
    for i, index in enumerate(tids_clean_list):
        try:
            av_filename = (av_predictions_dir / tracks[index]["path"]).with_suffix(
                ".npy"
            )
            # load and normalize
            data_av_time[index] = (np.load(av_filename) - 5) / 4
        except Exception:
            pass
        pbar_av_time.progress((i + 1) / len(tids_clean_list))
    pbar_av_time.empty()

    st.write(f"Loaded {len(data_av_time)} AV predictions")

    return data_av_time


data_av_time = load_av_time_data()

tids_clean = set(data_av_time.keys())

data_av_std = {i: np.std(v, axis=0) for i, v in data_av_time.items()}

low_p, high_p = st.slider("Arousal/Valence range (percentile)", 0, 100, (10, 90))

data_av_perc = {
    i: np.percentile(v, high_p, axis=0) - np.percentile(v, low_p, axis=0)
    for i, v in data_av_time.items()
}

thres_av_disp = st.slider("Arousal/Valence threshold (percentile)", 0.0, 1.0, 0.3)

tids_av_disp_low = {i for i, v in data_av_perc.items() if (v < thres_av_disp).any()}
tids_clean -= tids_av_disp_low

st.write(
    f"""
    tracks with low A/V disperssion : {len(tids_av_disp_low)}

    remaining tracks: {len(tids_clean)}
    """
)

sample_tid = list(tids_clean)[0]
sample = data_av_time[sample_tid]

sigma = st.slider("Gausian filter smoothing", 0, 100, 15)
sample_filt = gaussian_filter1d(sample, sigma, axis=0)

data_av_clean = {tid: data_av_time[tid] for tid in list(tids_clean)}


@st.cache_data
def smooth_data(data: dict, sigma: int):
    return {k: gaussian_filter1d(sample, sigma, axis=0) for k, sample in data.items()}


data_av_smooth = smooth_data(data_av_clean, sigma)


@st.cache_data
def diff_data(data: dict):
    return {k: np.diff(sample, axis=0) for k, sample in data.items()}


data_av_diff = diff_data(data_av_smooth)


def plot_av(tid: int, axvline_loc: float = None):
    sample = data_av_smooth[tid]

    formatter = DateFormatter("%M'%S''")

    emb2days = 63 * 256 / (16000 * 3600 * 24)
    time = np.linspace(0, len(sample) * emb2days, len(sample))

    fig, ax = plt.subplots()
    ax.plot(time, sample[:, 0], label="valence")
    ax.plot(time, sample[:, 1], label="arousal")
    ax.xaxis.set_major_formatter(formatter)

    if axvline_loc is not None:
        axvline_loc *= emb2days
        label = f"appx. peak: {formatter(axvline_loc)}"
        plt.axvline(axvline_loc, color="k", label=label)

    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)


plot_av(sample_tid)


@st.cache_data
def reduce_data(data_in: dict):
    # return {k: np.sum(sample, axis=0) for k, sample in data.items()}
    data = dict()
    for k, sample in data_in.items():
        reduced = np.mean(sample, axis=0)
        data[k] = {"arousal": reduced[1], "valence": reduced[0]}
    return data


# @st.cache_data
def reduce_max_abs(data_in: dict):
    thres = 15

    data = dict()
    for k, sample in data_in.items():
        if sample.shape[0] > 2 * thres:
            sample = sample[thres:-thres, :]

        absolute = np.abs(sample)
        argmax_loc = np.argmax(absolute, axis=0)

        data[k] = {
            "arousal": absolute[argmax_loc[1], 1],
            "valence": absolute[argmax_loc[0], 0],
            "arousal_loc": argmax_loc[1],
            "valence_loc": argmax_loc[0],
        }
    return data


data_av_diff_sum = reduce_data(data_av_diff)
data_av_diff_sum = pd.DataFrame.from_dict(data_av_diff_sum, orient="index")

data_av_diff_max = reduce_max_abs(data_av_diff)
data_av_diff_max = pd.DataFrame.from_dict(data_av_diff_max, orient="index")

st.dataframe(data_av_diff_max)

cluster_size = 3
trajectory_groups = defaultdict(dict)
trajectory_groups["arousal"]["ascending"] = data_av_diff_sum.nlargest(
    cluster_size, "arousal"
)
trajectory_groups["arousal"]["descending"] = data_av_diff_sum.nsmallest(
    cluster_size, "arousal"
)
trajectory_groups["arousal"]["peaks"] = data_av_diff_max.nlargest(
    cluster_size, "arousal"
)
trajectory_groups["valence"]["ascending"] = data_av_diff_sum.nlargest(
    cluster_size, "valence"
)
trajectory_groups["valence"]["descending"] = data_av_diff_sum.nsmallest(
    cluster_size, "valence"
)

trajectory_groups["valence"]["peaks"] = data_av_diff_max.nlargest(
    cluster_size, "valence"
)


for aspect in aspects:
    for traject_type in traject_types:
        st.write(f"## {aspect} {traject_type}")
        for tid in trajectory_groups[aspect][traject_type].index:
            jamendo_url = audio_url(tid)
            track = tracks[tid]
            tags = [t.split("---")[1] for t in track["tags"]]

            st.write("---")
            st.write(f"**Track {tid}** - tags: {tags}")
            st.audio(jamendo_url, format="audio/mp3", start_time=0)

            if traject_type == "peaks":
                max_loc = data_av_diff_max.loc[tid][f"{aspect}_loc"]
            else:
                max_loc = None
            plot_av(tid, axvline_loc=max_loc)
