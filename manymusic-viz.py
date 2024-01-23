import cmath
import json
from collections import defaultdict
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from matplotlib.dates import DateFormatter
from scipy.ndimage import gaussian_filter1d


sys.path.append("mtg-jamendo-dataset/scripts/")
import commons

data_dir = Path("data/")

av_predictions_dir = data_dir / "predictions" / "emomusic-msd-musicnn-2"

aspects = ("arousal", "valence")
traject_types = ("ascending", "descending", "peaks", "climax")

sampling_size = 100
example_size = 4
genre_threshold = 0.1


def audio_url(trackid):
    """Return the Jamendo URL for a given trackid."""
    return f"https://mp3d.jamendo.com/?trackid={trackid}&format=mp32#t=0,120"


def play(tid: str):
    """Play a track and print tags from its tid."""
    jamendo_url = audio_url(tid)
    track = tracks[tid]
    tags = [t.split("---")[1] for t in track["tags"]]

    st.write("---")
    st.write(f"**Track {tid}** - tags: {', '.join(tags)}")
    st.audio(jamendo_url, format="audio/mp3", start_time=0)


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


@st.cache_data
def load_av_time_data():
    """Load and prepare time-wise arousal and valence data in the streamlit cache."""
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
        label = f"location: {formatter(axvline_loc)}"
        plt.axvline(axvline_loc, color="k", label=label)

    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)

    plt.close()


st.write(
    """
# ManyMusic Dataset
## Interactive Visualization of MTG Jamendo Dataset
"""
)

data, tracks = load_data()
tids_init = set(tracks.keys())
tids_clean = tids_init


data_av_time = load_av_time_data()


st.write(
    """
## 1. Track duration filtering
"""
)

durations = np.array([v["duration"] for v in tracks.values()])
max_duration = int(np.max(durations) + 0.5)

dur_min, dur_max = st.slider("Minimum duration (seconds)", 0, max_duration, (60, 300))

tids_short = {tid for tid, values in tracks.items() if values["duration"] < dur_min}
tids_long = {tid for tid, values in tracks.items() if values["duration"] > dur_max}

tids_init = set(data_av_time.keys())
tids_clean = tids_init - tids_short - tids_long

st.write(
    f"""
    total tracks: {len(tids_init)}

    short tracks: {len(tids_short)}

    long tracks: {len(tids_long)}

    remaining tracks: {len(tids_clean)}
"""
)


st.write(
    """
    ## 2. Loudness filtering
    """
)

loudness_values = data["integrated_loudness"]
loud_range = st.slider("Loudness range (percentile)", 0, 100, (5, 95))
low_p, high_p = np.percentile(loudness_values, loud_range)
st.write(
    f"percentiles 5 and 95 correspond to integrated loudness values of {low_p:.2f} and {high_p:.1f}"
)

fig, ax = plt.subplots()
sns.histplot(loudness_values, ax=ax)
plt.axvline(low_p, color="r")
plt.axvline(high_p, color="r")

st.pyplot(fig)

tids_loud_l = set(data.index[loudness_values < low_p])
tids_loud_h = set(data.index[loudness_values > high_p])

tids_clean = tids_clean - tids_loud_l - tids_loud_h


st.write(
    f"""
    low loudness tracks: {len(tids_loud_l)}

    high loudness tracks: {len(tids_loud_h)}

    remaining tracks: {len(tids_clean)}
    """
)


st.write(
    """
    ## 3. False stereo filtering
    (non parametric)
    """
)
is_false_stereo = data["is_false_stereo"]

tids_false_stereo = set(data.index[data["is_false_stereo"]])
tids_clean = tids_clean - tids_false_stereo

st.write(
    f"""
    false stereo tracks: {len(tids_false_stereo)}

    remaining tracks: {len(tids_clean)}
    """
)


st.write(
    """
    ## 4. Clipping filtering
    """
)
n_peaks = data["peak_locations"].apply(lambda x: len(x))
n_peaks.rename("Number of peaks", inplace=True)

perc_peaks = st.slider("Number of clippings (percentile)", 0, 100, 90)
high_p = np.percentile(n_peaks, perc_peaks).astype(int)
st.write(f"percentile {perc_peaks} corresponds to {high_p} peaks per song")

fig, ax = plt.subplots()
sns.histplot(n_peaks, bins=100, log_scale=(0, 10), ax=ax)
plt.axvline(high_p, color="r")
st.pyplot(fig)

tids_peaks = set(data.index[n_peaks > high_p])
tids_clean = tids_clean - tids_peaks

st.write(
    f"""
    clipped tracks: {len(tids_peaks)}

    remaining tracks: {len(tids_clean)}
    """
)


st.write(
    """
    ## 5. Music style filtering
    We want to filter out tracks that do not belong clearly to any of the styles in the taxonomy.
    """
)
act_thres = st.slider(
    "Min value of the top activation in the Effnet-Discogs taxonomy", 0.0, 1.0, 0.2
)
data_styles = data.filter(like="genre_discogs400-discogs-effnet-1")
data_genre_not_present = data_styles[data_styles.max(axis=1) < act_thres]

tids_no_style = set(data_genre_not_present.index)
tids_clean -= tids_no_style

st.write(
    f"""
    tracks without style: {len(tids_no_style)}

    remaining tracks: {len(tids_clean)}
    """
)

tids_styles_available = tids_clean.intersection(set(data_styles.index))
styles_cumsum = data_styles.loc[list(tids_styles_available)].sum(axis=0)
genres = set(data_styles.columns.map(lambda x: x.split("---")[1]))
genres_cumsum = dict()
for style in styles_cumsum.index:
    for genre in genres:
        if genre in style:
            genres_cumsum[genre] = genres_cumsum.get(genre, 0) + styles_cumsum[style]

fig, ax = plt.subplots()
ax = sns.boxplot(x=genres_cumsum.keys(), y=genres_cumsum.values(), ax=ax)
ax.tick_params(labelrotation=90)
st.pyplot(fig)


st.write(
    """
    ## 6. Remove tracks with Non-Music style
    """
)

act_thres = st.slider(
    "Min value of the top non-music class to discard the track", 0.0, 1.0, 0.1
)
data_non_music = data.filter(like="Non-Music")
data_non_music_present = data_non_music[data_non_music.max(axis=1) > act_thres]

tids_non_music = set(data_non_music_present.index)
tids_clean -= tids_non_music

st.write(
    f"""
    non-music tracks: {len(tids_non_music)}

    remaining tracks: {len(tids_clean)}
    """
)

st.write("Examples of discarded tracks")

for tid in random.sample(list(tids_non_music), example_size):
    play(tid)
    st.dataframe(data_non_music_present.loc[tid].nlargest(3))


blacklist = (
    "mood/theme---xmas",
    "mood/theme---christmas",
    "mood/theme---advertising",
    "mood/theme---presentation",
    "mood/theme---backgrounds",
    "mood/theme---corporate",
    "mood/theme---background",
    "mood/theme---commercial",
    "mood/theme---motivational",
)

st.write(
    f"""
    ## 7. Remove tracks with black-listed mood/theme tags

    blacklist: {' ,'.join(blacklist)}
    """
)

blacklist_tids = set()

for tid, values in tracks.items():
    if set(values["tags"]).intersection(blacklist):
        blacklist_tids.add(tid)

tids_clean -= blacklist_tids

st.write(
    f"""
    tracks with blacklisted tags: {len(blacklist_tids)}

    remaining tracks: {len(tids_clean)}

    Examples of blacklisted tracks:
    """
)

for tid in random.sample(list(blacklist_tids), example_size):
    play(tid)


st.write(
    """
    ## 8. Remove tracks with too much/little AV dispersion
    """
)

data_av_std = {i: np.std(v, axis=0) for i, v in data_av_time.items()}

low_p, high_p = st.slider("Arousal/Valence range (percentile)", 0, 100, (10, 90))

data_av_perc = {
    i: np.percentile(v, high_p, axis=0) - np.percentile(v, low_p, axis=0)
    for i, v in data_av_time.items()
}

thres_av_disp = st.slider("Arousal/Valence threshold (percentile)", 0.0, 1.0, 0.15)

tids_av_disp_low = {i for i, v in data_av_perc.items() if (v < thres_av_disp).any()}
tids_clean -= tids_av_disp_low

st.write(
    f"""
    tracks with low A/V disperssion : {len(tids_av_disp_low)}

    remaining tracks: {len(tids_clean)}

    Examples of tracks with low A/V dispersion:
    """
)

for tid in random.sample(list(tids_av_disp_low), example_size):
    play(tid)

sample_tid = list(tids_clean)[0]
sample = data_av_time[sample_tid]


sigma = st.slider("Gausian filter smoothing", 0, 100, 5)
sample_filt = gaussian_filter1d(sample, sigma, axis=0)

data_av_clean = {tid: data_av_time[tid] for tid in list(tids_clean)}


@st.cache_data
def smooth_data(data: dict, sigma: int):
    return {k: gaussian_filter1d(sample, sigma, axis=0) for k, sample in data.items()}


data_av_smooth = smooth_data(data_av_clean, sigma)
plot_av(sample_tid)


@st.cache_data
def diff_data(data: dict):
    return {k: np.diff(sample, axis=0) for k, sample in data.items()}


data_av_diff = diff_data(data_av_smooth)


@st.cache_data
def reduce_data(data_in: dict):
    # return {k: np.sum(sample, axis=0) for k, sample in data.items()}
    data = dict()
    for k, sample in data_in.items():
        reduced = np.mean(sample, axis=0)
        data[k] = {"arousal": reduced[1], "valence": reduced[0]}
    return data


@st.cache_data
def reduce_max_abs(data_in: dict):
    thres = 15

    data = dict()
    shift = 0
    for k, sample in data_in.items():
        if sample.shape[0] > 2 * thres:
            sample = sample[thres:-thres, :]
            shift = thres

        absolute = np.abs(sample)
        argmax_loc = np.argmax(absolute, axis=0)

        data[k] = {
            "arousal": absolute[argmax_loc[1], 1],
            "valence": absolute[argmax_loc[0], 0],
            "arousal_loc": argmax_loc[1] + shift,
            "valence_loc": argmax_loc[0] + shift,
        }
    return data


@st.cache_data
def reduce_climax(data_in: dict):
    # new criterium -> locate for climax peak
    # - betwen 30% and 90%
    # needs to be a positive arousal peak
    # valence should be strengthen
    thres = 0.3

    data = dict()
    for k, sample in data_in.items():
        min_pos = int(thres * len(sample))
        # we only need arousal
        sample = sample[:, 1]
        # remove beginning and end
        sample = sample[min_pos:-min_pos]
        pos_diff = sample > 0

        # count build up samples
        pos_points = 0
        max_skips = 5
        skips = max_skips
        for climax_loc, point in enumerate(pos_diff):
            if point:
                pos_points += 1
                skips = max_skips
            else:
                skips -= 1
            if not skips:
                break

        # check at least at least some after-climax
        neg_thres = 5
        neg_points = 0
        max_skips = 5
        skips = max_skips
        for point in pos_diff[:climax_loc:-1]:
            if not point:
                neg_points += 1
                skips = max_skips
            else:
                skips -= 1
            if not skips:
                break

        pos_points += min_pos

        # do not consider sample if there is no after-climax
        if neg_points < neg_thres:
            pos_points = 0

        data[k] = {"arousal": pos_points}
    return data


data_av_diff_sum = reduce_data(data_av_diff)
data_av_diff_sum = pd.DataFrame.from_dict(data_av_diff_sum, orient="index")

data_av_diff_max = reduce_max_abs(data_av_diff)
data_av_diff_max = pd.DataFrame.from_dict(data_av_diff_max, orient="index")

data_a_climax = reduce_climax(data_av_diff)
data_a_climax_max = pd.DataFrame.from_dict(data_a_climax, orient="index")

trajectory_groups = defaultdict(dict)
trajectory_groups["arousal"]["ascending"] = data_av_diff_sum.nlargest(
    sampling_size, "arousal"
)
trajectory_groups["arousal"]["descending"] = data_av_diff_sum.nsmallest(
    sampling_size, "arousal"
)
trajectory_groups["arousal"]["peaks"] = data_av_diff_max.nlargest(
    sampling_size, "arousal"
)
trajectory_groups["valence"]["ascending"] = data_av_diff_sum.nlargest(
    sampling_size, "valence"
)
trajectory_groups["valence"]["descending"] = data_av_diff_sum.nsmallest(
    sampling_size, "valence"
)
trajectory_groups["valence"]["peaks"] = data_av_diff_max.nlargest(
    sampling_size, "valence"
)

trajectory_groups["arousal"]["climax"] = data_a_climax_max.nlargest(
    sampling_size, "arousal"
)

st.write("## Sample selection by A/V trajectories")

for aspect in aspects:
    for traject_type in traject_types:
        if traject_type not in trajectory_groups[aspect]:
            continue

        st.write(f"### {aspect} {traject_type}")
        for tid in trajectory_groups[aspect][traject_type].sample(example_size).index:
            play(tid)

            max_loc = None
            if traject_type == "peaks":
                max_loc = data_av_diff_max.loc[tid][f"{aspect}_loc"]
            if traject_type == "climax":
                max_loc = data_a_climax_max.loc[tid][f"{aspect}"]
            plot_av(tid, axvline_loc=max_loc)


st.write("## Sample selection by estimated music style")

data_styles = data.filter(like="genre_discogs400-discogs-effnet-1")

data_genres = data_styles.groupby(lambda x: x.split("---")[1], axis=1).max()
data_genres = data_genres[data_genres.index.isin(tids_clean)].copy()

genres = set(data_genres.columns)
genres_blacklist = set(["Non-Music", "Stage & Screen", "Children's"])
genres_good = genres - genres_blacklist

st.write(f"sampleing from {len(genres_good)} genres")


genre_groups = dict()
for genre in list(genres_good):
    st.write(f"### {genre}")
    genre_groups[genre] = data_genres.nlargest(sampling_size, genre)
    genre_groups[genre].drop(
        genre_groups[genre][genre_groups[genre][genre] < genre_threshold].index,
        inplace=True,
    )

    st.write(f"keeping {len(genre_groups[genre])}/{sampling_size} ids for {genre}")

    for tid in genre_groups[genre].sample(example_size).index:
        play(tid)


subset_data = defaultdict(dict)
tids_subset = set()
yid2src = dict()
for aspect in aspects:
    for traject_type in traject_types:
        if traject_type not in trajectory_groups[aspect]:
            continue
        subset_data[aspect][traject_type] = list(
            trajectory_groups[aspect][traject_type].index
        )
        tids_subset.update(subset_data[aspect][traject_type])
        for yid in subset_data[aspect][traject_type]:
            yid2src[yid] = f"{aspect}, {traject_type}"

for genre in genres_good:
    subset_data[genre] = list(genre_groups[genre].index)
    tids_subset.update(subset_data[genre])
    for yid in subset_data[genre]:
        yid2src[yid] = "genre"

yid2src = pd.DataFrame.from_dict(yid2src, orient="index")

st.write("## Ploting resulting sample in the A/V place")

data_c = data[data.index.isin(tids_subset)].copy()
data_c["source"] = yid2src


# av_models = ("emomusic", "muse", "deam")
av_models = ["emomusic"]
for dataset in av_models:
    fig, ax = plt.subplots()
    data_c[f"{dataset}-msd-musicnn-2---valence-norm"] = (
        data_c[f"{dataset}-msd-musicnn-2---valence"] - 5
    ) / 4
    data_c[f"{dataset}-msd-musicnn-2---arousal-norm"] = (
        data_c[f"{dataset}-msd-musicnn-2---arousal"] - 5
    ) / 4

    sns.scatterplot(
        data=data_c,
        x=f"{dataset}-msd-musicnn-2---valence-norm",
        y=f"{dataset}-msd-musicnn-2---arousal-norm",
        hue="source",
    ).set_title(dataset)
    plt.axvline(0, color="k")
    plt.axhline(0, color="k")
    st.pyplot(fig)

    plt.close(fig)

quad_rad_ss = {
    "A+V+": 0,
    "A-V+": -np.pi / 2,
    "A+V-": np.pi / 2,
    "A-V-": -np.pi,
}

quad_rad_es = {
    "A+V+": np.pi / 2,
    "A-V+": 0,
    "A+V-": np.pi,
    "A-V-": -np.pi / 2,
}

data_c["emomusic-msd-musicnn-2---av-polar-norm"] = [
    cmath.polar(
        complex(
            data_c["emomusic-msd-musicnn-2---valence-norm"][idx],
            data_c["emomusic-msd-musicnn-2---arousal-norm"][idx],
        )
    )
    for idx in data_c.index
]


def get_quadrant_ids(data, quadrant, field):
    quad_rad_s = quad_rad_ss[quadrant]
    quad_rad_e = quad_rad_es[quadrant]

    return data[data[field].apply(lambda x: x[1] > quad_rad_s and x[1] <= quad_rad_e)]


data_quadrants = {
    q: get_quadrant_ids(data_c, q, "emomusic-msd-musicnn-2---av-polar-norm")
    for q in ("A+V+", "A-V+", "A+V-", "A-V-")
}

for q, yids in data_quadrants.items():
    st.write(f"{q} has {len(yids)} ids.")


st.write("## Save resulting list of candidates")
with open("data/candidates.json", "w") as f:
    json.dump(subset_data, f)

st.write("## ok!")


# plot encode genre with color and AB curve/peaks with simbol, split in two plots/ AV trajectory and genres

# make player with tabs for every type of cluster of sounds with pages to listen/visualize the full selection of tracks.
