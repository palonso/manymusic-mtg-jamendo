import json
import random
import sys
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


sys.path.append("mtg-jamendo-dataset/scripts/")
import commons

data_dir = Path("data/")


aspects = ("arousal", "valence")
traject_types = ("ascending", "descending", "peaks", "climax")


example_size = 3


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


st.write(
    """
# ManyMusic Dataset
## Interactive Visualization of MTG Jamendo Dataset
"""
)

data, tracks = load_data()
tids_init = set(tracks.keys())
tids_clean = tids_init


st.write(
    """
## 1. Track duration filtering
"""
)

durations = np.array([v["duration"] for v in tracks.values()])
max_duration = int(np.max(durations) + 0.5)

dur_min, dur_max = st.slider("Minimum duration (seconds)", 0, max_duration, (180, 420))

tids_short = {tid for tid, values in tracks.items() if values["duration"] < dur_min}
tids_long = {tid for tid, values in tracks.items() if values["duration"] > dur_max}

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
    ## 6. Remove tracks with blacklisted styles
    """
)

act_thres = st.slider(
    "Min value of the top blacklisted classes to discard the track", 0.0, 1.0, 0.1
)
styles_blacklisted = ("Non-Music", "Chiptune")
for style in styles_blacklisted:
    data_style = data.filter(like=style)
    data_style_present = data_style[data_style.max(axis=1) > act_thres]

    tids_style = set(data_style_present.index)
    tids_clean -= tids_style

    st.write(
        f"""
        {style} tracks: {len(tids_style)}

        remaining tracks: {len(tids_clean)}
        """
    )

    st.write("Examples of discarded tracks")

    for tid in random.sample(list(tids_style), example_size):
        play(tid)
        st.dataframe(data_style_present.loc[tid].nlargest(3))


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


# st.write(
#     """
#     ## 8. Remove tracks with too much/little AV dispersion
#     """
# )

# data_av_std = {i: np.std(v, axis=0) for i, v in data_av_time.items()}

# low_p, high_p = st.slider("Arousal/Valence range (percentile)", 0, 100, (10, 90))

# data_av_perc = {
#     i: np.percentile(v, high_p, axis=0) - np.percentile(v, low_p, axis=0)
#     for i, v in data_av_time.items()
# }

# thres_av_disp = st.slider("Arousal/Valence threshold (percentile)", 0.0, 1.0, 0.15)

# tids_av_disp_low = {i for i, v in data_av_perc.items() if (v < thres_av_disp).any()}
# tids_clean -= tids_av_disp_low

# st.write(
#     f"""
#     tracks with low A/V disperssion: {len(tids_av_disp_low)}

#     remaining tracks: {len(tids_clean)}

#     Examples of tracks with low A/V dispersion:
#     """
# )
# for tid in random.sample(list(tids_av_disp_low), example_size):
#     play(tid)


st.write(
    """
    ## 9. Keep one track per album
    """
)

albums_set = set()
albums_list = list()
tids_album_duplicated = set()
for tid in tids_clean:
    album = tracks[tid]["album_id"]
    if album in albums_set:
        tids_album_duplicated.add(tid)
    albums_set.add(album)
    albums_list.append(album)

tids_clean -= tids_album_duplicated
albums_counter = Counter(albums_list)

top_albums = albums_counter.most_common(10)


st.write(
    f"""
    tracks belonging to the same album: {len(tids_album_duplicated)}

    remaining tracks: {len(tids_clean)}

    Top 10 albums with more tracks:
    {top_albums}
    """
)


data_styles = data.filter(like="genre_discogs400-discogs-effnet-1")

data_genres = data_styles.groupby(lambda x: x.split("---")[1], axis=1).max()
data_genres = data_genres[data_genres.index.isin(tids_clean)].copy()

genres = set(data_genres.columns)
genres_blacklist = set(["Non-Music", "Stage & Screen", "Children's"])
genres_good = genres - genres_blacklist

st.write(f"Saving {len(tids_clean)} clean tids")

with open(data_dir / "clean_tids.json", "w") as f:
    json.dump(list(tids_clean), f)
