import cmath
import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans
from matplotlib.dates import DateFormatter
from scipy.ndimage import gaussian_filter1d

sys.path.append("mtg-jamendo-dataset/scripts/")
import commons


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


def get_quadrant_ids(data, quadrant, field):
    """Get the ids of the samples in the specified quadrant."""

    quad_rad_ss = {"A+V+": 0, "A-V+": -np.pi / 2, "A+V-": np.pi / 2, "A-V-": -np.pi}
    quad_rad_es = {"A+V+": np.pi / 2, "A-V+": 0, "A+V-": np.pi, "A-V-": -np.pi / 2}

    quad_rad_s = quad_rad_ss[quadrant]
    quad_rad_e = quad_rad_es[quadrant]

    return data[data[field].apply(lambda x: x[1] > quad_rad_s and x[1] <= quad_rad_e)]


def load_av_time_data():
    """Load and prepare time-wise arousal and valence data in the streamlit cache."""
    data_av_time = dict()
    tids_clean_list = list(tids_clean)
    for index in tids_clean_list:
        try:
            av_filename = (av_predictions_dir / tracks[index]["path"]).with_suffix(
                ".npy"
            )
            # load and normalize
            data_av_time[index] = (np.load(av_filename) - 5) / 4
        except Exception:
            pass

    return data_av_time, set(tids_clean_list)


def plot_av(tid: int, axvline_loc: float = None, data: np.array = None):
    """Plot the arousal and valence curves for a given track id."""
    if data is None:
        sample = data_av_smooth[tid]
    else:
        sample = data

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
    plt.close()


def smooth_data(data: dict, sigma: int):
    """Smooth data using a gaussian filter."""
    return {k: gaussian_filter1d(sample, sigma, axis=0) for k, sample in data.items()}


parser = ArgumentParser()
parser.add_argument("--genre-threshold", type=float, default=0.1)
parser.add_argument("--n-samples-per-genre", type=int, default=200)
parser.add_argument("--smoothing-sigma", type=int, default=5)
parser.add_argument("--av-model", type=str, default="emomusic")
args = parser.parse_args()

genre_threshold = args.genre_threshold
n_samples_per_genre = args.n_samples_per_genre
smoothing_sigma = args.smoothing_sigma
av_model = args.av_model


data_dir = Path("data/")
av_predictions_dir = data_dir / "predictions" / "emomusic-msd-musicnn-2"
results_dir = (
    data_dir
    / f"clustering_genre_thres_{genre_threshold}_n_samples_{n_samples_per_genre}_smoothing_{smoothing_sigma}"
)

# Load ids
with open(data_dir / "clean_tids.json", "r") as f:
    tids_clean = set(json.load(f))


# Load data
data, tracks = load_data()

# Normalize AV
data[f"{av_model}-msd-musicnn-2---valence-norm"] = (
    data[f"{av_model}-msd-musicnn-2---valence"] - 5
) / 4
data[f"{av_model}-msd-musicnn-2---arousal-norm"] = (
    data[f"{av_model}-msd-musicnn-2---arousal"] - 5
) / 4

# Load AV timewise data
data_av_clean, tids_clean = load_av_time_data()
print(f"Kept {len(data_av_clean)} samples")

data_av_smooth = smooth_data(data_av_clean, smoothing_sigma)


data_styles = data.filter(like="genre_discogs400-discogs-effnet-1")
data_genres = data_styles.groupby(lambda x: x.split("---")[1], axis=1).max()
data_genres = data_genres[data_genres.index.isin(tids_clean)].copy()

genres = set(data_genres.columns)
genres_blacklist = set(["Non-Music", "Stage & Screen", "Children's"])
genres_good = genres - genres_blacklist


data_selected = dict()
for genre in list(genres_good):
    data_selected[genre] = dict()

    # Getting top activations for this genre
    data_genre = data_genres[data_genres[genre] > genre_threshold].copy()
    data_genre["source"] = "Not assigned"

    # Get AV data
    data_genre.loc[
        data_genre.index, f"{av_model}-msd-musicnn-2---valence-norm"
    ] = data.loc[data_genre.index, f"{av_model}-msd-musicnn-2---valence-norm"]
    data_genre.loc[
        data_genre.index, f"{av_model}-msd-musicnn-2---arousal-norm"
    ] = data.loc[data_genre.index, f"{av_model}-msd-musicnn-2---arousal-norm"]

    if len(data_genre) < n_samples_per_genre:
        print(f"Genre {genre} has {len(data_genre)} samples, using all of them.")
        data_selected[genre]["av_cluster_0"] = data_genre

    else:
        # get prototypical av curves for this genre
        data_av_genre = {
            k: v for k, v in data_av_smooth.items() if k in data_genre.index
        }
        tids_av_genre = list(data_av_genre.keys())
        data_av_genre_ts = to_time_series_dataset(list(data_av_genre.values()))

        n_clusters = 3
        n_samples_per_cluster = n_samples_per_genre // n_clusters

        print(f"training k-means for {genre} with {len(data_av_genre_ts)} samples")
        kmeans = TimeSeriesKMeans(
            n_clusters=n_clusters, metric="softdtw", max_iter_barycenter=10
        )
        y_distances = kmeans.fit_transform(data_av_genre_ts)

        np.save(results_dir / f"kmeans_centers_{genre}.npy", kmeans.cluster_centers_)

        sorting = np.argsort(y_distances, axis=0)
        indices = sorting[:n_samples_per_cluster, :]

        fig, ax = plt.subplots()

        for i_cluster in range(n_clusters):
            cluster_centroid = kmeans.cluster_centers_[i_cluster]
            cluster_centroid_mean = np.mean(cluster_centroid, axis=0)

            # plot_av(None, data=cluster_centroid)

            clust_sample_tids = [tids_av_genre[i] for i in indices[:, i_cluster]]
            data_selected[genre][f"av_cluster_{i_cluster}"] = data_genre.loc[
                clust_sample_tids
            ]

            data_genre.loc[clust_sample_tids, "source"] = f"C{i_cluster}"

            ax.annotate(
                f"C{i_cluster}", (cluster_centroid_mean[0], cluster_centroid_mean[1])
            )

        sns.scatterplot(
            data=data_genre,
            x=f"{av_model}-msd-musicnn-2---valence-norm",
            y=f"{av_model}-msd-musicnn-2---arousal-norm",
            hue="source",
        ).set_title(genre)
        plt.axvline(0, color="k")
        plt.axhline(0, color="k")

        results_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(results_dir / f"{genre}_av_scatter.png")
        plt.close(fig)

    data_genre["emomusic-msd-musicnn-2---av-polar-norm"] = [
        cmath.polar(
            complex(
                data_genre["emomusic-msd-musicnn-2---valence-norm"][idx],
                data_genre["emomusic-msd-musicnn-2---arousal-norm"][idx],
            )
        )
        for idx in data_genre.index
    ]

    data_quadrants = {
        q: get_quadrant_ids(data_genre, q, "emomusic-msd-musicnn-2---av-polar-norm")
        for q in ("A+V+", "A-V+", "A+V-", "A-V-")
    }

    for q, yids in data_quadrants.items():
        print(f"{q} has {len(yids)} ids.")


data_out = dict()
for k, v in data_selected.items():
    data_out[k] = dict()
    for k2, v2 in v.items():
        data_out[k][k2] = list(v2.index)

print("Save resulting list of candidates")
with open("data/candidates.json", "w") as f:
    json.dump(data_out, f)

print("done!")
