import cmath
import json
import sys
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans
from sklearn.metrics import silhouette_score

from utils import load_av_time_data, smooth_data, decimate_data, normalize_string

sys.path.append("mtg-jamendo-dataset/scripts/")
import commons


n_cluster_choices = [3, 5, 10]


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


parser = ArgumentParser()
parser.add_argument("--genre-threshold", type=float, default=0.1)
parser.add_argument("--n-samples-per-genre", type=int, default=200)
parser.add_argument("--smoothing-sigma", type=int, default=5)
parser.add_argument("--decimate-factor", type=int, default=5)
parser.add_argument("--av-model", type=str, default="emomusic")
parser.add_argument(
    "--norm", type=str, default="none", choices=["none", "minmax", "zscore"]
)
parser.add_argument("--force", action="store_true")
args = parser.parse_args()

genre_threshold = args.genre_threshold
n_samples_per_genre = args.n_samples_per_genre
smoothing_sigma = args.smoothing_sigma
decimate_factor = args.decimate_factor
av_model = args.av_model
norm_type = args.norm
force = args.force


data_dir = Path("data/")
av_predictions_dir = data_dir / "predictions" / "emomusic-msd-musicnn-2"
results_dir = (
    data_dir
    / "clustering"
    / f"clustering_genre_thres_{genre_threshold}_n_samples_{n_samples_per_genre}_smoothing_{smoothing_sigma}_decimate_{decimate_factor}_norm_{norm_type}"
)

results_dir.mkdir(parents=True, exist_ok=True)

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
data_av_clean, tids_clean = load_av_time_data(tids_clean, tracks)
print(f"Kept {len(data_av_clean)} samples")

data_av_smooth = smooth_data(data_av_clean, smoothing_sigma)

data_av_decimated = decimate_data(data_av_smooth, decimate_factor)


data_styles = data.filter(like="genre_discogs400-discogs-effnet-1")
data_genres = data_styles.groupby(lambda x: x.split("---")[1], axis=1).max()
data_genres = data_genres[data_genres.index.isin(tids_clean)].copy()

genres = set(data_genres.columns)
genres_blacklist = set(["Non-Music", "Stage & Screen", "Children's"])
genres_good = genres - genres_blacklist


data_selected = dict()
for genre in list(genres_good):
    genre_n = normalize_string(genre)
    results_file = results_dir / f"kmeans_centers_{genre_n}.npy"

    if results_file.exists() and not force:
        print(f"Skipping genre {genre}, already processed.")
        continue

    data_selected[genre] = dict()

    # Getting top activations for this genre
    data_genre = data_genres[data_genres[genre] > genre_threshold].copy()
    data_genre["source"] = "Not assigned"

    tids = list(data_genre.index)

    if len(tids) < n_samples_per_genre:
        print(f"Genre {genre} has {len(tids)} samples, using all of them.")
    else:
        for max_tracks_per_album in range(1, 10):
            print(f"Keeping {max_tracks_per_album} samples per album for genre {genre}")

            albums_dict = defaultdict(int)
            tids_album_duplicated = set()

            for tid in tids:
                album = tracks[tid]["album_id"]
                if albums_dict[album] >= max_tracks_per_album:
                    tids_album_duplicated.add(tid)
                albums_dict[album] += 1

            tids_clean = set(tids) - tids_album_duplicated

            if len(tids_clean) >= n_samples_per_genre:
                print(
                    f" {len(tids_clean)} samples, {max_tracks_per_album} tracks per album, enough."
                )
                break

        data_genre = data_genre.loc[list(tids_clean)]

    # Get AV data
    v_norm_field = f"{av_model}-msd-musicnn-2---valence-norm"
    a_norm_field = f"{av_model}-msd-musicnn-2---arousal-norm"

    data_genre.loc[data_genre.index, v_norm_field] = data.loc[
        data_genre.index, v_norm_field
    ]
    data_genre.loc[data_genre.index, a_norm_field] = data.loc[
        data_genre.index, a_norm_field
    ]

    if len(data_genre) < n_samples_per_genre:
        print(f"Genre {genre} has {len(data_genre)} samples, using all of them.")
        data_selected[genre]["av_cluster_0"] = data_genre

    else:
        # get prototypical av curves for this genre
        data_av_genre = {
            k: v for k, v in data_av_decimated.items() if k in data_genre.index
        }
        if norm_type == "zscore":
            data_av_genre = {
                k: (v - v.mean(axis=0)) / v.std(axis=0)
                for k, v in data_av_genre.items()
            }
        tids_av_genre = list(data_av_genre.keys())

        data_av_genre_ts = to_time_series_dataset(list(data_av_genre.values()))

        best_sil_score = -np.inf
        best_n_clusters = 0
        best_kmeans = None
        best_y_distances = None

        for n_clusters in n_cluster_choices:
            n_samples_per_cluster = n_samples_per_genre // n_clusters

            print(
                f"training k-means for {genre} with {len(data_av_genre_ts)} samples, and {n_clusters} clusters."
            )
            kmeans = TimeSeriesKMeans(
                n_clusters=n_clusters, metric="dtw", max_iter_barycenter=10
            )
            y_distances = kmeans.fit_transform(data_av_genre_ts)

            cluster_labels = kmeans.predict(data_av_genre_ts)

            # compute silhouette score on the time-averaged AV curves
            # (we have seen that av. values preserve most of the info).
            data_av_genre_ts_mean = np.array(
                [v.mean(axis=0) for v in data_av_genre.values()]
            )
            print("av. data shape", data_av_genre_ts_mean.shape)
            sil_score = silhouette_score(data_av_genre_ts_mean, cluster_labels)

            print(
                f"Silhouette score for {genre} with {n_clusters} clusters: {sil_score:.4f}"
            )

            if sil_score > best_sil_score:
                best_n_clusters = n_clusters
                best_sil_score = sil_score
                best_kmeans = kmeans
                best_y_distances = y_distances

            else:
                print("Silhouette score is not better than the best one, stopping.")
                print("Best silhouette score:", best_sil_score)
                print("Best number of clusters:", best_n_clusters)
                break

        np.save(
            results_file,
            best_kmeans.cluster_centers_,
        )

        n_samples_per_cluster = n_samples_per_genre // best_n_clusters

        sorting = np.argsort(best_y_distances, axis=0)
        indices = sorting[:n_samples_per_cluster, :]

        fig, ax = plt.subplots()

        for i_cluster in range(best_n_clusters):
            cluster_centroid = best_kmeans.cluster_centers_[i_cluster]
            cluster_centroid_mean = np.mean(cluster_centroid, axis=0)

            clust_sample_tids = [tids_av_genre[i] for i in indices[:, i_cluster]]
            data_selected[genre][f"av_cluster_{i_cluster}"] = data_genre.loc[
                clust_sample_tids
            ]

            data_genre.loc[clust_sample_tids, "source"] = f"av_cluster_{i_cluster}"

            # sort byt soruce column
            data_genre.sort_values("source", inplace=True)

            ax.annotate(
                f"C{i_cluster}",
                (cluster_centroid_mean[0], cluster_centroid_mean[1]),
            )

        sns.scatterplot(
            data=data_genre, x=v_norm_field, y=a_norm_field, hue="source"
        ).set_title(genre)

        plt.axvline(0, color="k")
        plt.axhline(0, color="k")

        results_dir.mkdir(parents=True, exist_ok=True)

        plt.savefig(results_dir / f"{genre_n}_av_scatter.png")
        plt.close(fig)

    p_norm_field = "emomusic-msd-musicnn-2---av-polar-norm"
    data_genre[p_norm_field] = [
        cmath.polar(
            complex(data_genre[v_norm_field][idx], data_genre[a_norm_field][idx])
        )
        for idx in data_genre.index
    ]

    data_quadrants = {
        q: get_quadrant_ids(data_genre, q, p_norm_field)
        for q in ("A+V+", "A-V+", "A+V-", "A-V-")
    }

    for q, yids in data_quadrants.items():
        print(f"{q} has {len(yids)} ids.")

results_file = results_dir / "candidates.json"
if results_file.exists():
    with open(results_file, "r") as f:
        data_out = json.load(f)
else:
    data_out = dict()

for k, v in data_selected.items():
    data_out[k] = dict()
    for k2, v2 in v.items():
        data_out[k][k2] = list(v2.index)

print("Save resulting list of candidates")
with open(results_dir / "candidates.json", "w") as f:
    json.dump(data_out, f)

print("done!")
