import json
from argparse import ArgumentParser
import pandas as pd
from collections import defaultdict
from pathlib import Path
import numpy as np

# generate a tsv combining several output jsons. Optionally, the resulting dataset can be split into equally sized chunks.

parser = ArgumentParser()
parser.add_argument("jsons", nargs="+", help="List of json files to combine.")
parser.add_argument("--output", "-o", help="Output tsv file.")
parser.add_argument(
    "--chunk-size", "-c", type=int, help="Split the dataset into chunks of this size."
)

args = parser.parse_args()

clusters = ["av_cluster_0", "av_cluster_1", "av_cluster_2"]
dfs = []
for json_file in args.jsons:
    with open(json_file) as f:
        json_data = json.load(f)
        file_stem = Path(json_file).stem

        genres = json_data.keys()

        data = defaultdict(list)
        for genre, values in json_data.items():
            if isinstance(values, list):
                tids = values
                data["tid"].extend(tids)
                for genre_i in genres:
                    data[genre].extend([genre == genre_i] * len(tids))
                for cluster_i in clusters:
                    data[cluster_i].extend([False] * len(tids))
            else:
                for cluster, tids in values.items():
                    data["tid"].extend(tids)
                    for genre_i in genres:
                        data[genre_i].extend([genre_i == genre] * len(tids))
                    for cluster_i in clusters:
                        data[cluster_i].extend([cluster_i == cluster] * len(tids))

        df = pd.DataFrame(data)
        # df.set_index("tid", inplace=True)
        dfs.append(df)


df = pd.concat(dfs, ignore_index=True)
df = df.groupby(["tid"]).agg(lambda col: any(col)).reset_index()


if args.chunk_size:
    n_chunks = np.ceil(len(df) / args.chunk_size)
    chunk_ids = np.repeat(np.arange(n_chunks, dtype=int), args.chunk_size)[: len(df)]
    df = df.sample(frac=1, random_state=42)
    df["chunk_id"] = chunk_ids


# save the dataset
df.to_csv(args.output, sep="\t", index=False)
