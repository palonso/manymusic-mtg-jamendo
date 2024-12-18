import pandas as pd

feature_file = "data/mtg-jamendo-predictions-algos.pk"
features = pd.read_pickle(feature_file)


features = features["integrated_loudness"]

features.to_pickle("data/integrated_loudness.pk")
