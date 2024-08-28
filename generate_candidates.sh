set -e

source venv/bin/activate


# streamlit run manymusic-viz.py
#
n_samples=170

python clustering.py --norm none --n-samples-per-genre $n_samples
python clustering.py --norm zscore --n-samples-per-genre ${n_samples}

python postprocess.py \
    data/clustering/clustering_genre_thres_0.1_n_samples_${n_samples}_smoothing_5_decimate_5_norm_none/candidates.json \
    data/clustering/clustering_genre_thres_0.1_n_samples_${n_samples}_smoothing_5_decimate_5_norm_zscore/candidates.json \
    --output data/candidates.tsv --chunk-size 200


