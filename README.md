# manymusic-mtg-jamendo
Streamlit app to create subsets of The MTG Jamendo dataset

# Install

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.py
```

## Usarge

1. Copy the data into `data/`.
The required files are: `mtg-jamendo-predictions-algos.pk`, `mtg-jamendo-predictions-av.pk`, ` mtg-jamendo-predictions.tsv`, and the timewise `predictions/`.

2. start the app: `streamlit run manymusic-viz.py`

3. The streamlit app will generate JSON file with the candidate MTG Jamendo ids for the ManyMusic dataset. The resulting ids are randomly sampled from a pool of valid ids created with several filter staged where the threshold can be updated by the user.
