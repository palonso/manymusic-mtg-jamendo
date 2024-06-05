# manymusic-mtg-jamendo
Streamlit app to create subsets of The MTG Jamendo dataset


# Install

1. Get [mtg-jamendo-dataset](https://github.com/MTG/mtg-jamendo-dataset) as a submodule:  `git submodule init && git submodule update`

2. Create and activate a virtual env: `python3 -m venv venv && source venv/bin/activate`

3. Install the Python dependencies: `pip install -r requirements.py`


## Usarge

1. Copy the data into `data/`.
The required files are: `mtg-jamendo-predictions-algos.pk`, `mtg-jamendo-predictions-av.pk`, ` mtg-jamendo-predictions.tsv`, and the timewise `predictions/`.

2. start the app: `streamlit run manymusic-viz.py`

3. The streamlit app will generate JSON file `data/clean_tids.json` with the candidate MTG Jamendo ids for the ManyMusic dataset. The resulting ids are randomly sampled from a pool of valid ids created with several filter staged where the threshold can be updated by the user.

4. Run `clustering.py` to generate a dictionary of tids sampled by applying clustering to the tracks belonging to the different genres. 

5. Run `postprocess.py` to generate a tsv combining several output jsons. Optionally, the resulting dataset can be split into equally sized chunks.
