# manymusic-mtg-jamendo
Streamlit app to create subsets of The MTG Jamendo dataset


# Install

0. Clone this repo:  `git clone https://github.com/seunggookim/manymusic-mtg-jamendo.git`

1. Get [mtg-jamendo-dataset](https://github.com/MTG/mtg-jamendo-dataset) as a submodule:  `git submodule init && git submodule update`

2. Create a virtual env:  `python3 -m venv venv`
3. Activate a virtual env:
   - Mac/Linux: `source venv/bin/activate`
   - Windows: `source venv/Scripts/activate`

4. Install the Python dependencies:  `pip install -r requirements.txt`


## (not needed for annotation) Generation of the ManyMusic song pre-selectiion

1. Copy the data into `data/`.
The required files are: `mtg-jamendo-predictions-algos.pk`, `mtg-jamendo-predictions-av.pk`, ` mtg-jamendo-predictions.tsv`, and the timewise `predictions/`.

2. start the app: `streamlit run manymusic-viz.py`

3. The streamlit app will generate JSON file `data/clean_tids.json` with the candidate MTG Jamendo ids for the ManyMusic dataset. The resulting ids are randomly sampled from a pool of valid ids created with several filter staged where the threshold can be updated by the user.

4. Run `python clustering.py` to generate a dictionary of tids sampled by applying clustering to the tracks belonging to the different genres. 

5. Run `python postprocess.py` to generate a tsv combining several output jsons. Optionally, the resulting dataset can be split into equally sized chunks.

## Annotation of the ManyMusic song pre-selection

1. Go to the cloned directory and activate the virtual environment (VENV):
   - Mac/Linux: `source venv/bin/activate`
   - Windows: `source venv/Scripts/activate`
2. Run the script: `streamlit run manymusic-annotator.py`
