import json
import sys
import uuid
from pathlib import Path
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

from utils import play

sys.path.append("mtg-jamendo-dataset/scripts/")
import commons


def generate_uuid():
    st.session_state.user_uuid = str(uuid.uuid4())


@st.cache_data
def load_data():
    """Load and prepare ground truth in the streamlit cache."""

    mtg_jamendo_file = "mtg-jamendo-dataset/data/autotagging.tsv"
    tracks, _, _ = commons.read_file(mtg_jamendo_file)
    return tracks


@st.cache_resource
def init():
    # Load ground truth data
    tracks = load_data()

    preselection_data = pd.read_csv(preselection_data_file, sep="\t")
    chunks = preselection_data["chunk_id"].unique()

    return tracks, preselection_data, chunks


@st.cache_resource(max_entries=1)
def retrieve_user_data(
    user_data_file: Path,
    preselection_data: pd.DataFrame,
    chunk_id: str,
) -> dict:
    """Retrieve user data from a file."""

    print(f"Retrieving user data, for chunk {chunk_id}.")

    # Get the tids for the selected chunk
    tids = list(
        preselection_data[preselection_data["chunk_id"] == int(chunk_id)]["tid"]
    )

    # Create a new annotation session
    new_session = {
        "start": datetime.now().isoformat(),
        "end": datetime.now().isoformat(),
        "chunk": chunk_id,
    }

    # Load user data
    if user_data_file.exists():
        with open(user_data_file, "r") as f:
            user_data = json.load(f)

        user_data["sessions"].append(new_session)
    # Initialise chunk dictionary otherwise
    else:
        user_data = {
            "annotations": dict(),
            "sessions": [new_session],
        }

    if chunk_id not in user_data["annotations"]:
        user_data["annotations"][chunk_id] = {k: dict() for k in tids}

    # Set the tid index
    tid_idx = 0
    if chunk_id in user_data["annotations"]:
        tid_idx = count_annotations(user_data["annotations"][chunk_id])

    if tid_idx > 0:
        st.write(f" Resuming annotation of chunk `{chunk_id}` at track `{tid_idx}`")

    st.session_state.tid_idx = tid_idx

    return user_data


def count_annotations(chunk_data: dict):
    """Count the number of annotations per chunk."""
    i = 0
    for v in chunk_data.values():
        if v:
            i += 1
        else:
            return i


def next_track(
    chunk_id: str,
    answer: str,
    tid: str,
):
    """Move to the next or previous track."""

    if answer == "previous":
        if st.session_state.tid_idx == 0:
            st.write("This is already the first track!")
        else:
            st.session_state.tid_idx -= 1
            st.write("Going to the previous track")
        return

    user_data["annotations"][chunk_id][tid] = {
        "answer": answer,
        "timestamp": str(datetime.now().isoformat()),
    }

    save_user_data(user_data, user_data_file)

    st.session_state.tid_idx += 1

    if answer == "all_good":
        st.write("All good! Next song.")

    elif answer == "bad_audio":
        st.write("Bad audio quality! Next song.")

    elif answer == "not_emotionally_conveying":
        st.write("Not emotionally conveying! Next song.")

    elif answer == "explicit_content":
        st.write("explicit content! Next song.")

    elif answer == "copyrighted_content":
        st.write("Copyrighted content detected! Next song.")

    elif answer == "other_reasons":
        st.write("Other reasons! Next song.")

    else:
        raise ValueError("Invalid answer.")


def save_user_data(user_data, user_data_file):
    """Save user data to a file."""

    # update the end timestamp
    user_data["sessions"][-1]["end"] = datetime.now().isoformat()

    if not user_data_file.parent.exists():
        user_data_file.parent.mkdir(parents=True)

    with open(user_data_file, "w") as f:
        json.dump(user_data, f)


choices = {
    "all_good": "✅ all good (a)",
    "bad_audio": "🔇 bad audio (s)",
    "not_emotionally_conveying": "😐 not emotional (d)",
    "explicit_content": "🤬 explicit content (f)",
    "copyrighted_content": "©️ copyrighted content (z)",
    "other_reasons": "👎 other reasons (x)",
}
choices_keys = list(choices.keys())

preselection_data_file = Path("data", "preselection.tsv")


# Generate or restore the UUID
if "user_uuid" not in st.session_state:
    st.session_state.user_uuid = None

if "tid_idx" not in st.session_state:
    st.session_state.tid_idx = 0

user_uuid = st.text_input("Insert an exisitng UUID ir create a new one.")

if user_uuid:
    st.session_state.user_uuid = user_uuid

st.button("Create UUID", on_click=generate_uuid)

if not st.session_state.user_uuid:
    st.write("Generate a user UUID or insert an existing one to continue.")
else:
    st.write(
        f"""User UUID: `{st.session_state.user_uuid}`

        Save your UUID to restore the annotation process latter.
        """
    )

    # main program
    user_data_file = Path("annotations", st.session_state.user_uuid, "annotations.json")
    tracks, preselection_data, chunks = init()

    chunk_id = st.selectbox("Select a chunk to annotate", chunks)
    chunk_id = str(chunk_id)

    st.caption(
        """
    The purpose of this annotation tool is to create a dataset of music suitable to evoke emotional responses.
    Our goal is to identify tracks that are unsuitable for this purpose and should be discarded.

    For every track, please select one of the following options:
    - `✅ all good!` The track has the potential to evoke emotions.
    - `🔇 bad audio quality` The track features audio quality problems that may interfere with the emotional response.
    - `😐 not emotionally conveying` the track is not emotionally conveying (e.g., elevator music, too repetitive, ...)
    - `🤬 explicit content` The track contains explicit content.
    - `©️  copyrighted content` The track contains copyrighted content from a recognized track.
    - `👎 other reasons` The track should not be included in the dataset for other reasons (contains irony, is attached to a specific event or ceremony, ...)
    """
    )

    # load user data
    user_data = retrieve_user_data(user_data_file, preselection_data, chunk_id)

    tids = list(user_data["annotations"][chunk_id].keys())
    st.write(f"Track `{st.session_state.tid_idx}/{len(tids)}`")

    if st.session_state.tid_idx >= len(tids):
        st.write(f"Chunk {chunk_id} completed.")
        st.stop()

    tid = int(tids[st.session_state.tid_idx])

    play(tid, tracks, autoplay=True)

    n_rows = 2
    n_cols = len(choices) // n_rows

    for row in range(n_rows):
        cols = st.columns(n_cols)
        for i, col in enumerate(cols):
            answer = choices_keys[i + row * n_cols]
            text = choices[answer]

            col.button(
                text,
                on_click=next_track,
                args=[chunk_id, answer, tid],
            )

    st.button(
        "⬅️  previous track",
        on_click=next_track,
        args=[chunk_id, "previous", tid],
    )

    save_user_data(user_data, user_data_file)


# Add keyboard shortcuts with JS
components.html(
    f"""
<script>
const doc = window.parent.document;
doc.addEventListener('keydown', function(e) {{
    switch (e.keyCode) {{
        case 65: // (65 = 'a' key) 
            const button_a = Array.from(doc.querySelectorAll('button'))
                                .find(btn => btn.innerText === '{choices["all_good"]}');
            if (button_a) {{
                button_a.click();
            }}
            break;

        case 83: // (83 = 's' key)
            const button_s = Array.from(doc.querySelectorAll('button'))
                                .find(btn => btn.innerText === '{choices["bad_audio"]}');
            if (button_s) {{
                button_s.click();
            }}
            break;

        case 68: // (68 = 'd' key)
            const button_d = Array.from(doc.querySelectorAll('button'))
                                .find(btn => btn.innerText === '{choices["not_emotionally_conveying"]}');
            if (button_d) {{
                button_d.click();
            }}
            break;

        case 70: // (70 = 'f' key)
            const button_f = Array.from(doc.querySelectorAll('button'))
                                .find(btn => btn.innerText === '{choices["explicit_content"]}');
            if (button_f) {{
                button_f.click();
            }}
            break;

        case 90: // (90 = 'z' key)
            const button_z = Array.from(doc.querySelectorAll('button'))
                                .find(btn => btn.innerText === '{choices["copyrighted_content"]}');
            if (button_z) {{
                button_z.click();
            }}
            break;

        case 88: // (88 = 'x' key)
            const button_x = Array.from(doc.querySelectorAll('button'))
                                .find(btn => btn.innerText === '{choices["other_reasons"]}');
            if (button_x) {{
                button_x.click();
            }}
            break;
    }}
}});
</script>

""",
    height=0,
    width=0,
)
