import sys
import streamlit as st
import uuid
from collections import Counter
from utils import play
import streamlit.components.v1 as components

sys.path.append("mtg-jamendo-dataset/scripts/")
import commons


def generate_uuid():
    st.session_state.user_uuid = uuid.uuid4()


@st.cache_data
def load_data():
    """Load and prepare ground truth in the streamlit cache."""

    mtg_jamendo_file = "mtg-jamendo-dataset/data/autotagging.tsv"
    tracks, _, _ = commons.read_file(mtg_jamendo_file)
    return tracks


def get_top_tags(tids: list, n_most_common: int = 5):
    """Get the top tags for a list of tids."""
    tags = []
    for tid in tids:
        track = tracks[tid]
        tags += [t.split("---")[1] for t in track["tags"]]

    return Counter(tags).most_common(n_most_common)


def next_track(answer: str):
    st.session_state.tid_idx += 1

    if answer == "all_good":
        st.write("All good! Next song.")

    elif answer == "bad_quality":
        st.write("Bad quality! Next song.")

    elif answer == "not_emotionally_conveying":
        st.write("Not emotionally conveying! Next song.")

    elif answer == "other_reasons":
        st.write("Other reasons! Next song.")

    else:
        raise ValueError("Invalid answer.")


if "user_uuid" not in st.session_state:
    st.session_state.user_uuid = None

if "tid_idx" not in st.session_state:
    st.session_state.tid_idx = 0

choices = {
    "all_good": "✅ all good! (a)",
    "bad_quality": "🔇 bad quality (s)",
    "not_emotionally_conveying": "😐 not emotionally conveying (d)",
    "other_reasons": "👎 other reasons (f)",
}
choices_keys = list(choices.keys())


st.write(
    """
    Insert UUID or genere a new one
    """
)

user_uuid = st.text_input("User UUID")

if user_uuid:
    st.session_state.user_uuid = user_uuid


st.button("Generate UUID", on_click=generate_uuid)

if not st.session_state.user_uuid:
    st.write("Generate a user UUID or insert an existing one.")
else:
    st.write(
        f"""User UUID: `{st.session_state.user_uuid}`. 

        Save it for future reference.
        """
    )

    st.write("Annotate this song")

    tracks = load_data()
    tids = list(tracks.keys())

    tid = tids[st.session_state.tid_idx]

    play(tid, tracks)

    cols = st.columns(len(choices))
    for i, col in enumerate(cols):
        key = choices_keys[i]
        text = choices[key]

        col.button(
            text,
            on_click=next_track,
            args=[key],
        )


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
                                .find(btn => btn.innerText === '{choices["bad_quality"]}');
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
                                .find(btn => btn.innerText === '{choices["other_reasons"]}');
            if (button_f) {{
                button_f.click();
            }}
            break;
    }}
}});
</script>

""",
    height=0,
    width=0,
)
