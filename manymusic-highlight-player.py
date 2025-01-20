import pickle as pk
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


LUFS_TARGET = -14


def wavesurfer_play(
    tid: str,
    gain: float = 1.0,
    start: int = 0,
    end: int = 120,
) -> None:
    """Play a track and print tags from its tid."""

    jamendo_url = f"https://mp3d.jamendo.com/?trackid={tid}&format=mp32#t={start},{end}"

    st.write("---")
    st.write(f"**Track {tid}**")

    html_code = f"""
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://unpkg.com/wavesurfer.js"></script>
    <style>
        #waveform {{
            width: 100%;
            height: 128px;
            margin: 0 auto;
        }}
        body {{
            text-align: center;
        }}
    </style>
</head>
<body>
    <div id="waveform"></div>

    <script type="text/javascript">
        document.addEventListener("DOMContentLoaded", function() {{
            var wavesurfer = WaveSurfer.create({{
                container: '#waveform',
                waveColor: 'violet',                progressColor: 'purple',
                height: 128,
                barWidth: 2
            }});

            // Load audio from a URL and autoplay
            wavesurfer.load('{jamendo_url}');
            wavesurfer.setVolume({gain});
            wavesurfer.setTime({start});
            wavesurfer.on('ready', function() {{
                wavesurfer.play();
            }});

            // Add play puse button
            var playButton = document.createElement('button');

            wavesurfer.on('play', function() {{
                playButton.innerHTML = 'Pause';
                playButton.style.position = 'center';
                playButton.style.fontSize = '20px';
                playButton.style.width = '100px';
                playButton.style.backgroundColor = 'white';
                playButton.style.border = '0px solid white';

                playButton.onclick = function() {{
                    wavesurfer.pause();
                    playButton.innerHTML = 'Play';
                    playButton.onclick = function() {{
                        wavesurfer.play();
                        playButton.innerHTML = 'Pause';
                    }};
                }};
                document.getElementById('waveform').appendChild(playButton);
            }});

            // Add event listeners for mouse events to redirect focus
            wavesurfer.on('interaction', function () {{
                document.activeElement.blur();  // Remove focus from the current element
                var focusElement = window.parent.document.querySelector('.main');
                focusElement.focus();
            }});
        }});
    </script>
</body>
</html>
"""

    # Embed the HTML code in the Streamlit app
    st.components.v1.html(html_code, height=170)


@st.cache_data
def load_data():
    """Load and prepare ground truth in the streamlit cache."""

    # mtg_jamendo_file = "mtg-jamendo-dataset/data/autotagging.tsv"
    # tracks, _, _ = commons.read_file(mtg_jamendo_file)

    lufs_file = "data/integrated_loudness.pk"
    lufs = pd.read_pickle(lufs_file)

    data_file = Path("data", "full_agreement_tracks.tsv")
    tids = pd.read_csv(data_file, sep="\t")
    tids = tids["track_id"].values.tolist()

    lufs.index = [str(i).split("/")[1] for i in lufs.index]

    av_data_file = Path("data", "data_av_clean.pk")
    with open(av_data_file, "rb") as f:
        av_data = pk.load(f)

    return tids, lufs, av_data


def get_tid():
    """Get the track id from the current index."""
    return str(TIDS[st.session_state.idx])


def next_idx():
    """Increment the index."""
    st.session_state.idx += 1


def previous_idx():
    """Reduce the index."""
    st.session_state.idx += -1


def get_gain(tid: str):
    """Compute the gain to reach the lufs target and return the gain."""
    # get value given index and column name
    lufs_tid = LUFS.loc[tid]

    # compute gain reduction to reach target loudness
    trim = LUFS_TARGET - lufs_tid

    # reduce gain if the track is too loud, otherwise keep it as it is
    if trim < 0:
        gain = 10 ** (trim / 20)
    else:
        gain = 1.0

    return gain


def get_highlight_timestamps(tid: str):
    dur_tgt = 15
    dur_av = 1
    k_size = dur_tgt // dur_av

    x = AV_DATA[int(tid)]
    x_avg_a, x_avg_v = np.mean(x, axis=0)

    # compute the average for x with a kernel of size 5
    x_a_smooth = np.convolve(x[:, 0], np.ones(k_size) / k_size, mode="same")
    x_v_smooth = np.convolve(x[:, 1], np.ones(k_size) / k_size, mode="same")

    a_a_diff = np.abs(x_a_smooth - x_avg_a)
    a_v_diff = np.abs(x_v_smooth - x_avg_v)

    avg_diff = np.mean([a_a_diff, a_v_diff], axis=0)
    middle_frame = np.argmin(avg_diff)

    start = int(middle_frame - k_size // 2)
    end = int(middle_frame + k_size // 2)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x)
    ax.axvline(start, color="red")
    ax.axvline(end, color="red")
    ax.set_title(f"Track {tid} highlight")

    st.pyplot(fig)

    return start, end


# init
if "idx" not in st.session_state:
    st.session_state.idx = 0

TIDS, LUFS, AV_DATA = load_data()

# get track info and play
tid = get_tid()
gain = get_gain(tid)

start, end = get_highlight_timestamps(tid)

wavesurfer_play(tid, gain=gain, start=start, end=end)


# manage buttons
prev_disabled = False
if st.session_state.idx == 0:
    prev_disabled = True

next_disabled = False
if st.session_state.idx == len(TIDS) - 1:
    next_disabled = True

cols = st.columns(4)
cols[0].button("previous", on_click=previous_idx, disabled=prev_disabled)
cols[3].button("next", on_click=next_idx, disabled=next_disabled)
