from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.dates import DateFormatter
import re

from scipy.ndimage import gaussian_filter1d
from scipy.signal import decimate


def load_av_time_data(
    tids: set,
    tracks: dict,
    av_predictions_dir: Path = Path("data/predictions/emomusic-msd-musicnn-2/"),
) -> tuple[dict, set]:
    """Load and prepare time-wise arousal and valence data in the streamlit cache."""

    data_av_time = dict()
    tids_list = list(tids)
    for index in tids_list:
        try:
            av_filename = (av_predictions_dir / tracks[index]["path"]).with_suffix(
                ".npy"
            )
            # load and normalize
            array = np.load(av_filename)
            data_av_time[index] = (array - 5) / 4

        except Exception:
            pass

    return data_av_time, set(tids_list)


def smooth_data(data: dict, sigma: int = 5) -> dict:
    """Smooth data using a gaussian filter."""

    return {k: gaussian_filter1d(sample, sigma, axis=0) for k, sample in data.items()}


def decimate_data(data: dict, factor: int = 5) -> dict:
    """Downsample data using ."""

    return {k: decimate(sample, factor, axis=0) for k, sample in data.items()}


def audio_url(trackid) -> str:
    """Return the Jamendo URL for a given trackid."""

    return f"https://mp3d.jamendo.com/?trackid={trackid}&format=mp32#t=0,120"


def play(tid: str, tracks: dict, autoplay: bool = False) -> None:
    """Play a track and print tags from its tid."""

    jamendo_url = audio_url(tid)
    # track = tracks[tid]
    # tags = [t.split("---")[1] for t in track["tags"]]

    st.write("---")
    st.write(f"**Track {tid}**")

    st.audio(jamendo_url, format="audio/mp3", start_time=0, autoplay=autoplay)


def wavesurfer_play(
    tid: str,
    tracks: dict,
    autoplay: bool = False,
    gain: float = 1.0,
) -> None:
    """Play a track and print tags from its tid."""

    jamendo_url = audio_url(tid)
    # track = tracks[tid]
    # tags = [t.split("---")[1] for t in track["tags"]]

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
                waveColor: 'violet',
                progressColor: 'purple',
                height: 128,
                barWidth: 2
            }});

            // Load audio from a URL and autoplay
            wavesurfer.load('{jamendo_url}');
            wavesurfer.setVolume({gain});
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


def plot_av(
    sample: np.ndarray,
    axvline_loc: float = 0,
) -> None:
    """Plot the arousal and valence curves for a given track id."""

    formatter = DateFormatter("%M'%S''")

    emb2days = 63 * 256 / (16000 * 3600 * 24)
    time = np.linspace(0, len(sample) * emb2days, len(sample))

    fig, ax = plt.subplots(figsize=(10, 2))
    ax.plot(time, sample[:, 0], label="valence")
    ax.plot(time, sample[:, 1], label="arousal")
    ax.xaxis.set_major_formatter(formatter)

    if axvline_loc != 0:
        axvline_loc *= emb2days
        label = f"location: {formatter(axvline_loc)}"
        plt.axvline(axvline_loc, color="k", label=label)

    ax.legend()
    fig.tight_layout()
    plt.grid()
    plt.ylim(-1, 1)
    st.pyplot(fig)
    plt.close()


def normalize_string(s: str) -> str:
    """Normalize a string for search purposes."""

    return re.sub("[^A-Za-z0-9]+", "_", s)
