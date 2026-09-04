# Free Local Video Toolkit

Generate full-length videos with **zero cost and no clip-length limit** — everything runs
on CPU with open-source libraries. No paid API keys required.

## What it does

| Mode | Input | Output |
|---|---|---|
| `slideshow` | A text script (+ optional images) | Narrated, captioned, Ken-Burns-style video — as long as your script is |
| `mp3video` | Any mp3/wav file | Animated waveform visualizer video synced to the audio |

## Honest limits (read this first)

- This does **not** do true AI-generated video frames (like Sora/Runway) — that needs a
  GPU with a lot of VRAM and no free tool does it "unlimited." This toolkit instead builds
  videos from **narration + images + motion + captions**, which is how most faceless
  YouTube/explainer channels actually make long videos anyway.
- `gTTS` (the default narrator) needs an internet connection (it's free, no API key,
  no rate limit for normal use) — fine on Colab. If you want fully offline narration,
  pass `--offline-tts` (uses `pyttsx3`, robotic but zero network calls).
- Video length is unlimited — it's just your script length. A 10-page script becomes a
  10+ minute video automatically.

## Setup (Google Colab — free, no GPU needed)

```python
!apt-get -qq install -y fonts-dejavu ffmpeg
!pip install -q -r requirements.txt
```

Upload `video_toolkit.py` and your script/images/audio to the Colab file panel, then run
from a cell:

```python
!python video_toolkit.py slideshow --script script.txt --out my_video.mp4
```

## Setup (your own machine)

```bash
pip install -r requirements.txt
# make sure ffmpeg is installed: sudo apt install ffmpeg  (Linux) / brew install ffmpeg (Mac)
```

## Usage

### 1. Script → narrated video
Write `script.txt` with one scene per paragraph (blank line = new scene):

```
The sun rises over a quiet mountain village.

A lone farmer walks toward the fields, tools in hand.

By noon, the whole village is alive with work and laughter.
```

```bash
python video_toolkit.py slideshow --script script.txt --out out.mp4
```

Optionally supply your own images (one per scene, same order):
```bash
python video_toolkit.py slideshow --script script.txt --images s1.jpg s2.jpg s3.jpg --out out.mp4
```

Fully offline (no internet for narration):
```bash
python video_toolkit.py slideshow --script script.txt --out out.mp4 --offline-tts
```

### 2. MP3 → visualizer video
```bash
python video_toolkit.py mp3video --audio song.mp3 --out song.mp4 --title "My Song"
```

## Extending this

- **Better visuals per scene**: swap `make_placeholder_image()` for calls to a free local
  image generator (e.g. a small Stable Diffusion checkpoint) if you get GPU access later —
  the rest of the pipeline (TTS, captions, stitching) doesn't need to change.
- **Real captions synced to speech**: for word-level timed captions instead of one caption
  per scene, run the narration audio through `faster-whisper` (CPU-friendly) and use its
  word timestamps to drive the caption clip timing.
- **Background music**: layer a second `AudioFileClip` under the narration with
  `CompositeAudioClip([narration, music.volumex(0.15)])`.
