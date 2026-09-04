"""
video_toolkit.py
=================
Free, unlimited, local video generation toolkit. No paid APIs, no GPU required.

Two modes:
  1. slideshow  -> turn a text script (+ optional images) into a narrated,
                   captioned, full-length video (Ken Burns pan/zoom + TTS + subtitles)
  2. mp3video   -> turn any mp3/wav into a video with an animated waveform /
                   spectrum visualizer (lyric-video / podcast-video style)

Requirements: pip install -r requirements.txt
(gTTS needs an internet connection to fetch speech audio; everything else is offline)

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
Slideshow from a script file (one scene per blank-line-separated paragraph):
    python video_toolkit.py slideshow --script script.txt --out out.mp4

Slideshow using your own images (one per scene, in order):
    python video_toolkit.py slideshow --script script.txt --images img1.jpg img2.jpg --out out.mp4

MP3 -> MP4 with waveform visualizer:
    python video_toolkit.py mp3video --audio song.mp3 --out song.mp4 --title "My Song"
--------------------------------------------------------------------------
"""

import argparse
import os
import sys
import shutil
import tempfile
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from moviepy.editor import (
    ImageClip, AudioFileClip, CompositeVideoClip, concatenate_videoclips,
    TextClip, VideoClip, CompositeAudioClip
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _load_font(size):
    """Try a few common fonts, fall back to PIL default if none exist."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "DejaVuSans-Bold.ttf",
    ]
    for c in candidates:
        try:
            return ImageFont.truetype(c, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _wrap_text(text, width_chars=40):
    return "\n".join(textwrap.wrap(text, width=width_chars))


def make_caption_image(text, size=(1280, 200), font_size=44):
    """Render a semi-transparent caption bar with white text + black outline."""
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = _load_font(font_size)
    wrapped = _wrap_text(text, width_chars=42)

    bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, align="center")
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (size[0] - tw) / 2
    y = (size[1] - th) / 2

    # background bar
    draw.rectangle([0, 0, size[0], size[1]], fill=(0, 0, 0, 140))

    # outline for readability
    for dx in (-2, 0, 2):
        for dy in (-2, 0, 2):
            if dx or dy:
                draw.multiline_text((x + dx, y + dy), wrapped, font=font,
                                     fill=(0, 0, 0, 255), align="center")
    draw.multiline_text((x, y), wrapped, font=font, fill=(255, 255, 255, 255), align="center")
    return np.array(img)


def make_placeholder_image(text, size=(1280, 720), seed=0):
    """Generate a simple gradient background with the scene text on it,
    used when the user hasn't supplied their own images."""
    rng = np.random.RandomState(seed)
    c1 = rng.randint(30, 90, size=3)
    c2 = rng.randint(90, 180, size=3)
    w, h = size
    grad = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        t = y / h
        grad[y, :, :] = (c1 * (1 - t) + c2 * t).astype(np.uint8)
    img = Image.fromarray(grad, "RGB").convert("RGBA")

    draw = ImageDraw.Draw(img)
    font = _load_font(56)
    wrapped = _wrap_text(text, width_chars=28)
    bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, align="center")
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x, y = (w - tw) / 2, (h - th) / 2
    for dx in (-2, 0, 2):
        for dy in (-2, 0, 2):
            if dx or dy:
                draw.multiline_text((x + dx, y + dy), wrapped, font=font,
                                     fill=(0, 0, 0, 255), align="center")
    draw.multiline_text((x, y), wrapped, font=font, fill=(255, 255, 255, 255), align="center")
    return img.convert("RGB")


def ken_burns_clip(pil_image, duration, zoom_start=1.0, zoom_end=1.15, size=(1280, 720)):
    """Slow pan/zoom effect on a still image, like classic documentary slideshows."""
    base = pil_image.resize(size).convert("RGB")
    arr = np.array(base)

    def make_frame(t):
        p = t / duration if duration > 0 else 0
        zoom = zoom_start + (zoom_end - zoom_start) * p
        h, w, _ = arr.shape
        new_w, new_h = int(w / zoom), int(h / zoom)
        x0 = (w - new_w) // 2
        y0 = (h - new_h) // 2
        cropped = arr[y0:y0 + new_h, x0:x0 + new_w]
        frame = np.array(Image.fromarray(cropped).resize(size))
        return frame

    return VideoClip(make_frame, duration=duration)


# ---------------------------------------------------------------------------
# Mode 1: slideshow (script -> narrated, captioned, full-length video)
# ---------------------------------------------------------------------------

def parse_script(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    scenes = [s.strip() for s in raw.split("\n\n") if s.strip()]
    return scenes


def tts_for_scene(text, out_path, lang="en"):
    """Free, unlimited narration using gTTS (needs internet). Swap for pyttsx3
    if you need fully offline narration (lower quality, but zero network calls)."""
    from gtts import gTTS  # noqa (import kept local in case gTTS isn't installed for mp3video-only use)
    tts = gTTS(text=text, lang=lang)
    tts.save(out_path)


def build_slideshow(script_path, images, out_path, size=(1280, 720), fps=24, offline_tts=False):
    scenes = parse_script(script_path)
    if not scenes:
        raise ValueError("Script file has no scenes (separate scenes with a blank line).")

    tmp_dir = tempfile.mkdtemp(prefix="slideshow_")
    clips = []

    try:
        for i, scene_text in enumerate(scenes):
            audio_path = os.path.join(tmp_dir, f"scene_{i}.mp3")

            if offline_tts:
                import pyttsx3
                engine = pyttsx3.init()
                engine.save_to_file(scene_text, audio_path.replace(".mp3", ".wav"))
                engine.runAndWait()
                audio_path = audio_path.replace(".mp3", ".wav")
            else:
                tts_for_scene(scene_text, audio_path)

            audio_clip = AudioFileClip(audio_path)
            duration = max(audio_clip.duration + 0.6, 2.0)  # small padding

            if images and i < len(images):
                pil_img = Image.open(images[i]).convert("RGB")
            else:
                pil_img = make_placeholder_image(scene_text, size=size, seed=i)

            visual = ken_burns_clip(pil_img, duration, size=size)
            caption_arr = make_caption_image(scene_text, size=(size[0], 200))
            caption_clip = (ImageClip(caption_arr)
                             .set_duration(duration)
                             .set_position(("center", "bottom")))

            scene_clip = (CompositeVideoClip([visual, caption_clip], size=size)
                          .set_audio(audio_clip)
                          .set_duration(duration))
            clips.append(scene_clip)

        final = concatenate_videoclips(clips, method="compose")
        final.write_videofile(out_path, fps=fps, codec="libx264", audio_codec="aac",
                               threads=4, preset="medium")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Mode 2: mp3 -> mp4 (waveform / spectrum visualizer video)
# ---------------------------------------------------------------------------

def build_waveform_video(audio_path, out_path, title=None, size=(1280, 720), fps=24,
                          bg_color=(15, 15, 25), bar_color=(80, 200, 255)):
    import librosa

    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    n_bars = 60
    hop = max(1, len(y) // (int(duration * fps) + 1))

    # Precompute a smoothed amplitude envelope per bar, per frame
    def frame_bars(t):
        center = int(t * sr)
        window = y[max(0, center - hop * 4): center + hop * 4]
        if len(window) == 0:
            return np.zeros(n_bars)
        chunks = np.array_split(window, n_bars)
        levels = np.array([np.sqrt(np.mean(c ** 2)) if len(c) else 0 for c in chunks])
        levels = levels / (levels.max() + 1e-6)
        return levels

    def make_frame(t):
        img = Image.new("RGB", size, bg_color)
        draw = ImageDraw.Draw(img)
        levels = frame_bars(t)
        bar_w = size[0] / n_bars
        max_h = size[1] * 0.6
        base_y = size[1] * 0.75
        for i, lvl in enumerate(levels):
            h = max(4, lvl * max_h)
            x0 = i * bar_w + 2
            x1 = (i + 1) * bar_w - 2
            y0 = base_y - h
            y1 = base_y + h * 0.25
            draw.rectangle([x0, y0, x1, y1], fill=bar_color)
        if title:
            font = _load_font(48)
            bbox = draw.textbbox((0, 0), title, font=font)
            tw = bbox[2] - bbox[0]
            draw.text(((size[0] - tw) / 2, size[1] * 0.08), title, font=font, fill=(255, 255, 255))
        return np.array(img)

    visual = VideoClip(make_frame, duration=duration)
    audio_clip = AudioFileClip(audio_path)
    final = visual.set_audio(audio_clip)
    final.write_videofile(out_path, fps=fps, codec="libx264", audio_codec="aac",
                           threads=4, preset="medium")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Free local video generation toolkit")
    sub = parser.add_subparsers(dest="mode", required=True)

    p1 = sub.add_parser("slideshow", help="Script -> narrated captioned video")
    p1.add_argument("--script", required=True, help="Text file, scenes separated by blank lines")
    p1.add_argument("--images", nargs="*", default=None, help="Optional images, one per scene")
    p1.add_argument("--out", required=True)
    p1.add_argument("--offline-tts", action="store_true",
                     help="Use pyttsx3 (no internet) instead of gTTS")

    p2 = sub.add_parser("mp3video", help="Audio file -> waveform visualizer video")
    p2.add_argument("--audio", required=True)
    p2.add_argument("--out", required=True)
    p2.add_argument("--title", default=None)

    args = parser.parse_args()

    if args.mode == "slideshow":
        build_slideshow(args.script, args.images, args.out, offline_tts=args.offline_tts)
        print(f"Done -> {args.out}")
    elif args.mode == "mp3video":
        build_waveform_video(args.audio, args.out, title=args.title)
        print(f"Done -> {args.out}")


if __name__ == "__main__":
    main()
