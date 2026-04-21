"""
Stage 4 — The "Hands" (Assembler)

Takes a Timeline JSON + the original audio file and renders the final MP4.

Steps:
  1. Normalize all source clips to TARGET_RESOLUTION @ TARGET_FPS via FFmpeg
  2. For each TimelineSegment, extract the sub-clip using MoviePy
  3. Apply speed correction (±MAX_SPEED_ADJUST) if clip duration ≠ beat gap
  4. Apply transition effect (cut / dissolve / zoom_in)
  5. Concatenate all segments
  6. Replace audio with original music track
  7. Export to output_path
"""

from __future__ import annotations

import os
from typing import List

from config import (
    DISSOLVE_DURATION,
    MAX_SPEED_ADJUST,
    TARGET_FPS,
    ZOOM_SCALE,
)
from utils.ffmpeg_utils import check_ffmpeg, normalize_all


def render(timeline: dict, music_path: str, output_path: str) -> None:
    """
    Render the final video from a timeline dict.
    Saves the output MP4 to output_path.
    """
    # Lazy import MoviePy so the CLI stays fast on --dry-run
    from moviepy import (
        AudioFileClip,
        CompositeVideoClip,
        VideoFileClip,
        concatenate_videoclips,
    )

    check_ffmpeg()

    segments = timeline.get("segments", [])
    if not segments:
        raise ValueError("Timeline has no segments. Run the director stage first.")

    print(f"[Stage 4] Assembling {len(segments)} segment(s) → {output_path}")

    # ── Step 1: Normalize all source clips ────────────────────────────────────
    unique_paths = list({s["clip_path"] for s in segments})
    print(f"          Normalizing {len(unique_paths)} unique clip(s)…")
    norm_map = normalize_all(unique_paths)

    # ── Step 2–4: Build each sub-clip with transitions ─────────────────────
    clips = []
    for i, seg in enumerate(segments):
        src_path = norm_map[seg["clip_path"]]
        music_gap = seg["music_end"] - seg["music_start"]
        clip_gap = seg["clip_end"] - seg["clip_start"]

        raw = VideoFileClip(src_path).subclipped(seg["clip_start"], seg["clip_end"])

        # Speed-correct if the clip is slightly too short or too long
        speed_ratio = clip_gap / music_gap if music_gap > 0 else 1.0
        max_adj = 1 + MAX_SPEED_ADJUST
        min_adj = 1 - MAX_SPEED_ADJUST
        if not (min_adj <= speed_ratio <= max_adj):
            # Clamp — if difference is too large, trim/pad instead
            speed_ratio = max(min_adj, min(max_adj, speed_ratio))

        if abs(speed_ratio - 1.0) > 0.01:
            raw = raw.with_speed_scaled(speed_ratio)

        # Force exact duration to match beat gap
        raw = raw.with_duration(music_gap)

        # Apply transition
        transition = seg.get("transition", "cut")
        clip = _apply_transition(raw, transition, i)

        clips.append(clip)
        print(f"          [{i + 1}/{len(segments)}] {os.path.basename(src_path)} "
              f"[{seg['clip_start']:.1f}–{seg['clip_end']:.1f}s] "
              f"→ music [{seg['music_start']:.1f}–{seg['music_end']:.1f}s] "
              f"({transition})")

    # ── Step 5: Concatenate ───────────────────────────────────────────────────
    print("          Concatenating clips…")
    final = concatenate_videoclips(clips, method="compose")

    # ── Step 6: Replace audio ─────────────────────────────────────────────────
    print("          Adding music track…")
    music = AudioFileClip(music_path).with_duration(final.duration)
    final = final.with_audio(music)

    # ── Step 7: Export ────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    print(f"          Rendering → {output_path}")
    final.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        fps=TARGET_FPS,
        preset="medium",
        logger="bar",
    )

    # Clean up
    final.close()
    music.close()
    for clip in clips:
        clip.close()

    print(f"          Done. Output: {output_path}")


# ── Transition helpers ────────────────────────────────────────────────────────

def _apply_transition(clip, transition: str, index: int):
    """
    Apply a visual transition to a clip.
      cut      → no effect (hard cut on beat)
      dissolve → cross-fade (handled by concatenate_videoclips with crossfade)
      zoom_in  → subtle Ken Burns zoom from 1.0 to ZOOM_SCALE
    """
    if transition == "cut":
        return clip

    if transition == "dissolve":
        # Add crossfade_in — concatenate_videoclips will blend adjacent clips
        return clip.with_effects([_CrossFadeIn(DISSOLVE_DURATION)])

    if transition == "zoom_in":
        return _zoom_effect(clip)

    return clip


def _zoom_effect(clip):
    """Gradually zoom from 1.0× to ZOOM_SCALE× over the clip duration."""
    import numpy as np

    duration = clip.duration

    def zoom_frame(get_frame, t):
        frame = get_frame(t)
        progress = t / duration if duration > 0 else 0
        scale = 1.0 + (ZOOM_SCALE - 1.0) * progress
        h, w = frame.shape[:2]
        new_h = int(h / scale)
        new_w = int(w / scale)
        top = (h - new_h) // 2
        left = (w - new_w) // 2
        cropped = frame[top:top + new_h, left:left + new_w]
        # Resize back to original dimensions using OpenCV
        import cv2
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    return clip.transform(zoom_frame)


class _CrossFadeIn:
    """MoviePy effect: fade in from black over `duration` seconds."""

    def __init__(self, duration: float):
        self.duration = duration

    def apply(self, clip):
        from moviepy import vfx
        return clip.with_effects([vfx.CrossFadeIn(self.duration)])
