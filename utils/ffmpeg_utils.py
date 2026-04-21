"""
FFmpeg utility helpers.

Handles:
  - Normalizing raw clips to a standard resolution/fps before processing
  - Checking that FFmpeg is available on PATH
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from config import TARGET_FPS, TARGET_RESOLUTION, TEMP_DIR


def check_ffmpeg() -> None:
    """Raise RuntimeError if FFmpeg is not available on PATH."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        raise RuntimeError(
            "FFmpeg not found on PATH.\n"
            "Install it with: winget install FFmpeg\n"
            "Then restart your terminal."
        )


def normalize_clip(input_path: str, force: bool = False) -> str:
    """
    Transcode a clip to TARGET_RESOLUTION @ TARGET_FPS using FFmpeg.
    Output is placed in TEMP_DIR with the same filename.
    Returns the path to the normalized file.

    If the normalized file already exists and force=False, skip re-encoding.
    """
    os.makedirs(TEMP_DIR, exist_ok=True)
    filename = os.path.basename(input_path)
    # Use .mp4 for all normalized output regardless of source extension
    stem = Path(filename).stem
    out_path = os.path.join(TEMP_DIR, f"{stem}_norm.mp4")

    if os.path.exists(out_path) and not force:
        return out_path

    width, height = TARGET_RESOLUTION
    # scale + pad to exact resolution, preserving aspect ratio
    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", vf,
        "-r", str(TARGET_FPS),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        out_path,
    ]

    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if result.returncode != 0:
        err = result.stderr.decode(errors="replace")
        raise RuntimeError(f"FFmpeg failed on {input_path}:\n{err}")

    return out_path


def normalize_all(clip_paths: list[str]) -> dict[str, str]:
    """
    Normalize a list of clips. Returns a mapping of original_path → normalized_path.
    Skips files that are already normalized.
    """
    mapping: dict[str, str] = {}
    for i, path in enumerate(clip_paths):
        print(f"          Normalizing [{i + 1}/{len(clip_paths)}] {os.path.basename(path)}")
        mapping[path] = normalize_clip(path)
    return mapping
