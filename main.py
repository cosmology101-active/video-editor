"""
Rocketry Video Editor Agent — CLI Entry Point

Usage:
  python main.py --clips ./footage --music track.mp3 --output reel.mp4

  # Skip re-tagging clips (reuse existing analysis/clip_database.json):
  python main.py --clips ./footage --music track.mp3 --skip-analysis

  # Preview the timeline without rendering:
  python main.py --clips ./footage --music track.mp3 --dry-run

  # Limit video length to 60 seconds:
  python main.py --clips ./footage --music track.mp3 --duration 60
"""

from __future__ import annotations

import json
import os
import sys
import time

import click
from dotenv import load_dotenv

load_dotenv()


@click.command()
@click.option("--clips", required=True, type=click.Path(exists=True),
              help="Folder containing raw video clips.")
@click.option("--music", required=True, type=click.Path(exists=True),
              help="Audio file (MP3, WAV, FLAC).")
@click.option("--output", default="output/reel.mp4", show_default=True,
              help="Output MP4 path.")
@click.option("--duration", type=float, default=None,
              help="Target reel length in seconds (default: full song).")
@click.option("--skip-analysis", is_flag=True, default=False,
              help="Reuse existing analysis/ JSON files (skips Stage 1 & 2).")
@click.option("--dry-run", is_flag=True, default=False,
              help="Print timeline JSON without rendering the video.")
def main(clips, music, output, duration, skip_analysis, dry_run):
    """AI-powered rocketry highlight reel generator."""
    _check_api_key()

    t_total = time.time()

    # ── Stage 1: Audio Analysis ───────────────────────────────────────────────
    if skip_analysis and _analysis_exists("audio_map.json"):
        print("[Stage 1] Skipping audio analysis (--skip-analysis).")
        from pipeline.audio_analyzer import load_audio_map
        audio_map = load_audio_map()
    else:
        from pipeline.audio_analyzer import analyze
        t = time.time()
        audio_map = analyze(music)
        print(f"          ✓ Done in {time.time() - t:.1f}s\n")

    # ── Stage 2: Clip Tagging ─────────────────────────────────────────────────
    if skip_analysis and _analysis_exists("clip_database.json"):
        print("[Stage 2] Skipping clip tagging (--skip-analysis).")
        from pipeline.clip_tagger import load_clip_database
        clip_db = load_clip_database()
    else:
        from pipeline.clip_tagger import tag_all
        t = time.time()
        clip_db = tag_all(clips)
        print(f"          ✓ Done in {time.time() - t:.1f}s\n")

    _validate_clip_database(clip_db)

    # ── Stage 3: Director (Claude) ────────────────────────────────────────────
    from pipeline.director import plan
    t = time.time()
    timeline = plan(audio_map, clip_db, target_duration=duration)
    print(f"          ✓ Done in {time.time() - t:.1f}s\n")

    if dry_run:
        print("\n[Dry run] Timeline JSON:\n")
        print(json.dumps(timeline, indent=2))
        print(f"\n[Dry run] {len(timeline['segments'])} segment(s) planned. No video rendered.")
        return

    # ── Stage 4: Assembly ─────────────────────────────────────────────────────
    from pipeline.assembler import render
    t = time.time()
    render(timeline, music, output)
    print(f"          ✓ Done in {time.time() - t:.1f}s\n")

    total = time.time() - t_total
    print(f"\n{'=' * 50}")
    print(f"  Reel ready: {output}")
    print(f"  Total time: {total:.1f}s")
    print(f"{'=' * 50}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check_api_key() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        click.echo(
            "Error: OPENAI_API_KEY is not set.\n"
            "Add OPENAI_API_KEY=sk-... to your .env file.",
            err=True,
        )
        sys.exit(1)


def _analysis_exists(filename: str) -> bool:
    from config import ANALYSIS_DIR
    return os.path.exists(os.path.join(ANALYSIS_DIR, filename))


def _validate_clip_database(clip_db: dict) -> None:
    clips = clip_db.get("clips", [])
    if not clips:
        click.echo(
            "Error: No clips were successfully tagged. "
            "Check that --clips points to a folder with video files.",
            err=True,
        )
        sys.exit(1)
    click.echo(f"          {len(clips)} clip(s) ready for editing.\n")


if __name__ == "__main__":
    main()
