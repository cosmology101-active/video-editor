"""
Stage 3 — The "Brain" (Director) — One-Shot Architect

Instead of a tool-use loop, the model receives the full audio map and clip
manifest in one prompt and returns a complete timeline in a single structured
JSON response (1 API call).

An optional sanity-check pass (1 more call) fixes boring sequences such as
three consecutive intro clips in a row.

This approach is:
  - Faster   (1–2 calls vs 20)
  - Cheaper  (no repeated context overhead)
  - Coherent (model sees the whole picture at once before deciding)
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Literal, Optional

from openai import OpenAI
from pydantic import BaseModel

from config import (
    ANALYSIS_DIR,
    DIRECTOR_SANITY_CHECK,
    OPENAI_MODEL,
)


# ── Output schema (Pydantic — fed to OpenAI structured output) ────────────────

class SegmentPlan(BaseModel):
    clip_id: str
    clip_start: float
    clip_end: float
    music_start: float
    music_end: float
    transition: Literal["cut", "dissolve", "zoom_in"]


class TimelinePlan(BaseModel):
    segments: List[SegmentPlan]
    director_notes: str   # brief creative rationale (not rendered, for debugging)


# ── Internal data model ───────────────────────────────────────────────────────

@dataclass
class TimelineSegment:
    clip_id: str
    clip_path: str
    clip_start: float
    clip_end: float
    music_start: float
    music_end: float
    transition: str


@dataclass
class Timeline:
    total_duration: float
    segments: List[TimelineSegment] = field(default_factory=list)


# ── Public API ────────────────────────────────────────────────────────────────

def plan(audio_map: dict, clip_db: dict, target_duration: Optional[float] = None) -> dict:
    """
    Generate a complete video timeline in 1–2 API calls.
    Returns the timeline dict and saves it to analysis/timeline.json.
    """
    duration = target_duration or audio_map["duration"]
    clips: Dict[str, dict] = {c["clip_id"]: c for c in clip_db.get("clips", [])}

    print(f"[Stage 3] Director: one-shot mode (model={OPENAI_MODEL}, target={duration:.1f}s)")
    print(f"          {len(clips)} clips in manifest")

    client = OpenAI()
    manifest = _build_manifest(audio_map, clip_db, duration)

    # ── Pass 1: Generate complete timeline ────────────────────────────────────
    print("          Pass 1: generating timeline …")
    plan_result = _call_structured(client, manifest, duration)
    print(f"          Pass 1 done: {len(plan_result.segments)} segment(s) planned")
    print(f"          Notes: {plan_result.director_notes[:120]}")

    # ── Pass 2: Sanity check (optional) ──────────────────────────────────────
    issues = _find_issues(plan_result, clips)
    if issues and DIRECTOR_SANITY_CHECK:
        print(f"          Pass 2: fixing {len(issues)} issue(s): {'; '.join(issues)}")
        plan_result = _call_sanity_check(client, manifest, plan_result, issues, duration)
        print(f"          Pass 2 done: {len(plan_result.segments)} segment(s)")
    elif not DIRECTOR_SANITY_CHECK:
        print("          Sanity check skipped (DIRECTOR_SANITY_CHECK=False)")
    else:
        print("          No issues found, skipping pass 2")

    # ── Validate and enrich with clip paths ───────────────────────────────────
    segments = _validate_and_enrich(plan_result.segments, clips, duration)

    timeline = Timeline(total_duration=duration, segments=segments)
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    out_path = os.path.join(ANALYSIS_DIR, "timeline.json")
    data = asdict(timeline)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"          Saved → {out_path}")
    return data


def load_timeline() -> dict:
    path = os.path.join(ANALYSIS_DIR, "timeline.json")
    with open(path) as f:
        return json.load(f)


# ── Manifest builder ──────────────────────────────────────────────────────────

def _build_manifest(audio_map: dict, clip_db: dict, duration: float) -> str:
    """
    Build a compact text manifest (not huge JSON) that fits cleanly in the
    prompt without wasting tokens on redundant structure.
    """
    bpm = audio_map.get("bpm", 0)
    beats = audio_map.get("beats", [])
    segments = audio_map.get("segments", [])

    # Phase windows with sample beat timestamps (every 4th beat to save tokens)
    phase_sections = []
    for seg in segments:
        phase_beats = [
            round(b, 2) for b in beats
            if seg["start"] <= b <= seg["end"]
        ][::4]  # every 4th beat
        phase_sections.append(
            f"  {seg['label']:8s}  {seg['start']:.1f}s – {seg['end']:.1f}s  "
            f"energy={seg['energy']:.3f}  "
            f"sample_beats={phase_beats[:8]}"
        )

    audio_block = (
        f"SONG: {duration:.1f}s total, {bpm:.1f} BPM, {len(beats)} beats\n"
        + "\n".join(phase_sections)
    )

    # Clip manifest — one line per clip, sorted by phase then aesthetic score
    clips = sorted(
        clip_db.get("clips", []),
        key=lambda c: (
            ["drop", "buildup", "outro", "intro"].index(c.get("phase_fit", "intro")),
            -c.get("aesthetic_score", 0),
        ),
    )
    clip_lines = []
    for c in clips:
        clip_lines.append(
            f"  {c['clip_id']}  dur={c['duration']:.1f}s  "
            f"phase={c['phase_fit']:8s}  "
            f"motion={c['motion_score']:.1f}  "
            f"aesth={c['aesthetic_score']:.1f}  "
            f"face={c['face_score']:.0f}  "
            f"tags=[{', '.join(c['tags'][:2])}]"
        )
    clips_block = "CLIPS:\n" + "\n".join(clip_lines)

    return f"{audio_block}\n\n{clips_block}"


# ── API calls ─────────────────────────────────────────────────────────────────

_SYSTEM = """\
You are an expert film director editing a rocketry highlight reel.

ROCKETRY NARRATIVE ARC (follow this sequence):
  intro   → lab work, assembly, travel      (low energy, wide shots)
  buildup → rail setup, countdown           (rising tension, medium shots)
  drop    → ignition, liftoff, fire/smoke   (MAX energy — only motion_score >= 6)
  outro   → parachute, recovery, team faces (warm, face_score clips preferred)

RULES:
- Place music_start and music_end on the provided sample_beats timestamps
- clip_start and clip_end must be within the clip's dur= value
- Do not use the same clip_id more than twice
- Avoid 3+ consecutive clips from the same phase
- Use transition="cut" on strong beats, "dissolve" on melodic passages, "zoom_in" entering the drop
- Cover the FULL song duration with no gaps and no overlaps
- Prefer higher aesth= scores when choosing between similar clips
"""

def _call_structured(client: OpenAI, manifest: str, duration: float) -> TimelinePlan:
    response = client.beta.chat.completions.parse(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": (
                    f"{manifest}\n\n"
                    f"Build the complete timeline for all {duration:.1f}s. "
                    f"Output every segment — do not truncate."
                ),
            },
        ],
        response_format=TimelinePlan,
    )
    return response.choices[0].message.parsed


def _call_sanity_check(
    client: OpenAI,
    manifest: str,
    original: TimelinePlan,
    issues: List[str],
    duration: float,
) -> TimelinePlan:
    issues_text = "\n".join(f"- {i}" for i in issues)
    original_json = original.model_dump_json(indent=2)

    response = client.beta.chat.completions.parse(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"{manifest}\n\nOriginal timeline:\n{original_json}"},
            {
                "role": "assistant",
                "content": original_json,
            },
            {
                "role": "user",
                "content": (
                    f"Fix these issues in the timeline:\n{issues_text}\n\n"
                    f"The full video is {duration:.1f}s. "
                    "Return the corrected full timeline. Keep all other segments unchanged."
                ),
            },
        ],
        response_format=TimelinePlan,
    )
    return response.choices[0].message.parsed


# ── Issue detection ───────────────────────────────────────────────────────────

def _find_issues(plan: TimelinePlan, clips: Dict[str, dict]) -> List[str]:
    """Detect common problems that warrant a sanity-check pass."""
    issues = []
    segs = plan.segments

    # 3+ consecutive clips from the same phase
    for i in range(len(segs) - 2):
        phases = [
            clips.get(segs[i + j].clip_id, {}).get("phase_fit", "?")
            for j in range(3)
        ]
        if len(set(phases)) == 1 and phases[0] in ("intro", "outro"):
            issues.append(
                f"Three consecutive '{phases[0]}' clips starting at "
                f"music_start={segs[i].music_start:.1f}s"
            )
            break  # report first occurrence only

    # Drop segments with low motion
    for seg in segs:
        clip = clips.get(seg.clip_id, {})
        if clip.get("phase_fit") == "drop" and clip.get("motion_score", 10) < 5:
            issues.append(
                f"Low-motion clip '{seg.clip_id}' (motion={clip.get('motion_score'):.1f}) "
                f"placed in the drop section"
            )
            break

    # Outro has no face clips
    outro_segs = [
        s for s in segs
        if clips.get(s.clip_id, {}).get("phase_fit") == "outro"
    ]
    if outro_segs and not any(
        clips.get(s.clip_id, {}).get("face_score", 0) > 0 for s in outro_segs
    ):
        issues.append("Outro section has no clips with visible faces (face_score=0)")

    return issues


# ── Validation ────────────────────────────────────────────────────────────────

def _validate_and_enrich(
    segs: List[SegmentPlan],
    clips: Dict[str, dict],
    duration: float,
) -> List[TimelineSegment]:
    """
    Validate model output and attach clip_path.
    Silently clamp/fix minor issues rather than crashing.
    """
    usage: Dict[str, int] = {}
    result: List[TimelineSegment] = []

    for seg in sorted(segs, key=lambda s: s.music_start):
        clip = clips.get(seg.clip_id)
        if not clip:
            print(f"          WARNING: unknown clip_id '{seg.clip_id}' — skipped")
            continue

        if usage.get(seg.clip_id, 0) >= 2:
            print(f"          WARNING: '{seg.clip_id}' used >2 times — skipped duplicate")
            continue

        # Clamp trim range to actual clip duration
        clip_dur = clip.get("duration", 0)
        clip_start = max(0.0, min(seg.clip_start, clip_dur - 0.1))
        clip_end = max(clip_start + 0.1, min(seg.clip_end, clip_dur))

        # Clamp music range to song duration
        music_start = max(0.0, seg.music_start)
        music_end = min(duration, seg.music_end)
        if music_end <= music_start:
            print(f"          WARNING: invalid music range for '{seg.clip_id}' — skipped")
            continue

        result.append(TimelineSegment(
            clip_id=seg.clip_id,
            clip_path=clip["path"],
            clip_start=round(clip_start, 3),
            clip_end=round(clip_end, 3),
            music_start=round(music_start, 3),
            music_end=round(music_end, 3),
            transition=seg.transition,
        ))
        usage[seg.clip_id] = usage.get(seg.clip_id, 0) + 1

    return result
