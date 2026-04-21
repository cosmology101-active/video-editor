"""
Stage 1 — The "Ear"

Analyzes a music track and returns a structured AudioMap containing:
  - BPM and beat timestamps
  - High-impact onset timestamps
  - Energy-based phase segments (intro / buildup / drop / outro)
  - Spectral centroid per segment (brightness)

Output is saved to analysis/audio_map.json and returned as a dict.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import List

import librosa
import numpy as np

from config import ANALYSIS_DIR, ENERGY_HIGH_PERCENTILE, ENERGY_LOW_PERCENTILE


@dataclass
class Segment:
    label: str          # intro | buildup | drop | outro
    start: float        # seconds
    end: float          # seconds
    energy: float       # mean RMS energy in this window
    centroid: float     # mean spectral centroid (Hz) — higher = brighter


@dataclass
class AudioMap:
    path: str
    duration: float
    bpm: float
    beats: List[float] = field(default_factory=list)
    onsets: List[float] = field(default_factory=list)
    segments: List[Segment] = field(default_factory=list)


def analyze(audio_path: str) -> dict:
    """
    Analyze the given audio file and return an AudioMap dict.
    Also writes analysis/audio_map.json.
    """
    print(f"[Stage 1] Loading audio: {audio_path}")
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    print(f"          Duration: {duration:.1f}s  |  Sample rate: {sr} Hz")

    # ── Beat tracking ─────────────────────────────────────────────────────────
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(tempo) if not hasattr(tempo, "__len__") else float(tempo[0])
    beats = librosa.frames_to_time(beat_frames, sr=sr).tolist()
    print(f"          BPM: {bpm:.1f}  |  Beats detected: {len(beats)}")

    # ── Onset detection ───────────────────────────────────────────────────────
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units="frames")
    onsets = librosa.frames_to_time(onset_frames, sr=sr).tolist()
    print(f"          Onsets detected: {len(onsets)}")

    # ── RMS energy per 1-second window ────────────────────────────────────────
    hop_length = sr  # 1 second per frame
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    # ── Spectral centroid per 1-second window ─────────────────────────────────
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]

    # ── Phase labelling ───────────────────────────────────────────────────────
    segments = _label_phases(rms, centroid, times, duration)
    print(f"          Segments: {[s.label for s in segments]}")

    audio_map = AudioMap(
        path=os.path.abspath(audio_path),
        duration=duration,
        bpm=bpm,
        beats=beats,
        onsets=onsets,
        segments=segments,
    )

    # ── Save to disk ──────────────────────────────────────────────────────────
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    out_path = os.path.join(ANALYSIS_DIR, "audio_map.json")
    data = asdict(audio_map)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"          Saved → {out_path}")

    return data


def load_audio_map() -> dict:
    """Load a previously saved audio_map.json from disk."""
    path = os.path.join(ANALYSIS_DIR, "audio_map.json")
    with open(path) as f:
        return json.load(f)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _label_phases(
    rms: np.ndarray,
    centroid: np.ndarray,
    times: np.ndarray,
    duration: float,
) -> List[Segment]:
    """
    Label each 1-second window with intro / buildup / drop / outro using
    RMS energy percentile thresholds and gradient direction.

    Groups consecutive identical labels into Segment objects.
    """
    low = np.percentile(rms, ENERGY_LOW_PERCENTILE)
    high = np.percentile(rms, ENERGY_HIGH_PERCENTILE)
    gradient = np.gradient(rms)

    labels = []
    for i, energy in enumerate(rms):
        if energy >= high:
            labels.append("drop")
        elif energy <= low:
            labels.append("intro")
        elif gradient[i] > 0:
            labels.append("buildup")
        else:
            labels.append("outro")

    # Group consecutive windows into segments
    segments: List[Segment] = []
    if not labels:
        return segments

    current_label = labels[0]
    seg_start = float(times[0]) if len(times) > 0 else 0.0

    for i in range(1, len(labels)):
        if labels[i] != current_label:
            seg_end = float(times[i])
            segments.append(Segment(
                label=current_label,
                start=seg_start,
                end=seg_end,
                energy=float(np.mean(rms[max(0, i - (i - _find_start_idx(segments, seg_start, times))):i])),
                centroid=float(np.mean(centroid[max(0, i - 1):i + 1])),
            ))
            current_label = labels[i]
            seg_start = float(times[i])

    # Final segment
    segments.append(Segment(
        label=current_label,
        start=seg_start,
        end=duration,
        energy=float(np.mean(rms)),
        centroid=float(np.mean(centroid)),
    ))

    return segments


def _find_start_idx(segments: list, seg_start: float, times: np.ndarray) -> int:
    """Return the frame index corresponding to seg_start."""
    for i, t in enumerate(times):
        if float(t) >= seg_start:
            return i
    return 0
