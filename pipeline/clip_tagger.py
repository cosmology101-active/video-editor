"""
Stage 2 — The "Eyes"

Scans a folder of raw video clips and builds a ClipDatabase JSON containing,
for each clip:
  - Semantic tags (CLIP ViT-B/32 scored against a rocketry vocabulary)
  - Aesthetic score 0–10 (CLIP similarity to a "beautiful cinematic photo" prompt)
  - Motion score 0–10 (Farneback optical flow magnitude)
  - Face bonus (MediaPipe: +2 if ≥ 3 faces detected)
  - Phase fit (intro / buildup / drop / outro)

Output is saved to analysis/clip_database.json and returned as a dict.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import open_clip

from config import (
    ANALYSIS_DIR,
    AESTHETIC_PROMPT,
    CLIP_MODEL_NAME,
    CLIP_PRETRAINED,
    CLIP_VOCABULARY,
    FACE_BONUS_SCORE,
    FACE_BONUS_THRESHOLD,
    PHASE_TAG_MAP,
)

# ── Supported video extensions ────────────────────────────────────────────────
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".mts", ".m4v", ".wmv"}

# ── Lazy-loaded globals (loaded once on first call) ───────────────────────────
_clip_model = None
_clip_preprocess = None
_clip_tokenizer = None
_face_detector = None


def _get_clip():
    global _clip_model, _clip_preprocess, _clip_tokenizer
    if _clip_model is None:
        print("          Loading CLIP model (first run may download weights)…")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
        )
        _clip_model = _clip_model.to(device).eval()
        _clip_tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
    return _clip_model, _clip_preprocess, _clip_tokenizer


def _get_face_detector():
    global _face_detector
    if _face_detector is None:
        try:
            import mediapipe as mp
            _face_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.5
            )
        except Exception:
            _face_detector = None  # MediaPipe unavailable; skip face scoring
    return _face_detector


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class ClipRecord:
    clip_id: str
    path: str
    duration: float
    fps: float
    resolution: Tuple[int, int]
    tags: List[str] = field(default_factory=list)
    tag_scores: Dict[str, float] = field(default_factory=dict)
    aesthetic_score: float = 0.0
    motion_score: float = 0.0
    face_score: float = 0.0
    phase_fit: str = "intro"


@dataclass
class ClipDatabase:
    clips: List[ClipRecord] = field(default_factory=list)


# ── Public API ────────────────────────────────────────────────────────────────

def tag_all(clips_folder: str) -> dict:
    """
    Scan clips_folder recursively, tag every video clip, and return a
    ClipDatabase dict. Saves to analysis/clip_database.json.
    """
    video_paths = _collect_videos(clips_folder)
    if not video_paths:
        raise FileNotFoundError(f"No video files found in: {clips_folder}")

    print(f"[Stage 2] Found {len(video_paths)} video clip(s) in '{clips_folder}'")

    model, preprocess, tokenizer = _get_clip()
    device = next(model.parameters()).device

    # Pre-compute text embeddings for all vocabulary labels + aesthetic prompt
    all_texts = CLIP_VOCABULARY + [AESTHETIC_PROMPT]
    text_tokens = tokenizer(all_texts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    vocab_features = text_features[: len(CLIP_VOCABULARY)]
    aesthetic_feature = text_features[len(CLIP_VOCABULARY):]

    db = ClipDatabase()

    for idx, path in enumerate(video_paths):
        clip_id = f"clip_{idx:04d}"
        print(f"          [{idx + 1}/{len(video_paths)}] {os.path.basename(path)}")
        try:
            record = _tag_clip(
                clip_id, path, model, preprocess, device,
                vocab_features, aesthetic_feature,
            )
            db.clips.append(record)
        except Exception as e:
            print(f"          WARNING: Skipping {path} — {e}")

    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    out_path = os.path.join(ANALYSIS_DIR, "clip_database.json")
    data = asdict(db)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"          Saved → {out_path}")

    return data


def load_clip_database() -> dict:
    """Load a previously saved clip_database.json from disk."""
    path = os.path.join(ANALYSIS_DIR, "clip_database.json")
    with open(path) as f:
        return json.load(f)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _collect_videos(folder: str) -> List[str]:
    paths = []
    for root, _, files in os.walk(folder):
        for fname in sorted(files):
            if os.path.splitext(fname)[1].lower() in VIDEO_EXTENSIONS:
                paths.append(os.path.join(root, fname))
    return paths


def _tag_clip(
    clip_id: str,
    path: str,
    model,
    preprocess,
    device,
    vocab_features: torch.Tensor,
    aesthetic_feature: torch.Tensor,
) -> ClipRecord:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("OpenCV could not open file")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0.0

    # Sample frames at 10%, 50%, 90% of the clip
    sample_positions = [int(total_frames * p) for p in (0.10, 0.50, 0.90)]
    frames_rgb: List[np.ndarray] = []
    for pos in sample_positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(pos, total_frames - 1)))
        ret, frame = cap.read()
        if ret:
            frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames_rgb:
        raise RuntimeError("Could not sample any frames")

    # ── CLIP scoring ──────────────────────────────────────────────────────────
    pil_frames = [_numpy_to_pil(f) for f in frames_rgb]
    image_tensors = torch.stack([preprocess(img) for img in pil_frames]).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_tensors)
        image_features /= image_features.norm(dim=-1, keepdim=True)

    # Average across sampled frames
    avg_image_feature = image_features.mean(dim=0, keepdim=True)
    avg_image_feature /= avg_image_feature.norm(dim=-1, keepdim=True)

    # Vocabulary similarity
    vocab_sims = (avg_image_feature @ vocab_features.T).squeeze(0).cpu().numpy()
    tag_scores: Dict[str, float] = {
        CLIP_VOCABULARY[i]: float(vocab_sims[i]) for i in range(len(CLIP_VOCABULARY))
    }
    top5_indices = np.argsort(vocab_sims)[::-1][:5]
    tags = [CLIP_VOCABULARY[i] for i in top5_indices]

    # Aesthetic score (0–10)
    aesthetic_sim = float((avg_image_feature @ aesthetic_feature.T).squeeze())
    # CLIP similarities are roughly in [-1, 1]; rescale to 0–10
    aesthetic_score = min(10.0, max(0.0, (aesthetic_sim + 1.0) * 5.0))

    # ── Motion score (optical flow) ───────────────────────────────────────────
    motion_score = _compute_motion_score(frames_rgb)

    # ── Face bonus (MediaPipe) ────────────────────────────────────────────────
    face_score = _compute_face_score(frames_rgb[len(frames_rgb) // 2])

    # ── Phase fit ─────────────────────────────────────────────────────────────
    phase_fit = _infer_phase(tags, motion_score)

    return ClipRecord(
        clip_id=clip_id,
        path=os.path.abspath(path),
        duration=round(duration, 3),
        fps=round(fps, 3),
        resolution=(width, height),
        tags=tags,
        tag_scores={k: round(v, 4) for k, v in tag_scores.items()},
        aesthetic_score=round(min(10.0, aesthetic_score + face_score), 2),
        motion_score=round(motion_score, 2),
        face_score=round(face_score, 2),
        phase_fit=phase_fit,
    )


def _numpy_to_pil(arr: np.ndarray):
    from PIL import Image
    return Image.fromarray(arr)


def _compute_motion_score(frames: List[np.ndarray]) -> float:
    """Compute mean optical flow magnitude between consecutive sampled frames."""
    if len(frames) < 2:
        return 0.0

    magnitudes = []
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    for frame in frames[1:]:
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray,
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        magnitudes.append(float(np.mean(mag)))
        prev_gray = curr_gray

    raw = float(np.mean(magnitudes))
    # Scale: typical max flow magnitude ~20px → score 10
    return min(10.0, raw * 0.5)


def _compute_face_score(frame: np.ndarray) -> float:
    """Return FACE_BONUS_SCORE if ≥ FACE_BONUS_THRESHOLD faces detected, else 0."""
    detector = _get_face_detector()
    if detector is None:
        return 0.0
    try:
        result = detector.process(frame)
        if result.detections and len(result.detections) >= FACE_BONUS_THRESHOLD:
            return FACE_BONUS_SCORE
    except Exception:
        pass
    return 0.0


def _infer_phase(tags: List[str], motion_score: float) -> str:
    """Map the top CLIP tags to a rocketry phase."""
    for phase, phase_tags in PHASE_TAG_MAP.items():
        for tag in tags:
            if tag in phase_tags:
                return phase
    # Fallback: high-motion clips that didn't match go to buildup
    if motion_score >= 6.0:
        return "buildup"
    return "intro"
