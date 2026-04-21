"""
Central configuration for the rocketry video editor agent.
Edit the CLIP_VOCABULARY and PHASE_THRESHOLDS to tune clip classification.
"""

# ── OpenAI model ─────────────────────────────────────────────────────────────
OPENAI_MODEL = "gpt-4o"
DIRECTOR_SANITY_CHECK = True   # run a second pass to fix boring sequences

# ── CLIP model ────────────────────────────────────────────────────────────────
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"

# Labels the CLIP model scores each video frame against.
# Add or remove labels to tune what the "Eyes" stage detects.
CLIP_VOCABULARY = [
    "rocket on launch rail",
    "ignition and fire",
    "smoke and liftoff",
    "rocket ascending into sky",
    "soldering in lab",
    "electronics and circuit boards",
    "team working in workshop",
    "team celebrating outdoors",
    "smiling faces and crowd",
    "parachute descent",
    "blue sky rocket tracking",
    "van driving on highway",
    "rocket assembly",
    "countdown display",
    "recovery team in field",
]

AESTHETIC_PROMPT = "a beautiful, sharp, well-lit cinematic photograph"

# ── Phase classification ───────────────────────────────────────────────────────
# Tags that map to each phase. First match wins.
PHASE_TAG_MAP = {
    "drop": ["ignition and fire", "smoke and liftoff", "rocket ascending into sky"],
    "buildup": ["rocket on launch rail", "countdown display"],
    "outro": ["parachute descent", "team celebrating outdoors", "smiling faces and crowd", "recovery team in field"],
    "intro": [
        "soldering in lab",
        "electronics and circuit boards",
        "team working in workshop",
        "van driving on highway",
        "rocket assembly",
        "blue sky rocket tracking",
    ],
}

# ── Energy segmentation thresholds ────────────────────────────────────────────
ENERGY_LOW_PERCENTILE = 30    # below this → intro
ENERGY_HIGH_PERCENTILE = 70   # above this → drop
# between low and high: gradient > 0 → buildup, else → outro

# ── Face detection ────────────────────────────────────────────────────────────
FACE_BONUS_THRESHOLD = 3      # minimum faces to award bonus score
FACE_BONUS_SCORE = 2.0

# ── Assembly ──────────────────────────────────────────────────────────────────
TARGET_RESOLUTION = (1920, 1080)
TARGET_FPS = 30
MAX_SPEED_ADJUST = 0.15       # ±15% clip speed allowed to fit beat gaps
DISSOLVE_DURATION = 0.3       # seconds for cross-fade transitions
ZOOM_SCALE = 1.08             # end scale for zoom_in transition

# ── Output directories ────────────────────────────────────────────────────────
ANALYSIS_DIR = "analysis"
TEMP_DIR = "temp/normalized"
