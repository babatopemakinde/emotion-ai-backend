"""
=============================================================================
 Multimodal Emotion AI v7.2 — FastAPI backend for Azure App Service
 Target: Autism Spectrum Disorder (ASD) Support System
 Modalities: Facial (EfficientNetB0) + Audio (MFCC-CNN)
 Deployment: FastAPI + Uvicorn on Azure App Service (Python 3.13 / TF 2.20)

 AZURE REVISION v7.2-azure
 ─────────────────────────────────────────────────────────────────────
 - Gradio entrypoint replaced with FastAPI + Uvicorn
 - All inference logic, models and helper classes preserved verbatim
 - File I/O uses /tmp for ephemeral Azure container storage
 - Endpoints:
     GET  /              → health check
     GET  /health        → detailed health + model status
     POST /predict       → multimodal inference (image + optional audio)
     GET  /policy        → current reinforcement policy
     GET  /history       → last N session records
     POST /feedback      → submit feedback rating for last session
=============================================================================
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import os
import sys
import io
import json
import base64
import datetime
import warnings
import logging
import traceback
import tempfile
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

# ── numeric / data ────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

# ── plotting ──────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── deep learning — Keras 3 / TF 2.20 ────────────────────────────────────────
import tensorflow as tf

try:
    import keras
    from keras import layers, regularizers
    from keras import applications as keras_applications
    _KERAS3 = True
except ImportError:
    keras = tf.keras
    from tensorflow.keras import layers, regularizers
    keras_applications = tf.keras.applications
    _KERAS3 = False

# ── audio / vision ────────────────────────────────────────────────────────────
import librosa
import cv2
from gtts import gTTS

# ── FastAPI ───────────────────────────────────────────────────────────────────
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# ── Azure Blob Storage ────────────────────────────────────────────────────────
from azure.storage.blob import BlobServiceClient

SHAP_AVAILABLE = False  # Disabled: incompatible with Keras 3

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# =============================================================================
# 0.  LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("EmotionAI")
logger.info(f"TF {tf.__version__} | Keras3={_KERAS3} | Python {sys.version.split()[0]}")

# =============================================================================
# 1.  CONSTANTS
# =============================================================================

TARGET_EMOTIONS: List[str] = ["anger", "fear", "joy", "Natural", "sadness", "surprise"]
EMOTION_TO_IDX: Dict[str, int] = {e: i for i, e in enumerate(TARGET_EMOTIONS)}
IDX_TO_EMOTION: Dict[int, str] = {i: e for e, i in EMOTION_TO_IDX.items()}
NUM_CLASSES: int = len(TARGET_EMOTIONS)

N_MFCC: int = 40
AUDIO_SR: int = 16_000
AUDIO_DUR: float = 3.0
AUDIO_SHAPE: Tuple[int, int, int] = (N_MFCC, 128, 3)
IMG_SIZE: int = 224

# Azure: use /tmp for writable ephemeral storage, or a mounted Azure File Share
BASE_DIR = Path(os.environ.get("EMOTIONAI_WORKSPACE", "/tmp/emotionai_workspace"))

# ── Azure Blob Storage config ─────────────────────────────────────────────────
# Set AZURE_STORAGE_CONNECTION_STRING and AZURE_BLOB_CONTAINER in your App
# Service environment variables (Application Settings).
AZURE_CONNECTION_STRING: str = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "")
AZURE_BLOB_CONTAINER:    str = os.environ.get("AZURE_BLOB_CONTAINER", "emotionai-models")

# Blob name → local filename inside BASE_DIR/models/final/
# Edit these names to match the exact filenames in your container.
MODEL_BLOBS: Dict[str, str] = {
    "vision_model.keras":      "vision_model.keras",
    "audio_model.keras":       "audio_model.keras",
    "joint_infer_model.keras": "joint_infer_model.keras",
}

EMOTION_COLORS = {
    "anger":    "#C0392B", "fear":     "#7D3C98",
    "joy":      "#1E8449", "sadness":  "#1A5276",
    "surprise": "#B7950B", "Natural":  "#1A7F74",
}
EMOTION_EMOJI = {
    "anger": "😠", "fear": "😨", "joy": "😄",
    "sadness": "😢", "surprise": "😮", "Natural": "😊",
}

# =============================================================================
# 2.  DIRECTORY SETUP
# =============================================================================

for _d in [
    "models/final", "reports/explainability/shap",
    "reports/metrics", "outputs/audio", "outputs/schedules", "data/memory",
]:
    (BASE_DIR / _d).mkdir(parents=True, exist_ok=True)

# =============================================================================
# 3.  CUSTOM SERIALISABLE LAYERS
# =============================================================================

class WeightedMul(layers.Layer):
    def call(self, inputs):
        x, w = inputs
        return x * w
    def get_config(self):
        return super().get_config()


class WeightedMulComplement(layers.Layer):
    def call(self, inputs):
        x, w = inputs
        return x * (1.0 - w)
    def get_config(self):
        return super().get_config()


class L1Normalise(layers.Layer):
    def call(self, x):
        return x / (tf.reduce_sum(x, axis=-1, keepdims=True) + 1e-9)
    def get_config(self):
        return super().get_config()


CUSTOM_OBJECTS = {
    "WeightedMul":           WeightedMul,
    "WeightedMulComplement": WeightedMulComplement,
    "L1Normalise":           L1Normalise,
}

# =============================================================================
# 4.  MODEL DEFINITIONS
# =============================================================================

def build_vision_model(num_classes: int = NUM_CLASSES) -> keras.Model:
    base = keras_applications.EfficientNetB0(
        include_top=False,
        weights=None,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        name="efficientnetb0",
    )
    base.trainable = False
    inp  = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="facial_input")
    x    = base(inp, training=False)
    x    = layers.GlobalAveragePooling2D()(x)
    x    = layers.BatchNormalization()(x)
    x    = layers.Dropout(0.4)(x)
    x    = layers.Dense(512, activation="relu")(x)
    x    = layers.BatchNormalization()(x)
    x    = layers.Dropout(0.3)(x)
    skip = layers.Dense(256, activation="relu")(x)
    skip = layers.BatchNormalization()(skip)
    xp   = layers.Dense(256, use_bias=False)(x)
    x    = layers.Add()([skip, xp])
    x    = layers.Activation("relu")(x)
    x    = layers.Dropout(0.2)(x)
    out  = layers.Dense(num_classes, name="face_logits")(x)
    return keras.Model(inp, out, name="VisionModel")


def build_audio_model(num_classes: int = NUM_CLASSES) -> keras.Model:
    inp = keras.Input(shape=AUDIO_SHAPE, name="audio_input")
    x   = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inp)
    x   = layers.BatchNormalization()(x)
    x   = layers.MaxPooling2D((2, 2))(x)
    x   = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.MaxPooling2D((2, 2))(x)
    x   = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.SpatialDropout2D(0.3)(x)
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.Dense(256, activation="relu")(x)
    x   = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, name="audio_logits")(x)
    return keras.Model(inp, out, name="AudioModel")


def build_joint_model(
    vision_model: keras.Model,
    audio_model: keras.Model,
    num_classes: int = NUM_CLASSES,
) -> Tuple[keras.Model, keras.Model]:
    face_inp  = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="joint_face_input")
    audio_inp = keras.Input(shape=AUDIO_SHAPE, name="joint_audio_input")
    face_logits  = vision_model(face_inp,  training=False)
    audio_logits = audio_model(audio_inp,  training=False)
    face_feat  = layers.Softmax(name="face_probs")(face_logits)
    audio_feat = layers.Softmax(name="audio_probs")(audio_logits)
    combined   = layers.Concatenate(name="gate_concat")([face_feat, audio_feat])
    gate_h     = layers.Dense(64, activation="relu", name="gate_hidden")(combined)
    gate       = layers.Dense(1, activation="sigmoid", name="gate")(gate_h)
    face_w     = WeightedMul(name="face_weighted")([face_feat, gate])
    audio_w    = WeightedMulComplement(name="audio_weighted")([audio_feat, gate])
    fused      = layers.Add(name="fused_probs")([face_w, audio_w])
    out        = L1Normalise(name="joint_output")(fused)
    joint_train = keras.Model(
        inputs=[face_inp, audio_inp], outputs=out, name="JointModel_train"
    )
    joint_infer = keras.Model(
        inputs=[face_inp, audio_inp], outputs=[out, gate], name="JointModel_infer"
    )
    return joint_train, joint_infer


def build_softmax_wrapper(logit_model: keras.Model, name: str) -> keras.Model:
    inp_specs = logit_model.inputs
    if len(inp_specs) == 1:
        new_inp = keras.Input(shape=inp_specs[0].shape[1:], name=inp_specs[0].name + "_sw")
        logits  = logit_model(new_inp)
    else:
        new_inps = [
            keras.Input(shape=s.shape[1:], name=s.name + "_sw")
            for s in inp_specs
        ]
        logits = logit_model(new_inps)
        new_inp = new_inps

    out = layers.Softmax(name="softmax_out")(logits)
    return keras.Model(new_inp, out, name=name)


# =============================================================================
# 5.  PREPROCESSING HELPER
# =============================================================================

def _efficientnet_preprocess(img_f32: np.ndarray) -> np.ndarray:
    return keras_applications.efficientnet.preprocess_input(img_f32)


# =============================================================================
# 6.  MODEL REGISTRY
# =============================================================================

class ModelRegistry:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    # ── Azure Blob download ───────────────────────────────────────────────────

    @staticmethod
    def _download_models_from_blob(dest_dir: Path) -> bool:
        """
        Download every file listed in MODEL_BLOBS from Azure Blob Storage into
        dest_dir.  Returns True if all blobs were downloaded successfully,
        False on any error (caller falls back to demo mode).

        Requires the env var AZURE_STORAGE_CONNECTION_STRING to be set.
        """
        if not AZURE_CONNECTION_STRING:
            logger.warning(
                "AZURE_STORAGE_CONNECTION_STRING is not set — skipping blob download."
            )
            return False

        try:
            service_client = BlobServiceClient.from_connection_string(
                AZURE_CONNECTION_STRING
            )
            container_client = service_client.get_container_client(
                AZURE_BLOB_CONTAINER
            )
        except Exception as exc:
            logger.error("Failed to create BlobServiceClient: %s", exc)
            return False

        all_ok = True
        for blob_name, local_filename in MODEL_BLOBS.items():
            local_path = dest_dir / local_filename
            # Skip if already downloaded (e.g. container restart with warm /tmp)
            if local_path.exists():
                logger.info("Blob already cached locally: %s — skipping download.", local_filename)
                continue

            logger.info("Downloading blob '%s' → %s …", blob_name, local_path)
            try:
                blob_client = container_client.get_blob_client(blob_name)
                with open(local_path, "wb") as f:
                    stream = blob_client.download_blob()
                    stream.readinto(f)
                size_mb = local_path.stat().st_size / (1024 ** 2)
                logger.info("Downloaded '%s' (%.1f MB).", blob_name, size_mb)
            except Exception as exc:
                logger.error("Failed to download blob '%s': %s", blob_name, exc)
                # Remove partial file so it isn't mistaken for a valid download
                if local_path.exists():
                    local_path.unlink()
                all_ok = False

        return all_ok

    # ── Main load ─────────────────────────────────────────────────────────────

    def load(self):
        if self._loaded:
            return

        logger.info("ModelRegistry: starting load sequence …")
        final_dir = BASE_DIR / "models/final"
        final_dir.mkdir(parents=True, exist_ok=True)

        vp = final_dir / "vision_model.keras"
        ap = final_dir / "audio_model.keras"
        jp = final_dir / "joint_infer_model.keras"

        # ── Step 1: attempt Azure Blob download ───────────────────────────────
        blobs_ready = self._download_models_from_blob(final_dir)

        # ── Step 2: load .keras files if present, else fall back to demo mode ─
        if blobs_ready and vp.exists() and ap.exists() and jp.exists():
            logger.info("Loading .keras models from local cache …")
            try:
                self.vision_model = keras.models.load_model(
                    str(vp), custom_objects=CUSTOM_OBJECTS
                )
                self.audio_model = keras.models.load_model(
                    str(ap), custom_objects=CUSTOM_OBJECTS
                )
                self.joint_infer = keras.models.load_model(
                    str(jp), custom_objects=CUSTOM_OBJECTS
                )
                self._demo_mode = False
                logger.info("All three .keras models loaded successfully.")
            except Exception as exc:
                logger.error(
                    "Failed to load one or more .keras models: %s — falling back to demo mode.",
                    exc,
                )
                self._demo_mode = True
                self.vision_model = build_vision_model()
                self.audio_model  = build_audio_model()
                _, self.joint_infer = build_joint_model(self.vision_model, self.audio_model)
        else:
            logger.warning(
                "Model files unavailable after blob download attempt — "
                "running in demo mode (random weights)."
            )
            self.vision_model = build_vision_model()
            self.audio_model  = build_audio_model()
            _, self.joint_infer = build_joint_model(self.vision_model, self.audio_model)
            self._demo_mode = True

        # ── Step 3: build softmax probability wrappers ────────────────────────
        self.vision_prob = build_softmax_wrapper(self.vision_model, "VisionProb")
        self.audio_prob  = build_softmax_wrapper(self.audio_model,  "AudioProb")

        # ── Step 4: warm-up forward passes ────────────────────────────────────
        dummy_img   = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), np.float32)
        dummy_audio = np.zeros((1, *AUDIO_SHAPE), np.float32)
        self.vision_prob(dummy_img, training=False)
        self.audio_prob(dummy_audio, training=False)
        self.joint_infer(
            {"joint_face_input": dummy_img, "joint_audio_input": dummy_audio},
            training=False,
        )

        logger.info("ModelRegistry ready. demo_mode=%s", self._demo_mode)
        self._loaded = True


registry = ModelRegistry()

# =============================================================================
# 7.  AUDIO PREPROCESSING
# =============================================================================

def extract_mfcc_spectrogram(filepath: str) -> Optional[np.ndarray]:
    try:
        y, _ = librosa.load(filepath, sr=AUDIO_SR, duration=AUDIO_DUR, mono=True)
        tlen = int(AUDIO_SR * AUDIO_DUR)
        y    = np.pad(y, (0, max(0, tlen - len(y))))[:tlen]
        mfcc   = cv2.resize(
            librosa.feature.mfcc(y=y, sr=AUDIO_SR, n_mfcc=N_MFCC), (128, N_MFCC)
        )
        delta1 = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        def _norm(a):
            s = a.std()
            return (a - a.mean()) / (s if s >= 1e-9 else 1.0)
        arr = np.stack([_norm(mfcc), _norm(delta1), _norm(delta2)], axis=-1).astype(np.float32)
        if arr.shape != AUDIO_SHAPE or np.isnan(arr).any() or np.isinf(arr).any():
            return None
        return arr
    except Exception as exc:
        logger.error(f"MFCC error: {exc}")
        return None

# =============================================================================
# 8.  HYBRID FUSION
# =============================================================================

class HybridFusion:
    @staticmethod
    def assess_image_quality(img_bgr: np.ndarray) -> str:
        if img_bgr is None or img_bgr.size == 0:
            return "unknown"
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        if float(gray.mean()) < 40:
            return "dark"
        if float(cv2.Laplacian(gray, cv2.CV_64F).var()) < 50:
            return "blurry"
        return "normal"

    @staticmethod
    def fuse(
        face_probs: np.ndarray, audio_probs: np.ndarray,
        learned_gate: float, image_quality: str,
    ) -> Tuple[np.ndarray, float, float]:
        rule_w  = 0.30 if image_quality in ("dark", "blurry") else 0.70
        face_w  = 0.5 * rule_w + 0.5 * float(learned_gate)
        audio_w = 1.0 - face_w
        fused   = face_probs * face_w + audio_probs * audio_w
        total   = fused.sum()
        fused   = fused / total if total >= 1e-9 else np.ones_like(fused) / len(fused)
        return fused, face_w, audio_w

# =============================================================================
# 9.  BEHAVIOURAL ENGINE
# =============================================================================

class BehavioralEngine:
    INTERVENTIONS = {
        "anger": {
            "message": "I can see you're feeling angry. Let's try counting to ten together.",
            "tasks":   ["Deep Breathing × 3", "Squeeze Stress Ball", "Calm Corner Break"],
            "cues":    ["Lower your voice", "Open hands", "Count slowly"],
        },
        "fear": {
            "message": "It's okay to feel scared — you are completely safe right now.",
            "tasks":   ["5-4-3-2-1 Grounding", "Weighted Blanket", "Check Safe Zone"],
            "cues":    ["Feet flat on floor", "Look around slowly", "Name 5 things you see"],
        },
        "joy": {
            "message": "I love that smile — amazing work today! Keep going!",
            "tasks":   ["Verbal Praise", "High Five", "Sticker Reward"],
            "cues":    ["Share the excitement", "Channel energy positively", "Celebrate!"],
        },
        "sadness": {
            "message": "It's okay to feel sad. Would you like some calming music?",
            "tasks":   ["Calming Music", "Comfort Object", "Quiet Sensory Space"],
            "cues":    ["Sit beside them", "Speak gently", "Offer favourite item"],
        },
        "surprise": {
            "message": "Wow, something unexpected! Let's take a breath together.",
            "tasks":   ["Wait Time (30 s)", "Reset Routine", "Preview Next Step"],
            "cues":    ["Reduce stimuli", "Use calm voice", "Show schedule"],
        },
        "Natural": {
            "message": "You're doing a wonderful job staying calm and focused!",
            "tasks":   ["Schedule Review", "Next Activity Preview", "Positive Reinforcement"],
            "cues":    ["Maintain routine", "Offer choice", "Praise calmness"],
        },
    }
    LOCATION_RULES = {
        "school":  "School environment — follow classroom protocol.",
        "home":    "Home environment — relaxed routine preferred.",
        "therapy": "Therapy session — therapist-led strategies.",
        "outdoor": "Outdoor — watch for sensory overload triggers.",
    }
    ROUTINE_RULES = {
        "morning routine": "Begin with visual schedule review.",
        "classroom":       "Use quiet signals; avoid disruption.",
        "lunch break":     "Allow unstructured wind-down time.",
        "therapy session": "Follow therapist protocol closely.",
    }
    TIME_RULES = {
        "morning":   "Energy building — set clear expectations.",
        "afternoon": "Possible fatigue — watch for frustration.",
        "evening":   "Wind-down mode — favour calming activities.",
    }

    def __init__(self):
        self.memory_path = BASE_DIR / "data/memory/interaction_history.json"
        self.policy_path = BASE_DIR / "data/memory/reinforcement_policy.json"
        self.memory  = self._load(self.memory_path, [])
        self.policy  = self._load(
            self.policy_path,
            {"sensitivity": 1.0, "feedback_count": 0,
             "emotion_freq": {e: 0 for e in TARGET_EMOTIONS}},
        )

    def _load(self, path, default):
        if path.exists():
            try:
                with open(path) as f:
                    return json.load(f)
            except Exception:
                pass
        return default

    def _save(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=4)
        except IOError:
            pass

    def save_interaction(self, data: dict):
        data["timestamp"] = str(datetime.datetime.now())
        em = data.get("emotion", "Natural")
        self.policy["emotion_freq"][em] = self.policy["emotion_freq"].get(em, 0) + 1
        self.policy["sensitivity"] = float(
            np.clip(self.policy["sensitivity"] + (int(data.get("feedback", 3)) - 3) * 0.05, 0.5, 2.0)
        )
        self.policy["feedback_count"] += 1
        self.memory.append(data)
        self._save(self.memory_path, {"history": self.memory[-200:]})
        self._save(self.policy_path, self.policy)

    def get_intervention(self, emotion, location="", routine="", time_of_day=""):
        base   = self.INTERVENTIONS.get(emotion, self.INTERVENTIONS["Natural"])
        ctx    = []
        loc_k  = next((k for k in self.LOCATION_RULES if k in location.lower()), None)
        rtn_k  = next((k for k in self.ROUTINE_RULES  if k in routine.lower()),  None)
        tim_k  = next((k for k in self.TIME_RULES     if k in time_of_day.lower()), None)
        if loc_k: ctx.append(self.LOCATION_RULES[loc_k])
        if rtn_k: ctx.append(self.ROUTINE_RULES[rtn_k])
        if tim_k: ctx.append(self.TIME_RULES[tim_k])
        hist = [r.get("emotion") for r in self.memory[-5:]]
        if hist.count(emotion) >= 3:
            ctx.append(f"Repeated '{emotion}' — consider escalating support.")
        return {**base, "context": ctx, "sensitivity": self.policy["sensitivity"]}

    def speak(self, text: str) -> Optional[str]:
        ts   = datetime.datetime.now().strftime("%H%M%S_%f")
        path = BASE_DIR / f"outputs/audio/speech_{ts}.mp3"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            gTTS(text=text, lang="en", slow=False).save(str(path))
            return str(path)
        except Exception as exc:
            logger.error(f"TTS failed: {exc}")
            return None

    def generate_schedule_card(self, tasks: List[str], emotion: str) -> Optional[np.ndarray]:
        COLORS = {
            "anger": (60, 80, 200), "fear": (200, 80, 140), "joy": (60, 180, 80),
            "sadness": (200, 130, 60), "surprise": (40, 180, 200), "Natural": (80, 160, 120),
        }
        bg  = COLORS.get(emotion, (120, 120, 120))
        img = np.ones((420, 680, 3), dtype=np.uint8) * 18
        cv2.rectangle(img, (0, 0), (680, 65), bg, -1)
        cv2.putText(
            img, f"{emotion.upper()} - Support Plan",
            (16, 44), cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2,
        )
        for i, t in enumerate(tasks[:4]):
            y = 110 + i * 75
            cv2.rectangle(img, (20, y - 28), (660, y + 28), (38, 38, 38), -1)
            cv2.rectangle(img, (20, y - 28), (660, y + 28), bg, 2)
            cv2.circle(img, (52, y), 16, bg, -1)
            cv2.putText(img, str(i + 1), (46, y + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(img, t, (78, y + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.72, (220, 220, 220), 2)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Encode to PNG base64 for JSON transport
        _, buf = cv2.imencode(".png", rgb)
        return base64.b64encode(buf).decode("utf-8")


engine = BehavioralEngine()

# =============================================================================
# 10.  CORE INFERENCE FUNCTION
# =============================================================================

def run_inference(
    image_bytes: bytes,
    audio_bytes: Optional[bytes],
    routine: str,
    time_of_day: str,
    location: str,
    feedback_rating: int,
) -> Dict[str, Any]:
    """
    Returns a dict with all inference results.
    Images are returned as base64-encoded PNG strings.
    Audio is returned as a base64-encoded MP3 string.
    """
    registry.load()

    # ── Decode image ──────────────────────────────────────────────────────────
    nparr = np.frombuffer(image_bytes, np.uint8)
    image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image_np is None:
        raise ValueError("Could not decode image. Ensure it is a valid JPEG/PNG.")
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    if image_np.ndim != 3 or image_np.shape[2] != 3:
        raise ValueError(f"Invalid image shape: {image_np.shape}")

    # ── Image preprocessing ───────────────────────────────────────────────────
    raw_rgb_u8 = cv2.resize(image_np, (IMG_SIZE, IMG_SIZE))
    if raw_rgb_u8.dtype != np.uint8:
        raw_rgb_u8 = np.clip(raw_rgb_u8, 0, 255).astype(np.uint8)

    img_f32   = _efficientnet_preprocess(raw_rgb_u8.astype(np.float32))
    img_batch = img_f32[np.newaxis]

    face_logits = registry.vision_prob(img_batch, training=False)
    face_probs  = np.array(face_logits)[0]

    # ── Audio preprocessing ───────────────────────────────────────────────────
    has_audio = False
    mfcc      = np.zeros(AUDIO_SHAPE, np.float32)
    audio_msg = "No audio — facial prediction only."

    if audio_bytes:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        try:
            feat = extract_mfcc_spectrogram(tmp_path)
            if feat is not None:
                mfcc      = feat
                has_audio = True
                audio_msg = "Multimodal fusion active."
            else:
                audio_msg = "Audio extraction failed — facial fallback."
        finally:
            os.unlink(tmp_path)

    audio_batch  = mfcc[np.newaxis]
    audio_logits = registry.audio_prob(audio_batch, training=False)
    audio_probs  = np.array(audio_logits)[0]

    # ── Joint inference ───────────────────────────────────────────────────────
    joint_out = registry.joint_infer(
        {"joint_face_input": img_batch, "joint_audio_input": audio_batch},
        training=False,
    )
    if isinstance(joint_out, (list, tuple)) and len(joint_out) >= 2:
        gate_val = float(np.array(joint_out[1]).flatten()[0])
    else:
        gate_val = 0.7

    img_quality         = HybridFusion.assess_image_quality(
        cv2.cvtColor(raw_rgb_u8, cv2.COLOR_RGB2BGR)
    )
    fused, face_w, audio_w = HybridFusion.fuse(
        face_probs, audio_probs, gate_val, img_quality
    )

    sens   = engine.policy.get("sensitivity", 1.0)
    fused  = np.power(fused, sens)
    fused /= fused.sum() + 1e-9

    pred_idx   = int(np.argmax(fused))
    emotion    = IDX_TO_EMOTION[pred_idx]
    confidence = float(fused[pred_idx])

    # ── Intervention ──────────────────────────────────────────────────────────
    interv    = engine.get_intervention(emotion, location, routine, time_of_day)
    tts_path  = engine.speak(interv["message"])
    sched_b64 = engine.generate_schedule_card(interv["tasks"], emotion)

    # ── TTS → base64 ──────────────────────────────────────────────────────────
    tts_b64 = None
    if tts_path and Path(tts_path).exists():
        with open(tts_path, "rb") as f:
            tts_b64 = base64.b64encode(f.read()).decode("utf-8")

    # ── Persist session ───────────────────────────────────────────────────────
    engine.save_interaction({
        "emotion": emotion, "confidence": confidence,
        "face_weight": round(face_w, 3), "audio_weight": round(audio_w, 3),
        "image_quality": img_quality, "learned_gate": round(gate_val, 3),
        "location": location, "routine": routine, "time": time_of_day,
        "feedback": int(feedback_rating),
        "modality": "Face+Audio" if has_audio else "Face-Only",
    })

    return {
        "emotion":         emotion,
        "confidence":      round(confidence, 4),
        "emoji":           EMOTION_EMOJI.get(emotion, ""),
        "color":           EMOTION_COLORS.get(emotion, "#888"),
        "modality":        "Face+Audio" if has_audio else "Face-Only",
        "audio_status":    audio_msg,
        "image_quality":   img_quality,
        "face_weight":     round(face_w, 3),
        "audio_weight":    round(audio_w, 3),
        "gate_value":      round(gate_val, 3),
        "sensitivity":     round(sens, 3),
        "probabilities":   {IDX_TO_EMOTION[i]: round(float(fused[i]), 4) for i in range(NUM_CLASSES)},
        "intervention": {
            "message":  interv["message"],
            "tasks":    interv["tasks"],
            "cues":     interv["cues"],
            "context":  interv.get("context", []),
        },
        "schedule_card_png_b64": sched_b64,
        "tts_mp3_b64":           tts_b64,
        "demo_mode":             registry._demo_mode,
    }


# =============================================================================
# 11.  FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="Multimodal Emotion AI API",
    description=(
        "ASD-support emotion recognition using EfficientNetB0 vision "
        "and MFCC-CNN audio fused via attention-gated cross-modal weighting. "
        "v7.2-azure — deployed on Azure App Service."
    ),
    version="7.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic response schemas ─────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    tf_version: str
    keras3: bool
    demo_mode: bool
    model_loaded: bool


class FeedbackRequest(BaseModel):
    rating: int   # 1–5


# ── Startup: warm up model ────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI startup: warming model registry …")
    registry.load()
    logger.info("Startup complete.")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    return {"message": "Multimodal Emotion AI v7.2-azure is running. Visit /docs for the API."}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    return HealthResponse(
        status="ok",
        version="7.2.0-azure",
        tf_version=tf.__version__,
        keras3=_KERAS3,
        demo_mode=getattr(registry, "_demo_mode", True),
        model_loaded=getattr(registry, "_loaded", False),
    )


@app.post("/predict", tags=["Inference"])
async def predict(
    image:           UploadFile = File(...,  description="Facial image (JPEG/PNG)"),
    audio:           Optional[UploadFile] = File(None, description="Vocal audio (WAV/MP3, optional)"),
    routine:         str  = Form("Classroom",  description="Current routine context"),
    time_of_day:     str  = Form("Morning",    description="Morning | Afternoon | Evening"),
    location:        str  = Form("School",     description="Environment label"),
    feedback_rating: int  = Form(3,            description="Intervention rating 1–5"),
):
    """
    Perform multimodal emotion recognition.

    - **image**: required; JPEG or PNG facial photo / frame.
    - **audio**: optional; WAV or MP3 vocal recording.
    - Returns emotion, confidence, intervention strategy, schedule card (PNG base64) and TTS audio (MP3 base64).
    """
    image_bytes = await image.read()
    audio_bytes = await audio.read() if audio else None

    if feedback_rating not in range(1, 6):
        raise HTTPException(status_code=422, detail="feedback_rating must be 1–5.")

    try:
        result = run_inference(
            image_bytes=image_bytes,
            audio_bytes=audio_bytes,
            routine=routine,
            time_of_day=time_of_day,
            location=location,
            feedback_rating=feedback_rating,
        )
        return JSONResponse(content=result)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception:
        logger.error("Unhandled inference error:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal inference error. Check server logs.")


@app.get("/policy", tags=["Session"])
async def get_policy():
    """Return the current reinforcement learning policy state."""
    return JSONResponse(content=engine.policy)


@app.get("/history", tags=["Session"])
async def get_history(n: int = 20):
    """Return the last *n* session records (default 20, max 200)."""
    n = min(n, 200)
    return JSONResponse(content={"history": engine.memory[-n:]})


@app.post("/feedback", tags=["Session"])
async def submit_feedback(body: FeedbackRequest):
    """
    Submit a retrospective feedback rating (1–5) for the most recent session.
    Adjusts the policy sensitivity accordingly.
    """
    if not engine.memory:
        raise HTTPException(status_code=404, detail="No session history found.")
    if body.rating not in range(1, 6):
        raise HTTPException(status_code=422, detail="rating must be 1–5.")

    last = engine.memory[-1]
    last["feedback"] = body.rating
    engine.policy["sensitivity"] = float(
        np.clip(
            engine.policy["sensitivity"] + (body.rating - 3) * 0.05,
            0.5, 2.0,
        )
    )
    engine.policy["feedback_count"] += 1
    engine._save(engine.policy_path, engine.policy)
    return {"updated_sensitivity": round(engine.policy["sensitivity"], 3)}


# =============================================================================
# 12.  ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting Multimodal Emotion AI v7.2-azure on port {port} …")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        workers=1,          # keep at 1 — TF/Keras not multi-process safe without care
        log_level="info",
    )
