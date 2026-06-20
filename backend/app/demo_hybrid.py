import json
import math
import pickle
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import config as ml_config
from app.schemas import FeatureReport, ScreeningResult


FULL_TRAINING_DIR = ml_config.RESULTS_DIR / "full_hybrid_cnn_appdr_full_training"
SEVERITY_MODEL_PATH = FULL_TRAINING_DIR / "hybrid_5class_best_model.pkl"
CNN_CHECKPOINT_PATH = (
    FULL_TRAINING_DIR
    / "cnn_sources"
    / "efficientnet_b3_full_training"
    / "best_model.pt"
)
BINARY_MODEL_PATH = ml_config.RESULTS_DIR / "binary" / "best_model.pkl"
BINARY_METADATA_PATH = ml_config.RESULTS_DIR / "binary" / "best_model_metadata.json"
BINARY_THRESHOLD_PATH = ml_config.RESULTS_DIR / "binary" / "optimal_threshold.json"
DEFAULT_BINARY_THRESHOLD = 0.20
SCREENING_MODEL_SOURCE = "appdr_binary_svm"
SEVERITY_MODEL_SOURCE = "full_training_hybrid_5class_xgboost"
CNN_SOURCE_ID = "efficientnet_b3_full_training"
CNN_INPUT_SIZE = 384
CNN_BACKBONE = "efficientnet_b3"

_DEMO_BUNDLE: dict[str, Any] | None = None
_DEMO_LOAD_ATTEMPTED = False


def classify_demo_hybrid(
    image_bgr: np.ndarray,
    features: FeatureReport,
) -> ScreeningResult | None:
    bundle = load_demo_bundle()
    if bundle is None:
        return None

    row = build_demo_feature_row(image_bgr, features, bundle)
    five_payload = bundle["five_model"]
    binary_payload = bundle["binary_model"]
    five_model = five_payload["model"]
    five_features = list(five_payload["features"])
    binary_model = binary_payload["model"]
    binary_features = list(binary_payload["features"])

    five_frame = ensure_feature_frame(row, five_features)
    binary_frame = ensure_feature_frame(row, binary_features)
    stage = int(five_model.predict(five_frame)[0])
    stage_probabilities = model_probabilities(five_model, five_frame, [0, 1, 2, 3, 4])
    stage_confidence = max(stage_probabilities.values()) if stage_probabilities else None
    referable_score = binary_score(binary_model, binary_frame)
    threshold = float(binary_payload.get("threshold") or 0.5)
    referable = bool(referable_score >= threshold)
    non_referable_score = float(np.clip(1.0 - referable_score, 0.0, 1.0))
    medical_label = ml_config.CLASS_NAMES.get(stage, "Medical severity label unavailable")
    probabilities = {
        ml_config.CLASS_NAMES[label]: float(stage_probabilities.get(label, 0.0))
        for label in [0, 1, 2, 3, 4]
    }
    probabilities["Non-Referable"] = non_referable_score
    probabilities["Referable"] = float(np.clip(referable_score, 0.0, 1.0))
    consistency = resolve_dual_model_screening(referable, stage)
    final_referable = bool(consistency["referable"])
    screening_status = str(consistency["label"])
    recommendation = str(consistency["recommendation"])
    screening = {
        "status": screening_status,
        "referable": final_referable,
        "rule": (
            f"Production binary screening threshold {threshold:.2f}; "
            "severity grades 2-4 require referable review."
        ),
        "recommendation": recommendation,
    }

    return ScreeningResult(
        classification=f"{screening_status}: {medical_label}",
        referable=final_referable,
        dr_probability=round(float(referable_score) * 100.0, 1),
        stage=stage,
        stage_label=medical_label,
        medical_label=medical_label,
        explanation=ml_config.CLASS_EXPLANATIONS.get(stage, medical_label),
        recommendation=recommendation,
        reason=(
            "Task-specific routing used the production AppDR binary screening model "
            "for referable screening and the full-training severity model with "
            "handcrafted and image-derived prediction features for severity support."
        ),
        disclaimer=(
            "This result is an automated screening support output and is not a "
            "final diagnosis. Please confirm with an ophthalmologist."
        ),
        model_type=ml_config.DUAL_MODEL_MODE,
        confidence=stage_confidence,
        confidence_label=confidence_label(stage_confidence),
        probabilities=probabilities,
        screening=screening,
        screening_recommendation=recommendation,
        consistency_status=str(consistency["status"]),
        raw_binary_prediction=int(referable),
        raw_severity_prediction=stage,
        binary_model_source=SCREENING_MODEL_SOURCE,
        severity_model_source=SEVERITY_MODEL_SOURCE,
    )


def resolve_dual_model_screening(
    binary_referable: bool,
    severity_grade: int,
) -> dict[str, object]:
    severity_requires_referral = severity_grade >= 2
    if binary_referable != severity_requires_referral:
        status = (
            "severity_escalation"
            if severity_requires_referral
            else "screening_severity_disagreement"
        )
        return {
            "status": status,
            "referable": True,
            "result": "referable_review",
            "label": "Referable / Needs ophthalmologist review",
            "recommendation": (
                "The screening and severity outputs require clinical confirmation."
            ),
        }

    if binary_referable:
        return {
            "status": "aligned",
            "referable": True,
            "result": "referable",
            "label": "Referable DR",
            "recommendation": "Ophthalmology evaluation is recommended.",
        }

    return {
        "status": "aligned",
        "referable": False,
        "result": "non_referable",
        "label": "Non-referable DR",
        "recommendation": (
            "Continue routine eye care and confirm the screening result with an "
            "ophthalmologist."
        ),
    }


def load_demo_bundle() -> dict[str, Any] | None:
    global _DEMO_BUNDLE, _DEMO_LOAD_ATTEMPTED
    if _DEMO_LOAD_ATTEMPTED:
        return _DEMO_BUNDLE

    _DEMO_LOAD_ATTEMPTED = True
    paths = {
        "five_model": SEVERITY_MODEL_PATH,
        "binary_model": BINARY_MODEL_PATH,
        "cnn_checkpoint": CNN_CHECKPOINT_PATH,
    }
    if not all(path.exists() for path in paths.values()):
        _DEMO_BUNDLE = None
        return None

    with paths["five_model"].open("rb") as file:
        five_model = pickle.load(file)
    with paths["binary_model"].open("rb") as file:
        binary_model = pickle.load(file)
    binary_metadata = load_json(BINARY_METADATA_PATH)
    binary_features = binary_metadata.get("feature_names", ml_config.FEATURE_NAMES)
    if not isinstance(binary_features, list) or not all(
        isinstance(name, str) for name in binary_features
    ):
        binary_features = list(ml_config.FEATURE_NAMES)
    threshold_payload = load_json(BINARY_THRESHOLD_PATH)
    threshold = float(threshold_payload.get("threshold", DEFAULT_BINARY_THRESHOLD))

    _DEMO_BUNDLE = {
        "five_model": five_model,
        "binary_model": {
            "model": binary_model,
            "features": binary_features,
            "threshold": threshold,
        },
        "cnn_checkpoint": paths["cnn_checkpoint"],
        "cnn_model": None,
        "device": None,
    }
    return _DEMO_BUNDLE


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def build_demo_feature_row(
    image_bgr: np.ndarray,
    features: FeatureReport,
    bundle: dict[str, Any],
) -> dict[str, float]:
    row = {name: float((features.expanded_features or {}).get(name, 0.0)) for name in ml_config.FEATURE_NAMES}
    cnn_row, _embedding = infer_demo_cnn(image_bgr, bundle)
    row.update(cnn_row)
    return row


def infer_demo_cnn(image_bgr: np.ndarray, bundle: dict[str, Any]) -> tuple[dict[str, float], np.ndarray]:
    import torch
    import torch.nn as nn
    from torchvision import transforms
    import timm

    class GeM(nn.Module):
        def __init__(self, p: float = 3.0, eps: float = 1e-6):
            super().__init__()
            self.p = nn.Parameter(torch.ones(1) * p)
            self.eps = eps

        def forward(self, x):
            return torch.nn.functional.avg_pool2d(
                x.clamp(min=self.eps).pow(self.p),
                (x.size(-2), x.size(-1)),
            ).pow(1.0 / self.p).flatten(1)

    class CnnSourceModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = timm.create_model(CNN_BACKBONE, pretrained=False, num_classes=0, global_pool="")
            self.pool = GeM()
            self.head = nn.Linear(int(self.encoder.num_features), 5)

        def forward_features_and_output(self, x):
            features = self.encoder(x)
            pooled = features if features.ndim == 2 else self.pool(features)
            output = self.head(pooled)
            return pooled, output

    device = bundle.get("device")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bundle["device"] = device
    model = bundle.get("cnn_model")
    if model is None:
        checkpoint = torch.load(bundle["cnn_checkpoint"], map_location=device, weights_only=False)
        model = CnnSourceModel().to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        bundle["cnn_model"] = model

    transform = transforms.Compose(
        [
            transforms.Resize((CNN_INPUT_SIZE, CNN_INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = transform(Image.fromarray(image_rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        pooled, logits_tensor = model.forward_features_and_output(tensor)
    embedding = pooled.detach().cpu().numpy()[0].astype(np.float32)
    logits = logits_tensor.detach().cpu().numpy()[0].astype(float)
    exp = np.exp(logits - np.max(logits))
    probs = (exp / exp.sum()).astype(float).tolist()
    pred_class = int(np.argmax(probs))
    severity = float(sum(idx * prob for idx, prob in enumerate(probs)))
    referable = float(sum(probs[2:]))
    top = sorted(probs, reverse=True)
    entropy = -sum(float(prob) * math.log(max(float(prob), 1e-9)) for prob in probs)
    row = {
        f"cnn_{CNN_SOURCE_ID}__severity": severity,
        f"cnn_{CNN_SOURCE_ID}__predicted_class": pred_class,
        f"cnn_{CNN_SOURCE_ID}__referable_probability": referable,
        f"cnn_{CNN_SOURCE_ID}__max_probability": float(max(probs)),
        f"cnn_{CNN_SOURCE_ID}__entropy": float(entropy),
        f"cnn_{CNN_SOURCE_ID}__uncertainty": float(1.0 - max(probs)),
        f"cnn_{CNN_SOURCE_ID}__margin": float(top[0] - top[1]),
    }
    for idx, value in enumerate(probs):
        row[f"cnn_{CNN_SOURCE_ID}__prob_class_{idx}"] = float(value)
    for idx, value in enumerate(logits):
        row[f"cnn_{CNN_SOURCE_ID}__logit_class_{idx}"] = float(value)
    return row, embedding


def ensure_feature_frame(row: dict[str, float], features: list[str]) -> pd.DataFrame:
    return pd.DataFrame([{name: float(row.get(name, 0.0)) for name in features}])


def model_probabilities(model: Any, frame: pd.DataFrame, labels: list[int]) -> dict[int, float]:
    if not hasattr(model, "predict_proba"):
        pred = int(model.predict(frame)[0])
        return {label: 1.0 if label == pred else 0.0 for label in labels}
    values = model.predict_proba(frame)[0]
    classes = getattr(model, "classes_", labels)
    return {int(label): float(prob) for label, prob in zip(classes, values)}


def binary_score(model: Any, frame: pd.DataFrame) -> float:
    if hasattr(model, "predict_proba"):
        values = model.predict_proba(frame)[0]
        classes = list(getattr(model, "classes_", [0, 1]))
        if 1 in classes:
            return float(values[classes.index(1)])
        return float(values[-1])
    if hasattr(model, "decision_function"):
        score = float(model.decision_function(frame)[0])
        return float(1.0 / (1.0 + math.exp(-score)))
    return float(model.predict(frame)[0])


def confidence_label(confidence: float | None) -> str:
    if confidence is None:
        return "Medium Confidence"
    if confidence >= 0.75:
        return "High Confidence"
    if confidence >= 0.45:
        return "Medium Confidence"
    return "Low Confidence"
