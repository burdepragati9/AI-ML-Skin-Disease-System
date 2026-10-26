import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils.model_manager import get_model_manager
from utils.prediction_comparison import (
    build_ai_verification_summary,
    compare_predictions,
    majority_vote,
    SUPPORTED_CLASSES,
)
from utils.ai_recognition import verify_multi_model_predictions_with_ai

LOGGER = logging.getLogger(__name__)


def _preprocess_image(image, target_size):
    image = image.convert("RGB")
    image = image.resize((target_size, target_size))

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array

def _safe_round(value: Any, ndigits: int = 2) -> float:
    try:
        return round(float(value), ndigits)
    except Exception:
        return 0.0


def predict_disease(image: Image.Image) -> Dict[str, Any]:
    try:
        manager = get_model_manager()

        model = manager.get_active_model()
        target_size = int(model.input_shape[1])

        image_array = _preprocess_image(image, target_size)

        model_predictions = manager.predict_with_all_models(image_array)
        comparison = compare_predictions(model_predictions)
        majority = majority_vote(model_predictions)
        soft = manager.soft_vote(model_predictions)
        # Self-learning eligibility metadata only.
        # Does not change the final prediction.
        probabilities = np.asarray(
            soft.get("probabilities") or [],
            dtype=np.float64,
        )

        probabilities = probabilities[
            np.isfinite(probabilities) & (probabilities > 0)
        ]

        if probabilities.size:
            probabilities = probabilities / probabilities.sum()
            entropy = float(-np.sum(probabilities * np.log(probabilities)))
        else:
            entropy = 0.0

        raw_ml_confidence = soft.get("selected_confidence")
        try:
            raw_ml_confidence = float(raw_ml_confidence)
            valid_ml_confidence = (
                np.isfinite(raw_ml_confidence)
                and 0.0 <= raw_ml_confidence <= 100.0
            )
        except (TypeError, ValueError, OverflowError):
            valid_ml_confidence = False

        ml_confidence = (
            round(raw_ml_confidence, 2)
            if valid_ml_confidence
            else None
        )

        threshold_triggered = (
            valid_ml_confidence and (ml_confidence < 80.0 or entropy > 0.9)
        )

        # AI verification (enabled via GEMINI_API_KEY/config inside ai_recognition)
        ai_result: Optional[Dict[str, Any]] = verify_multi_model_predictions_with_ai(
            image=image,
            model_predictions=model_predictions,
            comparison_summary=build_ai_verification_summary(
                model_predictions=model_predictions,
                comparison_result=comparison,
                majority_result=majority,
                soft_vote_result=soft,
                ai_result=None,
            ).get("summary", ""),
        )

        ai_verification_summary = build_ai_verification_summary(
            model_predictions=model_predictions,
            comparison_result=comparison,
            majority_result=majority,
            soft_vote_result=soft,
            ai_result=ai_result,
        )

        ai_candidate_disease = (
            ai_result.get("ai_prediction")
            if isinstance(ai_result, dict)
            else None
        )

        ai_candidate_confidence = (
            ai_result.get("ai_confidence")
            if isinstance(ai_result, dict)
            else None
        )

        try:
            valid_ai_confidence = (
                ai_candidate_confidence is not None
                and np.isfinite(float(ai_candidate_confidence))
                and 0 <= float(ai_candidate_confidence) <= 100
            )
        except (TypeError, ValueError, OverflowError):
            valid_ai_confidence = False

        self_learning_eligible = (
            threshold_triggered
            and ai_candidate_disease in SUPPORTED_CLASSES
            and valid_ai_confidence
        )

        LOGGER.warning(
            "[Self-learning Debug] threshold_triggered=%s, "
            "ml_confidence=%s, entropy=%s, "
            "ai_candidate_disease=%s, ai_candidate_confidence=%s",
            threshold_triggered,
            ml_confidence,
            _safe_round(entropy, 4),
            ai_candidate_disease,
            ai_candidate_confidence,
        )

        self_learning_candidate = {
            "eligible": bool(self_learning_eligible),
            "disease": ai_candidate_disease,
            "confidence": (
                _safe_round(ai_candidate_confidence, 2)
                if ai_candidate_confidence is not None
                else 0.0
            ),
            "ml_confidence": ml_confidence,
            "entropy": _safe_round(entropy, 4),
        }

        # Final selection uses ai_verification_summary which already applies override rules.
        predicted_disease = ai_verification_summary.get("final_class") or soft.get(
            "selected_class"
        )
        confidence = _safe_round(
            ai_verification_summary.get("final_confidence", soft.get("selected_confidence")),
            2,
        )

        available_models = manager.list_available_models()

        # Prediction source
        prediction_source = ai_verification_summary.get("source")
        if not prediction_source:
            prediction_source = "soft_voting"

        # =====================================================================
        # FINAL PREDICTION SAFETY VALIDATION GATE
        # =====================================================================
        # Ensure the final predicted_disease is always within the supported
        # 4-class scope. If an unsupported disease (e.g. from AI) reaches this
        # point, fall back to the best valid ML prediction.
        LOGGER.info("[Prediction Validation] Supported classes: %s", SUPPORTED_CLASSES)
        LOGGER.info("[Prediction Validation] ML prediction: %s", soft.get("selected_class", "N/A"))
        LOGGER.info("[Prediction Validation] ML confidence: %s", soft.get("selected_confidence", "N/A"))
        LOGGER.info("[Prediction Validation] AI prediction: %s",
                     ai_verification_summary.get("final_class", "N/A"))
        LOGGER.info("[Prediction Validation] AI confidence: %s",
                     ai_verification_summary.get("final_confidence", "N/A"))
        LOGGER.info("[Prediction Validation] AI prediction valid: %s",
                     str(predicted_disease in SUPPORTED_CLASSES).lower())

        if predicted_disease not in SUPPORTED_CLASSES:
            LOGGER.warning(
                "[Prediction Validation] Unsupported AI disease '%s' rejected. "
                "Falling back to valid ML prediction '%s' (confidence: %s).",
                predicted_disease,
                soft.get("selected_class", "Unknown"),
                soft.get("selected_confidence", 0.0),
            )
            predicted_disease = soft.get("selected_class", "Unknown")
            confidence = _safe_round(soft.get("selected_confidence", 0.0), 2)
            prediction_source = "majority_voting"
            # Also update ai_verification_summary so the returned dict is consistent
            ai_verification_summary["final_class"] = predicted_disease
            ai_verification_summary["final_confidence"] = confidence
            ai_verification_summary["source"] = prediction_source

        LOGGER.info("[Prediction Validation] Final disease: %s", predicted_disease)
        LOGGER.info("[Prediction Validation] Final source: %s", prediction_source)

        # Provide full fields requested
        return {
            "status": "success",
            "prediction_source": prediction_source,
            "predicted_disease": predicted_disease,
            "confidence": confidence,
            "model_predictions": model_predictions,
            "majority_vote": majority,
            "soft_vote": soft,
            "comparison": comparison,
            "available_models": available_models,
            "ai_verification": ai_verification_summary,
            "self_learning_candidate": self_learning_candidate,
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "predicted_disease": "Unknown",
            "confidence": 0.0,
            "model_predictions": [],
            "majority_vote": {},
            "soft_vote": {},
            "comparison": {},
            "available_models": [],
        }
