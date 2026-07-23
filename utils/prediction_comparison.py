"""utils.prediction_comparison

Prediction comparison layer for multi-model prediction system.

This module provides:
- compare_predictions: Compare predictions from multiple models
- select_best_prediction: Select best prediction based on confidence and consensus
- format_predictions_for_ai: Format predictions for AI verification
"""

import logging
from collections import Counter
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)

# The only disease classes supported by the ML system.
# Gemini must not introduce diseases outside this scope.
SUPPORTED_CLASSES = ["Acne", "Psoriasis", "Tinea", "Vitiligo"]


def compare_predictions(model_predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare predictions from multiple models.
    
    Args:
        model_predictions: List of prediction results from multiple models
            Each dict contains: model_name, model_type, predicted_class, confidence, all_predictions
    
    Returns:
        Dictionary containing:
            - consensus_class: Most commonly predicted class
            - consensus_count: Number of models predicting the consensus class
            - best_ml_prediction: Prediction with highest confidence
            - confidence_variance: Variance in confidence scores
            - prediction_agreement: Whether models agree on prediction
    """
    if not model_predictions:
        return {
            "consensus_class": "Unknown",
            "consensus_count": 0,
            "best_ml_prediction": None,
            "confidence_variance": 0.0,
            "prediction_agreement": False,
        }
    
    # Extract predictions
    predictions = [(p["predicted_class"], p["confidence"]) for p in model_predictions]
    classes = [p[0] for p in predictions]
    confidences = [p[1] for p in predictions]
    
    class_counts = Counter(classes)
    consensus_class = class_counts.most_common(1)[0][0]
    consensus_count = class_counts[consensus_class]
    
    # Find best prediction (highest confidence)
    best_prediction = max(model_predictions, key=lambda x: x["confidence"])
    
    # Calculate confidence variance
    import numpy as np
    confidence_variance = float(np.var(confidences)) if len(confidences) > 1 else 0.0
    
    # Determine if models agree
    prediction_agreement = consensus_count == len(model_predictions)
    
    return {
        "consensus_class": consensus_class,
        "consensus_count": consensus_count,
        "best_ml_prediction": best_prediction,
        "confidence_variance": confidence_variance,
        "prediction_agreement": prediction_agreement,
        "total_models": len(model_predictions),
        "model_vote_counts": dict(class_counts),
    }


def majority_vote(model_predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the majority-vote prediction and supporting model names."""
    if not model_predictions:
        return {
            "selected_class": "Unknown",
            "selected_confidence": 0.0,
            "supporting_models": [],
            "vote_counts": {},
            "method": "majority_voting",
        }

    vote_counts = Counter(p["predicted_class"] for p in model_predictions)
    selected_class, _ = vote_counts.most_common(1)[0]
    supporters = [p for p in model_predictions if p["predicted_class"] == selected_class]
    avg_confidence = float(np.mean([p["confidence"] for p in supporters])) if supporters else 0.0
    return {
        "selected_class": selected_class,
        "selected_confidence": avg_confidence,
        "supporting_models": [p["model_name"] for p in supporters],
        "vote_counts": dict(vote_counts),
        "method": "majority_voting",
    }


def build_ai_verification_summary(
    model_predictions: list[dict[str, Any]],
    comparison_result: dict[str, Any],
    majority_result: dict[str, Any],
    soft_vote_result: dict[str, Any],
    ai_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a concise verification decision from model agreement and optional Gemini output."""

    if not model_predictions:
        return {
            "summary": "No model predictions were available for verification.",
            "agreement": "none",
            "final_class": "Unknown",
            "final_confidence": 0.0,
            "source": "unavailable",
        }

    total_models = comparison_result["total_models"]
    consensus_class = comparison_result["consensus_class"]
    consensus_count = comparison_result["consensus_count"]

    soft_class = soft_vote_result["selected_class"]
    soft_confidence = soft_vote_result["selected_confidence"]

    best_model = comparison_result["best_ml_prediction"]

    agreement = (
        "agreement"
        if consensus_count == total_models
        else "disagreement"
    )

    # ==========================
    # Summary Text
    # ==========================
    if consensus_count == total_models:

        summary = (
            f"All {total_models} models predict {consensus_class}. "
            f"Prediction accepted with full agreement."
        )

    elif consensus_count > total_models / 2:

        summary = (
            f"{consensus_count}/{total_models} models predict "
            f"{consensus_class}. "
            f"Majority voting selected {consensus_class}. "
            f"Highest model confidence: "
            f"{best_model['model_type']} "
            f"({best_model['confidence']:.1f}%)."
        )

    else:

        summary = (
            f"The models disagree with no clear majority. "
            f"Soft voting selects {soft_class}. "
            f"Highest model confidence: "
            f"{best_model['model_type']} "
            f"({best_model['confidence']:.1f}%)."
        )

    # ==========================
    # FINAL DECISION
    # ==========================

    # Case 1: Full Agreement
    if consensus_count == total_models:

        final_class = consensus_class
        final_confidence = majority_result["selected_confidence"]
        source = "unanimous_voting"

    # Case 2: Majority Exists
    elif consensus_count > total_models / 2:

        final_class = consensus_class
        final_confidence = majority_result["selected_confidence"]
        source = "majority_voting"

    # Case 3: No Majority
    else:

        final_class = soft_class
        final_confidence = soft_confidence
        source = "soft_voting"

    # ==========================
    # Optional Gemini Override (with class-scope validation)
    # ==========================
    if ai_result:

        ai_prediction_raw = (
            ai_result.get("ai_prediction")
            or ai_result.get("disease")
        )

        ai_confidence = float(
            ai_result.get("ai_confidence")
            or ai_result.get("confidence")
            or 0.0
        )

        explanation = ai_result.get(
            "explanation",
            ""
        )

        # Validate that AI prediction is within the supported 4-class scope
        ai_prediction_valid = (
            ai_prediction_raw
            and str(ai_prediction_raw).lower() != "unknown"
            and str(ai_prediction_raw) in SUPPORTED_CLASSES
        )

        if (
            ai_prediction_valid
            and ai_confidence >= final_confidence + 10.0
        ):

            final_class = str(ai_prediction_raw)
            final_confidence = ai_confidence
            source = "ai_verification"

            summary += (
                f" AI verification favors "
                f"{final_class} "
                f"({final_confidence:.1f}%)."
            )

        elif ai_prediction_raw and str(ai_prediction_raw) not in SUPPORTED_CLASSES:

            LOGGER.warning(
                "AI returned unsupported disease '%s' — "
                "not overriding ML prediction '%s'.",
                ai_prediction_raw,
                final_class,
            )

            summary += (
                f" AI returned unsupported disease "
                f"'{ai_prediction_raw}' — "
                f"keeping ML prediction '{final_class}'."
            )

        elif explanation:

            summary += (
                f" AI verification note: "
                f"{explanation}"
            )

    return {
        "summary": summary,
        "agreement": agreement,
        "final_class": final_class,
        "final_confidence": final_confidence,
        "source": source,
    }


def select_best_prediction(
    model_predictions: list[dict[str, Any]],
    comparison_result: dict[str, Any]
) -> dict[str, Any]:
    """Select best prediction based on confidence and consensus.
    
    Args:
        model_predictions: List of prediction results from multiple models
        comparison_result: Result from compare_predictions
    
    Returns:
        Dictionary containing:
            - selected_class: Selected disease class
            - selected_confidence: Confidence score
            - selection_method: Method used (consensus, highest_confidence, or fallback)
            - supporting_models: List of models supporting this prediction
    """
    if not model_predictions:
        return {
            "selected_class": "Unknown",
            "selected_confidence": 0.0,
            "selection_method": "fallback",
            "supporting_models": [],
        }
    
    consensus_class = comparison_result["consensus_class"]
    consensus_count = comparison_result["consensus_count"]
    total_models = comparison_result["total_models"]
    best_prediction = comparison_result["best_ml_prediction"]
    
    # Selection logic:
    # 1. If all models agree, use consensus
    # 2. If majority agree (>50%), use consensus
    # 3. Otherwise, use highest confidence
    
    if consensus_count == total_models:
        # All models agree
        supporting_models = [
            p["model_name"] for p in model_predictions
            if p["predicted_class"] == consensus_class
        ]
        return {
            "selected_class": consensus_class,
            "selected_confidence": best_prediction["confidence"],
            "selection_method": "full_consensus",
            "supporting_models": supporting_models,
        }
    elif consensus_count > total_models / 2:
        # Majority agree
        supporting_models = [
            p["model_name"] for p in model_predictions
            if p["predicted_class"] == consensus_class
        ]
        # Use average confidence from supporting models
        supporting_confidences = [
            p["confidence"] for p in model_predictions
            if p["predicted_class"] == consensus_class
        ]
        avg_confidence = sum(supporting_confidences) / len(supporting_confidences)
        return {
            "selected_class": consensus_class,
            "selected_confidence": avg_confidence,
            "selection_method": "majority_consensus",
            "supporting_models": supporting_models,
        }
    else:
        # No consensus, use highest confidence
        return {
            "selected_class": best_prediction["predicted_class"],
            "selected_confidence": best_prediction["confidence"],
            "selection_method": "highest_confidence",
            "supporting_models": [best_prediction["model_name"]],
        }


def format_predictions_for_ai(
    model_predictions: list[dict[str, Any]],
    comparison_result: dict[str, Any],
    selected_prediction: dict[str, Any]
) -> str:
    """Format predictions for AI verification.
    
    Args:
        model_predictions: List of prediction results from multiple models
        comparison_result: Result from compare_predictions
        selected_prediction: Result from select_best_prediction
    
    Returns:
        Formatted string for AI verification
    """
    lines = []
    lines.append("Multi-Model Prediction Results:")
    lines.append("=" * 50)
    
    for pred in model_predictions:
        lines.append(
            f"{pred['model_type'].upper()}: {pred['predicted_class']} "
            f"({pred['confidence']:.1f}%)"
        )
    
    lines.append("")
    lines.append("Comparison Summary:")
    lines.append(f"Consensus: {comparison_result['consensus_class']} "
                f"({comparison_result['consensus_count']}/{comparison_result['total_models']} models)")
    lines.append(f"Agreement: {'Yes' if comparison_result['prediction_agreement'] else 'No'}")
    lines.append(f"Confidence Variance: {comparison_result['confidence_variance']:.2f}")
    
    lines.append("")
    lines.append("Selected Prediction:")
    lines.append(f"Class: {selected_prediction['selected_class']}")
    lines.append(f"Confidence: {selected_prediction['selected_confidence']:.1f}%")
    selection_method = selected_prediction.get("selection_method", selected_prediction.get("method", "unknown"))
    supporting_models = selected_prediction.get("supporting_models", [])
    lines.append(f"Method: {selection_method}")
    lines.append(f"Supporting Models: {', '.join(supporting_models)}")
    
    return "\n".join(lines)
