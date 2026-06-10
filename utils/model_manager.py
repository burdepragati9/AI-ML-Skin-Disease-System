"""utils.model_manager

Model manager for supporting multiple trained ML models.

This module provides:
- Load and manage multiple .keras models (CNN, MobileNet, EfficientNet, etc.)
- Model selection based on configuration or ensemble prediction
- Backward compatibility with single-model workflow
- Modular and scalable architecture for future model comparison

The manager uses the AVAILABLE_MODELS configuration from utils.config
to determine which models to load and their priorities.
"""

import json
import logging
from pathlib import Path
from typing import Any

import tensorflow as tf
import numpy as np

from utils.config import AVAILABLE_MODELS, ACTIVE_MODEL

LOGGER = logging.getLogger(__name__)


class ModelManager:
    """Manager for multiple trained ML models."""
    
    def __init__(self):
        self._models: dict[str, Any] = {}
        self._class_names: dict[str, list[str]] = {}
        self._active_model_name: str = ACTIVE_MODEL
        self._load_models()
    
    def _load_models(self) -> None:
        """Load all configured models from AVAILABLE_MODELS."""
        for model_name, model_config in AVAILABLE_MODELS.items():
            try:
                model_path = Path(model_config["path"])
                class_names_path = Path(model_config["class_names"])
                
                if not model_path.exists():
                    LOGGER.warning(f"Model file not found: {model_path}")
                    continue
                
                if not class_names_path.exists():
                    LOGGER.warning(f"Class names file not found: {class_names_path}")
                    continue
                
                # Load model
                model = tf.keras.models.load_model(model_path, compile=False)
                
                # Load class names
                class_names = json.loads(class_names_path.read_text(encoding="utf-8"))
                
                # Validate model output matches class names
                output_classes = int(model.output_shape[-1])
                if output_classes != len(class_names):
                    LOGGER.warning(
                        f"Model {model_name} output classes ({output_classes}) "
                        f"do not match class names ({len(class_names)})"
                    )
                    continue
                
                self._models[model_name] = model
                self._class_names[model_name] = class_names
                
                LOGGER.info(f"Loaded model: {model_name} ({model_config.get('type', 'unknown')})")
                
            except Exception as exc:
                LOGGER.error(f"Failed to load model {model_name}: {exc}")
    
    def get_model(self, model_name: str | None = None) -> Any:
        """Get a specific model by name, or the active model if name is None.
        
        Args:
            model_name: Name of the model to retrieve, or None for active model
            
        Returns:
            TensorFlow model object
            
        Raises:
            ValueError: If model not found
        """
        if model_name is None:
            model_name = self._active_model_name
        
        if model_name not in self._models:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(self._models.keys())}")
        
        return self._models[model_name]
    
    def get_class_names(self, model_name: str | None = None) -> list[str]:
        """Get class names for a specific model, or the active model if name is None.
        
        Args:
            model_name: Name of the model, or None for active model
            
        Returns:
            List of class names
            
        Raises:
            ValueError: If model not found
        """
        if model_name is None:
            model_name = self._active_model_name
        
        if model_name not in self._class_names:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(self._class_names.keys())}")
        
        return self._class_names[model_name]
    
    def get_active_model(self) -> Any:
        """Get the currently active model.
        
        Returns:
            TensorFlow model object
        """
        return self.get_model(self._active_model_name)
    
    def get_active_class_names(self) -> list[str]:
        """Get class names for the currently active model.
        
        Returns:
            List of class names
        """
        return self.get_class_names(self._active_model_name)
    
    def get_active_model_name(self) -> str:
        """Get the name of the currently active model.
        
        Returns:
            Name of the active model
        """
        return self._active_model_name
    
    def set_active_model(self, model_name: str) -> None:
        """Set the active model by name.
        
        Args:
            model_name: Name of the model to set as active
            
        Raises:
            ValueError: If model not found
        """
        if model_name not in self._models:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(self._models.keys())}")
        
        self._active_model_name = model_name
        LOGGER.info(f"Active model set to: {model_name}")
    
    def list_available_models(self) -> list[str]:
        """List all available model names.
        
        Returns:
            List of model names
        """
        return sorted(
            self._models.keys(),
            key=lambda name: AVAILABLE_MODELS.get(name, {}).get("priority", 999),
        )
    
    def get_model_info(self, model_name: str | None = None) -> dict[str, Any]:
        """Get information about a specific model.
        
        Args:
            model_name: Name of the model, or None for active model
            
        Returns:
            Dictionary with model information
        """
        if model_name is None:
            model_name = self._active_model_name
        
        if model_name not in self._models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self._models[model_name]
        config = AVAILABLE_MODELS.get(model_name, {})
        
        return {
            "name": model_name,
            "type": config.get("type", "unknown"),
            "priority": config.get("priority", 0),
            "input_shape": model.input_shape,
            "output_shape": model.output_shape,
            "num_classes": int(model.output_shape[-1]),
            "class_names": self._class_names[model_name],
            "is_active": model_name == self._active_model_name,
        }
    
    @staticmethod
    def _to_probabilities(raw_scores) -> np.ndarray:
        """Normalize a model output vector into validated probabilities."""
        scores = np.asarray(raw_scores, dtype=np.float64)
        if scores.ndim != 1:
            raise ValueError(f"Model output must be 1D per sample; got shape {scores.shape}")

        out_min = float(np.min(scores))
        out_max = float(np.max(scores))
        raw_sum = float(np.sum(scores))
        looks_like_probs = (
            out_min >= -1e-6
            and out_max <= 1.0 + 1e-6
            and abs(raw_sum - 1.0) <= 1e-2
        )

        if looks_like_probs:
            probs = scores
        else:
            exp = np.exp(scores - np.max(scores))
            probs = exp / np.sum(exp)

        prob_sum = float(np.sum(probs))
        if not (abs(prob_sum - 1.0) <= 1e-3):
            raise ValueError(f"Probability validation failed: sum(probs)={prob_sum}")
        if np.any(probs < -1e-6):
            raise ValueError(f"Probability validation failed: negative probs min={float(np.min(probs))}")
        return probs

    def predict_with_model(
        self,
        image_array,
        model_name: str | None = None
    ) -> tuple[str, float, list, np.ndarray]:
        """Make prediction using a specific model.
        
        Args:
            image_array: Preprocessed image array
            model_name: Name of the model to use, or None for active model
            
        Returns:
            Tuple of (predicted_class, confidence, all_predictions, probabilities)
        """
        model = self.get_model(model_name)
        class_names = self.get_class_names(model_name)
        
        # Make prediction
        raw_scores = model.predict(image_array, verbose=0)[0]
        probabilities = self._to_probabilities(raw_scores)
        
        # Get best prediction
        best_index = int(probabilities.argmax())
        confidence = float(probabilities[best_index]) * 100.0
        predicted_class = class_names[best_index]
        
        # Get all predictions sorted by confidence
        sorted_indices = probabilities.argsort()[::-1]
        all_predictions = [
            (class_names[int(i)], float(probabilities[int(i)]) * 100.0)
            for i in sorted_indices
        ]
        
        return predicted_class, confidence, all_predictions, probabilities

    def predict_with_all_models(self, image_array) -> list[dict[str, Any]]:
        """Make prediction using all available models.
        
        Args:
            image_array: Preprocessed image array
            
        Returns:
            List of dictionaries, each containing:
                - model_name: Name of the model
                - model_type: Type of the model (mobilenetv2, efficientnet, etc.)
                - predicted_class: Predicted disease class
                - confidence: Confidence score (0-100)
                - all_predictions: All predictions sorted by confidence
        """
        results = []
        
        for model_name in self.list_available_models():
            try:
                predicted_class, confidence, all_predictions, probabilities = self.predict_with_model(
                    image_array, model_name
                )
                
                model_info = self.get_model_info(model_name)
                
                results.append({
                    "model_name": model_name,
                    "model_type": model_info["type"],
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "all_predictions": all_predictions,
                    "top3": all_predictions[:3],
                    "probabilities": probabilities.tolist(),
                })
                
                LOGGER.info(f"Prediction from {model_name}: {predicted_class} ({confidence:.2f}%)")
                
            except Exception as exc:
                LOGGER.error(f"Failed to predict with model {model_name}: {exc}")
                continue
        
        return results

    def soft_vote(self, model_predictions: list[dict[str, Any]]) -> dict[str, Any]:
        """Average aligned class probabilities across loaded models."""
        if not model_predictions:
            return {
                "selected_class": "Unknown",
                "selected_confidence": 0.0,
                "probabilities": [],
                "top3": [],
                "method": "soft_voting",
                "selection_method": "soft_voting",
                "supporting_models": [],
            }

        class_names = self.get_active_class_names()
        probability_rows = [np.asarray(p["probabilities"], dtype=np.float64) for p in model_predictions]
        avg_probs = np.mean(np.stack(probability_rows, axis=0), axis=0)
        best_index = int(np.argmax(avg_probs))
        sorted_indices = np.argsort(avg_probs)[::-1]
        top3 = [(class_names[int(i)], float(avg_probs[int(i)]) * 100.0) for i in sorted_indices[:3]]

        return {
            "selected_class": class_names[best_index],
            "selected_confidence": float(avg_probs[best_index]) * 100.0,
            "probabilities": avg_probs.tolist(),
            "top3": top3,
            "method": "soft_voting",
            "selection_method": "soft_voting",
            "supporting_models": [p["model_name"] for p in model_predictions],
        }


# Global model manager instance
_model_manager: ModelManager | None = None


def get_model_manager() -> ModelManager:
    """Get the global model manager instance.
    
    Returns:
        ModelManager instance
    """
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


def reset_model_manager() -> None:
    """Reset the global model manager instance (useful for testing)."""
    global _model_manager
    _model_manager = None
