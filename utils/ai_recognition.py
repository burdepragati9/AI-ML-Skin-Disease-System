"""utils.ai_recognition

AI-based skin disease recognition using Google Gemini API.

This module provides:
- recognize_with_ai: Direct AI recognition for low-confidence ML predictions
- verify_prediction_with_ai: AI verification of ML predictions for improved accuracy

All AI requests are sanitized to remove PII before sending to external APIs.
"""

import base64
import io
import logging
from typing import Any

import google.generativeai as genai
from PIL import Image

from utils.config import GEMINI_API_KEY, GEMINI_MODEL_NAME
from utils.privacy import sanitize_payload

LOGGER = logging.getLogger(__name__)

# Configure Gemini API
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


def _image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string for API transmission."""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def _sanitize_ai_request_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Sanitize AI request payload to remove PII before sending to external API.
    
    This ensures no personal information (name, email, doctor ID, hospital name, etc.)
    is sent to AI services. Only image data and disease prediction data are allowed.
    """
    return sanitize_payload(payload)


def recognize_with_ai(image: Image.Image) -> dict[str, Any] | None:
    """Recognize skin disease using AI (Gemini) for low-confidence ML predictions.
    
    This function is called when ML confidence is low (<80%) or entropy is high (>0.9).
    It sends only the image to AI, with no PII.
    
    Args:
        image: PIL Image of the skin condition
        
    Returns:
        dict with keys: disease, confidence, severity, explanation
        or None if AI recognition fails
    """
    if not GEMINI_API_KEY:
        LOGGER.warning("GEMINI_API_KEY not configured, AI recognition unavailable")
        return None
    
    try:
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        
        # Convert image to bytes
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        image_bytes = buffered.getvalue()
        
        # Create sanitized payload - only image data, no PII
        payload = {
            "parts": [
                {
                    "mime_type": "image/jpeg",
                    "data": image_bytes
                },
                {
                    "text": """Analyze this skin image. This ML system supports only these four skin disease classes: Acne, Psoriasis, Tinea, Vitiligo.

You must select only one of these four classes.
If the image appears to show a different skin condition that is not one of these four classes, return "Unknown".
Never return any disease name outside this list.

Respond in this exact JSON format:
{
    "disease": "disease name or Unknown",
    "confidence": 0.0-100.0,
    "severity": "mild/moderate/severe",
    "explanation": "brief explanation"
}

Only respond with the JSON, no additional text."""
                }
            ]
        }
        
        # Sanitize payload to ensure no PII is sent
        sanitized_payload = _sanitize_ai_request_payload(payload)
        
        # Call Gemini API
        response = model.generate_content(
            [sanitized_payload["parts"][0], sanitized_payload["parts"][1]["text"]]
        )
        
        # Parse response
        result_text = response.text.strip()
        
        # Try to extract JSON from response
        import json
        import re
        
        # Find JSON in response
        json_match = re.search(r'\{[^}]+\}', result_text, re.DOTALL)
        if json_match:
            result_text = json_match.group(0)
        
        result = json.loads(result_text)
        
        # Validate required fields
        if not result.get("disease") or result.get("disease", "").lower() == "unknown":
            LOGGER.warning("AI returned unknown disease")
            return None
        
        return {
            "disease": result.get("disease", "Unknown"),
            "confidence": float(result.get("confidence", 0) or 0),
            "severity": result.get("severity", ""),
            "explanation": result.get("explanation", "")
        }
        
    except Exception as exc:
        LOGGER.error(f"AI recognition failed: {exc}")
        return None


def verify_prediction_with_ai(
    image: Image.Image,
    ml_prediction: str,
    ml_confidence: float
) -> dict[str, Any] | None:
    """Verify ML prediction using AI for improved accuracy.
    
    This function is called after ML prediction when confidence is above threshold
    to provide an additional layer of verification. It compares ML and AI predictions
    to improve overall confidence.
    
    Args:
        image: PIL Image of the skin condition
        ml_prediction: Disease predicted by ML model
        ml_confidence: Confidence score from ML model (0-100)
        
    Returns:
        dict with keys:
            - ai_prediction: Disease predicted by AI
            - ai_confidence: Confidence score from AI (0-100)
            - agreement: Whether ML and AI agree (True/False)
            - verification_source: "ML" or "AI" based on which to trust
        or None if verification fails
    """
    if not GEMINI_API_KEY:
        LOGGER.warning("GEMINI_API_KEY not configured, AI verification unavailable")
        return None
    
    try:
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        
        # Convert image to bytes
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        image_bytes = buffered.getvalue()
        
        # Create sanitized payload - only image and ML prediction, no PII
        payload = {
            "parts": [
                {
                    "mime_type": "image/jpeg",
                    "data": image_bytes
                },
                {
                    "text": f"""Analyze this skin image. The ML model predicted: "{ml_prediction}" with {ml_confidence:.1f}% confidence.

This ML system supports only these four skin disease classes: Acne, Psoriasis, Tinea, Vitiligo.
You must select only one of these four classes.
If the image appears to show a different skin condition that is not one of these four classes, return "Unknown".
Never return any disease name outside this list.

Verify if this prediction is correct. Respond in this exact JSON format:
{{
    "disease": "your predicted disease name or Unknown",
    "confidence": 0.0-100.0,
    "agreement": true/false,
    "explanation": "brief explanation"
}}

Only respond with the JSON, no additional text."""
                }
            ]
        }
        
        # Sanitize payload to ensure no PII is sent
        sanitized_payload = _sanitize_ai_request_payload(payload)
        
        # Call Gemini API
        response = model.generate_content(
            [sanitized_payload["parts"][0], sanitized_payload["parts"][1]["text"]]
        )
        
        # Parse response
        result_text = response.text.strip()
        
        # Try to extract JSON from response
        import json
        import re
        
        # Find JSON in response
        json_match = re.search(r'\{[^}]+\}', result_text, re.DOTALL)
        if json_match:
            result_text = json_match.group(0)
        
        result = json.loads(result_text)
        
        ai_disease = result.get("disease", "Unknown")
        ai_confidence = float(result.get("confidence", 0) or 0)
        agreement = result.get("agreement", False)
        
        # Determine which prediction to trust
        # If AI agrees with ML and has similar or higher confidence, trust ML
        # If AI disagrees and has higher confidence, trust AI
        verification_source = "ML"
        if agreement:
            verification_source = "ML"
        elif ai_confidence > ml_confidence + 10:  # AI significantly more confident
            verification_source = "AI"
        else:
            verification_source = "ML"
        
        return {
            "ai_prediction": ai_disease,
            "ai_confidence": ai_confidence,
            "agreement": agreement,
            "verification_source": verification_source,
            "explanation": result.get("explanation", "")
        }
        
    except Exception as exc:
        LOGGER.error(f"AI verification failed: {exc}")
        return None


def verify_multi_model_predictions_with_ai(
    image: Image.Image,
    model_predictions: list[dict[str, Any]],
    comparison_summary: str
) -> dict[str, Any] | None:
    """Verify multi-model predictions using AI for improved accuracy.
    
    This function is called after multi-model prediction to provide AI verification
    of the ensemble results. It analyzes all model outputs and suggests the best prediction.
    
    Args:
        image: PIL Image of the skin condition
        model_predictions: List of prediction results from multiple models
        comparison_summary: Formatted summary of model comparisons
        
    Returns:
        dict with keys:
            - ai_prediction: Disease predicted by AI
            - ai_confidence: Confidence score from AI (0-100)
            - verification_source: "ML" or "AI" based on which to trust
            - explanation: AI's explanation
        or None if verification fails
    """
    if not GEMINI_API_KEY:
        LOGGER.warning("GEMINI_API_KEY not configured, AI verification unavailable")
        return None
    
    try:
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        
        # Convert image to bytes
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        image_bytes = buffered.getvalue()
        
        # Create sanitized payload - only image and model predictions, no PII
        prompt_text = f"""Analyze this skin image and verify the multi-model predictions.

{comparison_summary}

This ML system supports only these four skin disease classes: Acne, Psoriasis, Tinea, Vitiligo.
You must select only one of these four classes.
If the image appears to show a different skin condition that is not one of these four classes, return "Unknown".
Never return any disease name outside this list.

Based on the image analysis and model predictions, provide your final assessment.
Respond in this exact JSON format:
{{
    "disease": "your final predicted disease name or Unknown",
    "confidence": 0.0-100.0,
    "verification_source": "ML" or "AI",
    "explanation": "brief explanation of your decision"
}}

Only respond with the JSON, no additional text."""
        
        payload = {
            "parts": [
                {
                    "mime_type": "image/jpeg",
                    "data": image_bytes
                },
                {
                    "text": prompt_text
                }
            ]
        }
        
        # Sanitize payload to ensure no PII is sent
        sanitized_payload = _sanitize_ai_request_payload(payload)
        
        # Call Gemini API
        response = model.generate_content(
            [sanitized_payload["parts"][0], sanitized_payload["parts"][1]["text"]]
        )
        
        # Parse response
        result_text = response.text.strip()
        
        # Try to extract JSON from response
        import json
        import re
        
        # Find JSON in response
        json_match = re.search(r'\{[^}]+\}', result_text, re.DOTALL)
        if json_match:
            result_text = json_match.group(0)
        
        result = json.loads(result_text)
        
        ai_disease = result.get("disease", "Unknown")
        ai_confidence = float(result.get("confidence", 0) or 0)
        verification_source = result.get("verification_source", "ML")
        
        return {
            "ai_prediction": ai_disease,
            "ai_confidence": ai_confidence,
            "verification_source": verification_source,
            "explanation": result.get("explanation", "")
        }
        
    except Exception as exc:
        LOGGER.error(f"Multi-model AI verification failed: {exc}")
        return None
