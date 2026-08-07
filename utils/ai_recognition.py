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
import traceback
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
    img_str = base64.b64encode(_image_to_jpeg_bytes(image)).decode()
    return img_str


def _image_to_jpeg_bytes(image: Image.Image) -> bytes:
    """Encode an RGB working copy as JPEG without mutating the uploaded image."""
    buffered = io.BytesIO()
    image_for_jpeg = image if image.mode == "RGB" else image.convert("RGB")
    image_for_jpeg.save(buffered, format="JPEG")
    return buffered.getvalue()


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
        
        image_bytes = _image_to_jpeg_bytes(image)
        
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
        
        image_bytes = _image_to_jpeg_bytes(image)
        
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

print("\n========== verify_multi_model_predictions_with_ai CALLED ==========")

def verify_multi_model_predictions_with_ai(
    image: Image.Image,
    model_predictions: list[dict[str, Any]],
    comparison_summary: str
) -> dict[str, Any] | None:
    """
    Verify multi-model predictions using Gemini AI.
    """

    if not GEMINI_API_KEY:
        LOGGER.warning("GEMINI_API_KEY not configured.")
        return None

    try:
        print("\n========== GEMINI DEBUG ==========")
        print("API KEY PRESENT :", bool(GEMINI_API_KEY))
        print("API KEY PREFIX  :", GEMINI_API_KEY[:10] if GEMINI_API_KEY else "None")
        print("API KEY LENGTH  :", len(GEMINI_API_KEY) if GEMINI_API_KEY else 0)
        print("MODEL NAME      :", GEMINI_MODEL_NAME)
        print("genai module    :", genai.__file__)
        print("==================================\n")

        # Configure Gemini again to ensure runtime uses the latest key
        genai.configure(api_key=GEMINI_API_KEY)

        model = genai.GenerativeModel(GEMINI_MODEL_NAME)

        image_bytes = _image_to_jpeg_bytes(image)

        prompt_text = f"""
Analyze this skin image and verify the multi-model predictions.

{comparison_summary}

This ML system supports only these four diseases:

- Acne
- Psoriasis
- Tinea
- Vitiligo

If the disease is outside this list return "Unknown".

Respond ONLY in JSON.

{{
    "disease":"Acne",
    "confidence":95,
    "verification_source":"ML",
    "explanation":"..."
}}
"""

        print("Sending request to Gemini...")

        print(type(image_bytes))
        print(len(image_bytes))
        print(prompt_text)

        response = model.generate_content(
    [
        image,
        prompt_text,
    ]
)

        print("Gemini response received.")
        print(response.text)

        import json
        import re

        result_text = response.text.strip()

        json_match = re.search(r"\{.*\}", result_text, re.DOTALL)

        if json_match:
            result_text = json_match.group(0)

        result = json.loads(result_text)

        return {
            "ai_prediction": result.get("disease", "Unknown"),
            "ai_confidence": float(result.get("confidence", 0)),
            "verification_source": result.get("verification_source", "ML"),
            "explanation": result.get("explanation", ""),
        }

    except Exception:
        print("\n========== GEMINI EXCEPTION ==========")
        traceback.print_exc()
        print("API KEY :", GEMINI_API_KEY[:10] if GEMINI_API_KEY else "None")
        print("MODEL   :", GEMINI_MODEL_NAME)
        print("======================================\n")

        LOGGER.exception("Multi-model AI verification failed")

        return None