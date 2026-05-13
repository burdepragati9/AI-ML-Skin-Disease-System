import json
import re

from PIL import Image

from utils.config import GEMINI_API_KEY, GEMINI_MODEL_NAME


AI_ANALYSIS_PROMPT = """
Analyze this skin image carefully.

Provide:
1. Possible skin condition
2. Common symptoms
3. Basic precautions
4. Simple skincare advice

Mention clearly:
This is NOT a medical diagnosis.
"""


def _gemini_model():
    import google.generativeai as genai

    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel(GEMINI_MODEL_NAME)


def analyze_with_ai(image: Image.Image) -> str:
    """Return a human-readable Gemini Vision analysis for the uploaded image."""
    if not GEMINI_API_KEY:
        return "AI analysis unavailable: API key is not configured."

    response = _gemini_model().generate_content([AI_ANALYSIS_PROMPT, image])
    return getattr(response, "text", "") or ""


def recognize_with_ai(image: Image.Image) -> dict | None:
    """Use Gemini Vision as a low-confidence fallback when configured."""
    if not GEMINI_API_KEY:
        return None

    try:
        prompt = (
            "Identify the most likely visible skin disease in this image. "
            "Return strict JSON only with these keys: disease, confidence, symptoms, "
            "precautions, skincare_advice, disclaimer, and note. Confidence must be from 0 to 100. "
            "Write symptoms, precautions, and skincare_advice as short user-friendly text. "
            "The disclaimer must clearly say this is not a medical diagnosis. "
            "If uncertain, use disease Unknown and a low confidence."
        )
        response = _gemini_model().generate_content([prompt, image])
        text = getattr(response, "text", "") or ""
        match = re.search(r"\{.*\}", text, re.S)
        payload = json.loads(match.group(0) if match else text)
        disease = str(payload.get("disease", "Unknown")).strip() or "Unknown"
        confidence = float(payload.get("confidence", 0))
        return {
            "disease": disease,
            "confidence": max(0.0, min(confidence, 100.0)),
            "symptoms": str(payload.get("symptoms", ""))[:800],
            "precautions": str(payload.get("precautions", ""))[:800],
            "skincare_advice": str(payload.get("skincare_advice", ""))[:800],
            "disclaimer": str(
                payload.get("disclaimer", "This is not a medical diagnosis.")
            )[:500],
            "note": str(payload.get("note", ""))[:800],
        }
    except Exception as exc:
        return {"disease": "Unknown", "confidence": 0.0, "note": f"AI fallback failed: {exc}"}
