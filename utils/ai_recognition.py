import json
import re

from PIL import Image

from utils.config import GEMINI_API_KEY, GEMINI_MODEL_NAME

from model.registry_utils import resolve_to_canonical



# Dermatology-focused prompt for strict JSON output
MEDICAL_DERM_PROMPT = """You are an expert dermatologist AI.

Analyze the uploaded skin disease image carefully and identify the most likely skin condition.

Return STRICT JSON only (no markdown, no backticks, no extra keys).

Return these keys exactly:
1) disease: string (choose ONE most likely disease)
2) confidence: number from 0 to 100
3) explanation: string (short medical explanation)
4) severity: string (choose ONE: Mild, Moderate, Severe)

Possible diseases include:
- Acne
- Psoriasis
- Eczema
- Vitiligo
- Tinea
- Molluscum Contagiosum
- Melanoma
- Contact Dermatitis

Do not return 'Unknown' unless absolutely impossible (e.g., the image is completely invalid/uninterpretable).

This is NOT a medical diagnosis. Add a clear disclaimer inside the 'explanation'."""



def _gemini_model():
    import google.generativeai as genai

    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel(GEMINI_MODEL_NAME)


def analyze_with_ai(image: Image.Image) -> str:
    """Return a human-readable Gemini Vision analysis for the uploaded image."""
    if not GEMINI_API_KEY:
        return "AI analysis unavailable: API key is not configured."

    # Reuse the strict JSON pipeline and then format it.
    result = recognize_with_ai(image)
    if not result:
        return "AI analysis unavailable."

    disease = result.get("disease", "")
    conf = result.get("confidence", 0)
    explanation = result.get("explanation", "")
    severity = result.get("severity", "")

    return f"Possible Skin Condition: {disease}\nConfidence: {conf:.1f}%\nSeverity: {severity}\nExplanation: {explanation}" 



def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def _normalize_disease_name(disease: str) -> str:
    """Normalize AI disease strings to our dataset/class names.

    This is intentionally conservative: it maps obvious synonyms to canonical labels.
    """
    raw = _normalize_text(disease)

    mapping = {
        "contact dermatitis": "Contact Dermatitis",
        "skin allergy": "Contact Dermatitis",
        "eczema": "Eczema",
        "atopic dermatitis": "Eczema",
        "fungal infection": "Tinea",
        "ringworm": "Tinea",
        "tinea": "Tinea",
        "molluscum contagiosum": "Molluscum Contagiosum",
        "molluscum": "Molluscum Contagiosum",
        "melanoma": "Melanoma",
        "psoriasis": "Psoriasis",
        "vitiligo": "Vitiligo",
        "acne": "Acne",
        "acne vulgaris": "Acne",
    }

    for k, v in mapping.items():
        if raw == k or raw.startswith(k):
            return v

    # If it already matches one of canonical disease names (case-insensitive), keep it.
    canonical = [
        "Acne",
        "Psoriasis",
        "Eczema",
        "Vitiligo",
        "Tinea",
        "Molluscum Contagiosum",
        "Melanoma",
        "Contact Dermatitis",
    ]
    for c in canonical:
        if raw == c.lower():
            return c

    return str(disease or "Unknown").strip() or "Unknown"


def recognize_with_ai(image: Image.Image) -> dict | None:
    """Use Gemini Vision as a low-confidence fallback when configured."""
    if not GEMINI_API_KEY:
        return None

    try:
        # Hardening: ensure valid RGB and provide bytes with explicit MIME via PIL->bytes.
        if image.mode != "RGB":
            image = image.convert("RGB")

        from io import BytesIO

        buf = BytesIO()
        # JPEG tends to be safest for Vision models.
        image.save(buf, format="JPEG", quality=90)
        img_bytes = buf.getvalue()

        response = _gemini_model().generate_content(
            [
                MEDICAL_DERM_PROMPT,
                {"mime_type": "image/jpeg", "data": img_bytes},
            ]
        )

        text = getattr(response, "text", "") or ""

        match = re.search(r"\{\s*\"disease\".*\}", text, re.S)
        raw_json = match.group(0) if match else text
        payload = json.loads(raw_json)

        disease_raw = payload.get("disease", "Unknown")
        disease = _normalize_disease_name(disease_raw)

        # Canonicalize using the registry (prevents folder/class drift).
        canonical = resolve_to_canonical(disease)
        disease = canonical or disease


        confidence = float(payload.get("confidence", 0))
        confidence = max(0.0, min(confidence, 100.0))

        explanation = str(payload.get("explanation", "") or "").strip()[:800]
        severity = str(payload.get("severity", "") or "").strip()

        # If the model returns generic/invalid outputs, treat as failure.
        generic = {
            "unknown",
            "unable to identify",
            "can't identify",
            "cannot identify",
            "uninterpretable",
            "not sure",
            "unsure",
        }

        if _normalize_text(disease_raw) in generic or _normalize_text(disease) in generic:
            raise ValueError("Generic/unknown AI disease output.")


        return {
            "disease": disease,
            "confidence": confidence,
            "explanation": explanation,
            "severity": severity,
        }

    except Exception as exc:
        # Return a hard-fail object; app.py will decide how to handle.
        return {
            "disease": "Unknown",
            "confidence": 0.0,
            "explanation": f"AI fallback failed: {exc}",
            "severity": "Mild",
        }


