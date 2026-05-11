import json
import os
from pathlib import Path

import google.generativeai as genai
import numpy as np
import streamlit as st
import tensorflow as tf
from dotenv import load_dotenv
from PIL import Image


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================
load_dotenv()

API_KEY = os.getenv("API_KEY")

if not API_KEY:

    st.error(
        "API_KEY not found.\n"
        "Please add it inside .env file."
    )

    st.stop()


# =========================================================
# CONFIGURE GEMINI AI
# =========================================================
genai.configure(api_key=API_KEY)

model_ai = genai.GenerativeModel(
    "gemini-2.5-flash"
)


# =========================================================
# PROJECT PATHS
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_ROOT
    / "model"
    / "skin_model.keras"
)

CLASS_NAMES_PATH = (
    PROJECT_ROOT
    / "model"
    / "class_names.json"
)

IMG_SIZE = 224

CONFIDENCE_THRESHOLD = 70.0


# =========================================================
# DISEASE DESCRIPTIONS
# =========================================================
DESCRIPTIONS = {

    "Acne": (
        "Pimples and oily skin caused "
        "by clogged pores."
    ),

    "Tinea": (
        "Fungal infection affecting skin."
    ),

    "Psoriasis": (
        "Chronic skin condition causing "
        "red and scaly patches."
    ),

    "Vitiligo": (
        "Loss of skin pigment resulting "
        "in white patches."
    ),
}


# =========================================================
# LOAD CNN MODEL
# =========================================================
@st.cache_resource
def load_model_and_labels():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "skin_model.keras not found."
        )

    if not CLASS_NAMES_PATH.exists():

        raise FileNotFoundError(
            "class_names.json not found."
        )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    class_names = json.loads(
        CLASS_NAMES_PATH.read_text(
            encoding="utf-8"
        )
    )

    return model, class_names


# =========================================================
# GEMINI AI ANALYSIS
# =========================================================
def analyze_with_ai(image):

    prompt = """
    Analyze this skin image carefully.

    Provide:
    1. Possible skin condition
    2. Common symptoms
    3. Basic precautions
    4. Simple skincare advice

    Mention clearly:
    This is NOT a medical diagnosis.
    """

    response = model_ai.generate_content(
        [
            prompt,
            image,
        ]
    )

    return response.text


# =========================================================
# STREAMLIT PAGE SETTINGS
# =========================================================
st.set_page_config(
    page_title="Skin Disease Detection",
    layout="centered",
)

st.title(
    "🧴 Skin Disease Detection System"
)

st.caption(
    "Hybrid AI + ML System "
    "(CNN + Gemini Vision AI)"
)

st.warning(
    "This application is for educational "
    "purposes only and is NOT a medical diagnosis."
)


# =========================================================
# LOAD MODEL
# =========================================================
try:

    model, class_names = (
        load_model_and_labels()
    )

except Exception as exc:

    st.error(str(exc))

    st.stop()


# =========================================================
# IMAGE UPLOAD
# =========================================================
uploaded_file = st.file_uploader(
    "Upload Skin Image",
    type=["jpg", "jpeg", "png"],
)


# =========================================================
# PREDICTION SECTION
# =========================================================
if uploaded_file is not None:

    try:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

    except Exception:

        st.error(
            "Invalid image file."
        )

        st.stop()

    # -----------------------------------------------------
    # DISPLAY IMAGE
    # -----------------------------------------------------
    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True,
    )

    # -----------------------------------------------------
    # PREPROCESS IMAGE
    # -----------------------------------------------------
    resized = image.resize(
        (IMG_SIZE, IMG_SIZE)
    )

    image_array = np.asarray(
        resized,
        dtype=np.float32,
    )

    image_array = np.expand_dims(
        image_array,
        axis=0,
    )

    # -----------------------------------------------------
    # CNN MODEL PREDICTION
    # -----------------------------------------------------
    scores = model.predict(
        image_array,
        verbose=0,
    )[0]

    best_index = int(
        np.argmax(scores)
    )

    confidence = (
        float(scores[best_index]) * 100
    )

    predicted_class = (
        class_names[best_index]
    )

    # -----------------------------------------------------
    # RESULT SECTION
    # -----------------------------------------------------
    st.subheader(
        "Prediction Result"
    )

    st.progress(
        min(confidence / 100, 1.0)
    )

    st.write(
        f"Confidence Score: "
        f"{confidence:.2f}%"
    )

    # =====================================================
    # HIGH CONFIDENCE → CNN RESULT
    # =====================================================
    if confidence >= CONFIDENCE_THRESHOLD:

        st.success(
            f"Predicted Disease: "
            f"{predicted_class}"
        )

        st.info(
            DESCRIPTIONS.get(
                predicted_class,
                "No description available.",
            )
        )

        st.success(
            "Prediction generated using "
            "CNN Deep Learning Model."
        )

    # =====================================================
    # LOW CONFIDENCE → GEMINI AI
    # =====================================================
    else:

        st.warning(
            f"Low confidence detected "
            f"({confidence:.2f}%)."
        )

        st.info(
            "Sending image to Gemini Vision AI..."
        )

        try:

            with st.spinner(
                "Gemini AI analyzing image..."
            ):

                ai_result = analyze_with_ai(
                    image
                )

            st.subheader(
                "Gemini AI Analysis"
            )

            st.write(ai_result)

            st.success(
                "Result generated using "
                "Gemini Vision AI because "
                "CNN confidence was low."
            )

        except Exception as exc:

            st.error(
                "AI analysis failed:\n"
                f"{str(exc)}"
            )

    # =====================================================
    # SHOW ALL PREDICTIONS
    # =====================================================
    st.subheader(
        "All Predictions"
    )

    sorted_indices = np.argsort(
        scores
    )[::-1]

    for index in sorted_indices:

        st.write(
            f"{class_names[index]} : "
            f"{float(scores[index]) * 100:.2f}%"
        )