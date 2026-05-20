# =========================================================
# FINAL UPDATED app.py
# HEATMAP ERROR FIXED VERSION
# =========================================================

import json
import textwrap
import unicodedata
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from PIL import Image
from PIL import ImageOps

from fpdf import FPDF

from analytics.admin_analytics import admin_summary, ai_recognized_images, training_status
from auth.doctor_auth import (
    assert_can_search,
    authenticate_doctor,
    consume_search,
    create_doctor,
    create_reset_token,
    get_doctor,
    reset_password,
    update_doctor_profile,
    usage_for_doctor,
)
from database.db import init_db
from history.search_history import doctor_search_stats, recent_searches, record_search
from training.self_learning import save_ai_prediction_for_learning, start_training_worker
from utils.ai_recognition import recognize_with_ai
from utils.config import (
    CLASS_NAMES_PATH as CONFIG_CLASS_NAMES_PATH,
    LOW_CONFIDENCE_THRESHOLD,
    MODEL_PATH as CONFIG_MODEL_PATH,
    PROFILE_PHOTO_PATH,
)
from utils.security import image_from_bytes, save_optimized_image, sanitize_text, validate_image_upload

# =========================================================
# PATHS
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = CONFIG_MODEL_PATH

CLASS_NAMES_PATH = CONFIG_CLASS_NAMES_PATH

# =========================================================
# SETTINGS
# =========================================================
IMG_SIZE = 160

CONFIDENCE_THRESHOLD = max(70.0, LOW_CONFIDENCE_THRESHOLD)

init_db()
start_training_worker()

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Skin Disease Detection",
    layout="centered",
)

# =========================================================
# BUTTON STYLE
# =========================================================
st.markdown("""
<style>

.stButton > button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 10px;
    height: 50px;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
    border: none;
}

.stButton > button:hover {
    background-color: #ff1f1f;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# SESSION STATE
# =========================================================
if "prediction_history" not in st.session_state:

    st.session_state.prediction_history = []

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model_and_labels(model_mtime: float, labels_mtime: float):

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


def _model_cache_key() -> tuple[float, float]:
    model_mtime = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else 0.0
    labels_mtime = CLASS_NAMES_PATH.stat().st_mtime if CLASS_NAMES_PATH.exists() else 0.0
    return model_mtime, labels_mtime

# =========================================================
# PREPROCESS IMAGE
# =========================================================
def preprocess_image(image):

    processed_image = image.resize(
        (IMG_SIZE, IMG_SIZE),
        Image.Resampling.LANCZOS,
    )

    # The model contains its own Rescaling layer, so inference inputs stay 0..255.
    image_array = np.asarray(
        processed_image,
        dtype=np.float32,
    )

    image_array = np.expand_dims(
        image_array,
        axis=0,
    )

    return processed_image, image_array


# =========================================================
# FIXED GRADCAM
# =========================================================
def generate_gradcam_heatmap(
    model,
    image_array,
    class_index=None,
):

    try:

        # ============================================
        # FIND LAST CONV LAYER
        # ============================================
        last_conv_layer = None

        for layer in reversed(model.layers):

            try:

                if len(layer.output.shape) == 4:

                    last_conv_layer = layer

                    break

            except:
                pass

        if last_conv_layer is None:

            return None

        # ============================================
        # CREATE GRAD MODEL
        # ============================================
        grad_model = tf.keras.models.Model(

            inputs=model.inputs,

            outputs=[
                last_conv_layer.output,
                model.output,
            ],
        )

        # ============================================
        # COMPUTE GRADIENTS
        # ============================================
        with tf.GradientTape() as tape:

            conv_outputs, predictions = grad_model(
                image_array
            )

            if class_index is None:

                class_index = tf.argmax(
                    predictions[0]
                )

            loss = predictions[:, class_index]

        grads = tape.gradient(
            loss,
            conv_outputs,
        )

        if grads is None:

            return None

        pooled_grads = tf.reduce_mean(

            grads,

            axis=(0, 1, 2),
        )

        conv_outputs = conv_outputs[0]

        heatmap = tf.reduce_sum(

            pooled_grads * conv_outputs,

            axis=-1,
        )

        heatmap = tf.maximum(
            heatmap,
            0,
        )

        max_val = tf.reduce_max(
            heatmap
        )

        if max_val == 0:

            return None

        heatmap /= max_val

        return heatmap.numpy()

    except Exception:

        return None

# =========================================================
# OVERLAY HEATMAP
# =========================================================
def overlay_gradcam_on_image(
    image,
    heatmap,
    alpha=0.4,
):

    heatmap = np.uint8(
        255 * heatmap
    )

    heatmap_img = Image.fromarray(
        heatmap
    ).resize(image.size)

    heatmap_array = np.array(
        heatmap_img
    )

    base_array = np.array(
        image
    ).astype(np.float32)

    overlay = base_array.copy()

    overlay[..., 0] = np.maximum(
        overlay[..., 0],
        heatmap_array,
    )

    blended = (
        base_array * (1 - alpha)
        + overlay * alpha
    )

    blended = np.clip(
        blended,
        0,
        255,
    )

    return blended.astype(np.uint8)

# =========================================================
# PDF GENERATION
# =========================================================
def _pdf_safe_text(value) -> str:

    normalized = unicodedata.normalize(
        "NFKD",
        str(value or ""),
    )

    return normalized.encode(
        "latin-1",
        "replace",
    ).decode(
        "latin-1"
    )


def generate_pdf(lines):

    pdf = FPDF()

    pdf.add_page()

    pdf.set_font(
        "Arial",
        size=12,
    )

    usable_width = pdf.w - pdf.l_margin - pdf.r_margin

    for line in lines:

        safe_line = _pdf_safe_text(
            line
        )

        if not safe_line.strip():

            pdf.ln(6)

            continue

        wrapped = textwrap.wrap(
            safe_line,
            width=90,
            break_long_words=True,
            break_on_hyphens=True,
        )

        for wrap_line in wrapped:

            pdf.set_x(
                pdf.l_margin
            )

            pdf.multi_cell(
                usable_width,
                10,
                wrap_line,
            )

    return bytes(
        pdf.output(dest="S")
    )


def render_ai_analysis(ai_result: dict | None) -> None:

    if not ai_result:
        return

    st.subheader("Gemini AI Analysis")

    disease = ai_result.get("disease", "")
    confidence = ai_result.get("confidence", 0)
    severity = ai_result.get("severity", "")
    explanation = ai_result.get("explanation", "")

    if disease:
        st.markdown(f"**Possible Skin Condition:** {disease}")

    if confidence is not None:
        try:
            st.markdown(f"**Confidence:** {float(confidence):.1f}%")
        except Exception:
            pass

    if severity:
        st.markdown(f"**Severity:** {severity}")

    if explanation:
        st.markdown(f"**Explanation:** {explanation}")



# =========================================================
# DOCTOR AUTHENTICATION UI
# =========================================================
def current_doctor():

    doctor_pk = st.session_state.get("doctor_pk")

    if not doctor_pk:

        return None

    return get_doctor(
        int(doctor_pk)
    )


def render_doctor_auth():

    st.title("Doctor Access")

    login_tab, signup_tab, forgot_tab, reset_tab = st.tabs(
        [
            "Login",
            "Signup",
            "Forgot Password",
            "Reset Password",
        ]
    )

    with login_tab:

        with st.form("doctor_login_form"):

            email = st.text_input("Email")

            password = st.text_input(
                "Password",
                type="password",
            )

            submitted = st.form_submit_button("Login")

        if submitted:

            doctor = authenticate_doctor(
                email,
                password,
            )

            if doctor:

                st.session_state.doctor_pk = doctor["id"]

                st.success("Login successful.")

                st.rerun()

            else:

                st.error("Invalid email or password.")

    with signup_tab:

        with st.form("doctor_signup_form"):

            profile = {
                "full_name": st.text_input("Full Name"),
                "doctor_id": st.text_input("Doctor ID"),
                "specialization": st.text_input("Specialization"),
                "clinic_name": st.text_input("Hospital/Clinic Name"),
                "email": st.text_input("Email"),
                "phone": st.text_input("Phone Number"),
                "experience": st.number_input("Experience", min_value=0, max_value=80, step=1),
                "location": st.text_input("Location"),
            }

            password = st.text_input(
                "Create Password",
                type="password",
            )

            submitted = st.form_submit_button("Create Account")

        if submitted:

            try:

                doctor_pk = create_doctor(
                    profile,
                    password,
                )

                st.session_state.doctor_pk = doctor_pk

                st.success("Doctor account created.")

                st.rerun()

            except Exception as exc:

                st.error(str(exc))

    with forgot_tab:

        email = st.text_input("Registered Email", key="forgot_email")

        if st.button("Generate Reset Token"):

            token = create_reset_token(
                email
            )

            if token:

                st.info("Use this reset token within one hour.")

                st.code(token)

            else:

                st.error("No doctor account found for this email.")

    with reset_tab:

        token = st.text_input("Reset Token")

        new_password = st.text_input(
            "New Password",
            type="password",
        )

        if st.button("Reset Password"):

            try:

                if reset_password(
                    token,
                    new_password,
                ):

                    st.success("Password reset successful.")

                else:

                    st.error("Invalid or expired reset token.")

            except Exception as exc:

                st.error(str(exc))


# =========================================================
# DOCTOR DASHBOARD UI
# =========================================================
def render_doctor_dashboard(doctor):

    st.title("Doctor Dashboard")

    if not doctor:

        st.warning("Please login as a doctor to view this dashboard.")

        render_doctor_auth()

        return

    usage = usage_for_doctor(
        int(doctor["id"])
    )

    stats = doctor_search_stats(
        int(doctor["id"])
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Searches", stats["total"])

    col2.metric("Free Searches Left", usage["remaining"])

    col3.metric("Used Searches", usage["used"])

    if usage["remaining"] <= 0:

        st.warning("Free search limit reached. Upgrade to premium to continue searching.")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:

        st.subheader("Most Searched Diseases")

        disease_df = pd.DataFrame(
            [dict(row) for row in stats["diseases"]]
        )

        if disease_df.empty:

            st.info("No disease searches yet.")

        else:

            st.bar_chart(
                disease_df,
                x="disease",
                y="count",
            )

    with chart_col2:

        st.subheader("AI vs ML Predictions")

        source_df = pd.DataFrame(
            [dict(row) for row in stats["sources"]]
        )

        if source_df.empty:

            st.info("No prediction source data yet.")

        else:

            st.bar_chart(
                source_df,
                x="prediction_source",
                y="count",
            )

    st.subheader("Recent Searches")

    disease_filter = st.text_input("Filter by disease")

    page_number = st.number_input(
        "History Page",
        min_value=1,
        step=1,
    )

    rows = recent_searches(
        int(doctor["id"]),
        limit=10,
        offset=(int(page_number) - 1) * 10,
        disease=sanitize_text(disease_filter),
    )

    if rows:

        st.dataframe(
            pd.DataFrame(
                [dict(row) for row in rows]
            )[
                [
                    "disease",
                    "confidence",
                    "prediction_source",
                    "created_at",
                ]
            ],
            use_container_width=True,
        )

    else:

        st.info("No search history found.")

    st.subheader("Most Searched Images")

    if stats["images"]:

        image_cols = st.columns(3)

        for index, row in enumerate(stats["images"]):

            with image_cols[index % 3]:

                st.image(
                    row["image_path"],
                    caption=f"{row['disease']} ({row['count']}x)",
                    use_container_width=True,
                )

    else:

        st.info("No repeated image searches yet.")

    st.subheader("Profile Management")

    with st.form("doctor_profile_form"):

        updated_profile = {
            "full_name": st.text_input("Full Name", value=doctor["full_name"]),
            "specialization": st.text_input("Specialization", value=doctor["specialization"]),
            "clinic_name": st.text_input("Hospital/Clinic Name", value=doctor["clinic_name"] or ""),
            "phone": st.text_input("Phone Number", value=doctor["phone"] or ""),
            "experience": st.number_input("Experience", min_value=0, max_value=80, value=int(doctor["experience"] or 0)),
            "location": st.text_input("Location", value=doctor["location"] or ""),
        }

        photo_upload = st.file_uploader(
            "Profile Photo",
            type=[
                "jpg",
                "jpeg",
                "png",
            ],
            key="profile_photo_upload",
        )

        submitted = st.form_submit_button("Save Profile")

    if submitted:

        try:

            if photo_upload:

                photo_bytes = validate_image_upload(
                    photo_upload
                )

                photo = image_from_bytes(
                    photo_bytes
                )

                photo_path = save_optimized_image(
                    photo,
                    PROFILE_PHOTO_PATH,
                    f"doctor_{doctor['id']}",
                )

                updated_profile["profile_photo"] = str(photo_path)

            update_doctor_profile(
                int(doctor["id"]),
                updated_profile,
            )

            st.success("Profile updated.")

            st.rerun()

        except Exception as exc:

            st.error(str(exc))


# =========================================================
# ADMIN ANALYTICS UI
# =========================================================
def render_admin_analytics():

    st.title("Admin Analytics")

    summary = admin_summary()

    status = training_status()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("AI Images", summary["total_ai"])

    col2.metric("Retrained Images", summary["retrained"])

    common = summary["most_common"]

    col3.metric("Most Added Disease", common["predicted_disease"] if common else "None")

    col4.metric("Accuracy Improvement", f"{summary['accuracy_improvement']:.2%}")

    st.subheader("Training Status")

    st.json(status)

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:

        disease_df = pd.DataFrame(
            [dict(row) for row in summary["disease_counts"]]
        )

        st.subheader("Disease Frequency")

        if disease_df.empty:

            st.info("No search data yet.")

        else:

            st.bar_chart(disease_df, x="disease", y="count")

    with chart_col2:

        source_df = pd.DataFrame(
            [dict(row) for row in summary["source_counts"]]
        )

        st.subheader("Prediction Sources")

        if source_df.empty:

            st.info("No source data yet.")

        else:

            st.bar_chart(source_df, x="prediction_source", y="count")

    st.subheader("AI-Recognized Images")

    ai_rows = ai_recognized_images(100)

    if ai_rows:

        st.dataframe(
            pd.DataFrame([dict(row) for row in ai_rows]),
            use_container_width=True,
        )

    else:

        st.info("No AI-recognized images have been stored yet.")

# =========================================================
# LOAD MODEL
# =========================================================
model, class_names = load_model_and_labels(*_model_cache_key())

try:

    model_input_shape = model.input_shape

    if (
        isinstance(model_input_shape, tuple)
        and len(model_input_shape) >= 3
        and model_input_shape[1]
        and model_input_shape[2]
    ):

        IMG_SIZE = int(model_input_shape[1])

except Exception:

    pass

# =========================================================
# SIDEBAR
# =========================================================
doctor = current_doctor()

page = st.sidebar.radio(
    "Navigation",
    [
        "Prediction",
        "Doctor Dashboard",
        "Doctor Login",
        "Admin Analytics",
    ],
)

if doctor:

    st.sidebar.success(
        f"Doctor: {doctor['full_name']}"
    )

    usage = usage_for_doctor(
        int(doctor["id"])
    )

    st.sidebar.caption(
        f"Free searches remaining: {usage['remaining']}"
    )

    if st.sidebar.button("Logout"):

        st.session_state.pop(
            "doctor_pk",
            None,
        )

        st.rerun()

if page == "Doctor Login":

    render_doctor_auth()

    st.stop()

if page == "Doctor Dashboard":

    render_doctor_dashboard(
        doctor
    )

    st.stop()

if page == "Admin Analytics":

    render_admin_analytics()

    st.stop()

st.sidebar.title(
    "Prediction History"
)

if st.sidebar.button(
    "Clear History"
):

    st.session_state.prediction_history = []

for item in reversed(
    st.session_state.prediction_history
):

        src = item.get("prediction_source", "")
        src_suffix = f" • {src}" if src else ""
        st.sidebar.write(
            f"{item['label']} ({item['confidence']:.2f}%){src_suffix}"
        )


# =========================================================
# TITLE
# =========================================================
st.title(
    "🧠 Skin Disease Detection"
)

st.markdown("---")

st.warning(
    "This app is for educational purposes only."
)

# =========================================================
# FILE UPLOAD
# =========================================================
uploaded_file = st.file_uploader(
    "Upload Skin Image",
    type=[
        "jpg",
        "jpeg",
        "png",
    ],
)

# =========================================================
# PREDICTION
# =========================================================
if uploaded_file is not None:

    try:

        upload_bytes = validate_image_upload(
            uploaded_file
        )

        image = image_from_bytes(
            upload_bytes
        )

    except Exception as exc:

        st.error(
            str(exc)
        )

        st.stop()

    st.image(
        image,
        caption="Uploaded Image",
        width=250,
    )

    processed_image, image_array = preprocess_image(
        image
    )

    if st.button(
        "🔍 Predict Disease"
    ):

        active_doctor = current_doctor()

        try:

            if active_doctor:

                assert_can_search(
                    int(active_doctor["id"])
                )

        except PermissionError as exc:

            st.error(
                str(exc)
            )

            st.warning(
                "Upgrade to premium to continue using doctor searches."
            )

            st.stop()

        # =============================================
        # PREDICT
        # =============================================
        with st.spinner("Analyzing image..."):

            raw_scores = model.predict(
                image_array,
                verbose=0,
            )[0]

        # ---- Runtime validation + confidence calibration (fixes UI inflation) ----
        raw_scores = np.asarray(raw_scores, dtype=np.float64)
        if raw_scores.ndim != 1:
            raise ValueError(f"Model output must be 1D per sample; got shape {raw_scores.shape}")

        # Treat model output as either probabilities (softmax) OR logits.
        # Heuristic: probabilities should be in [0,1] and sum ~= 1.
        out_min = float(np.min(raw_scores))
        out_max = float(np.max(raw_scores))
        raw_sum = float(np.sum(raw_scores))
        looks_like_probs = (
            out_min >= -1e-6
            and out_max <= 1.0 + 1e-6
            and abs(raw_sum - 1.0) <= 1e-2
        )

        if looks_like_probs:
            probs = raw_scores
            probs_source = "model_probabilities"
        else:
            # Convert logits -> softmax probabilities.
            exp = np.exp(raw_scores - np.max(raw_scores))
            probs = exp / np.sum(exp)
            probs_source = "softmax(model_logits)"

        # Required runtime validation
        prob_sum = float(np.sum(probs))
        if not (abs(prob_sum - 1.0) <= 1e-3):
            raise ValueError(f"Probability validation failed: sum(probs)={prob_sum}")

        if np.any(probs < -1e-6):
            raise ValueError(f"Probability validation failed: negative probs min={float(np.min(probs))}")

        sorted_indices = np.argsort(probs)[::-1]
        best_index = int(sorted_indices[0])
        second_index = int(sorted_indices[1]) if len(sorted_indices) > 1 else best_index

        # Required: softmax_output * 100 only ONCE
        ml_confidence = float(probs[best_index]) * 100.0
        predicted_class = class_names[best_index]

        # Top-3 for debug/UI sanity
        top3_indices = sorted_indices[:3]
        top3 = [
            {
                "idx": int(i),
                "class": class_names[int(i)],
                "prob": float(probs[int(i)]),
                "pct": float(probs[int(i)]) * 100.0,
            }
            for i in top3_indices
        ]

        # Debug logs (console)
        print("\n[ML DEBUG] raw_scores:", raw_scores)
        print("[ML DEBUG] probs_source:", probs_source)
        print("[ML DEBUG] probs (sum=%.6f):" % prob_sum, probs)
        print("[ML DEBUG] top-3:", top3)

        # Margin sanity check (probability-space)
        top1_prob = float(probs[best_index])
        top2_prob = float(probs[second_index])
        top1_minus_top2 = (top1_prob - top2_prob) * 100.0

        confidence = ml_confidence

        # Debug: print probability sanity summaries
        probs_max = float(np.max(probs))
        probs_entropy = float(-np.sum(np.clip(probs, 1e-12, 1.0) * np.log(np.clip(probs, 1e-12, 1.0))))
        print("[ML DEBUG] probs_max=%.6f entropy=%.6f" % (probs_max, probs_entropy))

        USE_MARGIN_MIN = 18.0  # tuned heuristic


        final_class = predicted_class
        final_confidence = confidence
        prediction_source = "ML"




        ai_result = None
        ai_fallback_status = "not_used"
        retraining_status = "not_required"
        ml_confidence = confidence

        # =============================================
        # AI FALLBACK
        # =============================================
        # =============================================
        # ML vs AI selection (required logic)
        # Requirement: Strict ML trust logic.
        # If confidence < 80 OR entropy > 0.9 => do NOT trust ML (use Gemini AI)
        # =============================================
        ML_CONFIDENCE_FALLBACK_MIN = 80.0
        ML_ENTROPY_FALLBACK_MAX = 0.9

        should_use_ml = not (
            confidence < ML_CONFIDENCE_FALLBACK_MIN
            or probs_entropy > ML_ENTROPY_FALLBACK_MAX
        )

        if should_use_ml and confidence > CONFIDENCE_THRESHOLD:




            final_class = predicted_class
            final_confidence = confidence
            prediction_source = "ML"

        else:


            ai_fallback_status = "triggered"
            prediction_source = "AI"


            with st.spinner("Predicted by AI due to low ML confidence..."): 
                ai_result = recognize_with_ai(image)

            ai_disease = (ai_result or {}).get("disease", "Unknown")
            ai_conf = float((ai_result or {}).get("confidence", 0) or 0.0)
            ai_explanation = (ai_result or {}).get("explanation", "")

            # Never accept generic AI disease results.
            if (
                ai_disease
                and str(ai_disease).strip().lower() != "unknown"
                and ai_conf > 0
            ):

                final_class = ai_disease
                final_confidence = ai_conf

                # Save for incremental training.
                learn_result = save_ai_prediction_for_learning(
                    image,
                    final_class,
                    final_confidence,
                    getattr(uploaded_file, "name", "ai_fallback_upload.jpg"),
                    int(active_doctor["id"]) if active_doctor else None,
                )

                if learn_result.get("duplicate"):
                    st.info("AI-recognized image already exists in the learning dataset.")
                    retraining_status = "duplicate"
                elif learn_result.get("saved"):
                    st.success("New AI-recognized image added to the learning queue.")
                    retraining_status = "queued"

            else:
                # AI failed; keep ML prediction (as final) instead of Unknown.
                final_class = predicted_class
                final_confidence = confidence
                prediction_source = "ML"
                retraining_status = "not_queued"
                ai_result = ai_result or {"disease": "Unknown", "confidence": 0.0, "explanation": ""}
                st.warning(
                    "AI fallback triggered, but returned an invalid/unreliable result. Keeping ML prediction."
                )


        # =============================================
        # SAVE HISTORY
        # =============================================
        # Save only valid final predictions (final_class is never forced to Unknown anymore).
        if final_class and str(final_class).strip().lower() not in {"", "unknown"}:
            st.session_state.prediction_history.append(
                {
                    "label": final_class,
                    "confidence": final_confidence,
                    "prediction_source": prediction_source,
                    "time": str(datetime.now()),
                }
            )


        try:

            record_search(
                int(active_doctor["id"]) if active_doctor else None,
                image,
                final_class,
                final_confidence,
                prediction_source,
                ai_fallback_status,
                retraining_status,
            )

            if active_doctor:

                consume_search(
                    int(active_doctor["id"])
                )

        except Exception as exc:

            st.warning(
                f"Search tracking warning: {str(exc)}"
            )

        # =============================================
        # RESULT
        # =============================================
        st.success(
            f"Prediction: {final_class}"
        )

        st.info(
            f"Confidence: {final_confidence:.2f}%"
        )

        if prediction_source == "ML":

            st.success("Predicted by ML")

        else:

            st.warning("Predicted by AI")
            st.info(
                f"ML confidence was low ({ml_confidence:.2f}%), so AI analysis was used."
            )

        st.progress(
            min(final_confidence, 100) / 100
        )

        # =============================================
        # LOW CONFIDENCE
        # =============================================
        if prediction_source == "AI":
            st.warning("Low ML confidence. AI fallback used.")
            render_ai_analysis(ai_result)
        elif confidence < CONFIDENCE_THRESHOLD:
            # ML was low confidence but AI failed; show AI diagnostics only if available.
            st.warning("Low confidence prediction (AI fallback may have failed).")
            render_ai_analysis(ai_result)


        # =============================================
        # HEATMAP
        # =============================================
        st.markdown("---")

        st.subheader("Heatmap")

        try:

            heatmap = generate_gradcam_heatmap(
                model,
                image_array,
                best_index,
            )

            if heatmap is not None:

                overlay = overlay_gradcam_on_image(
                    processed_image,
                    heatmap,
                )

                st.image(
                    overlay,
                    width=250,
                )

            else:

                st.warning(
                    "Heatmap could not be generated."
                )

        except Exception as exc:

            st.warning(
                f"Heatmap Error: {str(exc)}"
            )



        # =============================================
        # PDF REPORT
        # =============================================
        report_lines = [

            "Skin Disease Report",

            "",

            f"Prediction: {final_class}",

            f"Confidence: {final_confidence:.2f}%",

            f"Source: {prediction_source}",

            "",

            "Disclaimer:",

            "This is NOT a medical diagnosis.",
        ]

        pdf_bytes = generate_pdf(
            report_lines
        )

        st.download_button(

            label="📄 Download PDF",

            data=pdf_bytes,

            file_name="skin_report.pdf",

            mime="application/pdf",
        )
