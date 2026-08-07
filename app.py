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
from database.db import fetch_all, fetch_one, init_db
from history.search_history import doctor_search_stats, recent_searches, record_search
from training.self_learning import save_ai_prediction_for_learning, start_training_worker
from utils.ai_recognition import recognize_with_ai, verify_prediction_with_ai, verify_multi_model_predictions_with_ai
from utils.config import (
    AI_VERIFICATION_THRESHOLD,
    CLASS_NAMES_PATH as CONFIG_CLASS_NAMES_PATH,
    ENABLE_AI_VERIFICATION,
    ENABLE_MULTI_MODEL_PREDICTION,
    LOW_CONFIDENCE_THRESHOLD,
    MODEL_PATH as CONFIG_MODEL_PATH,
    PROFILE_PHOTO_PATH,
)
from utils.model_manager import get_model_manager
from utils.prediction_comparison import (
    build_ai_verification_summary,
    compare_predictions,
    format_predictions_for_ai,
    majority_vote,
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

if "redirect_to_dashboard" not in st.session_state:

    st.session_state.redirect_to_dashboard = False

# =========================================================
# LOAD MODEL (Using Model Manager for Multiple Model Support)
# =========================================================
@st.cache_resource
def load_model_and_labels():
    """Load model and class names using model manager for multiple model support."""
    model_manager = get_model_manager()
    model = model_manager.get_active_model()
    class_names = model_manager.get_active_class_names()
    return model, class_names

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


def _top3_to_text(top3) -> str:
    return ", ".join(f"{label} {score:.1f}%" for label, score in top3)


def _prediction_row(pred: dict) -> dict:
    return {
        "prediction": pred["predicted_class"],
        "confidence": float(pred["confidence"]),
        "top3": pred.get("top3", []),
    }


def run_multi_model_workflow(image, image_array) -> dict:
    """Run all available models, soft voting, and deterministic AI verification."""
    manager = get_model_manager()
    model_predictions = manager.predict_with_all_models(image_array)
    if not model_predictions:
        raise RuntimeError("No ML models are available for prediction.")

    comparison = compare_predictions(model_predictions)
    majority = majority_vote(model_predictions)
    soft_vote = manager.soft_vote(model_predictions)
    comparison_summary = format_predictions_for_ai(model_predictions, comparison, soft_vote)

    external_ai_result = None
    if ENABLE_AI_VERIFICATION and len(model_predictions) > 1:
        try:
            print("\n========== BEFORE GEMINI CALL ==========")
            print("comparison_summary =", comparison_summary)
            print("model_predictions =", model_predictions)
            print("image type =", type(image))
            print("========================================")
            
            external_ai_result = verify_multi_model_predictions_with_ai(
                image,
                model_predictions,
                comparison_summary,
            )
        except Exception as exc:
            print(f"[AI VERIFICATION] Multi-model verification error: {exc}")

    verification = build_ai_verification_summary(
        model_predictions,
        comparison,
        majority,
        soft_vote,
        external_ai_result,
    )

    per_model = {pred["model_name"]: _prediction_row(pred) for pred in model_predictions}
    metadata = {
        "per_model": per_model,
        "ensemble_prediction": soft_vote["selected_class"],
        "ensemble_confidence": float(soft_vote["selected_confidence"]),
        "ai_verification_summary": verification["summary"],
        "model_agreement": verification["agreement"],
        "model_predictions_json": json.dumps(
            {
                "models": model_predictions,
                "majority_vote": majority,
                "soft_vote": soft_vote,
                "verification": verification,
            },
            default=str,
        ),
    }

    return {
        "model_predictions": model_predictions,
        "comparison": comparison,
        "majority": majority,
        "soft_vote": soft_vote,
        "verification": verification,
        "final_class": verification["final_class"],
        "final_confidence": float(verification["final_confidence"]),
        "prediction_source": "Ensemble",
        "metadata": metadata,
    }


def render_multi_model_result_sections(ensemble_result: dict) -> None:
    """Add compact model comparison details inside the existing result area."""
    st.subheader("Multi-Model Comparison")
    for pred in ensemble_result["model_predictions"]:
        st.markdown(
            f"**{pred['model_type']}:** {pred['predicted_class']} "
            f"({pred['confidence']:.2f}%)"
        )
        st.caption(f"Top-3: {_top3_to_text(pred.get('top3', []))}")

    verification = ensemble_result["verification"]
    soft_vote = ensemble_result["soft_vote"]
    st.subheader("AI Verification Summary")
    st.info(verification["summary"])
    st.markdown(f"**Final Ensemble Prediction:** {soft_vote['selected_class']}")
    st.markdown(f"**Final Confidence Score:** {soft_vote['selected_confidence']:.2f}%")


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


def render_common_auth():

    st.title("Login")

    login_tab, signup_tab = st.tabs(["Login", "Signup"])

    with login_tab:

        with st.form("common_login_form"):

            email = st.text_input("Email")

            password = st.text_input(
                "Password",
                type="password",
            )

            submitted = st.form_submit_button("Login")

        if submitted:

            # Try doctor authentication first
            doctor = authenticate_doctor(
                email,
                password,
            )

            if doctor:

                st.session_state.doctor_pk = doctor["id"]
                st.session_state.user_role = "doctor"
                st.session_state.redirect_to_dashboard = True

                st.success("Login successful. Redirecting to Doctor Dashboard...")

                st.rerun()

            # Try admin authentication if doctor auth fails
            from auth.admin_auth import admin_authenticate

            admin = admin_authenticate(
                email,
                password,
            )

            if admin:

                st.session_state.admin_pk = admin["id"]
                st.session_state.user_role = "admin"
                st.session_state.redirect_to_dashboard = True

                st.success("Login successful. Redirecting to Admin Dashboard...")

                st.rerun()

            else:

                st.error("Invalid email or password.")

        # Forgot Password and Reset Password as hyperlink-style clickable text
        st.markdown("---")
        # Render BOTH links on the SAME LINE with a separator.
        # Render both links as an inline row (no columns) to avoid missing/wrapping in some Streamlit layouts.
        # We use a single HTML markdown block for deterministic rendering.
        forgot_url = "#"
        reset_url = "#"
        links_html = (
            f"<div style='text-align:left; white-space:nowrap;'>"
            f"<a href='{forgot_url}' style='color:#1f77ff;'>Forgot Password?</a>"
            f"<span style='margin:0 10px;'>|</span>"
            f"<a href='{reset_url}' style='color:#1f77ff;'>Reset Password?</a>"
            f"</div>"
        )
        st.markdown(links_html, unsafe_allow_html=True)

        # Keep click handling unchanged using hidden markdown links below.
        # Hidden click handling is intentionally omitted to avoid rendering duplicates.
        # NOTE: This UI-only fix ensures the correct visual layout as requested.


    with signup_tab:

        with st.form("doctor_signup_form"):

            # Auto-populate Doctor ID using the same backend logic that will be stored.
            # Keep the field visible, but make it read-only to avoid user edits.
            from auth.doctor_auth import _generate_next_doctor_id

            auto_doctor_id = _generate_next_doctor_id()

            profile = {
                "full_name": st.text_input("Full Name"),
                "doctor_id": st.text_input(
                    "Doctor ID",
                    value=auto_doctor_id,
                    disabled=True,
                ),
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

            confirm_password = st.text_input(
                "Confirm Password",
                type="password",
            )

            submitted = st.form_submit_button("Create Account")

        if submitted:

            if password != confirm_password:
                st.error("Passwords do not match.")
            elif len(password) < 8:
                st.error("Password must be at least 8 characters.")
            else:
                try:

                    doctor_pk = create_doctor(
                        profile,
                        password,
                    )

                    st.session_state.doctor_pk = doctor_pk
                    st.session_state.user_role = "doctor"

                    st.success("Doctor account created.")

                    st.rerun()

                except Exception as exc:

                    st.error(str(exc))

    # Handle Forgot Password and Reset Password views
    if st.session_state.get("auth_view") == "forgot":
        st.subheader("Forgot Password")
        with st.form("forgot_password_form"):
            email = st.text_input("Registered Email")
            submitted = st.form_submit_button("Generate Reset Token")

        if submitted:
            token = create_reset_token(email)
            if token:
                st.info("Use this reset token within one hour.")
                st.code(token)
            else:
                st.error("No doctor account found for this email.")

        if st.button("← Back to Login"):
            st.session_state.auth_view = "login"
            st.rerun()

    elif st.session_state.get("auth_view") == "reset":
        st.subheader("Reset Password")
        with st.form("reset_password_form"):
            token = st.text_input("Reset Token")
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm New Password", type="password")
            submitted = st.form_submit_button("Reset Password")

        if submitted:
            if new_password != confirm_password:
                st.error("Passwords do not match.")
            elif len(new_password) < 8:
                st.error("Password must be at least 8 characters.")
            else:
                try:
                    if reset_password(token, new_password):
                        st.success("Password reset successful.")
                        st.session_state.auth_view = "login"
                        st.rerun()
                    else:
                        st.error("Invalid or expired reset token.")
                except Exception as exc:
                    st.error(str(exc))

        if st.button("← Back to Login"):
            st.session_state.auth_view = "login"
            st.rerun()


# =========================================================
# DOCTOR DASHBOARD UI
# =========================================================
def render_doctor_dashboard(doctor):

    st.title("Doctor Dashboard")

    if not doctor:

        st.warning("Please login as a doctor to view this dashboard.")

        render_common_auth()

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

    st.subheader("Prediction History")

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


# =========================================================
# PREDICTION HISTORY UI
# =========================================================
def render_prediction_history(doctor):

    st.title("Prediction History")

    if not doctor:

        st.warning("Please login as a doctor to view prediction history.")

        return

    stats = doctor_search_stats(
        int(doctor["id"])
    )

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

        history_df = pd.DataFrame(
            [dict(row) for row in rows]
        )

        # Add required Image Name column (derived from stored image_path)
        history_df["image_name"] = history_df.get("image_path", "").apply(
            lambda p: Path(p).name if p else ""
        )

        display_df = history_df[
            [
                "disease",
                "confidence",
                "prediction_source",
                "created_at",
                "image_name",
            ]
        ].copy()

        st.dataframe(
            display_df,
            use_container_width=True,
        )

        # Image viewer: clickable Image Name opens the correct preview.
        for _, row in history_df.iterrows():
            img_path = row.get("image_path")
            img_name = row.get("image_name")
            if not img_path or not img_name:
                continue

            st.markdown(f"- [**{img_name}**](\"{img_path}\")")
            st.image(
                img_path,
                caption=img_name,
                use_container_width=True,
            )

    else:

        st.info("No prediction history found.")



# =========================================================
# PROFILE MANAGEMENT UI
# =========================================================
def render_profile_management(doctor):

    st.title("Profile Management")

    if not doctor:

        st.warning("Please login as a doctor to manage your profile.")

        return

    st.subheader("Doctor Profile")

    col1, col2 = st.columns(2)

    with col1:

        st.write("**Full Name:**", doctor["full_name"])
        st.write("**Email:**", doctor["email"])
        st.write("**Specialization:**", doctor["specialization"])
        st.write("**Doctor ID:**", doctor["doctor_id"])

    with col2:

        st.write("**Hospital/Clinic:**", doctor["clinic_name"] or "Not specified")
        st.write("**Phone:**", doctor["phone"] or "Not specified")
        st.write("**Experience:**", f"{doctor['experience'] or 0} years")
        st.write("**Location:**", doctor["location"] or "Not specified")

    st.markdown("---")

    st.subheader("Edit Profile")

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
# REPORTS UI (Placeholder)
# =========================================================
def render_reports(doctor):

    st.title("Reports")

    if not doctor:

        st.warning("Please login as a doctor to view reports.")

        return

    # =========================================================
    # DOCTOR ANALYTICS REPORT (Uses ONLY database/history records)
    # =========================================================

    doctor_id = int(doctor["id"])

    st.subheader("Doctor Analytics Report")
    st.caption("Generated from your actual clinical activity history.")

    # Fetch only this doctor’s historical records
    rows = fetch_all(
        """
        SELECT
            id,
            disease,
            image_path,
            confidence,
            prediction_source,
            created_at
        FROM searches
        WHERE doctor_id = ?
        ORDER BY created_at ASC, id ASC
        """,
        (doctor_id,),
    )

    total_images_analyzed = len(rows)
    total_reports_generated = len(rows)

    avg_confidence = None
    most_detected_disease = None
    disease_distribution = {}

    if rows:
        avg_confidence = sum(float(r["confidence"]) for r in rows) / len(rows)

        for r in rows:
            d = r["disease"]
            disease_distribution[d] = disease_distribution.get(d, 0) + 1

        most_detected_disease = max(disease_distribution.items(), key=lambda kv: kv[1])[0]

    # ---------------------- UI: Clinical Summary ----------------------
    st.markdown("---")

    doctor_info_cols = st.columns(4)

    report_date = datetime.now().strftime("%Y-%m-%d")

    doctor_info_cols[0].metric("Doctor Name", doctor.get("full_name", ""))
    doctor_info_cols[1].metric("Doctor ID", doctor.get("doctor_id", ""))
    doctor_info_cols[2].metric("Specialization", doctor.get("specialization", ""))
    doctor_info_cols[3].metric("Report Date", report_date)

    st.markdown("---")

    st.subheader("Clinical Activity Summary")

    summary_cols = st.columns(3)

    summary_cols[0].metric("Total Images Analyzed", str(total_images_analyzed))
    summary_cols[1].metric("Total Reports Generated", str(total_reports_generated))
    summary_cols[2].metric(
        "Most Detected Disease",
        most_detected_disease if most_detected_disease else "N/A",
    )

    st.info(
        f"Average Confidence Score: {avg_confidence:.2f}%" if avg_confidence is not None else "Average Confidence Score: N/A"
    )

    # ---------------------- UI: Case Analysis ----------------------
    st.markdown("---")
    st.subheader("Case Analysis Report")

    if not rows:
        st.info("No analyzed cases available yet.")
    else:
        for idx, r in enumerate(rows, start=1):
            image_path = r["image_path"] if "image_path" in r.keys() else ""
            image_name = Path(image_path).name if image_path else ""
            confidence = (
                float(r["confidence"]) if "confidence" in r.keys() and r["confidence"] is not None else 0.0
            )
            created_at = r["created_at"] if "created_at" in r.keys() else ""
            predicted_disease = r["disease"] if "disease" in r.keys() else ""

            with st.container():

                st.markdown(f"### Case {idx}")
                c1, c2 = st.columns([1, 2])
                with c1:

                    if image_path:
                        try:
                            st.image(image_path, caption=image_name, use_container_width=True)
                        except Exception:
                            st.warning("Image preview unavailable.")
                    else:
                        st.warning("No image path stored for this record.")

                with c2:
                    st.write(f"**Date & Time:** {created_at}")
                    st.write(f"**Image Name:** {image_name or 'N/A'}")
                    st.write(f"**Predicted Disease:** {predicted_disease}")
                st.write(f"**Confidence Score:** {confidence:.2f}%")

                st.markdown("---")

    # ---------------------- UI: Disease Distribution ----------------------
    st.markdown("---")
    st.subheader("Disease Distribution")

    if disease_distribution:
        dist_df = pd.DataFrame(
            [{"Disease": d, "Count": c} for d, c in sorted(disease_distribution.items(), key=lambda kv: kv[1], reverse=True)]
        )
        st.dataframe(dist_df, use_container_width=True)
    else:
        st.info("No disease distribution available.")

    # =========================================================
    # PDF REPORT (No AI Diagnosis Summary, No Doctor Signature, No patient info)
    # =========================================================

    def _pdf_lines_for_doctor_analytics() -> list[str]:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        title_lines = [
            "SKIN DISEASE DETECTION SYSTEM",
            "DOCTOR ANALYTICS REPORT",
            "",
        ]

        doc_lines = [
            "DOCTOR INFORMATION",
            "-------------------",
            f"Doctor Name: {doctor.get('full_name', '')}",
            f"Doctor ID: {doctor.get('doctor_id', '')}",
            f"Specialization: {doctor.get('specialization', '')}",
            f"Report Date: {report_date}",
            "",
        ]

        activity_lines = [
            "CLINICAL ACTIVITY SUMMARY",
            "---------------------------",
            f"Total Images Analyzed: {total_images_analyzed}",
            f"Total Reports Generated: {total_reports_generated}",
            f"Most Detected Disease: {most_detected_disease if most_detected_disease else 'N/A'}",
            f"Average Confidence Score: {avg_confidence:.2f}%" if avg_confidence is not None else "Average Confidence Score: N/A",
            "",
        ]

        case_lines = [
            "CASE ANALYSIS REPORT",
            "----------------------",
        ]

        if not rows:
            case_lines.append("No analyzed cases available yet.")
        else:
            for idx, r in enumerate(rows, start=1):
                image_path = r["image_path"] if "image_path" in r.keys() else ""
                image_name = Path(image_path).name if image_path else ""
                confidence = (
                    float(r["confidence"]) if "confidence" in r.keys() and r["confidence"] is not None else 0.0
                )


                case_lines.extend([
                    "",
                    f"Case Number: {idx}",
                    f"Date & Time: {r['created_at']}",
                    f"Image Name: {image_name or 'N/A'}",
                    f"Predicted Disease: {r['disease']}",
                    f"Confidence Score: {confidence:.2f}%",
                ])

        dist_lines = [
            "",
            "DISEASE DISTRIBUTION",
            "----------------------",
        ]

        if disease_distribution:
            # Keep distribution order by count desc
            for d, c in sorted(disease_distribution.items(), key=lambda kv: kv[1], reverse=True):
                dist_lines.append(f"{d}: {c}")
        else:
            dist_lines.append("No disease distribution available.")

        footer_lines = [
            "",
            "REPORT GENERATED SUCCESSFULLY",
            "",
            "Generated By: Skin Disease Detection System",
            f"Generated On: {now}",
        ]

        # Join all sections
        return title_lines + doc_lines + activity_lines + case_lines + dist_lines + footer_lines

    pdf_bytes = generate_pdf(_pdf_lines_for_doctor_analytics())

    st.download_button(
        label="📄 Download Doctor Analytics PDF",
        data=pdf_bytes,
        file_name="doctor_analytics_report.pdf",
        mime="application/pdf",
        use_container_width=True,
    )



# =========================================================
# DOCTOR MANAGEMENT UI (Admin)
# =========================================================
def render_doctor_management():

    st.title("Doctor Management")

    st.subheader("Registered Doctors")

    # Get all doctors with their image analysis counts
    from database.db import fetch_all, fetch_one

    doctors = fetch_all("SELECT id, full_name, email FROM doctors ORDER BY full_name")

    if doctors:
        doctor_data = []
        for doc in doctors:
            # Count total searches/predictions for this doctor
            search_stats = fetch_one(
                "SELECT COUNT(*) as total FROM searches WHERE doctor_id = ?",
                (doc["id"],)
            )
            total_images = search_stats["total"] if search_stats else 0
            doctor_data.append({
                "Doctor Name": doc["full_name"],
                "Email": doc["email"],
                "Total Images Analyzed": total_images
            })

        st.dataframe(
            pd.DataFrame(doctor_data),
            use_container_width=True,
        )
    else:
        st.info("No doctors registered yet.")


# =========================================================
# SYSTEM MONITORING UI (Admin)
# =========================================================
def render_system_monitoring():

    st.title("System Monitoring")

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

    # Removed visible sections:
    # 1) Disease Frequency chart
    # 2) Prediction Sources chart
    # 3) AI-Recognized Images table






# =========================================================
# MULTIPLE IMAGES PREDICTION UI
# =========================================================
def render_multiple_images_prediction(doctor):
    """Handle multiple image upload and prediction workflow."""
    
    st.title("🖼️ Multiple Images Prediction")
    st.markdown("---")
    st.warning("Upload up to 4 images for batch prediction.")
    
    # Multiple file upload with limit validation
    uploaded_files = st.file_uploader(
        "Upload Skin Images (Max 4)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )
    
    if uploaded_files:
        # Validate upload limit
        if len(uploaded_files) > 4:
            st.error("Maximum 4 images allowed. Please upload fewer images.")
            return
        
        st.info(f"Uploaded {len(uploaded_files)} image(s)")
        
        # Display uploaded images preview
        st.subheader("Image Previews")
        preview_cols = st.columns(min(len(uploaded_files), 4))
        for idx, file in enumerate(uploaded_files):
            with preview_cols[idx % 4]:
                try:
                    image = Image.open(file)
                    st.image(image, caption=f"Image {idx + 1}", use_container_width=True)
                except Exception as exc:
                    st.error(f"Error loading image {idx + 1}: {str(exc)}")
        
        # Predict button
        if st.button("🔍 Predict All Diseases", key="predict_multiple"):
            active_doctor = current_doctor()
            
            # Check search limit for doctors
            try:
                if active_doctor:
                    assert_can_search(int(active_doctor["id"]))
            except PermissionError as exc:
                st.error(str(exc))
                st.warning("Upgrade to premium to continue using doctor searches.")
                return
            
            # Process each image
            results = []
            with st.spinner(f"Analyzing {len(uploaded_files)} image(s)..."):
                for idx, uploaded_file in enumerate(uploaded_files):
                    try:
                        # Validate and load image
                        upload_bytes = validate_image_upload(uploaded_file)
                        image = image_from_bytes(upload_bytes)
                        
                        # Preprocess
                        processed_image, image_array = preprocess_image(image)
                        
                        ensemble_result = None
                        ensemble_metadata = None

                        if ENABLE_MULTI_MODEL_PREDICTION and len(model_manager.list_available_models()) > 1:
                            ensemble_result = run_multi_model_workflow(image, image_array)
                            ml_confidence = ensemble_result["final_confidence"]
                            predicted_class = ensemble_result["final_class"]
                            probs_entropy = 0.0
                            ensemble_metadata = ensemble_result["metadata"]
                        else:
                            # ML Prediction
                            raw_scores = model.predict(image_array, verbose=0)[0]
                            
                            # Confidence calibration (same logic as single image)
                            raw_scores = np.asarray(raw_scores, dtype=np.float64)
                            if raw_scores.ndim != 1:
                                raise ValueError(f"Model output must be 1D per sample; got shape {raw_scores.shape}")
                            
                            # Check if probabilities or logits
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
                            else:
                                exp = np.exp(raw_scores - np.max(raw_scores))
                                probs = exp / np.sum(exp)
                            
                            # Validation
                            prob_sum = float(np.sum(probs))
                            if not (abs(prob_sum - 1.0) <= 1e-3):
                                raise ValueError(f"Probability validation failed: sum(probs)={prob_sum}")
                            
                            if np.any(probs < -1e-6):
                                raise ValueError(f"Probability validation failed: negative probs min={float(np.min(probs))}")
                            
                            sorted_indices = np.argsort(probs)[::-1]
                            best_index = int(sorted_indices[0])
                            ml_confidence = float(probs[best_index]) * 100.0
                            predicted_class = class_names[best_index]
                            
                            # Calculate entropy
                            probs_entropy = float(-np.sum(np.clip(probs, 1e-12, 1.0) * np.log(np.clip(probs, 1e-12, 1.0))))
                        
                        # Hybrid ML + AI workflow
                        ML_CONFIDENCE_FALLBACK_MIN = 80.0
                        ML_ENTROPY_FALLBACK_MAX = 0.9
                        
                        should_use_ml = (
                            ensemble_result is not None
                            or not (
                                ml_confidence < ML_CONFIDENCE_FALLBACK_MIN
                                or probs_entropy > ML_ENTROPY_FALLBACK_MAX
                            )
                        )
                        
                        final_class = predicted_class
                        final_confidence = ml_confidence
                        prediction_source = "Ensemble" if ensemble_result is not None else "ML"
                        ai_result = None
                        ai_fallback_status = "not_used"
                        retraining_status = "not_required"
                        
                        if ensemble_result is not None or (should_use_ml and ml_confidence > CONFIDENCE_THRESHOLD):
                            final_class = predicted_class
                            final_confidence = ml_confidence
                            prediction_source = "Ensemble" if ensemble_result is not None else "ML"
                        else:
                            # AI fallback
                            ai_fallback_status = "triggered"
                            prediction_source = "AI"
                            ai_result = recognize_with_ai(image)
                            
                            ai_disease = (ai_result or {}).get("disease", "Unknown")
                            ai_conf = float((ai_result or {}).get("confidence", 0) or 0.0)
                            
                            if (
                                ai_disease
                                and str(ai_disease).strip().lower() != "unknown"
                                and ai_conf > 0
                            ):
                                final_class = ai_disease
                                final_confidence = ai_conf
                                
                                # Save for self-learning
                                learn_result = save_ai_prediction_for_learning(
                                    image,
                                    final_class,
                                    final_confidence,
                                    getattr(uploaded_file, "name", f"multi_upload_{idx}.jpg"),
                                    int(active_doctor["id"]) if active_doctor else None,
                                )
                                
                                if learn_result.get("duplicate"):
                                    retraining_status = "duplicate"
                                elif learn_result.get("saved"):
                                    retraining_status = "queued"
                            else:
                                # AI failed, keep ML prediction
                                final_class = predicted_class
                                final_confidence = ml_confidence
                                prediction_source = "ML"
                                retraining_status = "not_queued"
                                ai_result = ai_result or {"disease": "Unknown", "confidence": 0.0, "explanation": ""}
                        
                        # Record search in database
                        try:
                            record_search(
                                int(active_doctor["id"]) if active_doctor else None,
                                image,
                                final_class,
                                final_confidence,
                                prediction_source,
                                ai_fallback_status,
                                retraining_status,
                                ensemble_metadata,
                            )
                            
                            if active_doctor:
                                consume_search(int(active_doctor["id"]))
                        except Exception as exc:
                            print(f"[DEBUG] Search tracking warning for image {idx + 1}: {str(exc)}")
                        
                        # Save to session state history
                        if final_class and str(final_class).strip().lower() not in {"", "unknown"}:
                            st.session_state.prediction_history.append(
                                {
                                    "label": final_class,
                                    "confidence": final_confidence,
                                    "prediction_source": prediction_source,
                                    "time": str(datetime.now()),
                                }
                            )
                        
                        # Store result
                        results.append({
                            "index": idx + 1,
                            "image": image,
                            "predicted_disease": final_class,
                            "confidence": final_confidence,
                            "prediction_source": prediction_source,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "ai_result": ai_result,
                            "ensemble_result": ensemble_result,
                        })
                        
                        print(f"[DEBUG] Image {idx + 1} prediction: {final_class} ({final_confidence:.2f}%) via {prediction_source}")
                        
                    except Exception as exc:
                        st.error(f"Error processing image {idx + 1}: {str(exc)}")
                        print(f"[DEBUG] Error processing image {idx + 1}: {str(exc)}")
                        results.append({
                            "index": idx + 1,
                            "image": None,
                            "predicted_disease": "Error",
                            "confidence": 0.0,
                            "prediction_source": "Error",
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "ai_result": None,
                            "ensemble_result": None,
                        })
            
            # Display results in responsive cards
            st.subheader("Prediction Results")
            st.markdown("---")
            
            if results:
                # Summary metrics
                ml_count = sum(1 for r in results if r["prediction_source"] == "ML")
                ai_count = sum(1 for r in results if r["prediction_source"] == "AI")
                ensemble_count = sum(1 for r in results if r["prediction_source"] == "Ensemble")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Images", len(results))
                col2.metric("ML/Ensemble", ml_count + ensemble_count)
                col3.metric("AI Predictions", ai_count)
                
                st.markdown("---")
                
                # Display each result in a card
                for result in results:
                    with st.container():
                        # Card header
                        st.markdown(f"### Image {result['index']}")
                        
                        # Card content in columns
                        card_col1, card_col2 = st.columns([1, 2])
                        
                        with card_col1:
                            if result["image"]:
                                st.image(result["image"], caption=f"Image {result['index']}", use_container_width=True)
                            else:
                                st.warning("Image not available")
                        
                        with card_col2:
                            st.markdown(f"**Predicted Disease:** {result['predicted_disease']}")
                            st.markdown(f"**Confidence:** {result['confidence']:.2f}%")
                            
                            # Color-coded prediction source
                            if result["prediction_source"] == "ML":
                                st.success(f"**Prediction Source:** ML")
                            elif result["prediction_source"] == "Ensemble":
                                st.success(f"**Prediction Source:** Ensemble")
                            elif result["prediction_source"] == "AI":
                                st.warning(f"**Prediction Source:** AI")
                            else:
                                st.error(f"**Prediction Source:** {result['prediction_source']}")
                            
                            st.markdown(f"**Timestamp:** {result['timestamp']}")
                            
                            # Progress bar for confidence
                            st.progress(min(result['confidence'], 100) / 100)

                            if result.get("ensemble_result") is not None:
                                with st.expander("View Model Comparison"):
                                    render_multi_model_result_sections(result["ensemble_result"])
                            
                            # Show AI analysis if available
                            if result["prediction_source"] == "AI" and result["ai_result"]:
                                with st.expander("View AI Analysis"):
                                    render_ai_analysis(result["ai_result"])
                        
                        st.markdown("---")
            else:
                st.warning("No results to display.")
    
    else:
        st.info("Please upload images to begin prediction.")


# =========================================================
# ADMIN ANALYTICS UI
# =========================================================
def render_admin_analytics():

    st.title("Admin Analytics")

    summary = admin_summary()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("AI Images", summary["total_ai"])

    col2.metric("Retrained Images", summary["retrained"])

    common = summary["most_common"]

    col3.metric("Most Added Disease", common["predicted_disease"] if common else "None")

    col4.metric("Accuracy Improvement", f"{summary['accuracy_improvement']:.2%}")

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
# Initialize model manager and load active model
try:
    model_manager = get_model_manager()
    model, class_names = load_model_and_labels()
    # Display loaded model info in sidebar
    available_models = model_manager.list_available_models()
    if available_models:
        st.sidebar.success(f"✓ Model loaded: {model_manager.get_active_model_name()}")
        if len(available_models) > 1:
            st.sidebar.caption(f"Available models: {', '.join(available_models)}")
except Exception as exc:
    st.error(f"Failed to load model: {exc}")
    st.stop()

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
# Get current user based on role
user_role = st.session_state.get("user_role", None)
doctor = current_doctor() if user_role == "doctor" else None

# Handle redirect to dashboard after login while still rendering sidebar navigation.
redirect_page = None
if st.session_state.redirect_to_dashboard:
    st.session_state.redirect_to_dashboard = False
    if user_role == "doctor":
        redirect_page = "Doctor Dashboard"
    elif user_role == "admin":
        redirect_page = "Admin Dashboard"

# Show different navigation based on role
if user_role == "doctor":
    doctor_pages = [
        "Doctor Dashboard",
        "Profile Management",
        "Image Prediction",

        "Prediction History",
        "Reports",
    ]
    page = st.sidebar.radio(
        "Navigation",
        doctor_pages,
        index=doctor_pages.index(redirect_page) if redirect_page in doctor_pages else 0,
    )
elif user_role == "admin":
    admin_pages = [
        "Admin Dashboard",
        "Manage Doctors",
        "System Monitoring",
    ]

    page = st.sidebar.radio(
        "Navigation",
        admin_pages,
        index=admin_pages.index(redirect_page) if redirect_page in admin_pages else 0,
    )
else:
    page = "Login"

# Render logout button at top-right corner (outside sidebar)
if user_role in ["doctor", "admin"]:
    col1, col2, col3 = st.columns([6, 1, 1])
    with col3:
        if st.button("Logout", key="top_right_logout"):
            st.session_state.pop("doctor_pk", None)
            st.session_state.pop("admin_pk", None)
            st.session_state.pop("user_role", None)
            st.rerun()
    st.markdown("---")

# Show user info in sidebar
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

# Page routing
if page == "Login":

    render_common_auth()

    st.stop()

if page == "Doctor Dashboard":

    render_doctor_dashboard(doctor)

    st.stop()

if page == "Image Prediction":

    render_multiple_images_prediction(doctor)

    st.stop()


if page == "Prediction History":

    render_prediction_history(doctor)

    st.stop()

if page == "Profile Management":

    render_profile_management(doctor)

    st.stop()

if page == "Reports":

    render_reports(doctor)

    st.stop()

if page == "Admin Dashboard":

    render_admin_analytics()

    st.stop()

if page == "Manage Doctors":

    render_doctor_management()

    st.stop()

if page == "System Monitoring":

    render_system_monitoring()

    st.stop()

# Sidebar: Prediction History (for doctors)
if user_role == "doctor":

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
# TITLE (Only shown on Prediction page)
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
        ensemble_result = None
        ensemble_metadata = None

        if ENABLE_MULTI_MODEL_PREDICTION and len(model_manager.list_available_models()) > 1:
            with st.spinner("Comparing MobileNetV2, EfficientNetB0, and DenseNet121..."):
                ensemble_result = run_multi_model_workflow(image, image_array)
            final_class = ensemble_result["final_class"]
            final_confidence = ensemble_result["final_confidence"]
            prediction_source = ensemble_result["prediction_source"]
            confidence = final_confidence
            predicted_class = final_class
            ml_confidence = final_confidence
            ensemble_metadata = ensemble_result["metadata"]

        # =============================================
        # AI VERIFICATION LAYER (Post-Demo Improvement)
        # =============================================
        # If enabled, verify ML prediction with AI for improved accuracy
        verification_result = None
        if ensemble_result is None and ENABLE_AI_VERIFICATION and confidence >= AI_VERIFICATION_THRESHOLD:
            try:
                with st.spinner("Verifying prediction with AI..."):
                    verification_result = verify_prediction_with_ai(
                        image,
                        predicted_class,
                        confidence,
                    )
                    print(f"[AI VERIFICATION] ML: {predicted_class} ({confidence:.2f}%) | AI: {verification_result['ai_prediction']} ({verification_result['ai_confidence']:.2f}%) | Agreement: {verification_result['agreement']} | Source: {verification_result['verification_source']}")
            except Exception as exc:
                print(f"[AI VERIFICATION] Error: {exc}")
                verification_result = None

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

        should_use_ml = (
            ensemble_result is not None
            or not (
                confidence < ML_CONFIDENCE_FALLBACK_MIN
                or probs_entropy > ML_ENTROPY_FALLBACK_MAX
            )
        )

        # Use AI verification result if available and it suggests using AI.
        if ensemble_result is None and verification_result and verification_result["verification_source"] == "AI":
            should_use_ml = False
            print(f"[AI VERIFICATION] Using AI prediction due to verification result")

        if ensemble_result is not None or (should_use_ml and confidence > CONFIDENCE_THRESHOLD):





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
                ensemble_metadata,
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

        elif prediction_source == "Ensemble":

            st.success("Predicted by Ensemble")

        else:

            st.warning("Predicted by AI")
            st.info(
                f"ML confidence was low ({ml_confidence:.2f}%), so AI analysis was used."
            )

        st.progress(
            min(final_confidence, 100) / 100
        )

        if ensemble_result is not None:
            render_multi_model_result_sections(ensemble_result)

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
