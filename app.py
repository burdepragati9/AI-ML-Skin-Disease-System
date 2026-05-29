# =========================================================
# FINAL UPDATED app.py
# HEATMAP ERROR FIXED VERSION
# =========================================================

import json
import textwrap
import os
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

from analytics.admin_analytics import (
    admin_summary,
    ai_recognized_images,
    training_status,
    admin_summary_by_time,
    admin_prediction_source_counts_by_time,
    admin_disease_frequency_by_time,
)

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
from auth.admin_auth import admin_authenticate
from database.db import fetch_all, fetch_one, init_db
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
# AUTH VIEW NAV
# =========================================================
# Query params are accepted for deep links, but auth sub-section clicks are
# handled by Streamlit session state.
_q = st.query_params
if _q.get("doctor_auth_view"):
    st.session_state.doctor_auth_view = _q.get("doctor_auth_view")
    if _q.get("doctor_auth_view") in {"forgot", "reset"}:
        st.session_state.active_tab = "login"
if _q.get("admin_auth_view"):
    st.session_state.admin_auth_view = _q.get("admin_auth_view")


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

/* Auth selector: keep Streamlit's compact radio look like Login / Signup. */
[data-testid="stRadio"] [role="radiogroup"] {
    gap: 1rem;
}

[data-testid="stRadio"] [role="radiogroup"] label {
    align-items: center;
    cursor: pointer;
    color: #111827;
    font-size: 1rem;
    font-weight: 400;
    padding: 0 0.2rem 0.5rem 0;
}

.auth-links {
    font-size: 0.82rem;
    margin-top: 0.35rem;
}

.auth-links a {
    color: #1f77ff !important;
    text-decoration: none;
}

.auth-links a:hover {
    text-decoration: underline;
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


def set_doctor_auth_view(view: str) -> None:
    st.session_state.doctor_auth_view = view
    if view != "reset":
        st.session_state.pop("password_reset_token", None)
    try:
        st.query_params["doctor_auth_view"] = view
    except Exception:
        pass


def resolve_password_reset_identifier(identifier: str):
    value = sanitize_text(identifier, 180)
    if not value:
        return None

    return fetch_one(
        """
        SELECT email
        FROM doctors
        WHERE lower(email) = lower(?)
           OR lower(doctor_id) = lower(?)
           OR phone = ?
        LIMIT 1
        """,
        (value, value, value),
    )


def render_auth_state_links() -> None:
    st.markdown(
        """
        <div class="auth-links">
            <a href="?doctor_auth_view=forgot">Forgot Password?</a>
            <span>&nbsp;|&nbsp;</span>
            <a href="?doctor_auth_view=reset">Reset Password?</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_login_form():
    if "doctor_auth_view" not in st.session_state:
        st.session_state.doctor_auth_view = "login"

    if st.session_state.doctor_auth_view == "forgot":
        st.markdown("**Forgot Password?**")
        reset_identifier = st.text_input(
            "Email, Username, or Phone Number",
            key="forgot_password_email",
            help="Enter the email, doctor ID, or phone number linked to your doctor account.",
        )
        if st.button("Verify User", key="verify_forgot_password_user"):
            try:
                if not reset_identifier.strip():
                    st.error("Email, username, or phone number is required.")
                else:
                    doctor_row = resolve_password_reset_identifier(reset_identifier)
                    token = create_reset_token(doctor_row["email"]) if doctor_row else None
                    if token:
                        st.session_state.password_reset_token = token
                        st.success("User verified. Continue with password reset.")
                        set_doctor_auth_view("reset")
                        st.rerun()
                    else:
                        st.error("No doctor account found for those details.")
            except Exception:
                st.error("Could not verify the user. Please try again.")

        render_auth_state_links()

    elif st.session_state.doctor_auth_view == "reset":
        st.markdown("**Reset your password**")
        token = st.session_state.get("password_reset_token")
        if not token:
            token = st.text_input(
                "Reset Token",
                help="Use the token generated from Forgot Password.",
                key="reset_password_token",
            )
        new_password = st.text_input("New Password", type="password", key="reset_password_new")
        confirm_password = st.text_input("Confirm New Password", type="password", key="reset_password_confirm")

        if st.button("Reset Password", key="reset_password_submit"):
            try:
                if not str(token or "").strip():
                    st.error("Reset token is required.")
                elif new_password != confirm_password:
                    st.error("New password and confirm password do not match.")
                elif len(new_password or "") < 8:
                    st.error("Password must be at least 8 characters.")
                else:
                    if reset_password(token, new_password):
                        st.success("Password reset successful. Please login.")
                        set_doctor_auth_view("login")
                        st.rerun()
                    else:
                        st.error("Invalid or expired reset token.")
            except Exception:
                st.error("Password reset failed. Please try again.")

        render_auth_state_links()

    else:
        with st.form("doctor_login_form"):
            login_user = st.text_input("Username OR Email OR Phone Number", key="login_user")
            password = st.text_input("Password", type="password", key="login_password")
            hospital_name = st.text_input("Hospital Name", key="login_hospital_name")
            submitted = st.form_submit_button("Login")

        if submitted:
            try:
                doctor = authenticate_doctor(login_user, password)
            except Exception:
                doctor = None

            if doctor:
                st.session_state.doctor_pk = doctor["id"]
                st.session_state.redirect_to_dashboard = True
                st.session_state.admin_pk = None
                st.success("Login successful. Redirecting to Doctor Dashboard...")
                st.rerun()

            try:
                admin = admin_authenticate(login_user, password)
            except Exception:
                admin = None

            if admin:
                st.session_state.admin_pk = admin["id"]
                st.session_state.admin_redirect = True
                st.session_state.doctor_pk = None
                st.success("Login successful. Redirecting to Admin Dashboard...")
                st.rerun()

            st.error("Invalid username/email/phone or password.")

        render_auth_state_links()


def render_signup_form():
    with st.form("doctor_signup_form"):
        full_name = st.text_input("Full Name", key="signup_fullname")
        email = st.text_input("Email", key="signup_email")
        doctor_id = st.text_input("Doctor ID", key="signup_doctor_id")
        specialization = st.text_input("Specialization", key="signup_specialization")
        phone = st.text_input("Phone Number", key="signup_phone")
        hospital_name = st.text_input("Hospital Name", key="signup_hospital_name")
        username = st.text_input("Username", key="signup_username")
        password = st.text_input("Password", type="password", key="signup_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm_password")
        submitted = st.form_submit_button("Create Account")

    if submitted:
        if password != confirm_password:
            st.error("Password and Confirm Password must match.")
            return

        if not full_name.strip() or not email.strip() or not doctor_id.strip() or not specialization.strip() or not hospital_name.strip() or not username.strip():
            st.error("Full Name, Email, Doctor ID, Specialization, Hospital Name, and Username are required.")
            return

        profile = {
            "full_name": full_name,
            "doctor_id": doctor_id,
            "specialization": specialization,
            "clinic_name": hospital_name,
            "email": email,
            "phone": phone,
            "profile_photo": "",
            "experience": 0,
            "location": "",
        }

        try:
            doctor_pk = create_doctor(profile, password)
            st.session_state.doctor_pk = doctor_pk
            st.session_state.admin_pk = None
            st.session_state.redirect_to_dashboard = True
            st.success("Signup successful. Redirecting to Doctor Dashboard...")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))


def render_doctor_auth():

    st.title("Doctor Access")

    if "active_tab" not in st.session_state or st.session_state.active_tab not in {"login", "signup"}:
        st.session_state.active_tab = "login"

    st.radio(
        "Authentication",
        ["login", "signup"],
        format_func=lambda value: value.title(),
        horizontal=True,
        label_visibility="collapsed",
        key="active_tab",
    )

    active_tab = st.session_state.active_tab
    if active_tab == "login":
        render_login_form()
    elif active_tab == "signup":
        st.session_state.doctor_auth_view = "login"
        st.query_params["doctor_auth_view"] = "login"
        render_signup_form()




# =========================================================
# DOCTOR DASHBOARD UI
# =========================================================
def render_doctor_page_header(title: str, description: str = "") -> None:
    """Doctor page header.

    Matches the simpler, cleaner layout used elsewhere in the doctor UI.
    Avoids extra descriptive captions and automatic horizontal separators.
    """
    st.markdown(
        "<div style='margin: 0 0 8px 0; padding: 0; text-align: center;'>"
        f"<h1 style='margin: 0; font-size: 2rem; line-height: 1.2;'>{title}</h1>"
        "</div>",
        unsafe_allow_html=True,
    )

    return




def render_doctor_dashboard(doctor):

    render_doctor_page_header(
        "Doctor Dashboard",
    )


    # Doctor-specific dashboard must not reveal ML-vs-AI origin.
    # Do not show prediction_source analytics in any doctor UI.


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

    # Open metrics row (no boxed containers)
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Searches", stats["total"])

    col2.metric("Free Searches Left", usage["remaining"])

    col3.metric("Used Searches", usage["used"])


    if usage["remaining"] <= 0:

        st.warning("Free search limit reached. Upgrade to premium to continue searching.")

    # Open layout for charts: wider main graph area
    chart_col1, chart_col2 = st.columns([3, 1])

    with chart_col1:
        st.subheader("Most Searched Diseases")

        # NOTE: Doctor dashboard must not reveal any AI-vs-ML source info.

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



    # Intentionally do NOT show prediction source analytics on doctor UI.
    # Doctors must only see final disease results.

    # Side summary (simpler/open)
    with chart_col2:
        st.subheader("Recent Activity")
        st.metric("History Records", stats["total"])
        st.caption("Open Prediction History for filters and session-level entries.")

    st.subheader("Recent Searches")

    disease_filter = st.text_input("Filter by disease", key="dashboard_disease_filter")

    page_number = st.number_input(
        "History Page",
        min_value=1,
        step=1,
        key="dashboard_history_page",
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

                image_path = row["image_path"] if "image_path" in row.keys() else None
                if image_path and os.path.exists(image_path):

                    st.image(
                        image_path,
                        caption=f"{row['disease']} ({row['count']}x)",
                        use_container_width=True,
                    )

                else:

                    st.warning(
                        f"Image not found: {image_path}"
                    )

    else:

        st.info("No repeated image searches yet.")



def render_profile_management(doctor):
    render_doctor_page_header(
        "Profile Management",
    )

    with st.form("doctor_profile_form"):
        updated_profile = {
            "full_name": st.text_input("Full Name", value=doctor["full_name"]),
            "specialization": st.text_input("Specialization", value=doctor["specialization"]),
            "clinic_name": st.text_input(
                "Hospital/Clinic Name", value=doctor["clinic_name"] or ""
            ),
            "phone": st.text_input("Phone Number", value=doctor["phone"] or ""),
            "experience": st.number_input(
                "Experience",
                min_value=0,
                max_value=80,
                value=int(doctor["experience"] or 0),
            ),
            "location": st.text_input("Location", value=doctor["location"] or ""),
        }

        photo_upload = st.file_uploader(
            "Profile Photo",
            type=["jpg", "jpeg", "png"],
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

            st.success("Profile updated successfully")

            st.rerun()

        except Exception as exc:

            st.error("Unable to save profile. Please try again.")
            # Keep original exception details for server logs; UI stays user-friendly.
            print(f"[DEBUG] Profile save failed: {exc}")



def render_doctor_reports(doctor):
    render_doctor_page_header(
        "Reports",
        "Doctor-specific analytics based on your prediction and search history.",
    )

    doctor_id = int(doctor["id"])
    stats = doctor_search_stats(doctor_id)

    disease_df = pd.DataFrame([dict(row) for row in stats["diseases"]])

    total_searches = int(stats.get("total") or 0)
    unique_diseases = len(disease_df) if not disease_df.empty else 0
    most_searched = disease_df.iloc[0]["disease"] if not disease_df.empty else "None"

    # --- Summary cards (compact, equal width) ---
    card_col1, card_col2, card_col3, card_col4 = st.columns([1, 1, 1, 1])
    with card_col1:
        st.metric("Total Searches", f"{total_searches}")
    with card_col2:
        st.metric("Unique Diseases", f"{unique_diseases}")
    with card_col3:
        st.metric("Most Searched", f"{most_searched}")

    # Optional: average confidence from returned rows (best-effort)
    # (If created_at/confidence are missing in aggregate helper, keep it safe.)
    avg_conf = None
    try:
        rows = recent_searches(doctor_id, limit=200, offset=0, disease="")
        confs = []
        for r in rows:
            if r["confidence"] is not None:
                confs.append(float(r["confidence"]))
        if confs:
            avg_conf = sum(confs) / len(confs)
    except Exception:
        avg_conf = None

    with card_col4:
        st.metric(
            "Average Confidence",
            f"{avg_conf:.1f}%" if avg_conf is not None else "—",
        )

    st.markdown("---")

    # --- Charts section ---
    st.subheader("Disease Frequency")

    if disease_df.empty:
        st.info("No disease frequency data available yet.")
        return

    # Centered chart usage via container + responsive width
    chart_col = st.container()
    with chart_col:
        st.bar_chart(
            disease_df,
            x="disease",
            y="count",
            use_container_width=True,
        )

    # Table-style dataframe below chart
    st.dataframe(
        disease_df.rename(
            columns={
                "disease": "Disease",
                "count": "Count",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )



def render_prediction_history_page():
    render_doctor_page_header(
        "Prediction History",
    )

    # NOTE: This page is DB-driven (uploads are persisted in searches.image_path).
    # Do not modify ML/AI logic.

    with st.container():
        # Vertically stacked controls for professional dashboard alignment
        if st.button("Reset Filters", key="clear_history_main"):
            # Only affects filter/session UI; prediction records remain in DB.
            st.session_state.prediction_history = []
            st.session_state.history_disease_filter = ""
            st.session_state.history_page_number = 1
            st.rerun()

        disease_filter = st.text_input(
            "Filter by disease",
            key="history_disease_filter",
        )

        page_number = st.number_input(
            "Page",
            min_value=1,
            step=1,
            key="history_page_number",
        )




    limit = 10
    offset = (int(page_number) - 1) * limit

    doctor = _doctor
    if not doctor:
        st.warning("Please login as a doctor to view prediction history.")
        return

    rows = recent_searches(
        int(doctor["id"]),
        limit=limit,
        offset=offset,
        disease=sanitize_text(disease_filter),
    )

    if not rows:
        st.info("No prediction history found.")
        return

    # Table-style compact dashboard rendering
    table_rows = []
    for row in rows:
        disease = row["disease"] if row["disease"] else "Unknown"
        confidence = row["confidence"]
        created_at = row["created_at"]
        image_path = row["image_path"] if row["image_path"] else None

        try:
            confidence_f = float(confidence) if confidence is not None else 0.0
        except Exception:
            confidence_f = 0.0

        timestamp_text = str(created_at) if created_at is not None else ""
        if isinstance(created_at, (datetime,)):
            timestamp_text = created_at.strftime("%Y-%m-%d %H:%M:%S")

        has_image = bool(image_path and os.path.exists(image_path))
        table_rows.append(
            {
                "Disease": disease,
                "Confidence": f"{confidence_f:.2f}%",
                "Timestamp": timestamp_text,
                "Image": "✅" if has_image else "—",
                "image_path": image_path if has_image else None,
            }
        )

    st.write(
        """<div style='margin-top: 6px;'>""",
        unsafe_allow_html=True,
    )

    display_df = pd.DataFrame(table_rows)[
        ["Image", "Disease", "Confidence", "Timestamp"]
    ]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    # (Optional) thumbnails are intentionally omitted to keep the page compact.





# =========================================================
# MULTIPLE IMAGES PREDICTION UI
# =========================================================
def render_multiple_images_prediction(doctor):
    """Handle multiple image upload and prediction workflow."""
    
    st.title("🖼️ Multiple Images Prediction")
    st.caption("Upload up to 4 skin images for batch disease analysis.")
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
                        
                        should_use_ml = not (
                            ml_confidence < ML_CONFIDENCE_FALLBACK_MIN
                            or probs_entropy > ML_ENTROPY_FALLBACK_MAX
                        )
                        
                        final_class = predicted_class
                        final_confidence = ml_confidence
                        prediction_source = "ML"
                        ai_result = None
                        ai_fallback_status = "not_used"
                        retraining_status = "not_required"
                        
                        if should_use_ml and ml_confidence > CONFIDENCE_THRESHOLD:
                            final_class = predicted_class
                            final_confidence = ml_confidence
                            prediction_source = "ML"
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
                        })
            
            # Display results in responsive cards
            st.subheader("Prediction Results")

            if results:


                # Summary metrics
                ml_count = sum(1 for r in results if r["prediction_source"] == "ML")
                ai_count = sum(1 for r in results if r["prediction_source"] == "AI")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Images", len(results))
                # Doctor dashboard must not reveal AI-vs-ML origin (hide ML/AI counts)
                col2.metric("", "")
                col3.metric("", "")

                
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
                            
                            # Doctor UI must not reveal whether prediction came from AI or ML.
                            st.markdown(f"**Timestamp:** {result['timestamp']}")
                            
                            # Progress bar for confidence
                            st.progress(min(result['confidence'], 100) / 100)
                            
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

    # Admin-only time window filtering
    period = st.selectbox("Time Period", ["Weekly", "Monthly", "Yearly"], index=1)
    period_key = {
        "Weekly": "weekly",
        "Monthly": "monthly",
        "Yearly": "yearly",
    }[period]

    status = training_status()


    col1, col2, col3, col4 = st.columns(4)

    # Use time-windowed analytics for admin charts
    summary = admin_summary_by_time(period_key)

    col1.metric("AI Images", summary["total_ai"])

    # Note: retraining count is system-level; keep all-time behavior here
    col2.metric("Retrained Images", admin_summary()["retrained"])

    # Most common disease derived from selected time period
    common = (summary.get("disease_counts") or [])
    most_common_disease = common[0]["disease"] if common else "None"
    col3.metric("Most Added Disease", most_common_disease)

    # Accuracy improvement remains system-level (training logs not time-filtered in current helpers)
    latest_improvement = admin_summary()["accuracy_improvement"]
    col4.metric("Accuracy Improvement", f"{latest_improvement:.2%}")


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
# AUTH + SESSION + ROLE-BASED SIDEBAR
# =========================================================


def _is_doctor_logged_in() -> bool:
    return bool(st.session_state.get("doctor_pk"))


def _is_admin_logged_in() -> bool:
    return bool(st.session_state.get("admin_pk"))


def _doctor_session_guard() -> None:
    # Doctor-only pages should stop if doctor is missing
    if not _is_doctor_logged_in():
        st.warning("Please login as a doctor.")
        render_doctor_auth()
        st.stop()


def _admin_session_guard() -> None:
    if not _is_admin_logged_in():
        st.warning("Please login as an admin.")
        render_admin_auth_ui()
        st.stop()


def handle_doctor_logout() -> None:
    """Logout doctor session only."""
    st.session_state.pop("doctor_pk", None)
    st.session_state.pop("redirect_to_dashboard", None)
    st.session_state.pop("active_route", None)
    st.rerun()


def handle_admin_logout() -> None:
    """Logout admin session only."""
    st.session_state.pop("admin_pk", None)
    st.session_state.pop("admin_redirect", None)
    st.rerun()


def current_admin():
    admin_pk = st.session_state.get("admin_pk")
    if not admin_pk:
        return None
    return {"id": admin_pk, "full_name": "Admin"}


ADMIN_PAGES = [
    "Admin Dashboard",
    "Manage Doctors",
    "System Monitoring",
]

DOCTOR_PAGES = [
    "Doctor Dashboard",
    "Profile Management",
    "Image Prediction",
    "Prediction History",
    "Reports",
]



def render_admin_sidebar() -> str:
    """Render admin navigation on every admin dashboard page."""
    st.sidebar.title("Navigation")

    if st.session_state.get("admin_redirect"):
        st.session_state.admin_redirect = False
        st.session_state.admin_page = "Admin Dashboard"

    if st.session_state.get("admin_page") not in ADMIN_PAGES:
        st.session_state.admin_page = "Admin Dashboard"

    return st.sidebar.radio(
        "Navigation",
        ADMIN_PAGES,
        key="admin_page",
    )


def render_doctor_sidebar(doctor) -> str:
    """Render doctor navigation on every doctor dashboard page."""
    st.sidebar.title("Navigation")

    if st.session_state.get("redirect_to_dashboard"):
        st.session_state.redirect_to_dashboard = False
        st.session_state.doctor_page = "Doctor Dashboard"

    if st.session_state.get("doctor_page") not in DOCTOR_PAGES:
        st.session_state.doctor_page = "Doctor Dashboard"

    page = st.sidebar.radio(
        "Navigation",
        DOCTOR_PAGES,
        key="doctor_page",
    )

    try:
        st.sidebar.success(f"Doctor: {doctor['full_name']}")
        usage = usage_for_doctor(int(doctor["id"]))
        st.sidebar.caption(f"Free searches remaining: {usage['remaining']}")
    except Exception as exc:
        print(f"[DEBUG] usage sidebar error: {exc}")

    return page


# =========================================================
# ADMIN AUTH UI
# =========================================================

def render_admin_auth_ui():
    st.title("Admin Access")

    login_tab, signup_tab = st.tabs(["Login", "Signup"])

    with login_tab:
        with st.container(border=True):
            if "admin_auth_view" not in st.session_state:
                st.session_state.admin_auth_view = "login"

            with st.form("admin_login_form"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Login")

            if submitted:
                admin = admin_authenticate(email, password)
                if admin:
                    st.session_state.admin_pk = admin["id"]
                    st.session_state.admin_redirect = True
                    st.success("Login successful. Redirecting to Admin Dashboard...")
                    st.rerun()
                else:
                    st.error("Invalid email or password.")

            admin_link_col1, admin_link_col2, _admin_spacer = st.columns([1.2, 1.2, 6])
            with admin_link_col1:
                if st.button("Forgot Password", key="admin_forgot_password_link", type="tertiary"):
                    st.session_state.admin_auth_view = "forgot"
                    st.rerun()
            with admin_link_col2:
                if st.button("Reset Password", key="admin_reset_password_link", type="tertiary"):
                    st.session_state.admin_auth_view = "reset"
                    st.rerun()

            if st.session_state.admin_auth_view == "forgot":
                st.markdown("**Forgot Password?**")
                st.info("Password reset for admin is not configured in this demo. Contact the system administrator.")

            if st.session_state.admin_auth_view == "reset":
                st.markdown("**Reset your password**")
                st.text_input("New Password", type="password", key="admin_reset_password_new")
                st.text_input("Confirm New Password", type="password", key="admin_reset_password_confirm")
                if st.button("Reset Password", key="admin_reset_password_submit"):
                    st.info("Password reset for admin is not configured in this demo. Contact the system administrator.")


    with signup_tab:
        st.info("Admin accounts are configured via environment variables (ADMIN_EMAIL / ADMIN_PASSWORD).")


# =========================================================
# Admin + Doctor route selection
# =========================================================

# Precompute logged-in roles
_doctor = current_doctor() if _is_doctor_logged_in() else None
_admin = current_admin() if _is_admin_logged_in() else None

# Public access: single common login (auto role detection)
if not _is_doctor_logged_in() and not _is_admin_logged_in():
    render_doctor_auth()
    st.stop()

# If admin logged in, show admin sidebar exclusively
if _is_admin_logged_in():
    _admin_session_guard()

    admin_page = render_admin_sidebar()

    # Route: ensure Admin Dashboard UI actually renders.
    # (Previously an early st.stop() caused a blank page after login.)

    # Shared admin header actions (keep Logout visible across all admin pages)
    _left, _right = st.columns([8, 2])
    with _right:
        if st.button("Logout"):
            handle_admin_logout()

    # Render admin pages below
    if admin_page == "Admin Dashboard":

        # Debug: session + routing

        print(
            "\n[DEBUG][ADMIN] rendering Admin Dashboard | admin_redirect=",
            st.session_state.get("admin_redirect"),
            "admin_pk=",
            st.session_state.get("admin_pk"),
            "_admin=",
            bool(_admin),
            "admin_page=",
            admin_page,
            "selected=",
            st.session_state.get("active_route"),
        )

        # Pre-check analytics so we can show the required empty-state message
        # even if the downstream dashboard charts/tables are empty.
        show_empty_fallback = False
        try:
            snapshot = admin_summary_by_time("weekly")
            show_empty_fallback = not (
                snapshot.get("disease_counts") or snapshot.get("source_counts")
            )
            print(
                "[DEBUG][ADMIN] analytics empty-state decision=",
                show_empty_fallback,
                "weekly_disease_counts=",
                bool(snapshot.get("disease_counts")),
                "weekly_source_counts=",
                bool(snapshot.get("source_counts")),
            )
        except Exception as exc:
            print(f"[DEBUG][ADMIN] analytics empty-state pre-check failed: {exc}")

        # (Top analytics summary row intentionally removed as requested.)


        try:
            # Admin Dashboard is implemented by render_admin_analytics().
            render_admin_analytics()

            if show_empty_fallback:
                st.info("No analytics data available")


        except Exception as exc:
            print(f"[DEBUG][ADMIN] Admin Dashboard render failed: {exc}")
            st.error("Admin Dashboard failed to render. Please try again.")

        st.stop()



    if admin_page == "Manage Doctors":
        st.title("Doctor Management")
        try:
            docs = fetch_all(
                """
                SELECT
                    d.full_name AS "Doctor Name",
                    d.email AS "Email",
                    COUNT(s.id) AS "Total Images Analyzed",
                    d.doctor_id AS "Doctor ID",
                    d.specialization AS "Specialization",
                    d.clinic_name AS "Clinic Name",
                    d.phone AS "Phone",
                    d.experience AS "Experience",
                    d.location AS "Location"
                FROM doctors d
                LEFT JOIN searches s ON s.doctor_id = d.id
                GROUP BY d.id
                ORDER BY d.created_at DESC
                LIMIT 50
                """
            )
            if docs:
                st.dataframe(
                    pd.DataFrame([dict(d) for d in docs]),
                    use_container_width=True,
                )
            else:
                st.info("No doctors found.")
        except Exception as exc:
            st.error(f"Failed to load doctors: {exc}")
        st.stop()

    if admin_page == "System Monitoring":
        st.title("System Monitoring")
        st.subheader("Training Status")
        st.json(training_status())
        st.stop()

# If doctor logged in, show doctor sidebar exclusively
_doctor_session_guard()
if _is_doctor_logged_in() and _doctor:
    render_doctor_sidebar(_doctor)
    doctor_page = st.session_state.doctor_page

    # Top-right logout button in header area (modern alignment)
    _left, _right = st.columns([8, 2])
    with _right:
        if st.button("Logout"):
            handle_doctor_logout()

    if doctor_page == "Doctor Dashboard":
        render_doctor_dashboard(_doctor)
        st.stop()

    if doctor_page == "Profile Management":
        render_profile_management(_doctor)
        st.stop()

    if doctor_page == "Image Prediction":
        render_multiple_images_prediction(_doctor)
        st.stop()


    if doctor_page == "Prediction History":
        render_prediction_history_page()
        st.stop()

    if doctor_page == "Reports":
        render_doctor_reports(_doctor)
        st.stop()

# If neither role matched (shouldn't happen), show auth
render_doctor_auth()
st.stop()


# After login gating, show public prediction history ONLY in the logged-out minimal mode
# (Requirement: hide dashboard/sidebar options before login; prediction history sidebar is hidden here.)



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
        # Doctor UI must not expose prediction source origin.
        if final_class and str(final_class).strip().lower() not in {"", "unknown"}:
            st.session_state.prediction_history.append(
                {
                    "label": final_class,
                    "confidence": final_confidence,
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

            st.success("Predicted")

        else:

            st.warning("Predicted")
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
