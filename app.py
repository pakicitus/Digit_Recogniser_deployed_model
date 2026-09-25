"""
app.py
------
Streamlit front-end for the Handwritten Digit Recognition project.
This file is intentionally "thin" — all preprocessing logic lives in
preprocessing.py, all model logic lives in predict.py, and shared
helpers (charts, session state, CSS) live in utils.py.
"""

import streamlit as st
from streamlit_drawable_canvas import st_canvas

import preprocessing
import predict
import utils

# ============================== CONFIG ======================================
GITHUB_URL = "https://github.com/pakicitus/Digit_Recogniser_deployed_model"
LINKEDIN_URL = "www.linkedin.com/in/harsh-vats-788198253"

st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

utils.load_css("assets/style.css")
utils.init_session_state()

model = predict.load_trained_model()


# ============================== SHARED RENDER FUNCTION ======================
def render_prediction(result: dict, pipeline_images: dict, source_label: str):
    """
    Render the full results block (digit badge, metrics, gauge, probability
    chart, preprocessing preview, download) for a given prediction.
    Also logs the prediction into session history.
    """
    predicted_class = result["predicted_class"]
    confidence = result["confidence"]
    probabilities = result["probabilities"]
    inference_ms = result["inference_ms"]

    st.markdown("### 🎯 Prediction Result")

    col_badge, col_metrics = st.columns([1, 2])

    with col_badge:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="digit-badge">{predicted_class}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<p style="text-align:center;color:#8B92B0;">Predicted Digit</p>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_metrics:
        m1, m2, m3 = st.columns(3)
        m1.metric("Confidence", f"{confidence*100:.1f}%")
        m2.metric("Inference Time", f"{inference_ms:.1f} ms")
        m3.metric("Top Alternative", f"{predict.get_top_k_predictions(probabilities, 2)[1][0]}")
        st.progress(min(confidence, 1.0))

        if confidence >= 0.85:
            st.success(f"High confidence prediction ({confidence*100:.1f}%).")
        elif confidence >= 0.5:
            st.warning(f"Moderate confidence ({confidence*100:.1f}%) — the digit may be ambiguous.")
        else:
            st.warning(f"Low confidence ({confidence*100:.1f}%) — try drawing more clearly or centering the digit.")

    # ---- Gauge + probability chart ----
    g_col, p_col = st.columns([1, 2])
    with g_col:
        st.markdown("**Confidence Gauge**")
        st.plotly_chart(utils.create_confidence_gauge(confidence), use_container_width=True)
    with p_col:
        st.markdown("**Prediction Probabilities (0–9)**")
        st.plotly_chart(
            utils.create_probability_bar_chart(probabilities, predicted_class),
            use_container_width=True,
        )

    # ---- Preprocessing pipeline preview ----
    st.markdown("### 🔍 Preprocessing Pipeline")
    st.caption("This shows exactly what the network sees after preprocessing.")
    prev1, arrow1, prev2, arrow2, prev3 = st.columns([3, 1, 3, 1, 3])
    with prev1:
        st.image(pipeline_images["original_image"], caption="Original Drawing", use_container_width=True)
    with arrow1:
        st.markdown('<div class="pipeline-arrow">→</div>', unsafe_allow_html=True)
    with prev2:
        st.image(pipeline_images["processed_display"], caption="Processed 28×28", use_container_width=True)
    with arrow2:
        st.markdown('<div class="pipeline-arrow">→</div>', unsafe_allow_html=True)
    with prev3:
        st.markdown('<div class="glass-card" style="text-align:center;">', unsafe_allow_html=True)
        st.markdown(f'<div class="digit-badge" style="font-size:3rem;">{predicted_class}</div>', unsafe_allow_html=True)
        st.markdown('<p style="color:#8B92B0;">Prediction</p>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ---- Download ----
    png_bytes = utils.image_to_png_bytes(pipeline_images["processed_28x28"])
    st.download_button(
        label="⬇️ Download Processed Image (PNG)",
        data=png_bytes,
        file_name=f"digit_{predicted_class}_prediction.png",
        mime="image/png",
    )

    # ---- Log to history ----
    utils.add_prediction_to_history(
        source=source_label,
        predicted_class=predicted_class,
        confidence=confidence,
        thumbnail=pipeline_images["processed_28x28"],
    )

    st.session_state.last_result = result


def run_prediction_pipeline(pipeline_images: dict):
    """Run the model on preprocessed images and return the result dict, with a spinner."""
    with st.spinner("Running inference through the neural network..."):
        result = predict.predict_digit(model, pipeline_images["model_input"])
    return result


# ============================== SIDEBAR ======================================
with st.sidebar:
    st.markdown("## 🧠 Digit Recognizer")
    st.caption("Deep Neural Network · MNIST · Streamlit")

    st.markdown("---")

    with st.expander("📘 About Project", expanded=True):
        st.write(
            "This project demonstrates an end-to-end handwritten digit "
            "classifier built with a Deep Neural Network trained on the "
            "MNIST dataset, deployed as an interactive Streamlit web app. "
            "It was built as part of a Deep Learning learning journey."
        )

    with st.expander("🧬 Model Information"):
        st.markdown(
            """
            - **Dataset:** MNIST (60,000 train / 10,000 test)
            - **Type:** Fully-connected Deep Neural Network
            - **Input:** 28×28 grayscale image (flattened to 784)
            - **Output:** 10 classes (digits 0–9)
            - **Framework:** TensorFlow / Keras
            """
        )

    with st.expander("⚙️ How It Works"):
        st.markdown(
            """
            1. Draw a digit on the canvas
            2. Image is converted to grayscale
            3. Resized to 28×28 pixels
            4. Pixel values normalized to [0, 1]
            5. Fed into the trained neural network
            6. Network outputs a probability for each digit
            """
        )

    st.markdown("---")
    st.markdown(
        f"""
        <div class="sidebar-link-btn"><a href="{GITHUB_URL}" target="_blank">💻 View on GitHub</a></div>
        <div class="sidebar-link-btn"><a href="{LINKEDIN_URL}" target="_blank">🔗 Connect on LinkedIn</a></div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    if st.button("🔄 Reset Application", use_container_width=True):
        utils.reset_app()
        st.rerun()


# ============================== HERO HEADER ==================================
st.markdown('<div class="hero-title">🧠 Handwritten Digit Recognition</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">"Draw a digit and let a Deep Neural Network recognize it instantly."</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero-description">A portfolio project showcasing a fully-connected Deep Neural '
    'Network trained on the MNIST dataset — from raw pixels to prediction, entirely in your browser.</div>',
    unsafe_allow_html=True,
)

if model is None:
    st.stop()

st.markdown("<br>", unsafe_allow_html=True)

# ============================== MAIN INPUT TABS ==============================
tab_draw, tab_upload, tab_random = st.tabs(["✏️ Draw a Digit", "📤 Upload an Image", "🎲 Random MNIST Sample"])

# ---------------------------------------------------------------- DRAW TAB ---
with tab_draw:
    col_canvas, col_controls = st.columns([2, 1])

    with col_controls:
        st.markdown("#### 🎨 Canvas Controls")
        brush_width = st.slider("Brush Width", min_value=5, max_value=35, value=32)
        st.caption("💡 Brush width **32** tends to give the most accurate predictions — thick, bold strokes match MNIST's training style.")
        brush_color = st.color_picker("Brush Color", value="#FFFFFF")
        st.caption("Background is fixed to black to match MNIST's training format.")
        clear_clicked = st.button("🧹 Clear Canvas", use_container_width=True)
        if clear_clicked:
            st.session_state.canvas_key += 1
            st.rerun()

    with col_canvas:
        canvas_result = st_canvas(
            fill_color="rgba(255,255,255,1)",
            stroke_width=brush_width,
            stroke_color=brush_color,
            background_color="#000000",
            update_streamlit=True,
            height=280,
            width=280,
            drawing_mode="freedraw",
            key=f"canvas_{st.session_state.canvas_key}",
            return_image_data=True,
        )

    predict_clicked = st.button("🚀 Predict Digit", type="primary", key="predict_draw")

    if predict_clicked:
        try:
            pipeline_images = preprocessing.preprocess_canvas_drawing(canvas_result.image_data)
            result = run_prediction_pipeline(pipeline_images)
            render_prediction(result, pipeline_images, source_label="Draw")
        except ValueError as e:
            st.warning(f"⚠️ {e}")

# -------------------------------------------------------------- UPLOAD TAB ---
with tab_upload:
    st.markdown("#### 📤 Upload a digit image")
    st.caption("Works best with a single digit, roughly centered, on a plain background.")
    uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", width=180)
        if st.button("🚀 Predict Digit", type="primary", key="predict_upload"):
            try:
                pipeline_images = preprocessing.preprocess_uploaded_image(uploaded_file)
                result = run_prediction_pipeline(pipeline_images)
                render_prediction(result, pipeline_images, source_label="Upload")
            except ValueError as e:
                st.warning(f"⚠️ {e}")
    else:
        st.info("Upload a PNG or JPG image containing a single handwritten digit.")

# -------------------------------------------------------------- RANDOM TAB ---
with tab_random:
    st.markdown("#### 🎲 Try a random MNIST test sample")
    st.caption("Pulls a real example straight from the MNIST test set.")

    if st.button("🎲 Get Random Sample", key="get_random"):
        sample_array, true_label = utils.get_random_mnist_sample()
        st.session_state["random_sample"] = sample_array
        st.session_state["random_label"] = true_label

    if "random_sample" in st.session_state:
        sample_image = preprocessing.array_to_pil(st.session_state["random_sample"])
        st.image(sample_image.resize((140, 140), resample=0), caption=f"True Label: {st.session_state['random_label']}")

        if st.button("🚀 Predict This Sample", type="primary", key="predict_random"):
            normalized = st.session_state["random_sample"].astype("float32") / 255.0
            model_input = normalized.reshape(1, 28 * 28)
            pipeline_images = {
                "original_image": sample_image,
                "processed_28x28": sample_image,
                "processed_display": sample_image.resize((140, 140), resample=0),
                "model_input": model_input,
            }
            result = run_prediction_pipeline(pipeline_images)
            render_prediction(result, pipeline_images, source_label="Random Sample")
    else:
        st.info("Click **Get Random Sample** to pull a digit from the MNIST test set.")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")

# ============================== PREDICTION HISTORY ============================
hist_col1, hist_col2 = st.columns([4, 1])
with hist_col1:
    st.markdown("### 🕘 Prediction History")
with hist_col2:
    if st.button("🗑️ Clear History", use_container_width=True):
        utils.clear_history()
        st.rerun()

if st.session_state.history:
    for entry in st.session_state.history:
        h1, h2, h3, h4 = st.columns([1, 1, 3, 2])
        with h1:
            st.image(entry["thumbnail"], width=40)
        with h2:
            st.markdown(f'<div class="history-digit">{entry["digit"]}</div>', unsafe_allow_html=True)
        with h3:
            st.caption(f'Source: {entry["source"]}')
        with h4:
            st.caption(f'{entry["confidence"]*100:.1f}% · {entry["timestamp"]}')
else:
    st.info("No predictions yet. Draw, upload, or sample a digit to get started.")

st.markdown("<br>", unsafe_allow_html=True)

# ============================== MODEL DETAILS =================================
with st.expander("🧬 Model Architecture & Training Details"):
    d1, d2 = st.columns(2)
    with d1:
        st.markdown(
            """
            **Dataset:** MNIST handwritten digits
            **Input Layer:** 784 neurons (flattened 28×28 image)
            **Hidden Layers:** Dense(256, ReLU) → Dropout(0.2) → Dense(128, ReLU) → Dropout(0.2)
            **Output Layer:** Dense(10, Softmax)
            **Loss Function:** Categorical Crossentropy
            **Optimizer:** Adam
            """
        )
    with d2:
        st.metric("Total Parameters", "~235K")
        st.metric("Training Accuracy", "99.1%")
        st.metric("Validation Accuracy", "98.0%")
        st.metric("Test Accuracy", "97.8%")
    st.caption("Update the values above to match your actual trained model's metrics.")

# ============================== HOW IT WORKS ===================================
with st.expander("⚙️ How It Works — Full Pipeline"):
    steps = ["✏️ Draw Digit", "🖤 Grayscale", "🔲 Resize 28×28", "📊 Normalize", "🧠 Deep Neural Network", "🎯 Prediction"]
    cols = st.columns(len(steps))
    for c, step in zip(cols, steps):
        c.markdown(f'<div class="pipeline-step">{step}</div>', unsafe_allow_html=True)

# ============================== FOOTER =========================================
st.markdown(
    f"""
    <div class="app-footer">
        Built with 🧠 TensorFlow, Streamlit &amp; a lot of gradient descent ·
        <a href="{GITHUB_URL}" target="_blank">GitHub</a> ·
        <a href="{LINKEDIN_URL}" target="_blank">LinkedIn</a>
    </div>
    """,
    unsafe_allow_html=True,
)
