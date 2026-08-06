"""
utils.py
--------
Supporting helpers that don't belong in preprocessing.py or predict.py:
session-state / history management, chart builders, CSS injection,
random MNIST sampling, and image-download helpers.
"""

import io
import base64
from datetime import datetime

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from PIL import Image

# ---------------------------------------------------------------------------
# Theme colors — kept in one place so charts match assets/style.css
# ---------------------------------------------------------------------------
ACCENT_BLUE = "#6366F1"
ACCENT_PURPLE = "#A855F7"
ACCENT_GRADIENT = ["#6366F1", "#8B5CF6", "#A855F7"]
BG_CARD = "#161B2E"
TEXT_MUTED = "#8B92B0"
TEXT_PRIMARY = "#E8EAF6"
SUCCESS_GREEN = "#22C55E"


# ---------------------------------------------------------------------------
# CSS / assets
# ---------------------------------------------------------------------------
def load_css(css_path: str = "assets/style.css"):
    """Inject the custom stylesheet into the Streamlit app."""
    try:
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("Custom stylesheet not found — using default Streamlit theme.")


# ---------------------------------------------------------------------------
# Session state / history
# ---------------------------------------------------------------------------
def init_session_state():
    """Initialize all session_state keys used across the app, if not already set."""
    defaults = {
        "history": [],
        "canvas_key": 0,
        "last_result": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def add_prediction_to_history(source: str, predicted_class: int, confidence: float, thumbnail: Image.Image):
    """Append a prediction record to the session history (most recent first)."""
    st.session_state.history.insert(0, {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "source": source,
        "digit": predicted_class,
        "confidence": confidence,
        "thumbnail": thumbnail,
    })
    # Keep history bounded so the app doesn't balloon in memory
    st.session_state.history = st.session_state.history[:20]


def clear_history():
    st.session_state.history = []


def reset_app():
    """Clear all session state and force the canvas to remount blank."""
    st.session_state.history = []
    st.session_state.last_result = None
    st.session_state.canvas_key += 1


# ---------------------------------------------------------------------------
# MNIST sample loader
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Fetching MNIST test set...")
def load_mnist_test_set():
    """Load (and cache) the MNIST test split for the 'random sample' feature."""
    from tensorflow.keras.datasets import mnist
    (_, _), (x_test, y_test) = mnist.load_data()
    return x_test, y_test


def get_random_mnist_sample():
    """Pick a random image from the MNIST test set. Returns (28x28 uint8 array, true_label)."""
    x_test, y_test = load_mnist_test_set()
    idx = np.random.randint(0, len(x_test))
    return x_test[idx], int(y_test[idx])


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def create_probability_bar_chart(probabilities: np.ndarray, predicted_class: int) -> go.Figure:
    """Animated-feeling horizontal bar chart of the softmax distribution over digits 0-9."""
    digits = list(range(10))
    colors = [
        f"rgba(168, 85, 247, 1)" if d == predicted_class else "rgba(99, 102, 241, 0.35)"
        for d in digits
    ]

    fig = go.Figure(
        go.Bar(
            x=probabilities * 100,
            y=[str(d) for d in digits],
            orientation="h",
            marker=dict(
                color=colors,
                line=dict(color="rgba(255,255,255,0.08)", width=1),
            ),
            text=[f"{p*100:.1f}%" for p in probabilities],
            textposition="outside",
            textfont=dict(color=TEXT_PRIMARY, size=12),
            hovertemplate="Digit %{y}: %{x:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis=dict(range=[0, 105], showgrid=False, visible=False),
        yaxis=dict(
            autorange="reversed",
            showgrid=False,
            tickfont=dict(color=TEXT_PRIMARY, size=14, family="Sora, sans-serif"),
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=40, t=10, b=10),
        height=360,
        bargap=0.35,
        transition={"duration": 500, "easing": "cubic-in-out"},
        font=dict(family="Inter, sans-serif"),
    )
    return fig


def create_confidence_gauge(confidence: float) -> go.Figure:
    """Radial gauge showing the model's confidence in its top prediction."""
    pct = confidence * 100
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix": "%", "font": {"size": 36, "color": TEXT_PRIMARY, "family": "Sora, sans-serif"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": TEXT_MUTED, "tickfont": {"color": TEXT_MUTED}},
                "bar": {"color": ACCENT_PURPLE, "thickness": 0.28},
                "bgcolor": "rgba(255,255,255,0.03)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 50], "color": "rgba(239, 68, 68, 0.15)"},
                    {"range": [50, 80], "color": "rgba(234, 179, 8, 0.15)"},
                    {"range": [80, 100], "color": "rgba(34, 197, 94, 0.15)"},
                ],
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=30, b=10),
        height=220,
        font=dict(family="Inter, sans-serif", color=TEXT_PRIMARY),
    )
    return fig


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------
def image_to_png_bytes(image: Image.Image) -> bytes:
    """Convert a PIL image to PNG bytes, suitable for st.download_button."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def image_to_base64(image: Image.Image) -> str:
    """Base64-encode a PIL image (used for embedding thumbnails in custom HTML)."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()
