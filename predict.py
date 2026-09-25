"""
predict.py
----------
Handles everything related to the trained model itself: loading it
(cached so it only happens once per session) and running inference.
Kept separate from preprocessing.py and app.py for clean modularity.
"""

import time
import numpy as np
import streamlit as st
import tensorflow as tf

MODEL_PATH = "saved_model.keras"

# This model's output layer uses a LINEAR activation (raw logits), and was
# trained with SparseCategoricalCrossentropy(from_logits=True) — it was never
# meant to output probabilities directly. So we apply softmax ourselves,
# exactly mirroring the numerically-stable implementation from the training
# notebook (subtract max, exponentiate, normalize) rather than assuming
# model.predict() already returns a probability distribution.


@st.cache_resource(show_spinner="Loading model...")
def load_trained_model(model_path: str = MODEL_PATH):
    """
    Load the trained Keras model from disk. Cached via st.cache_resource
    so the (potentially expensive) load only happens once per server
    process, not on every rerun/interaction.

    Args:
        model_path: path to the .keras model file.

    Returns:
        A loaded tf.keras.Model, or None if loading failed.
    """
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as exc:
        st.error(
            f"⚠️ Could not load model from `{model_path}`. "
            f"Make sure the trained model file exists in the project root.\n\nDetails: {exc}"
        )
        return None


def softmax_from_logits(logits: np.ndarray) -> np.ndarray:
    """
    Numerically-stable softmax, matching the custom softmax function from the
    training notebook: subtract the max logit before exponentiating (for
    stability), then normalize by the sum. Vectorized here for speed, but
    mathematically identical to the notebook's loop-based version.

    Args:
        logits: raw model output, shape (10,) or (1, 10).

    Returns:
        np.ndarray of shape (10,), a valid probability distribution summing to 1.
    """
    logits = np.asarray(logits).reshape(-1)  # flatten to (10,)
    shifted = logits - np.max(logits)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)


def predict_digit(model, model_input: np.ndarray):
    """
    Run inference on a preprocessed (1, 784) input array.

    The model's output layer is linear (raw logits, not probabilities) —
    softmax is applied afterward via softmax_from_logits(), matching how
    the model was trained (SparseCategoricalCrossentropy(from_logits=True)).

    Args:
        model: loaded tf.keras.Model
        model_input: normalized, flattened input of shape (1, 784)

    Returns:
        dict with keys:
            predicted_class : int, 0-9
            confidence      : float, 0-1, probability of the predicted class
            probabilities   : np.ndarray shape (10,), full softmax distribution
            inference_ms    : float, wall-clock inference time in milliseconds
    """
    if model is None:
        raise RuntimeError("Model is not loaded. Cannot run prediction.")

    if model_input.shape != (1, 784):
        raise ValueError(f"Expected input shape (1, 784), got {model_input.shape}")

    start = time.perf_counter()
    logits = model.predict(model_input, verbose=0)
    probabilities = softmax_from_logits(logits)
    elapsed_ms = (time.perf_counter() - start) * 1000

    predicted_class = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_class])

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": probabilities,
        "inference_ms": elapsed_ms,
    }


def get_top_k_predictions(probabilities: np.ndarray, k: int = 3):
    """Return the top-k (digit, probability) pairs, sorted by descending probability."""
    top_indices = np.argsort(probabilities)[::-1][:k]
    return [(int(i), float(probabilities[i])) for i in top_indices]
