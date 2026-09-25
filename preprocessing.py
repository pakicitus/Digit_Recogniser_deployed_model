"""
preprocessing.py
-----------------
All image preprocessing logic lives here, kept deliberately separate from
prediction/inference (predict.py) and UI code (app.py).

Pipeline: raw drawing / uploaded image  ->  grayscale  ->  28x28  ->
normalized float32 array  ->  flattened vector ready for the Dense network.
"""

import numpy as np
from PIL import Image, ImageOps

# MNIST images are 28x28, pixel values scaled to [0, 1]
TARGET_SIZE = (28, 28)


def canvas_to_pil_image(canvas_image_data: np.ndarray) -> Image.Image:
    """
    Convert the RGBA numpy array returned by streamlit-drawable-canvas
    into a PIL Image.

    Args:
        canvas_image_data: RGBA array from st_canvas(...).image_data

    Returns:
        PIL.Image in RGBA mode.
    """
    if canvas_image_data is None:
        raise ValueError("No canvas data received. Please draw a digit first.")

    # canvas_image_data comes as float/uint8 RGBA array
    image_array = canvas_image_data.astype("uint8")
    return Image.fromarray(image_array, mode="RGBA")


def is_canvas_empty(canvas_image_data: np.ndarray, threshold: int = 10) -> bool:
    """
    Detects whether the user has actually drawn anything on the canvas.
    A blank canvas (all background pixels) has ~0 non-transparent / non-black
    pixels in the alpha or RGB channel.

    Args:
        canvas_image_data: RGBA array from the canvas widget.
        threshold: minimum number of "active" pixels required to count as a drawing.

    Returns:
        True if the canvas appears empty, False otherwise.
    """
    if canvas_image_data is None:
        return True

    rgb = canvas_image_data[:, :, :3]
    # Background is pure black (0,0,0). Any pixel brighter than a small
    # epsilon counts as "drawn".
    active_pixels = np.sum(np.any(rgb > 20, axis=-1))
    return active_pixels < threshold


def to_grayscale(image: Image.Image) -> Image.Image:
    """Flatten an RGBA/RGB image down to single-channel grayscale ('L')."""
    # Composite onto a black background first so transparency doesn't
    # get treated as white during the grayscale conversion.
    if image.mode == "RGBA":
        background = Image.new("RGBA", image.size, (0, 0, 0, 255))
        image = Image.alpha_composite(background, image)
    return image.convert("L")


def resize_to_28x28(image: Image.Image) -> Image.Image:
    """Resize (with high-quality antialiasing) to the model's expected 28x28 input."""
    return image.resize(TARGET_SIZE, Image.LANCZOS)


def normalize_image(image: Image.Image) -> np.ndarray:
    """
    Convert a grayscale PIL image into a normalized float32 numpy array
    with pixel values in [0, 1], matching MNIST training preprocessing.
    """
    array = np.array(image).astype("float32") / 255.0
    return array


def flatten_for_model(normalized_array: np.ndarray) -> np.ndarray:
    """
    Flatten the 28x28 normalized array into the (1, 784) shape expected
    by a Dense/Feed-forward network input layer, with a batch dimension.
    """
    return normalized_array.reshape(1, 28 * 28)


def preprocess_canvas_drawing(canvas_image_data: np.ndarray):
    """
    Full pipeline for a canvas drawing: raw RGBA -> grayscale -> 28x28 ->
    normalized -> model-ready tensor. Also returns intermediate images so
    the UI can render the "how preprocessing works" preview.

    Returns:
        dict with keys:
            original_image   : PIL.Image (grayscale, full resolution)
            processed_image   : PIL.Image (28x28, for display, upscaled for visibility)
            processed_28x28   : PIL.Image (raw 28x28, unscaled)
            model_input       : np.ndarray shape (1, 784), ready for model.predict
    """
    if is_canvas_empty(canvas_image_data):
        raise ValueError("Canvas is empty. Please draw a digit before predicting.")

    pil_image = canvas_to_pil_image(canvas_image_data)
    grayscale = to_grayscale(pil_image)
    resized = resize_to_28x28(grayscale)
    normalized = normalize_image(resized)
    model_input = flatten_for_model(normalized)

    return {
        "original_image": grayscale,
        "processed_28x28": resized,
        "processed_display": resized.resize((140, 140), Image.NEAREST),
        "model_input": model_input,
    }


def preprocess_uploaded_image(uploaded_file) -> dict:
    """
    Full pipeline for a user-uploaded image file. Handles arbitrary image
    modes/sizes and automatically inverts colors if the uploaded image looks
    like black-ink-on-white-paper (MNIST digits are white-on-black).

    Args:
        uploaded_file: file-like object from st.file_uploader

    Returns:
        Same dict shape as preprocess_canvas_drawing().
    """
    try:
        image = Image.open(uploaded_file)
    except Exception as exc:
        raise ValueError(f"Could not read the uploaded file as an image: {exc}")

    grayscale = to_grayscale(image.convert("RGBA"))
    resized = resize_to_28x28(grayscale)

    # Auto-invert: if the average pixel is bright (white-ish background),
    # assume dark ink on light paper and flip it to match MNIST's
    # white-digit-on-black-background convention.
    array_check = np.array(resized)
    if array_check.mean() > 127:
        resized = ImageOps.invert(resized)

    normalized = normalize_image(resized)
    model_input = flatten_for_model(normalized)

    return {
        "original_image": grayscale,
        "processed_28x28": resized,
        "processed_display": resized.resize((140, 140), Image.NEAREST),
        "model_input": model_input,
    }


def array_to_pil(array_28x28: np.ndarray) -> Image.Image:
    """Convert a raw 28x28 (0-1 or 0-255) numpy array (e.g. an MNIST sample) to a PIL image."""
    arr = array_28x28
    if arr.max() <= 1.0:
        arr = (arr * 255).astype("uint8")
    else:
        arr = arr.astype("uint8")
    return Image.fromarray(arr, mode="L")
