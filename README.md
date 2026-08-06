# 🧠 Handwritten Digit Recognition

> Draw a digit and let a Deep Neural Network recognize it instantly.

A polished, portfolio-ready Streamlit web application that deploys a **Deep Neural Network trained on the MNIST dataset** for real-time handwritten digit recognition — draw, upload, or sample a digit and watch the model classify it with full transparency into confidence, probabilities, and the preprocessing pipeline.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-informational)

---

## 📖 Project Overview

This project takes a Deep Neural Network built while learning deep learning fundamentals and turns it into a real, deployable AI product. Rather than a static notebook, it's a fully interactive web app with:

- A hand-drawn canvas that feeds directly into the model
- A clear, visual explanation of every preprocessing step
- Rich, animated visualizations of the model's confidence and reasoning

The model itself is a **fully-connected (Dense) Deep Neural Network** — no convolutional layers — trained on the classic **MNIST** dataset of 70,000 handwritten digit images.

---

## ✨ Features

- 🖌️ **Interactive drawing canvas** (via `streamlit-drawable-canvas`) with adjustable brush width and color
- 📤 **Upload an image** of a digit instead of drawing one
- 🎲 **Random MNIST sample** picker to test the model against real dataset examples
- 🎯 **Real-time prediction** with predicted digit, confidence score, and inference time (ms)
- 📊 **Animated probability chart** across all 10 digit classes, with the predicted digit highlighted
- 🌡️ **Confidence gauge** for an at-a-glance read on prediction certainty
- 🔍 **Preprocessing pipeline preview** — see the original drawing, the processed 28×28 image, and the final prediction side by side
- 🕘 **Prediction history** with thumbnails, timestamps, and a one-click clear
- ⬇️ **Download** the processed 28×28 image as a PNG
- 🧬 **Expandable model details** — architecture, parameters, and accuracy metrics
- ⚙️ **"How it works" walkthrough** of the full inference pipeline
- 🎨 **Modern, dark, AI-themed UI** with a blue → purple gradient identity, glass-style cards, and smooth animations
- 🔄 **Reset button** to clear the canvas, history, and app state in one click

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| UI / App Framework | Streamlit, `streamlit-drawable-canvas` |
| Deep Learning | TensorFlow / Keras |
| Data Handling | NumPy, Pandas |
| Image Processing | Pillow (PIL) |
| Visualization | Plotly, Matplotlib |

---

## 📁 Folder Structure

```
DigitRecognizer/
├── app.py                 # Streamlit UI — layout, tabs, sidebar, results rendering
├── preprocessing.py        # Image preprocessing pipeline (grayscale, resize, normalize)
├── predict.py               # Model loading + inference logic
├── utils.py                 # Session state, charts, history, CSS loading, MNIST sampling
├── train_model.py           # Optional script to train & export saved_model.keras
├── saved_model.keras        # Trained model (generate via train_model.py, or bring your own)
├── requirements.txt
├── assets/
│   └── style.css             # Custom dark / gradient theme
└── README.md
```

The app deliberately separates **preprocessing** (`preprocessing.py`) from **inference** (`predict.py`) so each piece stays testable and reusable on its own — e.g. you could swap in a CNN model without touching the preprocessing code.

---

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/DigitRecognizer.git
   cd DigitRecognizer
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Provide a trained model.** If you don't already have `saved_model.keras`, generate one:
   ```bash
   python train_model.py
   ```
   This trains the Dense network on MNIST and saves it to the project root. Training takes a few minutes on CPU.

---

## ▶️ How to Run

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (typically `http://localhost:8501`) in your browser.

---

## 🖼️ Screenshots

> _Add screenshots or a short GIF of the app here before publishing to GitHub/LinkedIn._

| Home | Prediction | Model Details |
|---|---|---|
| `assets/screenshot-home.png` | `assets/screenshot-prediction.png` | `assets/screenshot-model.png` |

---

## 🧬 Model Architecture

```
Input(784) → Dense(256, ReLU) → Dropout(0.2)
           → Dense(128, ReLU) → Dropout(0.2)
           → Dense(10, Softmax)
```

- **Loss:** Categorical Crossentropy
- **Optimizer:** Adam
- **Dataset:** MNIST (60,000 training / 10,000 test images)

Update `app.py`'s "Model Architecture & Training Details" section with your model's actual final metrics after training.

---

## 🚀 Future Improvements

- [ ] Swap the Dense network for a Convolutional Neural Network (CNN) for higher accuracy
- [ ] Add model comparison mode (Dense vs. CNN side-by-side)
- [ ] Deploy to Streamlit Community Cloud / Hugging Face Spaces with a live demo link
- [ ] Add batch prediction from a folder/zip of images
- [ ] Persist prediction history to a lightweight database instead of session state
- [ ] Add adversarial/robustness testing (noise, rotation, occlusion)
- [ ] Multi-language UI support

---

## 📄 License

This project is open-sourced under the MIT License — feel free to fork, adapt, and build on it.

---

## 🙌 Acknowledgements

Built as part of a Deep Learning learning journey, using the MNIST dataset (LeCun et al.).
