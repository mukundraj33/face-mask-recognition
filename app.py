from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from tensorflow import keras


APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "model" / "model.h5"
IMAGE_SIZE = (128, 128)
CLASS_NAMES = {
    0: "Without Mask",
    1: "With Mask",
}
CLASS_BADGES = {
    0: "Mask not detected",
    1: "Mask detected",
}


st.set_page_config(
    page_title="Face Mask Detection",
    page_icon=":mask:",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
        :root {
            --bg: #f6f8fb;
            --ink: #16202a;
            --muted: #607080;
            --line: #dce4ec;
            --panel: #ffffff;
            --teal: #0f766e;
            --teal-soft: #d9f5f1;
            --red: #c2410c;
            --red-soft: #ffedd5;
        }

        .stApp {
            background:
                radial-gradient(circle at 12% 18%, rgba(15, 118, 110, 0.09), transparent 26rem),
                radial-gradient(circle at 88% 4%, rgba(14, 165, 233, 0.10), transparent 24rem),
                var(--bg);
            color: var(--ink);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2.5rem;
            max-width: 1180px;
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        .hero {
            padding: 2rem 0 1rem;
            border-bottom: 1px solid rgba(22, 32, 42, 0.08);
            margin-bottom: 1.5rem;
        }

        .eyebrow {
            color: var(--teal);
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }

        .hero h1 {
            color: var(--ink);
            font-size: clamp(2.25rem, 5vw, 4.65rem);
            line-height: 0.96;
            letter-spacing: 0;
            margin: 0 0 0.85rem;
            max-width: 760px;
        }

        .hero p {
            color: var(--muted);
            font-size: 1.05rem;
            line-height: 1.65;
            max-width: 690px;
            margin: 0;
        }

        .metric-strip {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
            margin-top: 1.35rem;
            max-width: 720px;
        }

        .metric {
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid rgba(22, 32, 42, 0.08);
            border-radius: 8px;
            padding: 0.85rem 1rem;
        }

        .metric strong {
            display: block;
            color: var(--ink);
            font-size: 1.25rem;
            line-height: 1.1;
        }

        .metric span {
            color: var(--muted);
            font-size: 0.82rem;
        }

        .result-panel {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1.25rem;
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
        }

        .result-title {
            color: var(--muted);
            font-size: 0.82rem;
            font-weight: 800;
            text-transform: uppercase;
            margin-bottom: 0.45rem;
        }

        .prediction {
            font-size: clamp(1.7rem, 4vw, 2.7rem);
            font-weight: 850;
            line-height: 1.05;
            color: var(--ink);
            margin-bottom: 0.6rem;
        }

        .badge {
            display: inline-block;
            border-radius: 999px;
            padding: 0.35rem 0.75rem;
            font-size: 0.84rem;
            font-weight: 800;
            margin-bottom: 0.9rem;
        }

        .badge.ok {
            background: var(--teal-soft);
            color: var(--teal);
        }

        .badge.warn {
            background: var(--red-soft);
            color: var(--red);
        }

        .confidence-row {
            display: flex;
            justify-content: space-between;
            color: var(--muted);
            font-size: 0.92rem;
            margin-top: 0.65rem;
        }

        .hint {
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.55;
            margin-top: 0.85rem;
        }

        .stProgress > div > div > div > div {
            background-color: var(--teal);
        }

        div[data-testid="stFileUploader"] section {
            border: 1px dashed #9fb0bf;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.72);
        }

        div[data-testid="stFileUploader"] button {
            border-radius: 8px;
        }

        @media (max-width: 760px) {
            .metric-strip {
                grid-template-columns: 1fr;
            }

            .hero {
                padding-top: 1rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_mask_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    return keras.models.load_model(MODEL_PATH, compile=False)


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB").resize(IMAGE_SIZE)
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)


def predict_mask(image: Image.Image):
    model = load_mask_model()
    predictions = model.predict(preprocess_image(image), verbose=0)[0]
    predicted_index = int(np.argmax(predictions))
    confidence = float(predictions[predicted_index])
    probabilities = {
        CLASS_NAMES[index]: float(score)
        for index, score in enumerate(predictions)
        if index in CLASS_NAMES
    }
    return predicted_index, confidence, probabilities


def render_result(predicted_index: int, confidence: float, probabilities: dict[str, float]):
    badge_class = "ok" if predicted_index == 1 else "warn"
    st.markdown(
        f"""
        <div class="result-panel">
            <div class="result-title">Prediction</div>
            <div class="prediction">{CLASS_NAMES[predicted_index]}</div>
            <div class="badge {badge_class}">{CLASS_BADGES[predicted_index]}</div>
            <div class="confidence-row">
                <span>Model confidence</span>
                <strong>{confidence * 100:.2f}%</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.write("Class probabilities")
    for label, score in probabilities.items():
        st.progress(score, text=f"{label}: {score * 100:.2f}%")


def sample_image_paths():
    candidates = [
        APP_DIR / "images" / "sample1.jpg",
        APP_DIR / "images" / "sample2.jpg",
        APP_DIR / "sample_images" / "with_mask" / "with_mask_970.jpg",
        APP_DIR / "sample_images" / "without_mask" / "without_mask_834.jpg",
    ]
    return [path for path in candidates if path.exists()]


st.markdown(
    """
    <section class="hero">
        <div class="eyebrow">CNN image classifier</div>
        <h1>Face Mask Detection</h1>
        <p>
            Upload a clear face photo and the trained CNN model will classify it as
            with mask or without mask using the same preprocessing from the Colab notebook.
        </p>
        <div class="metric-strip">
            <div class="metric"><strong>128x128</strong><span>RGB model input</span></div>
            <div class="metric"><strong>92.79%</strong><span>reported test accuracy</span></div>
            <div class="metric"><strong>2 classes</strong><span>mask / no mask</span></div>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)


left_col, right_col = st.columns([1.05, 0.95], gap="large")

with left_col:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a JPG, JPEG, or PNG image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    selected_sample = None
    samples = sample_image_paths()
    if samples:
        sample_names = ["Use uploaded image"] + [path.name for path in samples]
        sample_choice = st.selectbox(
            "Or try a sample image",
            sample_names,
            index=0,
        )
        if sample_choice != "Use uploaded image":
            selected_sample = next(path for path in samples if path.name == sample_choice)

    image = None
    source_name = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        source_name = uploaded_file.name
    elif selected_sample is not None:
        image = Image.open(selected_sample)
        source_name = selected_sample.name

    if image is None:
        st.info("Upload an image or choose a sample to run the classifier.")
        st.markdown(
            '<p class="hint">For best results, use a front-facing image where the face is visible and not heavily cropped.</p>',
            unsafe_allow_html=True,
        )
    else:
        st.image(image, caption=source_name, use_container_width=True)

with right_col:
    st.subheader("Result")
    if image is None:
        st.markdown(
            """
            <div class="result-panel">
                <div class="result-title">Waiting for image</div>
                <div class="prediction">No prediction yet</div>
                <p class="hint">
                    The app will resize the image to 128x128, normalize pixel values,
                    and send it to your trained Keras model.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        try:
            with st.spinner("Classifying image..."):
                predicted_index, confidence, probabilities = predict_mask(image)
            render_result(predicted_index, confidence, probabilities)
        except Exception as exc:
            st.error("Could not run prediction.")
            st.exception(exc)


st.caption(
    "Model labels: 1 = With Mask, 0 = Without Mask. This demo is for portfolio use and should not be used as a medical or safety system."
)
