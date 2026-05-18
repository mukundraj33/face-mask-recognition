import streamlit as st
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Face Mask Detector",
    page_icon="😷",
    layout="centered"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown(
    """
    <style>
    .main {
        background-color: #0E1117;
    }

    .title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        color: white;
        margin-bottom: 10px;
    }

    .subtitle {
        text-align: center;
        color: #B0B3B8;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .result-box {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# =========================
class_names = ["With Mask 😷", "Without Mask ❌"]

# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

# =========================
# PREDICTION
# =========================
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Resize according to training size
    resized_image = image.resize((128, 128))

    # Convert image to array
    image_array = np.array(resized_image)

    # Normalize
    image_array = image_array / 255.0

    # Expand dimensions
    image_array = np.expand_dims(image_array, axis=0)

    # Prediction
    prediction = model.predict(image_array)

    predicted_class = np.argmax(prediction)

    confidence = float(np.max(prediction)) * 100

   # =========================
    # DISPLAY RESULT
    # =========================
    if predicted_class == 0:
        st.success(
            f"Prediction: {class_names[predicted_class]}\n\nConfidence: {confidence:.2f}%"
        )
    else:
        st.error(
            f"Prediction: {class_names[predicted_class]}\n\nConfidence: {confidence:.2f}%"
        )

    # Probability bars
    st.subheader("Prediction Probabilities")

    st.progress(int(prediction[0][0] * 100), text=f"With Mask 😷 : {prediction[0][0] * 100:.2f}%")

    st.progress(int(prediction[0][1] * 100), text=f"Without Mask ❌ : {prediction[0][1] * 100:.2f}%")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown(
    "<center>Built using TensorFlow, Streamlit, and CNN</center>",
    unsafe_allow_html=True
)