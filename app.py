import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
MODEL_PATH = "pneumonia_detector_mobilenetv2.keras"
#MODEL_PATH = "pneumonia_detector_vgg16.keras"  # Updated model path
IMG_SIZE = (224, 224)
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

st.set_page_config(
    page_title="Pneumonia X-Ray Classifier",
    page_icon="🫁",
    layout="centered"
)

# --------------------------------------------------------------------------
# Model loading (cached so it only loads once per session)
# --------------------------------------------------------------------------
@st.cache_resource
def load_model(path):
    try:
        return keras.models.load_model(path)
    except Exception as e:
        return None

model = load_model(MODEL_PATH)

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("About")
    st.write(
        "This app uses a transfer-learning CNN (MobileNetV2 backbone) trained on the "
        "[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) "
        "dataset to classify chest X-rays as **NORMAL** or **PNEUMONIA**."
    )
    st.warning(
        "⚠️ Research / educational tool only — not a certified diagnostic device. "
        "Do not use for real clinical decisions."
    )
    threshold = st.slider(
        "Decision threshold (PNEUMONIA if prediction ≥ this)",
        min_value=0.05, max_value=0.95, value=0.50, step=0.05,
        help="Lower the threshold to prioritize catching more pneumonia cases (higher recall) "
             "at the cost of more false alarms."
    )

# --------------------------------------------------------------------------
# Main UI
# --------------------------------------------------------------------------
st.title("🫁 Pneumonia Detection from Chest X-Rays")
st.write("Upload a chest X-ray image (JPEG/PNG) to get a prediction from the trained model.")

if model is None:
    st.error(
        f"Could not find or load a model at `{MODEL_PATH}`.\n\n"
        "Train and save a model first using `pneumonia_detection_transfer_learning.ipynb` "
        "(it saves `pneumonia_detector_mobilenetv2.keras`), then place that file in the "
        "same folder as this app — or update `MODEL_PATH` at the top of this script."
    )
    st.stop()

uploaded_file = st.file_uploader("Choose a chest X-ray image", type=["jpg", "jpeg", "png"])

col_upload_demo = st.container()

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Uploaded X-ray", use_container_width=True)

    # Preprocess
    resized = image.resize(IMG_SIZE)
    arr = keras.utils.img_to_array(resized) / 255.0
    arr = np.expand_dims(arr, axis=0)

    # Predict
    with st.spinner("Running inference..."):
        prob = float(model.predict(arr, verbose=0)[0][0])

    label = CLASS_NAMES[1] if prob >= threshold else CLASS_NAMES[0]
    confidence = prob if prob >= threshold else 1 - prob

    with col2:
        st.subheader("Prediction")
        if label == "PNEUMONIA":
            st.error(f"**{label}**")
        else:
            st.success(f"**{label}**")

        st.metric("Confidence", f"{confidence:.1%}")
        st.progress(prob)
        st.caption(f"Raw model output (P(PNEUMONIA)) = {prob:.4f} | threshold = {threshold:.2f}")

    st.divider()
    st.subheader("Interpretation")
    if label == "PNEUMONIA":
        st.write(
            "The model flagged patterns in this X-ray consistent with pneumonia in its training data "
            "(e.g. lung opacities). This is **not a diagnosis** — please consult a radiologist or physician."
        )
    else:
        st.write(
            "The model did not detect strong pneumonia-consistent patterns. This does **not rule out** "
            "illness — always confirm with a qualified clinician."
        )

else:
    st.info("👆 Upload an image to get started.")
    st.write(
        "Don't have a sample image handy? You can grab a few test images from the "
        "[Kaggle dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) "
        "(`chest_xray/test/NORMAL` and `chest_xray/test/PNEUMONIA` folders)."
    )

st.divider()
st.caption(
    "Model: MobileNetV2 transfer learning · Dataset: Kermany et al. / Paul Mooney (Kaggle) · "
    "Built for educational purposes."
)
