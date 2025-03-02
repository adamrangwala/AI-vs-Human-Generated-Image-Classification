import streamlit as st
import tensorflow as tf
import io
import json
import numpy as np
import os
from tensorflow.keras.models import load_model
import tempfile
from custom_objects import CustomDataAugmentation, ResNetPreprocessingLayer
from PIL import Image
from streamlit_extras.let_it_rain import rain

# Configure page
st.set_page_config(page_title="Keras Model Loader", layout="wide")

# Streamlit header
st.title("Keras Model Loader")
st.markdown("Upload a .keras or .h5 model file to view its architecture and weights.")

# File uploader
uploaded_file = st.file_uploader("Choose a model file", type=["keras", "h5"])

@st.cache_resource
def load_keras_model(model_path):
    """Loads and caches the Keras model to prevent reloading on each interaction."""
    custom_objects = {
        "CustomDataAugmentation": CustomDataAugmentation,
        "ResNetPreprocessingLayer": ResNetPreprocessingLayer
    }
    return load_model(model_path, custom_objects=custom_objects)

@st.cache_data
def preprocess_image(image, target_size=(640, 640)):
    """Caches image preprocessing to avoid redundant computation."""
    image = image.resize(target_size)
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@st.cache_data
def get_model_weights_info(model):
    """Caches model weight information to avoid redundant computations."""
    weights_info = []
    for i, layer in enumerate(model.layers):
        layer_weights = layer.get_weights()
        if layer_weights:
            for j, w in enumerate(layer_weights):
                weights_info.append({
                    "layer_name": layer.name,
                    "layer_index": i,
                    "weight_index": j,
                    "shape": w.shape,
                    "dtype": str(w.dtype),
                    "min": float(np.min(w)),
                    "max": float(np.max(w)),
                    "mean": float(np.mean(w)),
                    "std": float(np.std(w))
                })
    return weights_info

@st.cache_data
def serialize_model_architecture(model):
    """Caches model architecture serialization."""
    return json.dumps(model.get_config(), indent=2)

@st.cache_data
def predict_image(model, img_array):
    """Caches model predictions to prevent recomputing when the same image is used."""
    return model.predict(img_array)[0][0]

# Process uploaded model
if uploaded_file is not None:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_filename = tmp_file.name
        
        with st.spinner("Loading model..."):
            model = load_keras_model(tmp_filename)

        os.unlink(tmp_filename)
        st.success(f"Model loaded successfully from {uploaded_file.name}")

        # Tabs for different views
        tabs = st.tabs(["Prediction", "Summary", "Architecture", "Weights"])

        # Image upload for prediction
        with tabs[0]:
            pred_col1, pred_col2 = st.columns(2)
            uploaded_img = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            if uploaded_img is not None:
                with pred_col1:
                    image = Image.open(uploaded_img).convert("RGB")
                    st.image(image, caption="Uploaded Image", width=400)

                with st.spinner("Analyzing image..."):
                    image_array = preprocess_image(image)
                    probability = predict_image(model, image_array)

                    confidence = round(probability * 100, 2) if probability > 0.5 else round((1 - probability) * 100, 2)
                    label = "AI-Generated" if probability > 0.5 else "Human-Generated"
                    color = "red" if probability > 0.5 else "green"
                    emoji = "🤖" if probability > 0.5 else "🧑"
                    
                    with pred_col2:
                        st.markdown(f"<h1 style='text-align: center; color: {color};'>{label} Image</h1>", unsafe_allow_html=True)
                        st.markdown(f"<h2 style='text-align: center; color: black;'>{confidence}% confidence</h2>", unsafe_allow_html=True)
                        rain(emoji=emoji, font_size=80, falling_speed=5, animation_length=5)
        # Summary tab
        with tabs[1]:
            st.subheader("Model Summary")
            summary_io = io.StringIO()
            model.summary(print_fn=lambda x: summary_io.write(x + '\n'))
            st.code(summary_io.getvalue(), language="text")

        # Architecture tab
        with tabs[2]:
            st.subheader("Model Architecture")
            architecture_json = serialize_model_architecture(model)
            st.code(architecture_json, language="json")
            st.download_button("Download Architecture JSON", data=architecture_json, file_name="model_architecture.json", mime="application/json")

        # Weights tab
        with tabs[3]:
            st.subheader("Model Weights Information")
            weights_info = get_model_weights_info(model)
            st.dataframe(weights_info)
            st.download_button("Download Weights JSON", data=json.dumps(weights_info, indent=2), file_name="weights_info.json", mime="application/json")

     

    except Exception as e:
        st.error(f"Error loading the model: {str(e)}")
