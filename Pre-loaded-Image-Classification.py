import streamlit as st
import tensorflow as tf
import keras
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import os
from PIL import Image
import psutil

# Configure page
st.set_page_config(page_title="AI Image Identification", layout="wide")

# Memory monitoring function
def get_memory_usage():
    """Returns the current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

# Streamlit header
st.title("AI-Generated Image Identification")
st.markdown("Upload an image to check if it's AI-generated or human-generated.")

# Display memory usage if requested
if st.checkbox("Show memory usage", value=False):
    st.info(f"Current memory usage: {get_memory_usage():.2f} MB")

# Model path
MODEL_PATH = "model.keras"
st.write("File exists:", os.path.exists(MODEL_PATH))  
st.write("Keras version:", keras.__version__)

@st.cache_resource(show_spinner=False)
def load_model(path):
    """Loads the model"""
    try:
        with st.status("Loading model..."):
            # Configure TensorFlow for memory efficiency if GPU is available
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                tf.config.experimental.set_memory_growth(gpus[0], True)
            
            # Load model with CPU for stability
            with tf.device('/CPU:0'):
                model = keras.models.load_model(path)
                return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


def preprocess_image(image, target_size=(224, 224)):
    """Preprocesses image for ResNet50 model input"""
    try:
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Resize image to match ResNet50 input size
        image = image.resize(target_size, Image.LANCZOS)
        
        # Convert to array
        img_array = np.array(image, dtype=np.float32)
        
        # Preprocess image for ResNet50
        img_array = preprocess_input(img_array)  # This step is crucial
        
        # Add batch dimension
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

# Load the model
with st.spinner("Loading AI detection model..."):
    model = load_model(MODEL_PATH)
    
    if model is None:
        st.error("Failed to load the model. Please check that the model file exists.")
        st.stop()
    else:
        st.success("Model loaded successfully!")

# Try to determine input shape
try:
    input_shape = model.input_shape[1:3]
    if None in input_shape:
        input_shape = (224, 224)
except:
    input_shape = (224, 224)  # Default size

# Display model info
with st.expander("Model Information"):
    st.write(f"Model input shape: {input_shape}")

# Main interface
col1, col2 = st.columns([1, 1])

with col1:
    ai_col, human_col = st.columns([1, 1])
    with ai_col:
        if st.button("Sample AI Image"):
            uploaded_img="/ai_sample.jpg"
    with human_col:
        if st.button("Sample Human Image"):
            uploaded_img="/human_sample.jpg"
    st.subheader("Upload an image to analyze")
    uploaded_img = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_img is not None:
        try:
            # Check file size (max 10MB)
            if uploaded_img.size > 10 * 1024 * 1024:
                st.error("File is too large! Maximum size is 10MB.")
            else:
                # Display image
                image = Image.open(uploaded_img)
                st.image(image, caption="Uploaded Image", width=400)
                
                # Show image details
                width, height = image.size
                st.caption(f"Image dimensions: {width}x{height} pixels, {image.mode} mode")
                
                with st.spinner("Analyzing image..."):
                    st.caption(f"Resizing to {input_shape} for model input")
                    
                    # Preprocess and predict
                    image_array = preprocess_image(image, target_size=input_shape)
                    
                    if image_array is not None:
                        probability = float(model.predict(image_array))
                        st.write(probability)
                        if probability is not None:
                            with col2:
                                confidence = round(probability * 100, 2) if probability > 0.5 else round((1 - probability) * 100, 2)
                                label = "AI-Generated" if probability > 0.5 else "Human-Generated"
                                color = "red" if probability > 0.5 else "green"
                                emoji = "🤖" if probability > 0.5 else "🧑"

                                st.markdown("")
                                st.markdown("")
                                st.markdown("")
                                st.markdown(f"<h1 style='text-align: center; color: {color};'>{emoji} {label}</h1>", unsafe_allow_html=True)
                                st.markdown(f"<h2 style='text-align: center; color: black;'>{confidence}% confidence</h2>", unsafe_allow_html=True)
                                
                                # Show confidence meter
                                if probability > 0.5:
                                    ai_confidence = probability * 100
                                    st.progress(ai_confidence / 100)
                                    st.caption(f"AI confidence: {ai_confidence:.3f}%")
                                else:
                                    human_confidence = (1 - probability) * 100
                                    st.progress(human_confidence / 100)
                                    st.caption(f"Human confidence: {human_confidence:.3f}%")
                        else:
                            st.error("Failed to generate prediction.")
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

st.markdown("""
### How It Works

This tool uses a deep learning model to analyze images and determine if they were created by AI or humans.

1. Upload any JPEG or PNG image
2. The model will analyze the image
3. You'll receive a prediction with confidence score

**Note:** While this model is designed to detect AI-generated images, no detection system is perfect. Results should be considered as probabilities rather than definitive answers.
""")

# Sidebar information
st.sidebar.title("About")
st.sidebar.info("""
### AI Image Detection Tool

This application uses a neural network model trained to distinguish between AI-generated and human-created images.

The model analyzes visual patterns, inconsistencies, and artifacts that may indicate AI generation.

**Privacy Note:** Uploaded images are processed locally and are not stored.
""")
