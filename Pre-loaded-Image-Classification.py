import streamlit as st
import tensorflow as tf
import numpy as np
import os
import io
from PIL import Image
import traceback
import psutil

# Configure page
st.set_page_config(page_title="AI Image Identification", layout="wide")

# Memory monitoring function
def get_memory_usage():
    """Returns the current memory usage in MB"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)

# Streamlit header
st.title("AI-Generated Image Identification")
st.markdown("Upload an image to check if it's AI-generated or human-generated.")

# Display current memory usage if in debug mode
if st.checkbox("Show memory usage", value=False):
    memory_usage = get_memory_usage()
    memory_container = st.empty()
    memory_container.info(f"Current memory usage: {memory_usage:.2f} MB")

# Model path - assuming model.keras is in the same directory as the app
MODEL_PATH = "model.keras"

@st.cache_resource(show_spinner=False)
def load_model_from_path(path):
    """Loads the model from local path"""
    try:
        with st.status("Loading model..."):
            # Configure TensorFlow for memory efficiency
            tf.config.experimental.set_memory_growth(
                tf.config.experimental.list_physical_devices('GPU')[0], True
            ) if tf.config.experimental.list_physical_devices('GPU') else None
            
            # Set TensorFlow to log device placement if debugging
            tf.debugging.set_log_device_placement(False)
            
            # Try to load the model with CPU first for stability
            with tf.device('/CPU:0'):
                model = tf.keras.models.load_model(path)
                
            return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.code(traceback.format_exc(), language="text")
        return None

def preprocess_image(_image, target_size=(224, 224)):
    """Preprocesses image for model input"""
    try:
        # Convert to RGB if needed
        if _image.mode != "RGB":
            _image = _image.convert("RGB")
            
        # Resize with proper error handling
        try:
            _image = _image.resize(target_size, Image.LANCZOS)
        except (AttributeError, ValueError):
            # Fallback to NEAREST for problematic images
            _image = _image.resize(target_size, Image.NEAREST)
            
        # Convert to array
        img_array = np.array(_image, dtype=np.float32)  # Specify dtype for consistency
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def predict_image(model, img_array):
    """Makes prediction on the image"""
    if img_array is None:
        return None
        
    try:
        # Use try/except with graceful fallback
        try:
            # Predictions on CPU for stability
            with tf.device('/CPU:0'):
                result = model.predict(img_array, verbose=0)
                
            # Handle different output shapes
            if isinstance(result, list):
                return result[0][0] if len(result) > 0 and len(result[0]) > 0 else 0.5
            elif result.ndim > 1:
                return result[0][0] if result.shape[0] > 0 and result.shape[1] > 0 else 0.5
            else:
                return result[0] if result.size > 0 else 0.5
                
        except (IndexError, AttributeError, ValueError) as e:
            st.warning(f"Prediction shape issue: {str(e)}. Model may have unexpected output format.")
            return 0.5
            
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

# File size validator for user images
def validate_file_size(file, max_size_mb=10):
    """Validates that the uploaded file isn't too large"""
    if file.size > max_size_mb * 1024 * 1024:
        st.error(f"File is too large! Maximum size is {max_size_mb}MB.")
        return False
    return True

# Load the model once when the app starts
with st.spinner("Loading AI detection model..."):
    model = load_model_from_path("/model.keras")
    
    if model is None:
        st.error("Failed to load the model. Please check that the model file exists in the app directory.")
        st.stop()
    else:
        st.success("Model loaded successfully!")

# Get model input shape
try:
    input_shape = model.input_shape[1:3]
    if None in input_shape:
        input_shape = (224, 224)  # Default
except (IndexError, AttributeError):
    input_shape = (224, 224)  # Default

# Display model info
with st.expander("Model Information"):
    st.write(f"Model input shape: {input_shape}")
    
    # Display summary
    try:
        summary_io = io.StringIO()
        model.summary(print_fn=lambda x: summary_io.write(x + '\n'))
        st.code(summary_io.getvalue(), language="text")
    except Exception as e:
        st.error(f"Error displaying model summary: {str(e)}")

# Main interface for image upload and prediction
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Upload an image to analyze")
    uploaded_img = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_img is not None:
        try:
            # Validate image size
            if not validate_file_size(uploaded_img):
                st.warning("Image is too large, please upload a smaller image.")
            else:
                # Display image
                image = Image.open(uploaded_img)
                if image is None:
                    st.error("Failed to load image. Please try another file.")
                    st.stop()
                    
                st.image(image, caption="Uploaded Image", width=400)
                
                # Show image details
                width, height = image.size
                st.caption(f"Image dimensions: {width}x{height} pixels, {image.mode} mode")
                
                with st.spinner("Analyzing image..."):
                    st.caption(f"Resizing to {input_shape} for model input")
                    
                    # Preprocess and predict
                    image_array = preprocess_image(image, target_size=input_shape)
                    
                    if image_array is not None:
                        probability = predict_image(model, image_array)
                        
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
                                    st.caption(f"AI confidence: {ai_confidence:.1f}%")
                                else:
                                    human_confidence = (1 - probability) * 100
                                    st.progress(human_confidence / 100)
                                    st.caption(f"Human confidence: {human_confidence:.1f}%")
                        else:
                            st.error("Failed to generate prediction.")
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.code(traceback.format_exc(), language="text")
            
    st.markdown("""
    ### How It Works
    
    This tool uses a deep learning model to analyze images and determine if they were created by AI or humans.
    
    1. Upload any JPEG or PNG image
    2. The model will analyze the image
    3. You'll receive a prediction with confidence score
    
    **Note:** While this model is designed to detect AI-generated images, no detection system is perfect. Results should be considered as probabilities rather than definitive answers.
    """)

# Additional information in sidebar
st.sidebar.title("About")
st.sidebar.info("""
### AI Image Detection Tool

This application uses a neural network model trained to distinguish between AI-generated and human-created images.

The model analyzes visual patterns, inconsistencies, and artifacts that may indicate AI generation.

**Privacy Note:** Uploaded images are processed locally and are not stored.
""")
