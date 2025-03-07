import streamlit as st
import tensorflow as tf
import keras
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

# Model path - assuming model is in the same directory as the app
MODEL_PATH = "model"  # Directory containing the SavedModel

@st.cache_resource(show_spinner=False)
def load_saved_model(path):
    """Loads a TensorFlow SavedModel"""
    try:
        with st.status("Loading model..."):
            # Configure TensorFlow for memory efficiency
            tf.config.experimental.set_memory_growth(
                tf.config.experimental.list_physical_devices('GPU')[0], True
            ) if tf.config.experimental.list_physical_devices('GPU') else None
            
            # Try to load the model with CPU for stability
            with tf.device('/CPU:0'):
                # For TensorFlow SavedModel format
                try:
                    # First try loading as TF SavedModel
                    model = tf.saved_model.load(path)
                    # Get the serving signature
                    serving_key = list(model.signatures.keys())[0]  # Usually 'serving_default'
                    st.info(f"Loaded SavedModel with signature: {serving_key}")
                    return model, serving_key
                except Exception as e:
                    st.warning(f"Failed to load as SavedModel, trying Keras TFSMLayer: {str(e)}")
                    # Try loading as TFSMLayer
                    try:
                        # Try to find the serving endpoint
                        endpoints = tf.saved_model.list_all_signatures(path)
                        call_endpoint = list(endpoints.keys())[0] if endpoints else 'serving_default'
                        model = keras.layers.TFSMLayer(path, call_endpoint=call_endpoint)
                        return model, None
                    except Exception as e2:
                        # Last resort: try loading as Keras model
                        st.warning(f"Failed as TFSMLayer, trying as Keras model: {str(e2)}")
                        model = keras.models.load_model(path)
                        return model, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.code(traceback.format_exc(), language="text")
        return None, None

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
        
        # Normalize the image (common preprocessing)
        img_array = img_array / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def predict_image(model, serving_key, img_array):
    """Makes prediction on the image using either SavedModel or Keras model"""
    if img_array is None:
        return None
        
    try:
        # Different prediction approaches based on model type
        if serving_key is not None:
            # For SavedModel with signatures
            try:
                with tf.device('/CPU:0'):
                    # Convert to tensor
                    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
                    # Get prediction function from signature
                    predict_fn = model.signatures[serving_key]
                    # Make prediction
                    result = predict_fn(tf.constant(img_tensor))
                    
                    # Extract result from output tensor
                    # The exact output key may vary, try common ones
                    output_keys = list(result.keys())
                    if output_keys:
                        result_tensor = result[output_keys[0]]
                        result_value = result_tensor.numpy()
                        
                        # Handle different output shapes
                        if result_value.ndim > 1:
                            return result_value[0][0] if result_value.shape[0] > 0 and result_value.shape[1] > 0 else 0.5
                        else:
                            return result_value[0] if result_value.size > 0 else 0.5
                    else:
                        st.warning("No output keys found in model result")
                        return 0.5
            except Exception as e:
                st.warning(f"Error with SavedModel prediction: {str(e)}, falling back to direct call")
                # Fallback to direct call
                result = model(img_array)
                if isinstance(result, dict):
                    first_key = list(result.keys())[0]
                    return result[first_key].numpy()[0][0]
                else:
                    return result.numpy()[0][0]
        else:
            # For Keras model or TFSMLayer
            with tf.device('/CPU:0'):
                result = model(img_array, training=False)
                
                # Handle TFSMLayer output
                if isinstance(result, dict):
                    # Get the first output if it's a dictionary
                    first_key = list(result.keys())[0]
                    result_value = result[first_key].numpy()
                elif hasattr(result, 'numpy'):
                    # Regular tensor output
                    result_value = result.numpy()
                else:
                    # Direct numpy array or other
                    result_value = np.array(result)
                
                # Handle different output shapes
                if result_value.ndim > 1:
                    return result_value[0][0] if result_value.shape[0] > 0 and result_value.shape[1] > 0 else 0.5
                else:
                    return result_value[0] if result_value.size > 0 else 0.5
                
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.code(traceback.format_exc(), language="text")
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
    model, serving_key = load_saved_model(MODEL_PATH)
    
    if model is None:
        st.error("Failed to load the model. Please check that the model directory exists.")
        st.stop()
    else:
        st.success("Model loaded successfully!")
        if serving_key:
            st.info(f"Using signature: {serving_key}")

# Try to determine input shape
try:
    # For TF SavedModel, try to get input shape from signature
    if serving_key is not None and hasattr(model.signatures[serving_key], 'inputs'):
        input_tensor = list(model.signatures[serving_key].inputs.values())[0]
        input_shape = input_tensor.shape.as_list()[1:3]
    else:
        # For Keras models
        input_shape = getattr(model, 'input_shape', None)
        if input_shape:
            input_shape = input_shape[1:3]
    
    # Default if we couldn't determine
    if input_shape is None or None in input_shape:
        input_shape = (224, 224)  # Default common size
except Exception:
    input_shape = (224, 224)  # Default common size

# Display model info
with st.expander("Model Information"):
    st.write(f"Model input shape: {input_shape}")
    st.write(f"Model type: {'TensorFlow SavedModel' if serving_key else 'Keras Model or TFSMLayer'}")
    
    # Try to display signature info for SavedModel
    if serving_key is not None:
        st.subheader("Model Signature Details")
        try:
            signature = model.signatures[serving_key]
            st.write("Inputs:")
            for i, (name, tensor) in enumerate(signature.inputs.items()):
                st.write(f"  {i+1}. {name}: shape={tensor.shape}, dtype={tensor.dtype}")
            
            st.write("Outputs:")
            for i, (name, tensor) in enumerate(signature.outputs.items()):
                st.write(f"  {i+1}. {name}: shape={tensor.shape}, dtype={tensor.dtype}")
        except Exception as e:
            st.warning(f"Could not display signature details: {e}")

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
                        probability = predict_image(model, serving_key, image_array)
                        
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
