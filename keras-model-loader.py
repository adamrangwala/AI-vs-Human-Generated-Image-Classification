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
import gc
import traceback
import psutil  

# Configure page
st.set_page_config(page_title="Keras Model Loader", layout="wide")

# Memory monitoring function
def get_memory_usage():
    """Returns the current memory usage in MB"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)

# Streamlit header
st.title("Keras Model Loader")
st.markdown("Upload a .keras or .h5 model file to view its architecture and weights.")

# Display current memory usage if in debug mode
if st.checkbox("Show memory usage", value=False):
    memory_usage = get_memory_usage()
    memory_container = st.empty()
    memory_container.info(f"Current memory usage: {memory_usage:.2f} MB")

# File size validator
def validate_file_size(file, max_size_mb=500):
    """Validates that the uploaded file isn't too large"""
    if file.size > max_size_mb * 1024 * 1024:
        st.error(f"File is too large! Maximum size is {max_size_mb}MB.")
        return False
    return True

# File uploader with size limit warning
st.warning("Note: Large model files (>100MB) may cause memory issues. Consider using a smaller or quantized model.")
uploaded_file = st.file_uploader("Choose a model file", type=["keras", "h5"])

@st.cache_resource(show_spinner=False)
def load_keras_model(model_path, custom_objects_dict=None):
    """Loads and caches the Keras model to prevent reloading on each interaction."""
    try:
        if custom_objects_dict is None:
            custom_objects_dict = {
                "CustomDataAugmentation": CustomDataAugmentation,
                "ResNetPreprocessingLayer": ResNetPreprocessingLayer
            }
        
        # Configure TensorFlow for memory efficiency
        tf.config.experimental.set_memory_growth(
            tf.config.experimental.list_physical_devices('GPU')[0], True
        ) if tf.config.experimental.list_physical_devices('GPU') else None
        
        # Set TensorFlow to log device placement if debugging
        tf.debugging.set_log_device_placement(False)
        
        # Try to load the model with CPU first for stability
        with tf.device('/CPU:0'):
            model = load_model(model_path, custom_objects=custom_objects_dict)
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.code(traceback.format_exc(), language="text")
        return None

@st.cache_data
def preprocess_image(_image, target_size=(640, 640)):
    """Caches image preprocessing to avoid redundant computation."""
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

@st.cache_data
def get_model_weights_info(model):
    """Caches model weight information with memory management."""
    try:
        weights_info = []
        total_weights = len(model.layers)
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, layer in enumerate(model.layers):
            layer_weights = layer.get_weights()
            if layer_weights:
                for j, w in enumerate(layer_weights):
                    # Use a small slice of very large weight arrays for stats
                    if w.size > 10_000_000:  # If weights are over 10M elements
                        sample = w.flatten()[:1_000_000]  # Sample first million
                        weights_info.append({
                            "layer_name": layer.name,
                            "layer_index": i,
                            "weight_index": j,
                            "shape": str(w.shape),  # Convert to string for large tuples
                            "size_mb": float(w.nbytes / (1024 * 1024)),
                            "dtype": str(w.dtype),
                            "min": float(np.min(sample)),
                            "max": float(np.max(sample)),
                            "mean": float(np.mean(sample)),
                            "std": float(np.std(sample)),
                            "sampled": True
                        })
                    else:
                        weights_info.append({
                            "layer_name": layer.name,
                            "layer_index": i,
                            "weight_index": j,
                            "shape": str(w.shape),
                            "size_mb": float(w.nbytes / (1024 * 1024)),
                            "dtype": str(w.dtype),
                            "min": float(np.min(w)),
                            "max": float(np.max(w)),
                            "mean": float(np.mean(w)),
                            "std": float(np.std(w)),
                            "sampled": False
                        })
            
            # Update progress every few layers
            if i % max(1, total_weights // 10) == 0:
                progress = min(1.0, (i + 1) / total_weights)
                progress_bar.progress(progress)
                status_text.text(f"Analyzing layer {i+1}/{total_weights}: {layer.name}")
        
        progress_bar.progress(1.0)
        status_text.empty()
        progress_bar.empty()
        
        return weights_info
    except Exception as e:
        st.error(f"Error analyzing weights: {str(e)}")
        return []

@st.cache_data
def serialize_model_architecture(model):
    """Caches model architecture serialization with error handling."""
    try:
        config = model.get_config()
        return json.dumps(config, indent=2)
    except Exception as e:
        st.error(f"Error serializing model architecture: {str(e)}")
        return json.dumps({"error": str(e)})

@st.cache_data
def predict_image(model, img_array):
    """Caches model predictions with error handling."""
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

# Process uploaded model
if uploaded_file is not None:
    if not validate_file_size(uploaded_file, max_size_mb=500):
        st.stop()
        
    try:
        # Progress indicator for model loading
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("Saving uploaded model to temporary file...")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_filename = tmp_file.name
        
        progress_bar.progress(0.3)
        status_text.text("Loading model (this may take a while for large models)...")
        
        with st.spinner("Loading model..."):
            model = load_keras_model(tmp_filename)

        # Clean up temp file
        try:
            os.unlink(tmp_filename)
        except Exception as e:
            st.warning(f"Could not delete temporary file: {str(e)}")
        
        progress_bar.progress(1.0)
        status_text.empty()
        progress_bar.empty()
        
        if model is None:
            st.error("Failed to load model. Please check the model format and try again.")
            st.stop()
            
        st.success(f"Model loaded successfully from {uploaded_file.name}")

        # Force garbage collection
        gc.collect()

        # Tabs for different views
        tabs = st.tabs(["Prediction", "Summary", "Architecture", "Weights"])

        # Image upload for prediction
        with tabs[0]:
            pred_col1, pred_col2 = st.columns(2)
            
            with pred_col1:
                st.subheader("Upload an image to analyze")
                uploaded_img = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
                
                if uploaded_img is not None:
                    try:
                        # Validate image size
                        if not validate_file_size(uploaded_img, max_size_mb=10):
                            st.warning("Image is too large, please upload a smaller image.")
                        else:
                            # Display image
                            image = Image.open(uploaded_img)
                            if image is None:
                                st.error("Failed to load image. Please try another file.")
                            st.image(image, caption="Uploaded Image", width=400)
                            
                            # Show image details
                            width, height = image.size
                            st.caption(f"Image dimensions: {width}x{height} pixels, {image.mode} mode")
                            
                            with st.spinner("Analyzing image..."):
                                # Get input shape from model
                                try:
                                    input_shape = model.input_shape[1:3]
                                    if None in input_shape:
                                        input_shape = (640, 640)  # Default
                                except (IndexError, AttributeError):
                                    input_shape = (640, 640)  # Default
                                    
                                st.caption(f"Resizing to {input_shape} for model input")
                                
                                # Preprocess and predict
                                image_array = preprocess_image(image, target_size=input_shape)
                                
                                if image_array is not None:
                                    probability = predict_image(model, image_array)
                                    
                                    if probability is not None:
                                        with pred_col2:
                                            confidence = round(probability * 100, 2) if probability > 0.5 else round((1 - probability) * 100, 2)
                                            label = "AI-Generated" if probability > 0.5 else "Human-Generated"
                                            color = "red" if probability > 0.5 else "green"
                                            emoji = "🤖" if probability > 0.5 else "🧑"
                                            
                                            st.markdown(f"<h1 style='text-align: center; color: {color};'>{label} Image</h1>", unsafe_allow_html=True)
                                            st.markdown(f"<h2 style='text-align: center; color: black;'>{confidence}% confidence</h2>", unsafe_allow_html=True)
                                            
                                            # Use try/except for rain animation
                                            try:
                                                rain(emoji=emoji, font_size=80, falling_speed=5, animation_length=5)
                                            except Exception as e:
                                                st.warning(f"Could not display animation: {str(e)}")
                                                st.markdown(f"<h1 style='text-align: center;'>{emoji}</h1>", unsafe_allow_html=True)
                                    else:
                                        st.error("Failed to generate prediction.")
                    except Exception as e:
                        st.error(f"Error processing image: {str(e)}")
                        st.code(traceback.format_exc(), language="text")

        # Summary tab
        with tabs[1]:
            st.subheader("Model Summary")
            try:
                summary_io = io.StringIO()
                
                # Safely generate summary
                try:
                    model.summary(print_fn=lambda x: summary_io.write(x + '\n'))
                    st.code(summary_io.getvalue(), language="text")
                except ValueError:
                    st.warning("Could not generate complete summary. The model structure may be complex.")
                    st.write("Basic model information:")
                    st.write(f"Number of layers: {len(model.layers)}")
                    st.write(f"Input shape: {model.input_shape}")
                    st.write(f"Output shape: {model.output_shape}")
            except Exception as e:
                st.error(f"Error displaying model summary: {str(e)}")

        # Architecture tab
        with tabs[2]:
            st.subheader("Model Architecture")
            try:
                architecture_json = serialize_model_architecture(model)
                
                # Display a sample of the architecture if it's very large
                if len(architecture_json) > 500000:
                    st.warning("Model architecture is very large. Showing first 100KB...")
                    st.code(architecture_json[:100000], language="json")
                else:
                    st.code(architecture_json, language="json")
                    
                st.download_button("Download Architecture JSON", data=architecture_json, file_name="model_architecture.json", mime="application/json")
            except Exception as e:
                st.error(f"Error displaying model architecture: {str(e)}")

        # Weights tab
        with tabs[3]:
            st.subheader("Model Weights Information")
            try:
                st.warning("For large models, this may take a while and consume significant memory.")
                if st.button("Analyze Weights"):
                    weights_info = get_model_weights_info(model)
                    
                    # Convert to DataFrame for better display
                    try:
                        import pandas as pd
                        df = pd.DataFrame(weights_info)
                        
                        # Calculate total model size
                        total_mb = df['size_mb'].sum() if 'size_mb' in df.columns else 0
                        st.info(f"Total model size: {total_mb:.2f} MB")
                        
                        st.dataframe(df)
                    except Exception:
                        st.json(weights_info)
                        
                    # Provide download
                    st.download_button("Download Weights JSON", 
                                    data=json.dumps(weights_info, indent=2), 
                                    file_name="weights_info.json", 
                                    mime="application/json")
                    
                    # Force garbage collection after weights analysis
                    gc.collect()
            except Exception as e:
                st.error(f"Error analyzing weights: {str(e)}")
                
    except Exception as e:
        st.error(f"Error loading the model: {str(e)}")
        st.code(traceback.format_exc(), language="text")
        
    finally:
        # Final cleanup
        if 'tmp_filename' in locals() and os.path.exists(tmp_filename):
            try:
                os.unlink(tmp_filename)
            except:
                pass
