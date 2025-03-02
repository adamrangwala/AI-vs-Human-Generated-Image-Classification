import streamlit as st
import tensorflow as tf
import io
import base64
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

# Add CSS for better styling
st.markdown("""
<style>
    .upload-container {
        border: 2px dashed #ccc;
        border-radius: 5px;
        padding: 20px;
        text-align: center;
    }
    .model-details {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Streamlit header
st.title("Keras Model Loader")
st.markdown("Upload a .keras or .h5 model file to view its architecture and weights.")

# Create upload widget for model
uploaded_file = st.file_uploader("Choose a model file", type=["keras", "h5"], 
                               help="Upload your TensorFlow/Keras model (.keras or .h5 format)")

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)

def display_model_summary(model):
    # Capture the summary output
    summary_io = io.StringIO()
    model.summary(print_fn=lambda x: summary_io.write(x + '\n'))
    model_summary = summary_io.getvalue()
    summary_io.close()
    
    st.code(model_summary, language="text")

def serialize_model_architecture(model):
    # Get the model config in JSON format
    config = model.get_config()
    config_str = json.dumps(config, indent=2, cls=NumpyEncoder)
    return config_str

def get_model_weights_info(model):
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

def download_model_code(model):
    """Generate Python code to recreate the model"""
    model_json = model.to_json()
    model_config = json.loads(model_json)
    
    # Start building the Python code
    code = """
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import numpy as np
import json

# Model architecture in JSON format
model_json = '''
{}
'''

# Load model from JSON
model = model_from_json(model_json)

# Compile the model
model.compile(
    optimizer='adam',  # Replace with your optimizer
    loss='categorical_crossentropy',  # Replace with your loss function
    metrics=['accuracy']  # Replace with your metrics
)

print("Model loaded successfully!")
print(model.summary())
    """.format(model_json)
    
    return code
    
# Function to preprocess the image
def preprocess_image(image, target_size=(640,640)):
    """Preprocess the image to be compatible with the model"""
    # Resize image
    image = image.resize(target_size)
    # Convert to array and normalize
    img_array = np.array(image)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to make prediction
def predict_image(model, img_array):
    """Make prediction using the model"""
    
    prediction = model.predict(img_array)
    
    # Binary classification where 0=Human, 1=AI
    probability = prediction[0][0]
    return probability

if uploaded_file is not None:
    try:
        # Create a temporary file to save the uploaded model
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_filename = tmp_file.name
        
        # Load the model from the temporary file
        custom_objects = {
          "CustomDataAugmentation": CustomDataAugmentation,
          "ResNetPreprocessingLayer": ResNetPreprocessingLayer
        }
        with st.spinner("Loading model..."):
            model = load_model(tmp_filename, custom_objects=custom_objects)
        
        # Clean up the temporary file
        os.unlink(tmp_filename)
        
        # Display success message
        st.success(f"Model loaded successfully from {uploaded_file.name}")

        # Create tabs for different views
        tabs = st.tabs(["Summary", "Architecture", "Weights", "Code Generation"])
        
        # Tab 1: Model Summary
        with tabs[0]:
            st.subheader("Model Summary")
            display_model_summary(model)
        
        # Tab 2: Architecture
        with tabs[1]:
            st.subheader("Model Architecture")
            architecture_json = serialize_model_architecture(model)
            st.code(architecture_json, language="json")
            
            # Download button for architecture
            st.download_button(
                label="Download Architecture as JSON",
                data=architecture_json,
                file_name=f"{os.path.splitext(uploaded_file.name)[0]}_architecture.json",
                mime="application/json"
            )
        
        # Tab 3: Weights Information
        with tabs[2]:
            st.subheader("Model Weights Information")
            weights_info = get_model_weights_info(model)
            
            # Display weights info as table
            if weights_info:
                st.dataframe(weights_info)
                
                # Download button for weights info
                weights_json = json.dumps(weights_info, indent=2, cls=NumpyEncoder)
                st.download_button(
                    label="Download Weights Info as JSON",
                    data=weights_json,
                    file_name=f"{os.path.splitext(uploaded_file.name)[0]}_weights_info.json",
                    mime="application/json"
                )
            else:
                st.info("No weights found in the model.")
        
        # Tab 4: Code Generation
        with tabs[3]:
            st.subheader("Python Code to Recreate Model")
            model_code = download_model_code(model)
            st.code(model_code, language="python")
            
            # Download button for code
            st.download_button(
                label="Download Python Code",
                data=model_code,
                file_name=f"{os.path.splitext(uploaded_file.name)[0]}_model_code.py",
                mime="text/plain"
            )
            
            # Additional options for saving the model
            st.subheader("Save Model Options")
            save_format = st.radio(
                "Select format to save the model:",
                options=["SavedModel", "H5", "TF-Lite", "JSON Only"],
                horizontal=True
            )
            
            if st.button("Prepare Model for Download"):
                with st.spinner("Preparing model..."):
                    if save_format == "SavedModel":
                        # Save as SavedModel in a temporary directory
                        with tempfile.TemporaryDirectory() as tmp_dir:
                            save_path = os.path.join(tmp_dir, "model")
                            model.save(save_path, save_format="tf")
                            # Zip the directory
                            import shutil
                            zip_path = os.path.join(tmp_dir, "model.zip")
                            shutil.make_archive(os.path.join(tmp_dir, "model"), 'zip', save_path)
                            
                            with open(zip_path, "rb") as f:
                                st.download_button(
                                    label="Download SavedModel as ZIP",
                                    data=f,
                                    file_name=f"{os.path.splitext(uploaded_file.name)[0]}_savedmodel.zip",
                                    mime="application/zip"
                                )
                    
                    elif save_format == "H5":
                        # Save as H5 in a temporary file
                        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
                            model.save(tmp_file.name, save_format="h5")
                            with open(tmp_file.name, "rb") as f:
                                st.download_button(
                                    label="Download H5 Model",
                                    data=f,
                                    file_name=f"{os.path.splitext(uploaded_file.name)[0]}_model.h5",
                                    mime="application/octet-stream"
                                )
                            os.unlink(tmp_file.name)
                    
                    elif save_format == "TF-Lite":
                        # Convert to TFLite
                        converter = tf.lite.TFLiteConverter.from_keras_model(model)
                        tflite_model = converter.convert()
                        
                        st.download_button(
                            label="Download TF-Lite Model",
                            data=tflite_model,
                            file_name=f"{os.path.splitext(uploaded_file.name)[0]}_model.tflite",
                            mime="application/octet-stream"
                        )
                    
                    elif save_format == "JSON Only":
                        model_json = model.to_json()
                        st.download_button(
                            label="Download Model JSON",
                            data=model_json,
                            file_name=f"{os.path.splitext(uploaded_file.name)[0]}_model.json",
                            mime="application/json"
                        )
        
    except Exception as e:
        st.error(f"Error loading the model: {str(e)}")
        st.exception(e)
    
    # Image upload
    uploaded_img = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
   
    st.markdown("### Or try a sample image:")
    sample_col1, sample_col2 = st.columns(2)
    
    with sample_col1:
        if st.button("Sample Human Photo"):
            # Replace with path to your sample human image
            sample_img_path = "samples/human_sample.jpg"
            if os.path.exists(sample_img_path):
                with open(sample_img_path, "rb") as file:
                    uploaded_img = io.BytesIO(file.read())
            else:
                st.warning("Sample image not found. Please check the path.")
    
    with sample_col2:
        if st.button("Sample AI Photo"):
            # Replace with path to your sample AI image
            sample_img_path = "samples/ai_sample.jpg"
            if os.path.exists(sample_img_path):
                with open(sample_img_path, "rb") as file:
                    uploaded_img = io.BytesIO(file.read())
            else:
                st.warning("Sample image not found. Please check the path.")
                
    if uploaded_img is not None:
        try:
            # Open and display the image
            image = Image.open(uploaded_img).convert("RGB")

            with sample_col1:
                st.markdown("<div class='image-container'>", unsafe_allow_html=True)
                st.image(image, caption="Uploaded Image", width=400)
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Add a spinner while processing
            with st.spinner("Analyzing image..."):

                # preprocess_image
                image_array = preprocess_image(image)
                
                # Make prediction
                probability = predict_image(model, image_array)
                
                # Display result
                if probability > 0.5:
                    confidence = round(probability * 100, 2)
                    with sample_col2:
                        st.markdown(f"<div class='result-header ai-result'>AI-Generated <span class='confidence'>{confidence}%</span> confidence</div>", unsafe_allow_html=True)
                        rain(emoji="👨", font_size=54, falling_speed=5, animation_length="infinite") 
                else:
                    confidence = round((1 - probability) * 100, 2)
                    with sample_col2:
                        st.markdown(f"<div class='result-header human-result'>Human-Generated <span class='confidence'>{confidence}%</span> confidence</div>", unsafe_allow_html=True)
                        rain(emoji="🤖", font_size=54, falling_speed=5, animation_length="infinite") 
                
                # Add explanation
                st.markdown("### How it works")
                st.write("""
                Our model analyzes various features in the image to determine if it was created by AI or a human. 
                AI-generated images often have subtle patterns, inconsistencies in details like hands, eyes, 
                or backgrounds, and other artifacts that the model has learned to recognize.
                """)
                
                # Disclaimer
                st.markdown("---")
                st.caption("""
                **Disclaimer**: While this model strives for accuracy, it may not be perfect. The rapidly evolving 
                field of AI image generation means new models may produce images that are increasingly difficult to distinguish.
                """)
                         
        except Exception as e:
                st.error(f"Error processing image: {e}")   
    else:   
            st.info("Upload/Select an image file to begin.")

else:
    # Display instructions when no file is uploaded
    st.info("Upload a .keras or .h5 model file to begin.")
    

