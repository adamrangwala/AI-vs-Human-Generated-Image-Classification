import streamlit as st
import tensorflow as tf
import io
import base64
import json
import numpy as np
import os
from tensorflow.keras.models import load_model
import tempfile

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

# Create upload widget
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

if uploaded_file is not None:
    try:
        # Create a temporary file to save the uploaded model
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_filename = tmp_file.name
        
        # Load the model from the temporary file
        with st.spinner("Loading model..."):
            model = load_model(tmp_filename)
        
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
else:
    # Display instructions when no file is uploaded
    st.info("Upload a .keras or .h5 model file to begin.")
    
    # Example section
    with st.expander("How to use this app"):
        st.markdown("""
        ### Instructions:
        1. Click on the file uploader above to select your Keras model file (.keras or .h5 format)
        2. The app will load your model and display its architecture, summary, and weights information
        3. You can download various representations of your model:
           - Architecture as JSON
           - Weights information as JSON
           - Python code to recreate the model
           - The model in different formats (SavedModel, H5, TF-Lite)
        
        ### Supported Features:
        - Works with most TensorFlow/Keras model architectures
        - Analyzes model layers and weights
        - Generates code to recreate your model
        - Multiple export formats
        """)
