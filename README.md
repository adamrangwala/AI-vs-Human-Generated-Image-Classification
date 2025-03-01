# AI vs Human Image Classifier

This <a href="https://ai-human-generated-image-classifier-nagba8rmfff8dvfxwpzs8c.streamlit.app/"> Streamlit application </a> uses a pre-trained Keras model to classify images as either AI-generated or human-created. Upload an image, and the app will analyze it, displaying the classification result with a confidence percentage.

## Features

- Simple drag-and-drop interface for image upload
- Real-time classification of images
- Confidence score display
- Sample images for testing
- Responsive design

## Setup and Deployment

### Local Development

1. Clone this repository:

    'git clone https://github.com/adamrangwala/ai-human-classifier.git'
    'cd ai-human-classifier'

2. Install the required dependencies:
    'pip install -r requirements.txt'
  
4. Create the following directories and add your files:
- Create a `model` directory and place your Keras model (`ai_human_classifier.h5`) inside
- Create a `samples` directory and add sample images (`human_sample.jpg` and `ai_sample.jpg`)

4. Run the Streamlit app:
    'streamlit run app.py'

### Deploying to Streamlit Cloud

1. Push your repository to GitHub
2. Log in to [Streamlit Cloud](https://streamlit.io/cloud)
3. Create a new app and connect it to your GitHub repository
4. Set the main file path to `app.py`
5. Deploy the app

## Model Information

This application uses a fine-tuned ResNet50 model trained on a dataset of AI-generated and human-created images. The model analyzes visual patterns and artifacts that may not be obvious to the human eye.

## Customization

You can customize the application by:
- Updating the model path in the `load_model()` function
- Changing the preprocessing steps in the `preprocess_image()` function to match your model's requirements
- Modifying the CSS styling in the `st.markdown()` section
- Adding additional sample images in the `samples` directory

## License

MIT
