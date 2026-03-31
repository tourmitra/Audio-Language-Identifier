import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image

# Configuration
IMG_HEIGHT = 128
IMG_WIDTH = 128
CLASS_NAMES = ['English', 'Hindi']
MODEL_PATH = 'language_id_model.keras'

# Set page configuration
st.set_page_config(page_title="Language Identifier", page_icon="🎙️", layout="centered")

@st.cache_resource
def load_model():
    # Load the trained Keras model
    try:
        model = keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

def main():
    st.title("🎙️ Language Identification from Spectrograms")
    st.write("Upload an audio spectrogram to classify whether the spoken language is **English** or **Hindi**.")

    model = load_model()

    if model is None:
        st.warning("Please ensure the model file `language_id_model.keras` is placed in the same directory.")
        return

    # File uploader for the spectrogram image
    uploaded_file = st.file_uploader("Choose a spectrogram image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Open and display the uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        
        # Two-column layout for better aesthetics
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Uploaded Spectrogram", use_column_width=True)
        
        with col2:
            st.write("### Analysis Results")
            with st.spinner('Classifying...'):
                # Preprocess the image to match exactly what the model expects
                img = image.resize((IMG_WIDTH, IMG_HEIGHT))
                
                # Convert to array and scale (0-255 -> 0-1) as done during training
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = img_array / 255.0
                
                # Add batch dimension
                img_array = np.expand_dims(img_array, axis=0)
                
                # Make prediction
                prediction = model.predict(img_array)[0][0]
                
                # Interpret results (0 is English, 1 is Hindi)
                is_hindi = prediction > 0.5
                predicted_class = CLASS_NAMES[1] if is_hindi else CLASS_NAMES[0]
                confidence = prediction if is_hindi else (1 - prediction)
                
                # Display output based on confidence
                color = "green" if confidence > 0.8 else "orange"
                st.markdown(f"**Predicted Language:** <span style='color:{color}; font-size:24px;'>{predicted_class}</span>", unsafe_allow_html=True)
                st.write(f"**Confidence:** {confidence * 100:.2f}%")
                
                # Show a progress bar for the visual effect
                st.progress(float(confidence))

if __name__ == '__main__':
    main()
