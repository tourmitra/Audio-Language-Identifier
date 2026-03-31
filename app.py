import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import librosa
import librosa.display
import matplotlib.pyplot as plt
import io
import tempfile
import os

# Configuration
IMG_HEIGHT = 128
IMG_WIDTH = 128
CLASS_NAMES = ['English', 'Hindi']
MODEL_PATH = 'language_id_model.keras'

st.set_page_config(page_title="Language Identifier", page_icon="🎙️", layout="centered")

@st.cache_resource
def load_model():
    try:
        return keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

def audio_to_spectrogram(audio_bytes, suffix):
    """
    Converts raw audio bytes into a mel-spectrogram image using librosa and matplotlib.
    """
    # Save uploaded audio to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
        temp_audio.write(audio_bytes)
        temp_audio_path = temp_audio.name
        
    try:
        # Load audio (automatically handles resampling if necessary)
        y, sr = librosa.load(temp_audio_path, sr=None)
        
        # Create standard mel spectrogram
        mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
        
        # Plot to an image buffer without axes
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.axis('off') # Remove axes for pure spectrogram image
        librosa.display.specshow(mel_spect_db, sr=sr, x_axis=None, y_axis=None, ax=ax, fmax=8000)
        
        # Remove padding
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        
        # Return as PIL Image matching the model's training pipeline expected format
        return Image.open(buf).convert('RGB')
    except Exception as e:
        st.error(f"Failed to process audio file: {e}")
        return None
    finally:
        # Cleanup temporary audio file
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

def main():
    st.title("🎙️ Language Identification")
    st.write("Upload a short audio clip to classify whether the spoken language is **English** or **Hindi**.")

    model = load_model()

    if model is None:
        st.warning("Please ensure the model file `language_id_model.keras` is placed in the same directory.")
        return

    # Updated file uploader to accept audio formats
    uploaded_file = st.file_uploader("Choose an audio file (WAV, MP3, etc.)", type=["wav", "mp3", "ogg", "flac"])

    if uploaded_file is not None:
        # Display an audio player so the user can listen to the uploaded file
        st.audio(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with st.spinner('Converting audio to spectrogram & analyzing...'):
            # Convert the raw audio to a spectrogram image
            file_extension = os.path.splitext(uploaded_file.name)[1]
            image = audio_to_spectrogram(uploaded_file.read(), file_extension)
            
            if image is not None:
                with col1:
                    st.image(image, caption="Generated Spectrogram", use_column_width=True)
                
                with col2:
                    st.write("### Analysis Results")
                    
                    # Preprocess for model (Resize to 128x128, scale 0-1, add batch dimension)
                    img = image.resize((IMG_WIDTH, IMG_HEIGHT))
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    img_array = img_array / 255.0
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    # Make prediction
                    prediction = model.predict(img_array)[0][0]
                    
                    is_hindi = prediction > 0.5
                    predicted_class = CLASS_NAMES[1] if is_hindi else CLASS_NAMES[0]
                    confidence = prediction if is_hindi else (1 - prediction)
                    
                    color = "green" if confidence > 0.8 else "orange"
                    st.markdown(f"**Predicted Language:** <span style='color:{color}; font-size:24px;'>{predicted_class}</span>", unsafe_allow_html=True)
                    st.write(f"**Confidence:** {confidence * 100:.2f}%")
                    st.progress(float(confidence))

if __name__ == '__main__':
    main()
