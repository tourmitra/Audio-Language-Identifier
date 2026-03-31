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

def process_audio(audio_bytes, suffix):
    """
    Converts raw audio bytes into an image array matching the exact parameters used during training.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
        temp_audio.write(audio_bytes)
        temp_audio_path = temp_audio.name
        
    try:
        # Convert audio to mel-spectrogram array using user's exact parameters
        y, sr = librosa.load(temp_audio_path, sr=22050, mono=True)

        mel_spec_db = librosa.power_to_db(
            librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000),
            ref=np.max
        )

        # Render to in-memory image
        fig, ax = plt.subplots(figsize=(IMG_WIDTH / 100, IMG_HEIGHT / 100), dpi=100)
        librosa.display.specshow(mel_spec_db, sr=sr, fmax=8000, ax=ax)
        ax.axis("off")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        buf.seek(0)

        # Return exact img_array conversion
        img_array = np.array(
            Image.open(buf).convert("RGB").resize((IMG_WIDTH, IMG_HEIGHT)),
            dtype=np.float32
        ) / 255.0
        
        return img_array
    except Exception as e:
        st.error(f"Failed to process audio file: {e}")
        return None
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

def main():
    st.title("🎙️ Language Identification")
    st.write("Upload a short audio clip (or an existing spectrogram) to classify whether the spoken language is **English** or **Hindi**.")

    model = load_model()

    if model is None:
        st.warning("Please ensure the model file `language_id_model.keras` is placed in the same directory.")
        return

    # Accept both audio and image files
    uploaded_file = st.file_uploader("Choose a file (Audio or Image)", type=["wav", "mp3", "ogg", "flac", "png", "jpg", "jpeg"])

    if uploaded_file is not None:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        img_array = None
        
        if file_extension in ['.png', '.jpg', '.jpeg']:
            # Handle image uploads directly (useful for testing Colab images)
            image = Image.open(uploaded_file).convert('RGB')
            # Preprocess image
            img = image.resize((IMG_WIDTH, IMG_HEIGHT))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = img_array / 255.0
        else:
            # Handle audio uploads
            st.audio(uploaded_file)
            with st.spinner('Converting audio & analyzing...'):
                img_array = process_audio(uploaded_file.read(), file_extension)
        
        if img_array is not None:
            st.write("### Analysis Results")
            
            # Add batch dimension
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
