# 🎙️ Audio Language Identifier

## Project Overview
The **Audio Language Identifier** is a machine learning application designed to classify spoken audio as either **English** or **Hindi**. 

Instead of traditional audio processing, this project leverages computer vision to solve an audio problem. Raw audio clips are dynamically converted into Mel-Spectrograms (visual representations of sound frequencies). These spectrogram images are then passed through a Convolutional Neural Network (CNN) capable of distinguishing the unique visual patterns of both languages to give an accurate prediction.

## Tech Stack Used
- **Frontend / Deployment**: [Streamlit](https://streamlit.io/)
- **Machine Learning**: TensorFlow & Keras (CNN Architecture)
- **Audio Processing**: Librosa
- **Data Visualization & Image Processing**: Matplotlib, Pillow (PIL), NumPy
- **Environment**: Python 3.11

## Links

- **Deployment Link**: [Streamlit](https://audio-language-identifier.streamlit.app)
- **Colab Notebook Link**: [Colab](https://colab.research.google.com/drive/1dTcJhjbJCCeb8RPXn5zxddErfquAIEyK#scrollTo=pMpwnlO1dasK)
- **GitHub Repository**: [GitHub](https://github.com/tourmitra/Audio-Language-Identifier)
- **Dataset Used**: [Kaggle](https://www.kaggle.com/datasets/abhay242/english-hindi-audio-spectrograms)
