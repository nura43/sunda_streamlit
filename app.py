import streamlit as st
import numpy as np
import librosa
import pickle
from tensorflow.keras.models import load_model

# Load model dan label encoder
model = load_model("model_sunda_dnn.h5", compile=False)
with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# Fungsi prediksi suara
def predict_audio(audio_file):
    y, sr = librosa.load(audio_file, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    feature = np.mean(mfcc.T, axis=0).reshape(1, -1)
    prediction = model.predict(feature)
    label_index = np.argmax(prediction)
    label = encoder.inverse_transform([label_index])[0]
    return label

# Halaman utama Streamlit
st.title("🎙️ Pengenalan Suara Bahasa Sunda ke Bahasa Inggris")
uploaded_file = st.file_uploader("Unggah file suara (.wav)", type="wav")

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    hasil_prediksi = predict_audio(uploaded_file)
    st.write("📝 **Hasil Transkripsi (Label):**", hasil_prediksi)

    # Kamus terjemahan label ke Bahasa Inggris
    kamus_terjemahan = {
        "abdi_hoyong_tuang": "I want to eat",
        "abdi_geulis": "I am beautiful",
        "abdi_bade_ka_pasar": "I am going to the market",
        "hatur_nuhun": "Thank you",
        "kamana": "Where are you going?",
        "sampurasun": "Excuse me",
        "kumaha_damang": "How are you?",
        "bade_meli_naon": "What do you want to buy?",
        "namina_saha": "What is your name?",
        "nuju_naon": "What are you doing?"
    }

    terjemahan = kamus_terjemahan.get(hasil_prediksi, "❗ Terjemahan tidak ditemukan.")
    st.write("🌐 **Terjemahan Bahasa Inggris:**", terjemahan)
