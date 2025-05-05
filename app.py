import streamlit as st
import os
import soundfile as sf
import io
from transformers import pipeline

# Load Hugging Face API Key from environment variables
HUGGINGFACE_API_KEY = os.getenv("HF_API_KEY")

# Hugging Face model identifiers
ASR_MODEL = "openai/whisper-small"
TEXT_GEN_MODEL = "mistralai/Mistral-7B-Instruct"
TTS_MODEL = "facebook/mms-tts-eng"

# Load Hugging Face models
asr_pipeline = pipeline("automatic-speech-recognition", model=ASR_MODEL)
text_gen_pipeline = pipeline("text-generation", model=TEXT_GEN_MODEL, tokenizer=TEXT_GEN_MODEL, token=HUGGINGFACE_API_KEY)
tts_pipeline = pipeline("text-to-speech", model=TTS_MODEL, token=HUGGINGFACE_API_KEY)

# Streamlit UI
st.title("🎙️ AI Voice Chatbot")
st.write("Speak to the AI, and it will reply with voice!")

# Upload or record audio
audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])
record_audio = st.button("Record Audio")

if record_audio:
    st.warning("Recording is not supported in Streamlit yet. Please upload an audio file instead.")

if audio_file is not None:
    st.audio(audio_file, format="audio/wav")

    # Read audio file in memory
    audio_bytes = audio_file.read()

    # Convert audio to text (Speech-to-Text)
    asr_response = asr_pipeline(audio_bytes)
    user_text = asr_response["text"]

    st.subheader("🗣️ Recognized Speech")
    st.write(user_text)

    # Generate AI response (Chatbot)
    prompt = f"User: {user_text}\nAI:"
    ai_response = text_gen_pipeline(prompt, max_length=100, num_return_sequences=1, return_full_text=False)[0]["generated_text"]

    st.subheader("🤖 AI Response")
    st.write(ai_response)

    # Convert AI response to speech (Text-to-Speech)
    tts_response = tts_pipeline(ai_response)

    # Convert to WAV format
    audio_output = io.BytesIO()
    sf.write(audio_output, tts_response["audio"], samplerate=tts_response["sampling_rate"], format="wav")
    audio_output.seek(0)

    st.subheader("🔊 AI Voice Response")
    st.audio(audio_output, format="audio/wav")
