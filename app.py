import os
import streamlit as st
from pytube import YouTube
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator

# App Title
st.title("🎬 YouTube Translator App (Python 3.11 Compatible)")

# Input YouTube URL
youtube_url = st.text_input("Enter YouTube video URL")

if youtube_url:
    try:
        with st.spinner("Downloading YouTube video..."):
            yt = YouTube(youtube_url)
            stream = yt.streams.filter(only_audio=True).first()
            audio_file_path = stream.download(filename="audio.mp4")
            st.success("Audio downloaded!")
            st.write(" Saved to:", audio_file_path)
    except Exception as e:
        st.error(f" Failed to download video: {e}")
        st.stop()

    try:
        with st.spinner("Transcribing audio..."):
            model = WhisperModel("base", device="cpu")  # Use "cuda" if you have GPU
            segments, info = model.transcribe(audio_file_path, beam_size=5)
            transcription = " ".join([segment.text for segment in segments])
            st.success(" Transcription complete!")
            st.subheader("Transcript")
            st.text_area("Transcript", transcription, height=200)
    except Exception as e:
        st.error(f" Transcription failed: {e}")
        st.stop()

    # Translation
    st.subheader("🌍 Translate Transcript")
    target_lang = st.selectbox(
        "Choose target language", 
        ["en", "hi", "ta", "te", "fr", "de", "es", "zh", "ar"]
    )

    if st.button("Translate"):
        if not transcription:
            st.warning("⚠️ Nothing to translate!")
        else:
            try:
                with st.spinner(f"Translating to '{target_lang}'..."):
                    translator = GoogleTranslator(source="auto", target=target_lang)
                    translated_text = translator.translate(transcription)
                    st.success(" Translation complete!")
                    st.subheader("🈂 Translated Text")
                    st.text_area("Translated Transcript", translated_text, height=200)
            except Exception as e:
                st.error(f" Translation error: {e}")
