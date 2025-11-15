import streamlit as st
import tempfile
import requests
import os

#----- Gemini API Helper ----#
import google.generativeai as genai

def get_gemini_analysis(transcript, domain, round_type, feedback_tone, model_name, api_key):
    genai.configure(api_key=api_key)
    prompt = (
        f"You are an AI Interview Mentor and Analyzer.\n"
        f"Interview/Discussion Domain: {domain}\n"
        f"Round type: {round_type}\n"
        f"Feedback Tone: {feedback_tone}\n\n"
        f"Given the following interview/group discussion transcript, perform:\n"
        f"- For each participant: analyze tone, sentiment (positive/neutral/negative etc), clarity, confidence, empathy, and communication quality.\n"
        f"- Extract and summarize key points mentioned by each participant.\n"
        f"- Give overall summary of the discussion.\n"
        f"- Suggest concrete areas of improvement.\n"
        f"Produce a clear, structured report suitable for both participants and mentors, using bullet points and short paragraphs. Avoid generic advice; be context-aware.\n\n"
        f"Transcript:\n{transcript}\n"
    )
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    return response.text

#----------- Whisper Transcription -----------#
def transcribe_with_whisper(audio_path):
    try:
        import openai
        # Exp: Set your OpenAI API key as an env variable or input field if you want to use Whisper API.
        openai.api_key = os.environ.get("OPENAI_API_KEY", "")
        with open(audio_path, "rb") as f:
            transcript = openai.Audio.transcribe("whisper-1", f)
        return transcript["text"]
    except Exception as e:
        st.error(f"Transcription failed: {str(e)}")
        return None

#------------- Streamlit UI --------------#
st.set_page_config(page_title="AI Interview Analyzer", layout="wide")

st.title("🎤 AI Interview/Discussion Analyzer")
st.write("Upload an audio file (interview/discussion) or paste a text transcript. Get instant, structured feedback like a true AI mentor. Powered by Gemini API.")

input_mode = st.radio("Input Mode:", ["Audio File (auto transcription)", "Text Transcript"])

domain = st.selectbox("Select Domain/Context:", ["Tech", "Managerial", "HR", "Group Discussion", "Other"])
round_type = st.selectbox("Round Type:", ["Screening", "Technical", "Managerial", "HR", "Final", "Other"])
feedback_tone = st.selectbox("Feedback Tone:", ["Professional", "Encouraging", "Critical", "Neutral"])

# Gemini API config
with st.expander("🔐 Gemini API Configuration"):
    api_key = st.text_input("Enter your Gemini API key (Safe: Not shared or stored online):", type="password")
    gemini_model = st.selectbox("Gemini Model", ["gemini-pro", "gemini-1.5-pro", "gemini-ultra"], index=0)

transcript = None
if input_mode == "Audio File (auto transcription)":
    audio_file = st.file_uploader("Upload audio (.wav, .mp3, .m4a)", type=["wav", "mp3", "m4a"])
    if audio_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(audio_file.read())
            tmp.flush()
            st.info("Transcribing audio file (Whisper)...")
            transcript = transcribe_with_whisper(tmp.name)
            if transcript:
                st.success("Transcription successful. See text below:")
                st.text_area("Transcript", transcript, height=200)
elif input_mode == "Text Transcript":
    transcript = st.text_area("Paste full transcript here:", height=300)

analyze_btn = st.button("🔎 Analyze Interview/Discussion", disabled=not transcript or not api_key)

if analyze_btn and transcript and api_key:
    st.info("Analyzing with Gemini, please wait (~15-30s)...")
    try:
        response = get_gemini_analysis(transcript, domain, round_type, feedback_tone, gemini_model, api_key)
        st.subheader("📝 Interview/Discussion Feedback Report:")
        st.markdown(response)
    except Exception as e:
        st.error(f"Gemini API call failed: {str(e)}")

st.markdown("---")
st.caption("Built for interview and group discussion feedback | Gemini API & OpenAI Whisper support | Streamlit deploy ready")

# Optional: Add demo transcript, example usage, or PDF generation for bonus features!
