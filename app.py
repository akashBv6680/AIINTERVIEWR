import streamlit as st
import tempfile
import requests
import os

# NOTE: This application requires two external dependencies:
# 1. The google-genai library (for analysis)
# 2. The openai library (for Whisper transcription)
# You must install both: pip install google-genai openai streamlit

#----- Gemini API Helper ----#
import google.generativeai as genai

def get_gemini_analysis(transcript, domain, round_type, feedback_tone, model_name, api_key):
    """Configures the Gemini client and requests the interview analysis."""
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        # Handle configuration error if API key is malformed
        st.error(f"Failed to configure Gemini API: {e}")
        return None

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
    
    # Use the selected model name
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        # Catch API errors like invalid model name or connection issues
        st.error(f"Gemini generation failed: {e}")
        return None


#----------- Whisper Transcription -----------#
def transcribe_with_whisper(audio_path):
    """
    Transcribes the audio file using the OpenAI Whisper API.
    Requires the 'openai' library and an OPENAI_API_KEY environment variable.
    """
    try:
        import openai
        # IMPORTANT: Replace the dummy key with a way to securely set your actual OpenAI key
        openai.api_key = os.environ.get("OPENAI_API_KEY", "") 

        if not openai.api_key:
             st.error("OpenAI API Key not set. Please set the OPENAI_API_KEY environment variable to use audio transcription.")
             return None

        with open(audio_path, "rb") as f:
            # Using the recommended standard Whisper model
            transcript_response = openai.audio.transcriptions.create(
                model="whisper-1", 
                file=f
            )
        return transcript_response.text
    except ImportError:
        st.error("The 'openai' library is not installed. Please install it to use audio transcription (pip install openai).")
        return None
    except Exception as e:
        st.error(f"Whisper Transcription failed: {str(e)}")
        return None

#------------- Streamlit UI --------------#
st.set_page_config(page_title="AI Interview Analyzer", layout="wide")

st.title("🎤 AI Interview/Discussion Analyzer")
st.write("Upload an audio file (interview/discussion) or paste a text transcript. Get instant, structured feedback from an AI mentor. Powered by the Gemini API.")

st.markdown("---")

# Layout columns for settings
col1, col2, col3 = st.columns(3)

with col1:
    domain = st.selectbox("Select Domain/Context:", ["Tech", "Managerial", "HR", "Group Discussion", "Other"])
with col2:
    round_type = st.selectbox("Round Type:", ["Screening", "Technical", "Managerial", "HR", "Final", "Other"])
with col3:
    feedback_tone = st.selectbox("Feedback Tone:", ["Professional", "Encouraging", "Critical", "Neutral"])

st.markdown("---")

# Gemini API config
with st.expander("🔐 Gemini API Configuration & Model"):
    # Using the non-pro flash model as requested for this fast analysis task
    gemini_model = "gemini-2.5-flash"
    st.markdown(f"**Gemini Model Used:** `{gemini_model}` (High-speed, cost-effective analysis)")
    api_key = st.text_input("Enter your Gemini API key (Safe: Not shared or stored online):", type="password")


input_mode = st.radio("Input Mode:", ["Text Transcript", "Audio File (auto transcription)"])

transcript = None
audio_file_path = None

if input_mode == "Audio File (auto transcription)":
    st.warning("Note: Audio transcription uses the **OpenAI Whisper API** and requires the `OPENAI_API_KEY` environment variable to be set.")
    audio_file = st.file_uploader("Upload audio (.wav, .mp3, .m4a)", type=["wav", "mp3", "m4a"])
    
    if audio_file is not None:
        # Save file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=audio_file.name.split('.')[-1]) as tmp:
            tmp.write(audio_file.read())
            audio_file_path = tmp.name
        
        st.info("Transcribing audio file (Whisper)...")
        transcript = transcribe_with_whisper(audio_file_path)
        
        # Clean up temporary file
        if audio_file_path and os.path.exists(audio_file_path):
            os.remove(audio_file_path)

        if transcript:
            st.success("Transcription successful. Review the text below:")
            st.text_area("Transcript (Editable)", transcript, height=200)

elif input_mode == "Text Transcript":
    transcript = st.text_area("Paste full transcript here:", 
                              placeholder="e.g., Interviewer: Can you describe a challenging project?\nCandidate: I once had to integrate a new system...",
                              height=300)

analyze_btn = st.button("🔎 Analyze Interview/Discussion", disabled=not transcript or not api_key)

st.markdown("---")

if analyze_btn and transcript and api_key:
    # Use a spinner for better UX while waiting
    with st.spinner(f"Analyzing with {gemini_model}, please wait..."):
        response = get_gemini_analysis(transcript, domain, round_type, feedback_tone, gemini_model, api_key)
        
        if response:
            st.success("Analysis Complete!")
            st.subheader("📝 Interview/Discussion Feedback Report:")
            st.markdown(response)

st.markdown("---")
st.caption("Built for interview and group discussion feedback | Analysis powered by Gemini API | Audio transcription requires OpenAI Whisper API.")
