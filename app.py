import streamlit as st
import requests
import json
import plotly.express as px
import pandas as pd

# --- GEMINI CONFIGURATION ---
# Using the recommended, active model for fast, general tasks
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_API_VERSION = "v1"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/{GEMINI_API_VERSION}/models/{GEMINI_MODEL}:generateContent"

# --- CORE FUNCTION: CALL GEMINI API ---

def call_gemini(prompt, api_key, format_json=False):
    """
    Calls the Gemini API with a given prompt.
    If format_json is True, it forces the model to output valid JSON.
    """
    headers = {"Content-Type": "application/json"}
    
    # 1. Start with the mandatory contents field
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
    }
    
    # 2. Add the configuration field ONLY if requesting JSON format
    if format_json:
        # **CORRECTION**: 'generationConfig' must be at the root level alongside 'contents'
        data["generationConfig"] = {
            "responseMimeType": "application/json"
            # We rely on the detailed instruction in the prompt to define the schema
        }
        
    endpoint = f"{GEMINI_API_URL}?key={api_key}"
    
    try:
        # Pass the constructed data dictionary to requests.post
        res = requests.post(endpoint, headers=headers, json=data)
        res.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        output = res.json()
        
        # Safely extract the text response
        # Note: For JSON output, the text contains the JSON string
        return output["candidates"][0]["content"]["parts"][0]["text"]
    except requests.exceptions.HTTPError as http_err:
        error_msg = f"Gemini API HTTP Error: {http_err} - {res.text}"
        st.error(error_msg)
        return json.dumps({"error": error_msg}) if format_json else error_msg
    except Exception as e:
        error_msg = f"Gemini API Error or parsing issue: {e}"
        st.error(error_msg)
        return json.dumps({"error": error_msg}) if format_json else error_msg

# --- PROMPT TEMPLATE AND PARSING ---

JSON_SCHEMA = """
{
    "summary": "Overall summary of the discussion.",
    "key_points_speaker_A": "Key points mentioned by Speaker A.",
    "key_points_speaker_B": "Key points mentioned by Speaker B.",
    "feedback_speaker_A": {
        "confidence_score": "Score (1-10)",
        "clarity_score": "Score (1-10)",
        "empathy_score": "Score (1-10)",
        "sentiment_label": "Positive/Neutral/Negative",
        "improvement_suggestions": ["Suggestion 1", "Suggestion 2"]
    },
    "feedback_speaker_B": {
        "confidence_score": "Score (1-10)",
        "clarity_score": "Score (1-10)",
        "empathy_score": "Score (1-10)",
        "sentiment_label": "Positive/Neutral/Negative",
        "improvement_suggestions": ["Suggestion 1", "Suggestion 2"]
    }
}
"""

def generate_analysis_prompt(transcript, domain, round_type):
    """Generates the detailed, structured prompt for Gemini."""
    
    prompt = f"""
    You are an expert AI Interview Analyzer and mentor. Your task is to analyze the following conversation transcript.
    The domain is '{domain}' and the round type is '{round_type}'.
    
    Analyze the conversation for two speakers, 'Speaker A' and 'Speaker B'.
    
    Perform the following analysis for both speakers:
    1. Overall Sentiment and Tone (Positive, Neutral, Negative, Confident, Nervous, etc.).
    2. Clarity and Coherence Score (1-10).
    3. Empathy and Emotional Intelligence Score (1-10).
    4. Confidence Score (1-10).
    5. Specific, actionable improvement suggestions.
    
    Conversation Transcript:
    ---
    {transcript}
    ---
    
    Generate the final analysis STRICTLY as a single JSON object that conforms to this schema. DO NOT include any text outside the JSON block.
    {JSON_SCHEMA}
    """
    return prompt

# --- STREAMLIT APP LAYOUT ---

st.set_page_config(page_title="AI Interview Analyzer (Gemini)", layout="wide")
st.title("🗣️ AI Interview Analyzer (Gemini 2.5 Flash)")
st.markdown("---")

st.markdown("""
    **Purpose:** Generate instant, structured feedback on interview/discussion transcripts, acting as an AI mentor.
    This prototype supports analysis for two speakers (Speaker A and Speaker B).
""")

# 1. Input Section
st.sidebar.header("Analyzer Configuration")

# Context Selection
domain = st.sidebar.selectbox("Domain/Industry", ["Tech/IT", "Managerial/Leadership", "HR/Behavioral", "Sales/Marketing", "General Discussion"], index=0)
round_type = st.sidebar.selectbox("Round Type", ["One-on-One Interview", "Group Discussion", "Presentation/Pitch"], index=0)

# API Key
if "GEMINI_API_KEY" in st.secrets:
    gemini_api_key = st.secrets["GEMINI_API_KEY"]
    st.sidebar.success("Gemini API Key loaded from `st.secrets`.")
else:
    gemini_api_key = st.sidebar.text_input("Enter your Gemini API Key:", type="password")

# --- Sample Transcript for quick demo ---
SAMPLE_TRANSCRIPT = """
Speaker A: Good morning. Thank you for taking the time to meet with me. I'm really excited about this Senior Developer role.
Speaker B: Good morning. Let's dive right in. Can you describe a complex technical challenge you've solved recently?
Speaker A: (Hesitantly) Uh, yes. So, we had this legacy system... it was, uh, really slow. I felt very nervous about touching it, honestly. But, I decided to refactor the database connection layer entirely. It took longer than expected, but eventually, we reduced the latency by almost 40%. It showed me the value of, like, persistence.
Speaker B: That sounds impressive. How did you communicate the risks and the timeline to your non-technical stakeholders?
Speaker A: I made sure to meet them daily. I used simple analogies, saying things like, "We're replacing the old engine with a new one," so they wouldn't feel confused. It's important to be empathetic to their lack of technical knowledge.
Speaker B: I agree. That demonstrates strong emotional intelligence.
"""

transcript_input = st.text_area(
    "Paste Conversation Transcript Here (Max ~200 lines for free model)",
    SAMPLE_TRANSCRIPT,
    height=300,
    help="Format the transcript with speaker labels (e.g., 'Speaker A: ...', 'Speaker B: ...')"
)

analyze_button = st.button("🧠 Analyze Conversation", type="primary")

# --- 2. Analysis & Output Section ---
if analyze_button and transcript_input and gemini_api_key:
    if not gemini_api_key.startswith("AIza"):
        st.error("Invalid or incomplete Gemini API Key. Please check your input.")
        st.stop()
        
    st.header("Detailed Analysis Report")
    st.markdown("---")
    
    with st.spinner("🚀 AI Mentor is analyzing the conversation..."):
        prompt = generate_analysis_prompt(transcript_input, domain, round_type)
        # Request JSON output here
        json_output = call_gemini(prompt, gemini_api_key, format_json=True)

    try:
        report_data = json.loads(json_output)
        
        if "error" in report_data:
            st.error(f"Analysis failed. Please check the logs or try a shorter transcript. Error: {report_data['error']}")
            st.stop()

        # --- Overall Summary ---
        st.subheader("📝 Overall Discussion Summary")
        st.info(report_data.get('summary', 'N/A'))
        
        # --- Key Points ---
        col_kp_a, col_kp_b = st.columns(2)
        with col_kp_a:
            st.markdown("#### Key Points: Speaker A")
            st.markdown(report_data.get('key_points_speaker_A', 'N/A'))
        with col_kp_b:
            st.markdown("#### Key Points: Speaker B (Interviewer/Partner)")
            st.markdown(report_data.get('key_points_speaker_B', 'N/A'))

        st.markdown("---")

        # --- Detailed Feedback & Visualizations ---
        st.subheader("📈 Quantitative Scorecard & Feedback")
        
        # Prepare Data for Scores
        scores_a = report_data['feedback_speaker_A']
        scores_b = report_data['feedback_speaker_B']
        
        # Ensure scores are integers before plotting
        metrics_df = pd.DataFrame({
            'Metric': ['Confidence', 'Clarity', 'Empathy'],
            'Speaker A': [int(scores_a['confidence_score']), int(scores_a['clarity_score']), int(scores_a['empathy_score'])],
            'Speaker B': [int(scores_b['confidence_score']), int(scores_b['clarity_score']), int(scores_b['empathy_score'])]
        })
        
        # 1. Score Metrics
        col_a_sum, col_b_sum = st.columns(2)
        
        with col_a_sum:
            st.markdown("### Speaker A (Candidate)")
            st.metric("Sentiment/Tone", scores_a['sentiment_label'])
        with col_b_sum:
            st.markdown("### Speaker B (Interviewer/Partner)")
            st.metric("Sentiment/Tone", scores_b['sentiment_label'])

        # 2. Radar Chart
        st.markdown("##### Performance Score Radar Chart")
        fig = px.line_polar(
            metrics_df.melt(id_vars='Metric', var_name='Speaker', value_name='Score'), 
            r='Score', 
            theta='Metric', 
            color='Speaker', 
            line_close=True,
            range_r=[0, 10]
        )
        fig.update_traces(fill='toself')
        st.plotly_chart(fig, use_container_width=True)

        # 3. Improvement Suggestions
        st.markdown("---")
        st.subheader("⭐ Areas for Improvement (AI Mentor Guidance)")
        
        col_sugg_a, col_sugg_b = st.columns(2)
        
        with col_sugg_a:
            st.markdown("#### Suggestions for Speaker A:")
            for item in scores_a['improvement_suggestions']:
                st.markdown(f"- **{item}**")
        with col_sugg_b:
            st.markdown("#### Suggestions for Speaker B:")
            for item in scores_b['improvement_suggestions']:
                st.markdown(f"- **{item}**")
                
    except json.JSONDecodeError:
        st.error("AI Analysis failed to return valid JSON. This usually happens if the model response format is broken. Please check the raw output below.")
        st.code(json_output) # Show raw output for debugging
    except KeyError as ke:
        st.error(f"AI Analysis failed to find expected key in JSON structure. Key missing: {ke}. Please check the raw output below.")
        st.code(json_output)

elif analyze_button and not gemini_api_key:
    st.error("Please enter your Gemini API Key to run the analysis.")
