"""
IntiqAI - Main Streamlit Application

AI-powered voice-based technical interview system.
Run with: streamlit run app.py
"""

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

import sys
import os

# --- SET API KEY BEFORE IMPORTING WORKFLOW ---
from dotenv import load_dotenv
load_dotenv()

# Ensure Google API key is set from environment variables
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError(
        "GOOGLE_API_KEY environment variable is not set. "
        "Please create a .env file with your API key. "
        "See .env.example for reference."
    )


# ============================================================================
# STANDARD IMPORTS
# ============================================================================

import streamlit as st
import pandas as pd
import tempfile
import hashlib
import time
from datetime import datetime
import pytz
from concurrent.futures import ThreadPoolExecutor, as_completed

# Audio recorder
from audio_recorder_streamlit import audio_recorder

# LangChain messages
from langchain_core.messages import HumanMessage, AIMessage


# ============================================================================
# LOCAL IMPORTS
# ============================================================================

# CV Filtering module
from cv_filtering import (
    Config,
    make_llm,
    llm_json,
    JD_FIX_PROMPT,
    evaluate_single_resume,
    results_to_dataframe,
    save_excels,
    save_jd_pdf,
    extract_position_from_jd,
    normalize_improved_jd,
    load_candidates_from_excel,
    sort_candidates,
    read_pdf_text_from_path,
    clean_whitespace,
    ensure_dir,
)

# Interview session module
from interview_session import (
    get_audio_duration,
    prepare_audio_for_playback,
    transcribe_audio,
    generate_tts,
    USE_OPTIMIZED,
)

# Workflow
from src.dynamic_workflow import build_workflow, AgentState

# Timing instrumentation
from src.timing_instrumentation import ensure_session, timing_start, timing_end, TIME_REPORT_DIR
from src.pdf_utils_time_analysis import generate_time_analysis_pdf

# Fundamental Knowledge Assessment modules
import sys
sys.path.append("Fundamental knowledge Assessment")
from core.evaluator import EvaluationEngine
from core.question_generator import QuestionGenerator
from core.jd_parser import JDParser
import json


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(page_title="IntiqAI", layout="wide", initial_sidebar_state="expanded")


# ============================================================================
# CUSTOM CSS
# ============================================================================

CUSTOM_CSS = """
<style>
    /* Main container styling */
    .main {
        padding: 0rem 1rem;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }

    .main-header p {
        color: rgba(255, 255, 255, 0.9);
        margin: 0.5rem 0 0 0;
        font-size: 1.2rem;
    }

    /* Hide Streamlit's default footer */
    footer {visibility: hidden;}
    .stApp > footer {visibility: hidden;}

    /* Smooth scrolling */
    html {
        scroll-behavior: smooth;
    }

    /* Better spacing for main content */
    .main .block-container {
        padding-bottom: 2rem;
    }

    /* Transcript styling */
    .transcript-container {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(0, 0, 0, 0.05);
    }

    /* Ensure page is scrollable */
    .stApp {
        overflow-y: auto;
    }

    /* Message styling */
    .candidate-message {
        background: #f7f7f7;
        padding: 1rem 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }

    .candidate-message strong {
        color: #667eea !important;
    }

    .candidate-message {
        color: #333 !important;
    }

    .recruiter-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 4px solid #764ba2;
        color: white !important;
    }

    .recruiter-message strong {
        color: white !important;
    }

    .recruiter-message span {
        color: white !important;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: #f8f9fa;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }

    /* Audio player styling */
    .stAudio {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }

    /* Spinner styling */
    .stSpinner > div {
        border-color: #667eea;
    }

    /* Info box styling */
    .stAlert {
        border-radius: 10px;
        border: none;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
    }

    /* Input field styling */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        transition: all 0.3s ease;
    }

    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }

    /* Selectbox styling */
    .stSelectbox > div > div {
        border-radius: 8px;
    }

    /* Number input styling */
    .stNumberInput > div > div > input {
        border-radius: 8px;
    }

    /* File uploader styling */
    .stFileUploader {
        border-radius: 10px;
    }

    /* Success/Error messages */
    .stSuccess {
        border-left: 4px solid #10b981;
    }

    .stError {
        border-left: 4px solid #ef4444;
    }

    .stWarning {
        border-left: 4px solid #f59e0b;
    }

    .stInfo {
        border-left: 4px solid #3b82f6;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================================
# HEADER
# ============================================================================

st.markdown("""
<div class="main-header">
    <h1>IntiqAI</h1>
    <p>AI-powered voice-based technical interviews</p>
</div>
""", unsafe_allow_html=True)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "app_page": "cv_filtering_setup",
        "candidates_df": None,
        "selected_candidate_index": None,
        "state": None,
        "last_processed_audio": None,
        "job_description": "",
        "position": "AI Specialist",
        "filtering_in_progress": False,
        "filtering_status": "",
        "excel_output_path": None,
        "all_results_path": None,
        "time_analysis_generated": False,
        "stt_model": "whisper_base",
        "stt_model_label": "Whisper Base (balanced)",
        "use_tts_cache": True,
        "timings": [],
        "turn_idx": 0,
        "resumes_dir": "",
        "fundamental_assessment_completed": False,
        "fundamental_assessment_results": None,
        "fundamental_questions": [],
        "fundamental_responses": [],
        "fundamental_evaluation": None,
        "fundamental_questions_generated": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ============================================================================
# WORKFLOW CACHING
# ============================================================================

@st.cache_resource
def get_workflow():
    """Get cached workflow instance."""
    return build_workflow()


workflow = get_workflow()

# Initialize timing state
ensure_session(st.session_state)


# ============================================================================
# UI HELPER FUNCTIONS
# ============================================================================

def play_audio_html5(audio_path: str, wait: bool = True) -> float:
    """
    Play audio using Streamlit's native audio player with autoplay.

    Args:
        audio_path: Path to the audio file
        wait: Whether to wait for audio to finish playing

    Returns:
        Duration of the audio in seconds
    """
    if not os.path.exists(audio_path):
        st.error("Audio file not found")
        return 0.0

    try:
        audio_bytes, audio_format, duration = prepare_audio_for_playback(audio_path)

        if audio_bytes is None:
            st.error("Failed to prepare audio for playback")
            return 0.0

        # Show visual indicator
        st.markdown("""
        <div style="padding: 12px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 8px; margin-bottom: 10px;">
            <div style="color: white; font-weight: bold; font-size: 14px;">
                🔊 Recruiter Voice Response
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Use Streamlit's native audio player with autoplay
        st.audio(audio_bytes, format=audio_format, autoplay=True)

        # Wait for audio to finish playing if requested
        if wait:
            time.sleep(duration + 1)

        return duration

    except Exception as e:
        st.error(f"Error playing audio: {str(e)}")
        return 0.0


# ============================================================================
# PAGE: CV FILTERING SETUP
# ============================================================================

def render_cv_filtering_setup_page():
    """Render the CV filtering setup page."""
    st.markdown("""
    <div style="background: white; padding: 2rem; border-radius: 15px; margin-bottom: 2rem; box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
        <h2 style="color: #333; margin-bottom: 1rem;">📋 CV Filtering Setup</h2>
        <p style="color: #666;">Configure the CV filtering process by providing the directory with CV PDF files and the job description.</p>
    </div>
    """, unsafe_allow_html=True)

    # Directory path input with file browser helper
    def browse_directory():
        """Open a directory picker dialog using tkinter"""
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)

            directory = filedialog.askdirectory(
                title="Select Directory with CV PDF Files",
                initialdir=st.session_state.get("resumes_dir", os.getcwd())
            )

            root.destroy()
            return directory
        except Exception as e:
            st.error(f"Error opening directory browser: {str(e)}")
            return None

    def on_browse_click():
        """Callback to handle directory selection before UI reruns"""
        selected_dir = browse_directory()
        if selected_dir:
            st.session_state.resumes_dir = selected_dir
            st.session_state.resumes_dir_input = selected_dir

    col1, col2 = st.columns([4, 1])
    with col1:
        resumes_dir = st.text_input(
            "📁 Directory Path with CV PDF Files:",
            value=st.session_state.resumes_dir,
            placeholder=r"e.g., C:\Users\96658\Desktop\TalentTalk\uploaded_resumes",
            help="Enter the full path to the directory containing the CV PDF files, or click 'Browse' to select a directory",
            key="resumes_dir_input"
        )

        if resumes_dir != st.session_state.resumes_dir:
            st.session_state.resumes_dir = resumes_dir

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.button("📂 Browse", on_click=on_browse_click, use_container_width=True, help="Open file browser", key="browse_dir_btn")

    # Path validation and directory listing
    current_dir = st.session_state.resumes_dir

    if current_dir:
        if os.path.exists(current_dir) and os.path.isdir(current_dir):
            try:
                pdf_files = [f for f in os.listdir(current_dir) if f.lower().endswith('.pdf')]
                if pdf_files:
                    st.success(f"✅ Directory found with {len(pdf_files)} PDF file(s)")
                    with st.expander(f"📋 View PDF files in directory ({len(pdf_files)} files)"):
                        for pdf_file in sorted(pdf_files)[:20]:
                            st.text(f"  • {pdf_file}")
                        if len(pdf_files) > 20:
                            st.text(f"  ... and {len(pdf_files) - 20} more files")
                else:
                    st.warning("⚠️ Directory exists but contains no PDF files")
            except Exception as e:
                st.error(f"❌ Error reading directory: {str(e)}")
        elif os.path.exists(current_dir):
            st.error("❌ Path exists but is not a directory")
        else:
            st.error("❌ Directory path does not exist")

    # Job description PDF upload
    st.markdown("**📝 Job Description (PDF File):**")
    jd_file = st.file_uploader(
        "Upload Job Description PDF",
        type=["pdf"],
        help="Upload the job description as a PDF file",
        label_visibility="collapsed"
    )

    # Extract text from uploaded PDF
    job_description = ""
    if jd_file is not None:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(jd_file.read())
                tmp_path = tmp_file.name

            job_description = read_pdf_text_from_path(tmp_path)
            job_description = clean_whitespace(job_description)

            os.unlink(tmp_path)

            if job_description:
                st.success(f"✅ Job description loaded successfully ({len(job_description)} characters)")
                with st.expander("📄 Preview Job Description (first 500 characters)"):
                    st.text(job_description[:500] + "..." if len(job_description) > 500 else job_description)
        except Exception as e:
            st.error(f"❌ Error reading PDF file: {str(e)}")
            job_description = ""
    elif st.session_state.get("job_description"):
        job_description = st.session_state.job_description
        st.info("ℹ️ Using previously loaded job description. Upload a new PDF to replace it.")

    # Start filtering button
    start_filtering = st.button("🚀 Start CV Filtering Process", type="primary", use_container_width=True)

    if start_filtering:
        if not resumes_dir or not os.path.exists(resumes_dir):
            st.error("❌ Please enter a valid directory path.")
        elif not job_description or len(job_description.strip()) < 50:
            st.error("❌ Please upload a valid job description PDF file (at least 50 characters after extraction).")
        else:
            st.session_state.resumes_dir = resumes_dir
            st.session_state.job_description = job_description
            st.session_state.filtering_in_progress = True
            st.session_state.app_page = "filtering_process"
            st.rerun()

    # If we already have candidates loaded, show option to go to candidate list
    if st.session_state.candidates_df is not None:
        st.markdown("---")
        if st.button("👥 View Candidate List", use_container_width=True):
            st.session_state.app_page = "candidate_list"
            st.rerun()


# ============================================================================
# PAGE: FILTERING PROCESS
# ============================================================================

def render_filtering_process_page():
    """Render the filtering process page."""
    st.markdown("""
    <div style="background: white; padding: 2rem; border-radius: 15px; margin-bottom: 2rem; box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
        <h2 style="color: #333; margin-bottom: 1rem;">🔄 CV Filtering in Progress</h2>
    </div>
    """, unsafe_allow_html=True)

    status_placeholder = st.empty()
    progress_placeholder = st.empty()

    try:
        # Setup configuration
        out_dir = os.path.join(os.getcwd(), "Filtered Resumes")
        improved_jd_dir = os.path.join(os.getcwd(), "Improved Job Descriptions")
        ensure_dir(out_dir)
        ensure_dir(improved_jd_dir)

        cfg = Config(
            jd_text=st.session_state.job_description,
            resumes_dir=st.session_state.resumes_dir,
            out_dir=out_dir,
            Improved_JD_dir=improved_jd_dir,
            model="gemini-2.5-flash",
            temperature=0.2,
            high_score_threshold=8
        )

        # Initialize LLM
        status_placeholder.info("🔧 Initializing AI model...")
        llm = make_llm(cfg)

        # Step 1: Fix and improve JD
        status_placeholder.info("📝 Processing and improving job description...")
        jd_fix_result = llm_json(llm, JD_FIX_PROMPT, jd_text=cfg.jd_text)

        raw_improved = jd_fix_result.get("improved_job_description", "")
        fixed_jd = normalize_improved_jd(raw_improved)

        must_haves = jd_fix_result.get("must_have_requirements", []) or []
        nice_haves = jd_fix_result.get("nice_to_have_requirements", []) or []

        # Normalize possible string outputs into lists
        if isinstance(must_haves, str):
            must_haves = [x.strip("-• \t") for x in must_haves.splitlines() if x.strip()]
        if isinstance(nice_haves, str):
            nice_haves = [x.strip("-• \t") for x in nice_haves.splitlines() if x.strip()]

        # Extract position from JD
        status_placeholder.info("🔍 Extracting job position...")
        position = extract_position_from_jd(llm, fixed_jd)
        st.session_state.position = position

        # Save improved JD
        improved_jd_pdf_path = save_jd_pdf(fixed_jd, must_haves, nice_haves, cfg.Improved_JD_dir)
        status_placeholder.success(f"✅ Job description improved and saved to: {improved_jd_pdf_path}")

        # Step 2: Load resumes
        status_placeholder.info("📚 Loading CV files...")
        resumes = []
        pdf_files = [f for f in os.listdir(cfg.resumes_dir) if f.lower().endswith(".pdf")]

        if not pdf_files:
            st.error(f"❌ No PDF files found in {cfg.resumes_dir}")
            st.session_state.app_page = "cv_filtering_setup"
            st.rerun()

        for fname in sorted(pdf_files):
            path = os.path.join(cfg.resumes_dir, fname)
            text = clean_whitespace(read_pdf_text_from_path(path))
            resumes.append({"name": fname, "path": path, "text": text})

        status_placeholder.success(f"✅ Loaded {len(resumes)} CV files")

        # Step 3: Skip Google Drive upload
        uploaded_resumes = []
        for r in resumes:
            uploaded_resumes.append({"name": r["name"], "text": r["text"], "link": r["name"]})

        # Step 4: Evaluate resumes
        status_placeholder.info("🤖 Evaluating CVs against job description...")
        results = []

        def process_item(item):
            return evaluate_single_resume(llm, fixed_jd, must_haves, nice_haves, item["text"], item["link"])

        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = {executor.submit(process_item, item): item for item in uploaded_resumes}
            completed = 0
            total = len(futures)
            for f in as_completed(futures):
                completed += 1
                progress_placeholder.progress(completed / total, text=f"Evaluating CV {completed}/{total}...")
                results.append(f.result())

        # Step 5: Save Excel files
        status_placeholder.info("💾 Saving results to Excel...")
        df = results_to_dataframe(results)
        all_path, high_path = save_excels(df, cfg.out_dir, cfg.high_score_threshold)

        # Load the high scoring candidates
        st.session_state.candidates_df = load_candidates_from_excel(high_path, error_callback=st.error)
        if st.session_state.candidates_df is not None:
            st.session_state.candidates_df = sort_candidates(st.session_state.candidates_df)
            st.session_state.excel_output_path = high_path
            st.session_state.all_results_path = all_path
            status_placeholder.success(f"✅ Filtering complete! Found {len(st.session_state.candidates_df)} high-scoring candidates.")
            st.success(f"📊 Results saved to:\n- All results: {all_path}\n- High scoring: {high_path}")
            st.session_state.app_page = "candidate_list"
            st.rerun()
        else:
            st.error("❌ Failed to load candidates from Excel file.")
            st.session_state.app_page = "cv_filtering_setup"
            st.rerun()

    except Exception as e:
        st.error(f"❌ Error during filtering: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        if st.button("🔙 Back to Setup"):
            st.session_state.app_page = "cv_filtering_setup"
            st.rerun()


# ============================================================================
# PAGE: CANDIDATE LIST
# ============================================================================

def render_candidate_list_page():
    """Render the candidate list page."""
    if st.session_state.candidates_df is None:
        st.warning("No candidates loaded. Please run CV filtering first.")
        if st.button("🔙 Back to Setup"):
            st.session_state.app_page = "cv_filtering_setup"
            st.rerun()
        return

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
        <div style="background: white; padding: 2rem; border-radius: 15px; margin: 2rem 0; box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
            <h2 style="color: #333; margin-bottom: 1.5rem;">👥 Candidate List</h2>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔙 Back to Setup", use_container_width=True):
            st.session_state.app_page = "cv_filtering_setup"
            st.rerun()

    for index, row in st.session_state.candidates_df.iterrows():
        st.markdown(f"""
        <div style="background: white; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.05); border: 1px solid #f0f0f0;">
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([3, 1, 1])

        with col1:
            st.markdown(f"### 👤 {row['First Name']} {row['Last Name']}")

        with col2:
            # Show CV Filtering Score
            if 'Overall Fit' in st.session_state.candidates_df.columns:
                cv_score = row.get('Overall Fit', 'N/A')
                if cv_score != 'N/A':
                    st.metric("CV Score", f"{cv_score}/10")
                else:
                    st.write("CV Score: N/A")
            
            # Show Fundamental Assessment Score if available
            if 'Fundamental Knowledge Score' in st.session_state.candidates_df.columns:
                fundamental_score = row.get('Fundamental Knowledge Score')
                if pd.notna(fundamental_score) and isinstance(fundamental_score, (int, float)):
                    score_color = "green" if fundamental_score >= 80 else "red"
                    st.markdown(f"""
                    <div style="background: {'#d1fae5' if fundamental_score >= 80 else '#fef3c7'}; 
                                padding: 0.5rem; border-radius: 8px; margin-top: 0.5rem; 
                                border-left: 4px solid {'#10b981' if fundamental_score >= 80 else '#f59e0b'};">
                        <strong>Fundamental Score:</strong> {int(fundamental_score)}/100<br>
                        <small>Minimum: 80</small>
                    </div>
                    """, unsafe_allow_html=True)

        with col3:
            # Check if assessment already completed for this candidate
            assessment_completed = False
            score_meets_requirement = False
            
            if st.session_state.selected_candidate_index == index:
                assessment_completed = st.session_state.get("fundamental_assessment_completed", False)
                if assessment_completed:
                    evaluation = st.session_state.get("fundamental_evaluation")
                    if evaluation:
                        score = evaluation.get('overall_score', 0)
                        score_meets_requirement = score >= 80
            
            # Also check Excel file for score if available
            if not score_meets_requirement and 'Fundamental Knowledge Score' in st.session_state.candidates_df.columns:
                candidate_score = row.get('Fundamental Knowledge Score')
                if pd.notna(candidate_score) and isinstance(candidate_score, (int, float)):
                    score_meets_requirement = candidate_score >= 80
                    assessment_completed = True
            
            if assessment_completed and score_meets_requirement:
                if st.button(f"Proceed to Interview", key=f"interview_{index}", use_container_width=True):
                    st.session_state.selected_candidate_index = index
                    candidate_info = st.session_state.candidates_df.loc[index]
                    
                    resume_text = candidate_info.get('Full Resume', '')
                    if not resume_text or len(str(resume_text).strip()) < 50:
                        st.error(f"Resume text is missing or too short for candidate {candidate_info.get('First Name', '')} {candidate_info.get('Last Name', '')}. Please check the Excel file.")
                        st.stop()
                    
                    position = st.session_state.get("position", "AI Specialist")
                    if not position or position.strip() == "":
                        position = "AI Specialist"
                    
                    st.session_state.state = AgentState(
                        mode="friendly",
                        num_of_q=2,
                        num_of_follow_up=1,
                        position=position,
                        company_name="Prince Mogrin University",
                        messages=[],
                        evaluation_result="",
                        report="",
                        pdf_path=None,
                        resume_path=None,
                        questions_path=None,
                        resume_text=str(resume_text)
                    )
                    
                    st.session_state.app_page = "interview"
                    st.session_state.timings = []
                    st.session_state.turn_idx = 0
                    st.session_state.time_analysis_generated = False
                    st.session_state.auto_start_interview = True
                    st.session_state.interview_stage = "switching"
                    st.rerun()
            elif assessment_completed and not score_meets_requirement:
                # Show that assessment was completed but score is too low - allow viewing results
                if st.button(f"View Assessment Results", key=f"view_assessment_{index}", use_container_width=True):
                    st.session_state.selected_candidate_index = index
                    # Try to load evaluation from session state first
                    evaluation = st.session_state.get("fundamental_evaluation")
                    if evaluation and st.session_state.get("fundamental_assessment_completed", False):
                        # Evaluation already in session state, just navigate
                        st.session_state.app_page = "fundamental_assessment"
                        st.rerun()
                    else:
                        # Load from Excel if available
                        candidate_score = row.get('Fundamental Knowledge Score')
                        if pd.notna(candidate_score):
                            # Set assessment as completed and navigate to view results
                            st.session_state.fundamental_assessment_completed = True
                            # Create a basic evaluation structure from Excel data
                            st.session_state.fundamental_evaluation = {
                                'overall_score': int(candidate_score),
                                'recommendation': row.get('Fundamental Recommendation', 'Not Recommended (Rejection)'),
                                'qualitative_feedback': f"Assessment completed with a score of {int(candidate_score)}/100. This score does not meet the minimum requirement of 80/100.",
                                'strengths': [],
                                'weaknesses': ['Score below minimum requirement'],
                                'criterion_scores': {
                                    'technical_accuracy': int(candidate_score),
                                    'completeness': int(candidate_score),
                                    'relevance': int(candidate_score),
                                    'practicality': int(candidate_score)
                                },
                                'question_scores': {}
                            }
                            # Try to load question scores from Excel if available
                            question_scores_str = row.get('Fundamental Question Scores')
                            if pd.notna(question_scores_str) and question_scores_str:
                                try:
                                    import json
                                    st.session_state.fundamental_evaluation['question_scores'] = json.loads(str(question_scores_str))
                                except:
                                    pass
                        st.session_state.app_page = "fundamental_assessment"
                        st.rerun()
            else:
                if st.button(f"Begin Assessment", key=f"assessment_{index}", use_container_width=True):
                    st.session_state.selected_candidate_index = index
                    st.session_state.fundamental_assessment_completed = False
                    st.session_state.fundamental_questions_generated = False
                    st.session_state.fundamental_questions = []
                    st.session_state.fundamental_responses = []
                    st.session_state.fundamental_evaluation = None
                    st.session_state.app_page = "fundamental_assessment"
                    st.rerun()

        # Add justification in an expander below each candidate
        if 'Justification' in st.session_state.candidates_df.columns:
            justification = row.get('Justification', 'No justification available.')
            if justification and str(justification).strip() and str(justification) != 'nan':
                with st.expander("📋 View Justification", expanded=False):
                    st.markdown(f"""
                    <div style="background: #f9fafb; padding: 1rem; border-radius: 8px; border-left: 4px solid #667eea;
                                color: #333; line-height: 1.6;">
                        {str(justification)}
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # Add section for non-qualified candidates
    if st.session_state.all_results_path and os.path.exists(st.session_state.all_results_path):
        try:
            all_results_df = pd.read_excel(st.session_state.all_results_path)

            qualified_names = set()
            if st.session_state.candidates_df is not None:
                for _, row in st.session_state.candidates_df.iterrows():
                    first_name = str(row.get('First Name', '')).strip()
                    last_name = str(row.get('Last Name', '')).strip()
                    qualified_names.add((first_name.lower(), last_name.lower()))

            non_qualified = []
            has_justification_col = 'Justification' in all_results_df.columns
            for _, row in all_results_df.iterrows():
                first_name = str(row.get('First Name', '')).strip()
                last_name = str(row.get('Last Name', '')).strip()
                name_key = (first_name.lower(), last_name.lower())
                if name_key not in qualified_names:
                    if has_justification_col:
                        justification = row.get('Justification', 'No justification available.')
                        justification_text = justification if justification and str(justification).strip() and str(justification) != 'nan' else 'No justification available.'
                    else:
                        justification_text = 'No justification available.'
                    non_qualified.append({
                        'First Name': first_name,
                        'Last Name': last_name,
                        'Justification': justification_text
                    })

            if non_qualified:
                st.markdown("---")
                st.markdown("""
                <div style="background: white; padding: 2rem; border-radius: 15px; margin: 2rem 0; box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
                    <h2 style="color: #333; margin-bottom: 1.5rem;">❌ Non-Qualified Candidates</h2>
                </div>
                """, unsafe_allow_html=True)

                for idx, candidate in enumerate(non_qualified):
                    st.markdown(f"""
                    <div style="background: #fef2f2; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;
                                box-shadow: 0 2px 5px rgba(0,0,0,0.05); border: 1px solid #fecaca; border-left: 4px solid #ef4444;">
                    """, unsafe_allow_html=True)

                    st.markdown(f"### 👤 {candidate['First Name']} {candidate['Last Name']}")

                    with st.expander("📋 View Justification", expanded=False):
                        st.markdown(f"""
                        <div style="background: #fff; padding: 1rem; border-radius: 8px; border-left: 4px solid #ef4444;
                                    color: #333; line-height: 1.6;">
                            {candidate['Justification']}
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("</div>", unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"⚠️ Could not load non-qualified candidates: {str(e)}")


# ============================================================================
# HELPER FUNCTION: Update Excel with Assessment Results
# ============================================================================

def update_excel_with_assessment_results(candidate_index: int, evaluation: dict):
    """Update Excel file with fundamental assessment results."""
    try:
        excel_path = st.session_state.excel_output_path
        if not excel_path or not os.path.exists(excel_path):
            st.warning("Excel file path not found. Assessment results will not be saved to Excel.")
            return False
        
        # Read the Excel file
        df = pd.read_excel(excel_path)
        
        # Add new columns if they don't exist
        if 'Fundamental Knowledge Score' not in df.columns:
            df['Fundamental Knowledge Score'] = None
        if 'Fundamental Recommendation' not in df.columns:
            df['Fundamental Recommendation'] = None
        if 'Fundamental Assessment Date' not in df.columns:
            df['Fundamental Assessment Date'] = None
        if 'Fundamental Question Scores' not in df.columns:
            df['Fundamental Question Scores'] = None
        
        # Update the row for current candidate
        df.at[candidate_index, 'Fundamental Knowledge Score'] = evaluation.get('overall_score', 0)
        df.at[candidate_index, 'Fundamental Recommendation'] = evaluation.get('recommendation', 'N/A')
        df.at[candidate_index, 'Fundamental Assessment Date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Store question scores as JSON string
        question_scores = evaluation.get('question_scores', {})
        if question_scores:
            df.at[candidate_index, 'Fundamental Question Scores'] = json.dumps(question_scores)
        
        # Save the updated Excel file
        df.to_excel(excel_path, index=False)
        return True
        
    except Exception as e:
        st.error(f"Error updating Excel file: {str(e)}")
        return False


# ============================================================================
# PAGE: FUNDAMENTAL KNOWLEDGE ASSESSMENT
# ============================================================================

def render_fundamental_assessment_page():
    """Render the fundamental knowledge assessment page."""
    if st.session_state.candidates_df is None or st.session_state.selected_candidate_index is None:
        st.warning("No candidate selected. Please select a candidate from the list.")
        if st.button("Back to Candidate List"):
            st.session_state.app_page = "candidate_list"
            st.rerun()
        return
    
    candidate_info = st.session_state.candidates_df.loc[st.session_state.selected_candidate_index]
    candidate_name = f"{candidate_info['First Name']} {candidate_info['Last Name']}"
    
    # Check for Groq API key
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("GROQ_API_KEY not found. Please set it in your .env file to use Fundamental Knowledge Assessment.")
        if st.button("Back to Candidate List"):
            st.session_state.app_page = "candidate_list"
            st.rerun()
        return
    
    # Initialize assessment system
    if 'assessment_system' not in st.session_state:
        try:
            model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
            st.session_state.assessment_system = {
                'evaluator': EvaluationEngine(groq_api_key, model),
                'question_generator': QuestionGenerator(groq_api_key, model),
            }
        except Exception as e:
            st.error(f"Error initializing assessment system: {str(e)}")
            return
    
    # Get JD and role from session state
    job_description = st.session_state.get("job_description", "")
    role = st.session_state.get("position", "AI Specialist")
    
    if not job_description or len(job_description.strip()) < 50:
        st.error("Job description not found. Please run CV filtering first.")
        if st.button("Back to CV Filtering"):
            st.session_state.app_page = "cv_filtering_setup"
            st.rerun()
        return
    
    # Header
    st.markdown(f"""
    <div style="background: white; padding: 2rem; border-radius: 15px; margin-bottom: 2rem; box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
        <h2 style="color: #333; margin-bottom: 1rem;">Fundamental Knowledge Assessment</h2>
        <p style="color: #666; margin: 0;">Candidate: <strong>{candidate_name}</strong> | Role: <strong>{role}</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown(f"""
    <div style="background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
        <strong>Candidate:</strong><br>
        {candidate_name}
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.info(f"**Role:** {role}")
    st.sidebar.info(f"**Model:** {os.getenv('GROQ_MODEL', 'llama-3.1-8b-instant')}")
    
    # Configuration: Number of questions
    if 'fundamental_questions_count' not in st.session_state:
        # Get from env var or use default
        st.session_state.fundamental_questions_count = int(os.getenv("FUNDAMENTAL_QUESTIONS_COUNT", 5))
    
    num_questions = st.sidebar.number_input(
        "Number of Questions",
        min_value=1,
        max_value=10,
        value=st.session_state.fundamental_questions_count,
        help="Set the number of fundamental questions to generate (1-10)",
        key="num_questions_input"
    )
    
    # Update session state if changed
    if num_questions != st.session_state.fundamental_questions_count:
        st.session_state.fundamental_questions_count = num_questions
        # Reset questions if count changed
        if st.session_state.fundamental_questions_generated:
            st.session_state.fundamental_questions_generated = False
            st.session_state.fundamental_questions = []
            st.session_state.fundamental_responses = []
    
    # Check if assessment is already completed - if so, show results directly
    if st.session_state.get("fundamental_assessment_completed", False) and st.session_state.get("fundamental_evaluation"):
        # Skip to results display
        pass
    # Step 1: Generate Questions
    elif not st.session_state.fundamental_questions_generated:
        st.subheader("Step 1: Generate Assessment Questions")
        
        st.info(f"Configured to generate **{num_questions}** fundamental question(s).")
        
        if st.button("Generate Fundamental Questions", type="primary", use_container_width=True):
            with st.spinner(f"Generating {num_questions} fundamental questions for {role}..."):
                try:
                    questions = st.session_state.assessment_system['question_generator'].generate_fundamental_questions(
                        role, job_description, num_questions=num_questions
                    )
                    st.session_state.fundamental_questions = questions
                    st.session_state.fundamental_responses = [""] * len(questions)
                    st.session_state.fundamental_questions_generated = True
                    st.session_state.fundamental_assessment_completed = False
                    st.rerun()
                except Exception as e:
                    st.error(f"Error generating questions: {str(e)}")
                    # Use fallback questions
                    try:
                        qg = QuestionGenerator(groq_api_key, os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"))
                        fallback_questions = qg._get_fallback_questions(role)[:num_questions]
                        st.session_state.fundamental_questions = fallback_questions
                        st.session_state.fundamental_responses = [""] * len(fallback_questions)
                        st.session_state.fundamental_questions_generated = True
                        st.session_state.fundamental_assessment_completed = False
                        st.info("Using fallback questions instead.")
                        st.rerun()
                    except Exception as e2:
                        st.error(f"Failed to load fallback questions: {str(e2)}")
    
    # Step 2: Answer Questions (only show if not already completed)
    if not st.session_state.get("fundamental_assessment_completed", False) and st.session_state.fundamental_questions_generated and st.session_state.fundamental_questions:
        st.markdown("---")
        st.subheader(f"Step 2: Answer Fundamental Questions")
        st.info(f"{len(st.session_state.fundamental_questions)} questions - Answer each question concisely to test your fundamental knowledge.")
        
        # Initialize responses if needed
        if len(st.session_state.fundamental_responses) != len(st.session_state.fundamental_questions):
            st.session_state.fundamental_responses = [""] * len(st.session_state.fundamental_questions)
        
        # Display each question
        for i, question in enumerate(st.session_state.fundamental_questions):
            st.markdown(f"### Question {i+1}/{len(st.session_state.fundamental_questions)}")
            
            col_type, col_diff = st.columns(2)
            with col_type:
                st.info(f"**Type:** {question['type'].title()}")
            with col_diff:
                st.info(f"**Difficulty:** {question['difficulty'].title()}")
            
            st.markdown(f"**{question['question']}**")
            
            # Show expected keywords if available
            if question.get('expected_keywords'):
                with st.expander("Expected keywords (hint)"):
                    st.write(", ".join(question['expected_keywords']))
            
            # Response input
            response = st.text_area(
                f"Your Answer (Question {i+1}):",
                height=150,
                placeholder="Type your answer here...",
                value=st.session_state.fundamental_responses[i],
                key=f"fundamental_response_{i}"
            )
            
            # Update response in session state
            st.session_state.fundamental_responses[i] = response
            
            st.markdown("---")
        
        # Submit for Evaluation
        st.markdown("### Step 3: Submit for Evaluation")
        
        # Check if all questions have responses
        all_answered = all(response.strip() for response in st.session_state.fundamental_responses)
        
        if not all_answered:
            st.warning("Please answer all questions before submitting.")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Submit All Answers for Evaluation", 
                        type="secondary", 
                        use_container_width=True,
                        disabled=not all_answered,
                        key="fundamental_evaluate_btn"):
                with st.spinner("Evaluating fundamental knowledge..."):
                    try:
                        # Prepare responses data
                        responses_data = []
                        for i, question in enumerate(st.session_state.fundamental_questions):
                            responses_data.append({
                                "question": question['question'],
                                "type": question['type'],
                                "difficulty": question['difficulty'],
                                "expected_keywords": question.get('expected_keywords', []),
                                "response": st.session_state.fundamental_responses[i],
                                "response_time": 0.0
                            })
                        
                        # Evaluate
                        evaluation = st.session_state.assessment_system['evaluator'].evaluate_fundamental_responses(
                            role,
                            job_description[:1000],
                            st.session_state.fundamental_questions,
                            responses_data
                        )
                        
                        st.session_state.fundamental_evaluation = evaluation
                        st.session_state.fundamental_assessment_completed = True
                        
                        # Update Excel file
                        update_excel_with_assessment_results(
                            st.session_state.selected_candidate_index,
                            evaluation
                        )
                        
                        st.success("Assessment completed successfully!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error during evaluation: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
    
    # Step 3: Show Results
    if st.session_state.fundamental_assessment_completed and st.session_state.fundamental_evaluation:
        st.markdown("---")
        st.subheader("Assessment Results")
        
        evaluation = st.session_state.fundamental_evaluation
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Fundamental Score", f"{evaluation['overall_score']}/100", 
                      help="Score based on essential knowledge for the role")
        with col2:
            rec = evaluation['recommendation']
            st.metric("Recommendation", rec, 
                      help="Hiring recommendation based on fundamental knowledge")
        
        st.markdown("---")
        
        st.markdown("##### Qualitative Feedback")
        st.info(evaluation['qualitative_feedback'])
        
        # Display criterion scores
        st.markdown("##### Criterion Scores")
        if 'criterion_scores' in evaluation:
            crit_df = pd.DataFrame([
                {"Criterion": "Technical Accuracy", "Score": evaluation['criterion_scores'].get('technical_accuracy', 0)},
                {"Criterion": "Completeness", "Score": evaluation['criterion_scores'].get('completeness', 0)},
                {"Criterion": "Relevance", "Score": evaluation['criterion_scores'].get('relevance', 0)},
                {"Criterion": "Practicality", "Score": evaluation['criterion_scores'].get('practicality', 0)}
            ])
            st.bar_chart(crit_df.set_index('Criterion'))
        
        # Display question scores if available
        if 'question_scores' in evaluation and evaluation['question_scores']:
            st.markdown("##### Question-wise Scores")
            scores_data = []
            for q_key, score in evaluation['question_scores'].items():
                scores_data.append({
                    "Question": q_key[:50] + "..." if len(q_key) > 50 else q_key,
                    "Score": score,
                })
            if scores_data:
                scores_df = pd.DataFrame(scores_data)
                st.bar_chart(scores_df.set_index('Question'))
        
        st.markdown("##### Strengths")
        if evaluation.get('strengths'):
            for s in evaluation['strengths']:
                st.write(f"• {s}")
        else:
            st.write("No specific strengths identified.")
        
        st.markdown("##### Weaknesses")
        if evaluation.get('weaknesses'):
            for w in evaluation['weaknesses']:
                st.write(f"• {w}")
        else:
            st.write("No significant weaknesses identified.")
        
        # Check if score meets minimum requirement (80)
        score = evaluation.get('overall_score', 0)
        meets_requirement = score >= 80
        
        st.markdown("---")
        
        if not meets_requirement:
            # Show apology message if score is below 80
            st.markdown("""
            <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); 
                        padding: 2rem; border-radius: 15px; margin: 2rem 0; 
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-left: 5px solid #f59e0b;">
                <h3 style="color: #92400e; margin: 0 0 1rem 0; font-size: 1.5rem;">
                    Assessment Result
                </h3>
                <p style="color: #78350f; margin: 0.5rem 0; font-size: 1.1rem; line-height: 1.6;">
                    Thank you for completing the Fundamental Knowledge Assessment. 
                    We appreciate the time and effort you've invested in this process.
                </p>
                <p style="color: #78350f; margin: 1rem 0 0.5rem 0; font-size: 1.1rem; line-height: 1.6;">
                    Unfortunately, your assessment score of <strong>{score}/100</strong> does not meet 
                    our minimum requirement of 80/100 to proceed to the voice interview stage.
                </p>
                <p style="color: #78350f; margin: 1rem 0 0; font-size: 1.1rem; line-height: 1.6;">
                    We encourage you to continue developing your skills and consider applying again in the future. 
                    We wish you the best of luck in your career journey.
                </p>
            </div>
            """.format(score=score), unsafe_allow_html=True)
            
            # Show feedback for improvement
            st.markdown("### Feedback for Improvement")
            st.info("""
            Based on your assessment, we recommend focusing on the following areas:
            - Review the fundamental concepts related to the role
            - Practice with hands-on projects to strengthen practical knowledge
            - Consider additional training or certification in key technical areas
            - Stay updated with industry best practices and trends
            """)
        else:
            # Show success message and allow proceeding
            st.markdown("""
            <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); 
                        padding: 2rem; border-radius: 15px; margin: 2rem 0; 
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-left: 5px solid #10b981;">
                <h3 style="color: #065f46; margin: 0 0 1rem 0; font-size: 1.5rem;">
                    Congratulations!
                </h3>
                <p style="color: #047857; margin: 0.5rem 0; font-size: 1.1rem; line-height: 1.6;">
                    You have successfully passed the Fundamental Knowledge Assessment with a score of <strong>{score}/100</strong>.
                </p>
                <p style="color: #047857; margin: 1rem 0 0; font-size: 1.1rem; line-height: 1.6;">
                    You may now proceed to the voice interview stage.
                </p>
            </div>
            """.format(score=score), unsafe_allow_html=True)
            
            # Button to proceed to interview (only if score >= 80)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("Proceed to Voice Interview", type="primary", use_container_width=True):
                    # Initialize interview state
                    candidate_info = st.session_state.candidates_df.loc[st.session_state.selected_candidate_index]
                    resume_text = candidate_info.get('Full Resume', '')
                    
                    if not resume_text or len(str(resume_text).strip()) < 50:
                        st.error(f"Resume text is missing or too short. Please check the Excel file.")
                        st.stop()
                    
                    position = st.session_state.get("position", "AI Specialist")
                    if not position or position.strip() == "":
                        position = "AI Specialist"
                    
                    st.session_state.state = AgentState(
                        mode="friendly",
                        num_of_q=2,
                        num_of_follow_up=1,
                        position=position,
                        company_name="Prince Mogrin University",
                        messages=[],
                        evaluation_result="",
                        report="",
                        pdf_path=None,
                        resume_path=None,
                        questions_path=None,
                        resume_text=str(resume_text)
                    )
                    
                    st.session_state.app_page = "interview"
                    st.session_state.timings = []
                    st.session_state.turn_idx = 0
                    st.session_state.time_analysis_generated = False
                    st.session_state.auto_start_interview = True
                    st.session_state.interview_stage = "switching"
                    st.rerun()
        
        # Button to go back
        if st.button("Back to Candidate List"):
            # Preserve assessment data when going back so it can be viewed again
            # The evaluation is already saved in Excel and session state
            st.session_state.app_page = "candidate_list"
            st.rerun()


# ============================================================================
# PAGE: INTERVIEW
# ============================================================================

def render_interview_page():
    """Render the interview page."""
    # Check if fundamental assessment is completed
    if not st.session_state.get("fundamental_assessment_completed", False):
        st.warning("Please complete the Fundamental Knowledge Assessment before proceeding to the interview.")
        if st.button("Go to Assessment"):
            st.session_state.app_page = "fundamental_assessment"
            st.rerun()
        return
    
    # Check if score meets minimum requirement (80)
    evaluation = st.session_state.get("fundamental_evaluation")
    if evaluation:
        score = evaluation.get('overall_score', 0)
        if score < 80:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); 
                        padding: 2rem; border-radius: 15px; margin: 2rem 0; 
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-left: 5px solid #f59e0b;">
                <h3 style="color: #92400e; margin: 0 0 1rem 0; font-size: 1.5rem;">
                    Access Restricted
                </h3>
                <p style="color: #78350f; margin: 0.5rem 0; font-size: 1.1rem; line-height: 1.6;">
                    Your Fundamental Knowledge Assessment score of <strong>{score}/100</strong> does not meet 
                    our minimum requirement of 80/100 to proceed to the voice interview stage.
                </p>
                <p style="color: #78350f; margin: 1rem 0 0; font-size: 1.1rem; line-height: 1.6;">
                    We appreciate your interest and encourage you to continue developing your skills.
                </p>
            </div>
            """.format(score=score), unsafe_allow_html=True)
            
            if st.button("Back to Candidate List"):
                st.session_state.app_page = "candidate_list"
                st.rerun()
            return
    
    if st.session_state.candidates_df is None or st.session_state.selected_candidate_index is None:
        st.warning("No candidate selected. Please select a candidate from the list.")
        if st.button("🔙 Back to Candidate List"):
            st.session_state.app_page = "candidate_list"
            st.rerun()
        return

    candidate_info = st.session_state.candidates_df.loc[st.session_state.selected_candidate_index]

    # Sidebar setup
    st.sidebar.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;">
        <h3 style="color: white; margin: 0;">⚙️ Interview Setup</h3>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown(f"""
    <div style="background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
        <strong>👤 Candidate:</strong><br>
        {candidate_info['First Name']} {candidate_info['Last Name']}
    </div>
    """, unsafe_allow_html=True)

    current_mode = st.session_state.state.get("mode", "friendly")
    current_position = st.session_state.state.get("position", st.session_state.position)
    current_company = st.session_state.state.get("company_name", "Prince Mogrin University")
    current_num_q = st.session_state.state.get("num_of_q", 2)
    current_num_follow = st.session_state.state.get("num_of_follow_up", 1)

    mode = st.sidebar.selectbox("Interviewer Mode", ["friendly", "formal", "technical"],
                                index=["friendly", "formal", "technical"].index(current_mode))

    st.sidebar.text_input("Position", value=current_position, disabled=True, key="position_display")
    st.sidebar.text_input("Company Name", value=current_company, disabled=True)

    num_of_q = st.sidebar.number_input("Number of Technical Questions", min_value=1, max_value=10, value=current_num_q)
    num_of_follow_up = st.sidebar.number_input("Number of Follow-up Questions", min_value=0, max_value=3, value=current_num_follow)

    params_changed = (
        mode != st.session_state.state.get("mode") or
        num_of_q != st.session_state.state.get("num_of_q") or
        num_of_follow_up != st.session_state.state.get("num_of_follow_up")
    )

    if params_changed:
        if st.sidebar.button("Update Parameters"):
            st.session_state.state["mode"] = mode
            st.session_state.state["num_of_q"] = num_of_q
            st.session_state.state["num_of_follow_up"] = num_of_follow_up
            st.sidebar.success("Parameters updated!")

    st.sidebar.header("Upload Files")
    st.sidebar.subheader("Interview Questions (Optional)")

    questions_file = st.sidebar.file_uploader("Upload Questions (PDF)", type=["pdf"], key="questions_uploader")
    if questions_file:
        questions_dir = "./uploaded_questions"
        os.makedirs(questions_dir, exist_ok=True)
        questions_path = os.path.join(questions_dir, questions_file.name)
        with open(questions_path, "wb") as f:
            f.write(questions_file.read())
        if st.session_state.state.get("questions_path") != questions_path:
            st.session_state.state["questions_path"] = questions_path
            st.sidebar.success(f"Questions uploaded: {questions_file.name}")

    st.sidebar.subheader("Speech-to-Text Engine")
    stt_options = {
        "Whisper Tiny (fastest, lower accuracy)": "whisper_tiny",
        "Whisper Base (balanced)": "whisper_base",
        "Whisper Small (higher accuracy)": "whisper_small",
        "Whisper Medium (slower, high accuracy)": "whisper_medium",
        "Whisper Large (slowest, highest accuracy)": "whisper_large",
        "AssemblyAI (Best - Cloud)": "assemblyai_best",
    }
    stt_labels = list(stt_options.keys())
    current_label = next((label for label, value in stt_options.items() if value == st.session_state.stt_model), "Whisper Base (balanced)")
    try:
        default_index = stt_labels.index(current_label)
    except ValueError:
        default_index = stt_labels.index("Whisper Base (balanced)")

    selected_label = st.sidebar.selectbox("Choose STT Model", stt_labels, index=default_index)
    selected_model = stt_options[selected_label]
    st.session_state.stt_model = selected_model
    st.session_state.stt_model_label = selected_label

    if selected_model.startswith("assemblyai"):
        if not os.getenv("ASSEMBLYAI_API_KEY"):
            st.sidebar.error("AssemblyAI API key not found. Please set it in your environment.")
        else:
            st.sidebar.info("AssemblyAI runs in the cloud using your AssemblyAI API quota.")
    else:
        st.sidebar.caption("Whisper runs locally. Larger models require more CPU/GPU resources.")

    # Process message function
    def process_message(user_input):
        if not st.session_state.state.get("position"):
            st.error("Please enter a position in the sidebar before starting the interview.")
            return

        if not user_input or len(user_input.strip()) < 3:
            st.error("Invalid input: message is too short or empty.")
            return

        st.session_state.state["messages"].append(HumanMessage(content=user_input.strip()))

        st.session_state.turn_idx += 1
        turn = st.session_state.turn_idx

        try:
            if not st.session_state.state.get("resume_text"):
                st.error("❌ Resume text is missing. Cannot proceed with interview.")
                return

            with st.spinner("Recruiter..."):
                ai_tok = timing_start("ai_generation", {"len_messages": len(st.session_state.state["messages"])})
                result = workflow.invoke(st.session_state.state)
                st.session_state.timings.append(timing_end(ai_tok, turn))

                for key, value in result.items():
                    if key == "messages":
                        st.session_state.state["messages"] = value
                    else:
                        st.session_state.state[key] = value

                if not st.session_state.state.get("messages"):
                    st.error("❌ No response from AI. The workflow may have encountered an error.")
                    return

                ai_message = st.session_state.state["messages"][-1]
                if isinstance(ai_message, AIMessage):
                    ai_text = ai_message.content

                    if ai_text and ("error" in ai_text.lower() or "apologize" in ai_text.lower() or "encountered" in ai_text.lower()):
                        st.warning("⚠️ The AI workflow may have encountered an error. Check the console for details.")
                        st.code(f"AI Response: {ai_text}\n\nState: {st.session_state.state.get('position', 'N/A')}\nResume length: {len(str(st.session_state.state.get('resume_text', '')))}")

                    # Handle empty response - retry if needed
                    if not ai_text or ai_text.strip() == "":
                        if hasattr(ai_message, 'tool_calls') and ai_message.tool_calls:
                            st.info("🔧 AI is retrieving information from knowledge base...")
                            # Re-invoke workflow to process tool calls
                            result = workflow.invoke(st.session_state.state)
                            for key, value in result.items():
                                if key == "messages":
                                    st.session_state.state["messages"] = value
                                else:
                                    st.session_state.state[key] = value
                            # Check if we now have a response
                            if st.session_state.state.get("messages"):
                                new_ai_message = st.session_state.state["messages"][-1]
                                if isinstance(new_ai_message, AIMessage) and new_ai_message.content:
                                    ai_text = new_ai_message.content
                        else:
                            st.warning("Recruiter Response is empty. Retrying...")
                            # Retry once more
                            result = workflow.invoke(st.session_state.state)
                            for key, value in result.items():
                                if key == "messages":
                                    st.session_state.state["messages"] = value
                                else:
                                    st.session_state.state[key] = value
                            if st.session_state.state.get("messages"):
                                new_ai_message = st.session_state.state["messages"][-1]
                                if isinstance(new_ai_message, AIMessage) and new_ai_message.content:
                                    ai_text = new_ai_message.content

                    # Display and generate TTS if we have content
                    if ai_text and ai_text.strip():
                        st.subheader("Recruiter Response")
                        st.write(ai_text)

                        with st.spinner("Generating voice..."):
                            try:
                                tts_tok = timing_start("tts_generate", {"chars": len(ai_text), "use_cache": st.session_state.use_tts_cache})
                                audio_path, tts_error = generate_tts(ai_text, use_cache=st.session_state.use_tts_cache)
                                st.session_state.timings.append(timing_end(tts_tok, turn))

                                if tts_error:
                                    st.error(f"Voice generation failed: {tts_error}")
                                elif audio_path and os.path.exists(audio_path):
                                    pb_tok = timing_start("playback_wait", {"audio_path": audio_path})
                                    waited = play_audio_html5(audio_path)
                                    pb_timing = timing_end(pb_tok, turn)
                                    pb_timing.meta["waited_s"] = waited
                                    st.session_state.timings.append(pb_timing)
                                else:
                                    st.error("Audio file not found")

                            except Exception as e:
                                st.error(f"Voice generation error: {str(e)}")
                                import traceback
                                traceback.print_exc()
                    else:
                        st.error("❌ Could not get a response from the AI. Please try speaking again.")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    # Interview header
    st.markdown(f"""
    <div style="background: white; padding: 1.5rem; border-radius: 15px; margin-bottom: 2rem; box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
        <h2 style="color: #333; margin: 0;">🎙️ Interviewing: {candidate_info['First Name']} {candidate_info['Last Name']}</h2>
        <p style="color: #666; margin: 0.5rem 0 0 0;">Position: {st.session_state.state.get("position", "AI Specialist")} at {st.session_state.state.get("company_name", "Prince Mogrin University")}</p>
    </div>
    """, unsafe_allow_html=True)

    # Auto-start interview if flagged
    if hasattr(st.session_state, 'auto_start_interview') and st.session_state.auto_start_interview:
        st.session_state.auto_start_interview = False

        st.session_state.state["messages"].append(HumanMessage(content="[Interview Started]"))

        with st.spinner("Starting interview..."):
            st.session_state.turn_idx += 1
            turn = st.session_state.turn_idx

            try:
                if not st.session_state.state.get("resume_text"):
                    raise ValueError("Resume text is missing. Please check the candidate data.")

                if not st.session_state.state.get("position"):
                    st.session_state.state["position"] = "AI Specialist"

                ai_tok = timing_start("ai_generation", {"len_messages": len(st.session_state.state["messages"])})
                result = workflow.invoke(st.session_state.state)
                st.session_state.timings.append(timing_end(ai_tok, turn))

                for key, value in result.items():
                    if key == "messages":
                        st.session_state.state["messages"] = value
                    else:
                        st.session_state.state[key] = value

                if st.session_state.state["messages"] and st.session_state.state["messages"][0].content == "[Interview Started]":
                    st.session_state.state["messages"].pop(0)

                if st.session_state.state["messages"]:
                    ai_message = st.session_state.state["messages"][-1]
                    if isinstance(ai_message, AIMessage):
                        ai_text = ai_message.content

                        if ai_text and ("error" in ai_text.lower() or "apologize" in ai_text.lower()):
                            st.warning(f"⚠️ AI returned a potential error message. Check the workflow configuration.")

                        if ai_text and ai_text.strip():
                            tts_tok = timing_start("tts_generate", {"chars": len(ai_text)})
                            audio_path, tts_error = generate_tts(ai_text, use_cache=True)
                            st.session_state.timings.append(timing_end(tts_tok, turn))

                            if not tts_error and audio_path and os.path.exists(audio_path):
                                st.session_state.intro_audio_path = audio_path
                                st.session_state.intro_audio_turn = turn

                st.rerun()

            except Exception as e:
                error_msg = f"Failed to start interview: {str(e)}"
                st.error(error_msg)
                import traceback
                st.code(traceback.format_exc())
                if st.session_state.state.get("messages") and st.session_state.state["messages"][0].content == "[Interview Started]":
                    st.session_state.state["messages"].pop(0)

    # Display transcript
    if st.session_state.state["messages"]:
        st.markdown("""
        <div class="transcript-container">
            <h3 style="color: #333; margin-bottom: 1.5rem;">💬 Interview Transcript</h3>
        </div>
        """, unsafe_allow_html=True)

        transcript_container = st.container()
        with transcript_container:
            for m in st.session_state.state["messages"]:
                if isinstance(m, HumanMessage):
                    if m.content != "[Interview Started]":
                        st.markdown(f"""
                        <div class="candidate-message" style="color: #333333 !important;">
                            <strong style="color: #667eea !important; font-size: 1.1em;">👤 Candidate:</strong><br>
                            <span style="color: #333333 !important; font-size: 1em; line-height: 1.6;">{m.content}</span>
                        </div>
                        """, unsafe_allow_html=True)
                elif isinstance(m, AIMessage):
                    if m.content and m.content.strip():
                        has_tool_calls = hasattr(m, 'tool_calls') and m.tool_calls
                        if not has_tool_calls:
                            st.markdown(f"""
                            <div class="recruiter-message" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white !important; padding: 1rem 1.5rem; border-radius: 15px; margin: 1rem 0; border-left: 4px solid #764ba2;">
                                <strong style="color: white !important; font-size: 1.1em;">🤖 Recruiter:</strong><br>
                                <span style="color: white !important; font-size: 1em; line-height: 1.6;">{m.content}</span>
                            </div>
                            """, unsafe_allow_html=True)

    # Play intro audio if available
    if hasattr(st.session_state, 'intro_audio_path') and st.session_state.intro_audio_path:
        audio_path = st.session_state.intro_audio_path
        turn = st.session_state.intro_audio_turn

        with open(audio_path, 'rb') as audio_file:
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/mp3', autoplay=True)

        duration = get_audio_duration(audio_path)
        pb_tok = timing_start("playback_wait", {"audio_path": audio_path})
        pb_timing = timing_end(pb_tok, turn)
        pb_timing.meta["waited_s"] = duration
        st.session_state.timings.append(pb_timing)

        del st.session_state.intro_audio_path
        del st.session_state.intro_audio_turn

    # Check if interview has ended
    interview_ended = False
    for msg in reversed(st.session_state.state.get("messages", [])):
        if isinstance(msg, AIMessage) and "that's it for today" in msg.content.lower():
            interview_ended = True
            break

    # Only show microphone if interview hasn't ended
    if not interview_ended:
        st.markdown("""
        <div style="position: sticky; bottom: 0; background: linear-gradient(to top, rgba(255,255,255,0.98), rgba(255,255,255,0.95));
                    backdrop-filter: blur(10px); padding: 1.5rem; margin-top: 2rem; border-radius: 15px 15px 0 0;
                    box-shadow: 0 -4px 20px rgba(0, 0, 0, 0.15); border-top: 2px solid rgba(102, 126, 234, 0.2); z-index: 100;">
            <div style="max-width: 800px; margin: 0 auto; text-align: center;">
                <h4 style="color: #333; margin-bottom: 0.5rem; font-size: 1.2rem;">🎤 Ready to Answer?</h4>
                <p style="color: #666; margin-bottom: 1rem; font-size: 0.9rem;">Click the microphone button below to start recording your response</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            audio_bytes = audio_recorder(
                text="Click to start/stop recording",
                recording_color="#764ba2",
                neutral_color="#667eea",
                icon_name="microphone",
                pause_threshold=10.0,
                sample_rate=44100,
                icon_size="3x",
                key="fixed_recorder"
            )

        if audio_bytes:
            audio_hash = hashlib.md5(audio_bytes).hexdigest()

            if audio_hash != st.session_state.last_processed_audio:
                st.audio(audio_bytes, format="audio/wav")

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(audio_bytes)
                    audio_path = tmp.name

                stt_label = st.session_state.get("stt_model_label", "Whisper Base (balanced)")
                with st.spinner(f"Transcribing audio ({stt_label})..."):
                    stt_tok = timing_start("stt_transcribe", {"format": "wav", "model": st.session_state.stt_model})
                    transcribed_text, stt_error = transcribe_audio(audio_path, stt_model=st.session_state.stt_model, max_wait=180)
                    st.session_state.timings.append(timing_end(stt_tok, st.session_state.turn_idx + 1))
                    os.remove(audio_path)

                    if stt_error:
                        st.error(f"Transcription error: {stt_error}")
                    elif transcribed_text and len(transcribed_text.strip()) >= 3:
                        st.write(f"**You said:** {transcribed_text}")

                        st.session_state.last_processed_audio = audio_hash
                        process_message(transcribed_text.strip())
                        st.rerun()
                    else:
                        st.warning("No speech detected or audio too short. Please try again.")

    # Auto-generate Time Analysis PDF when interview ends
    if interview_ended and not st.session_state.time_analysis_generated and st.session_state.get("timings"):
        try:
            tz = pytz.timezone('Asia/Kuwait')
            dt_display = datetime.now(tz).strftime("%Y-%m-%d %H-%M-%S")
            candidate_name = f"{candidate_info['First Name']} {candidate_info['Last Name']}"
            position = st.session_state.state.get("position", "AI Specialist")

            pdf_path = generate_time_analysis_pdf(
                candidate_name=candidate_name,
                position=position,
                timings=st.session_state.timings,
                dt_display_local=dt_display,
                tz_label="UTC+3"
            )
            st.session_state.time_analysis_generated = True
        except Exception:
            pass

    if interview_ended and not st.session_state.state.get("evaluation_result"):
        st.markdown("""
        <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); padding: 1.5rem; border-radius: 15px; margin: 2rem 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h3 style="color: white; margin: 0 0 1rem 0;">✅ Interview Completed!</h3>
            <p style="color: rgba(255,255,255,0.9); margin: 0;">Click the button below to generate the evaluation and report.</p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("📊 Generate Evaluation and Report", type="primary", use_container_width=True):
            with st.spinner("🔄 Generating evaluation..."):
                try:
                    from src.dynamic_workflow import evaluator, report_writer, pdf_generator_node

                    current_state = st.session_state.state.copy()
                    current_state["evaluation_result"] = ""
                    current_state["report"] = ""
                    current_state["pdf_path"] = None

                    with st.spinner("📝 Analyzing responses..."):
                        eval_result = evaluator(current_state)
                        st.session_state.state["evaluation_result"] = eval_result["evaluation_result"]

                    with st.spinner("📄 Writing report..."):
                        current_state["evaluation_result"] = st.session_state.state["evaluation_result"]
                        report_result = report_writer(current_state)
                        st.session_state.state["report"] = report_result["report"]

                    with st.spinner("📥 Creating PDF..."):
                        current_state["report"] = st.session_state.state["report"]
                        pdf_result = pdf_generator_node(current_state)
                        st.session_state.state["pdf_path"] = pdf_result["pdf_path"]

                    st.success("✅ Evaluation and report generated successfully!")
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Error generating evaluation: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

    # Display evaluation result
    evaluation_result = st.session_state.state.get("evaluation_result")
    if evaluation_result:
        st.markdown("""
        <div style="background: white; padding: 2rem; border-radius: 15px; margin: 2rem 0; box-shadow: 0 2px 10px rgba(0,0,0,0.05); border-left: 5px solid #10b981;">
            <h3 style="color: #333; margin-bottom: 1rem; display: flex; align-items: center;">
                <span style="font-size: 1.5em; margin-right: 0.5rem;">📊</span>
                Evaluation Result
            </h3>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background: #f9fafb; padding: 1.5rem; border-radius: 10px; color: #333; line-height: 1.8;">
            {evaluation_result}
        </div>
        """, unsafe_allow_html=True)
    elif interview_ended:
        st.markdown("""
        <div style="background: #fef3c7; padding: 1.5rem; border-radius: 15px; margin: 2rem 0; border-left: 5px solid #f59e0b;">
            <h3 style="color: #92400e; margin-bottom: 0.5rem;">📊 Evaluation Result</h3>
            <p style="color: #78350f; margin: 0;">No evaluation available yet. Please click "Generate Evaluation and Report" above to create the evaluation.</p>
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.state.get("report"):
        st.markdown("""
        <div style="background: white; padding: 2rem; border-radius: 15px; margin: 2rem 0; box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
            <h3 style="color: #333; margin-bottom: 1rem;">📄 HR Report</h3>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(st.session_state.state["report"])

        pdf_path = st.session_state.state.get("pdf_path")
        if pdf_path and os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                st.download_button("📥 Download PDF Report", f,
                                 file_name=os.path.basename(pdf_path),
                                 type="primary",
                                 use_container_width=True)

    if interview_ended:
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Back to Candidate List"):
                st.session_state.app_page = "candidate_list"
                st.session_state.selected_candidate_index = None
                st.session_state.state = None
                st.session_state.last_processed_audio = None
                st.session_state.timings = []
                st.session_state.turn_idx = 0
                st.session_state.time_analysis_generated = False
                st.rerun()
        with col2:
            if st.button("🔙 Back to CV Filtering Setup"):
                st.session_state.app_page = "cv_filtering_setup"
                st.session_state.selected_candidate_index = None
                st.session_state.state = None
                st.session_state.last_processed_audio = None
                st.session_state.timings = []
                st.session_state.turn_idx = 0
                st.session_state.time_analysis_generated = False
                st.rerun()


# ============================================================================
# PAGE ROUTING
# ============================================================================

PAGE_HANDLERS = {
    "cv_filtering_setup": render_cv_filtering_setup_page,
    "filtering_process": render_filtering_process_page,
    "candidate_list": render_candidate_list_page,
    "fundamental_assessment": render_fundamental_assessment_page,
    "interview": render_interview_page,
}


def main():
    """Main entry point."""
    current_page = st.session_state.app_page
    if current_page in PAGE_HANDLERS:
        PAGE_HANDLERS[current_page]()
    else:
        st.error(f"Unknown page: {current_page}")
        st.session_state.app_page = "cv_filtering_setup"
        st.rerun()


if __name__ == "__main__":
    main()
