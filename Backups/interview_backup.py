# interview.py - WITH AUTOMATIC SILENT TIMING ANALYSIS

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
# --------------------------------------------------

import streamlit as st
import pandas as pd
import io
import tempfile
import hashlib
import time
import base64
from datetime import datetime
import pytz
import re
import json
from dataclasses import dataclass
from typing import List
from PyPDF2 import PdfReader
from pydantic import BaseModel, field_validator
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from audio_recorder_streamlit import audio_recorder

# LangChain for CV filtering
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from tenacity import retry, stop_after_attempt, wait_exponential

# Google Drive (optional)
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# Parallel Processing
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- NOW SAFE TO IMPORT WORKFLOW ---
from src.dynamic_workflow import build_workflow, AgentState
from langchain_core.messages import HumanMessage, AIMessage
# -----------------------------------

# Import optimized audio utils (using Edge TTS instead of ElevenLabs)
try:
    from utils.audio_utils_optimized import transcribe_audio_file_optimized as transcribe_audio_file
    from utils.audio_utils_optimized import elevenlabs_tts_optimized as elevenlabs_tts  # Now uses Edge TTS
    USE_OPTIMIZED = True
except ImportError:
    from utils.audio_utils import transcribe_audio_file
    # Fallback TTS function
    def elevenlabs_tts(text, api_key=None, voice_id="andrew", use_cache=True):
        return None, "Audio utils not available"
    USE_OPTIMIZED = False

# TIMING ANALYSIS IMPORTS (NEW)
from timing_instrumentation import ensure_session, timing_start, timing_end, TIME_REPORT_DIR
from src.pdf_utils_time_analysis import generate_time_analysis_pdf

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

if not os.getenv("ASSEMBLYAI_API_KEY"):
    # Optional: Don't error out immediately if using Whisper local
    pass 

# ============================================================================
# CV FILTERING FUNCTIONS (from CV-Filtering-Langchain.py)
# ============================================================================

# Google Drive credentials (optional)
SCOPES = ["https://www.googleapis.com/auth/drive.file"]

def get_credentials():
    """Get Google Drive credentials. Returns None if credentials file not found."""
    creds = None
    possible_paths = [
        "client_secret.json",
        r"C:\Users\96658\Desktop\TalentTalk\client_secret_2_744938258489-91hb253qiop2e8rl8uuj9duhh6fn1dj4.apps.googleusercontent.com.json"
    ]
    
    client_secrets_file = None
    for path in possible_paths:
        if os.path.exists(path):
            client_secrets_file = path
            break
    
    if not client_secrets_file:
        return None

    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                client_secrets_file,
                SCOPES
            )
            creds = flow.run_local_server(port=0)

        with open("token.json", "w") as token:
            token.write(creds.to_json())

    return creds

CACHE_FILE = "drive_upload_cache.json"

def load_cache():
    """Load the upload cache from disk"""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_cache(cache):
    """Save the upload cache to disk"""
    try:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        pass

def upload_to_drive(file_path, folder_id):
    """Upload file to Google Drive with duplicate detection."""
    creds = get_credentials()
    if not creds:
        return None
    
    filename = os.path.basename(file_path)
    cache = load_cache()
    cache_key = f"{folder_id}/{filename}"
    if cache_key in cache:
        return cache[cache_key]
    
    try:
        service = build("drive", "v3", credentials=creds)
        query = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
        results = service.files().list(
            q=query,
            fields="files(id, name, webViewLink)",
            pageSize=1
        ).execute()
        
        existing_files = results.get('files', [])
        if existing_files:
            drive_link = existing_files[0].get('webViewLink')
            cache[cache_key] = drive_link
            save_cache(cache)
            return drive_link
        
        file_metadata = {"name": filename, "parents": [folder_id]}
        media = MediaFileUpload(file_path, mimetype="application/pdf")
        uploaded = service.files().create(
            body=file_metadata, media_body=media, fields="webViewLink"
        ).execute()
        drive_link = uploaded.get("webViewLink")
        cache[cache_key] = drive_link
        save_cache(cache)
        return drive_link
    except Exception as e:
        return None

@dataclass
class Config:
    jd_text: str  # Changed from jd_path to jd_text
    resumes_dir: str
    out_dir: str
    Improved_JD_dir: str
    model: str = "gemini-2.5-flash-lite"
    temperature: float = 0.2
    high_score_threshold: int = 8

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def read_pdf_text_from_path(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join([page.extract_text() or "" for page in reader.pages])

def clean_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())

def make_llm(cfg: Config):
    return ChatGoogleGenerativeAI(model=cfg.model, temperature=cfg.temperature)

JD_FIX_PROMPT = ChatPromptTemplate.from_template("""
You are an experienced HR and talent acquisition specialist.

Goal: Clean and structure this job description so it is optimized for **internal screening and CV evaluation**, not for job advertising.

Do the following:
- Keep the same job role and core requirements.
- Organize the description into clear, structured sections useful for recruiters, such as:
  - Job Title
  - Location (if provided)
  - Employment Type (if provided)
  - Summary / Purpose of the Role
  - Key Responsibilities
  - Must-Have Requirements (hard screening criteria)
  - Nice-to-Have / Preferred Requirements
  - Technical Skills
  - Soft Skills
- Make the language clear, neutral, and ATS-friendly.
- Remove marketing language, fluff, and employer-branding content that is not relevant for screening.
- Do NOT add sections like "About Us", "Benefits", "Perks", or "Why Join Us" unless they already exist and are essential.
- Remove bias or gender-coded language.
- Clarify any ambiguous requirements where possible.

IMPORTANT:
- Explicitly extract two lists from the job description:
  1) must_have_requirements: hard screening criteria (e.g., required degree/cert, years, core skills, mandatory domain).
  2) nice_to_have_requirements: preferences (helpful but not mandatory).
- Keep each requirement short and atomic (one idea per bullet).
- Do NOT invent requirements that are not supported by the text.

Return JSON ONLY in exactly this format (no extra text):

{{
  "improved_job_description": "",
  "high_level_summary": "",
  "detailed_changes": [""],
  "must_have_requirements": [""],
  "nice_to_have_requirements": [""]
}}

Raw Job Description:
{jd_text}
""")

RECRUITER_PROMPT = ChatPromptTemplate.from_template("""

You are a recruiter evaluating a candidate's resume against a structured internal job description.

You are given:
- A detailed job description optimized for screening.
- A list of MUST-HAVE requirements.
- A list of NICE-TO-HAVE requirements.
- The candidate's resume.

Your evaluation must be fair, realistic, recruiter-like, and NOT overly strict or keyword-dependent.

====================================================================
INTERPRETING MUST-HAVES (FAIR, REALISTIC, HUMAN-LIKE)
====================================================================
A MUST-HAVE requirement may be counted as **satisfied** if ANY of the following are true:

1. **Explicitly satisfied**  
   The resume directly states the required skill, tool, qualification, or experience.

2. **Implicitly or likely satisfied**  
   Very closely related experience strongly implies the requirement.
   Examples:
   - “Next.js” → implies React
   - “REST APIs” → implies HTTP/API fundamentals
   - “GitHub collaboration” → implies Git
   - “Statistical modeling” → implies analytical skills

3. **Partially satisfied**  
   Some relevant evidence appears, even if not exact.

4. **UNIVERSAL HIERARCHY RULE (GLOBAL SKILL LOGIC)**  
   If the candidate demonstrates a **higher-level, more advanced, more senior, or more specialized version** of a requirement,  
   then the lower-level requirement is automatically considered satisfied.

   Examples:
   - Master’s degree → satisfies Bachelor’s requirement
   - Senior-level duties → satisfy junior/mid-level requirements
   - Advanced Excel/Power BI → satisfies basic Excel
   - Leading teams → satisfies teamwork/collaboration
   - Full-stack engineering → satisfies backend or frontend basics
   - Advanced ML models → satisfies basic ML fundamentals

   This applies to **education, skills, experience, and tools**.

A MUST-HAVE should be marked **missing** ONLY IF:
- No explicit match,
- No implied/related evidence,
- No partial match,
- No higher-level equivalent.

Do NOT penalize candidates for:
- Different terminology,
- Summarized resumes,
- Not repeating keywords word-for-word.

====================================================================
SCORING FRAMEWORK (HUMAN-LIKE)
====================================================================





Detailed scoring:
10: ALL must-haves satisfied + strong alignment + ALL nice-to-haves + strong reward factor
9: ALL must-haves satisfied + MANY nice-to-haves + low/medium risk
7–8: ALL must-haves satisfied + SOME nice-to-haves + moderate alignment (7 = moderate risk or weaker depth, 8 = solid alignment and lower risk)
6: ALL must-haves satisfied but noticeable risk or limited relevance
5: ONE must-have missing but otherwise strong candidate
3–4: MORE THAN ONE must-have missing, partial relevance
1–2: MANY must-haves missing, low relevance

Nice-to-have guardrail:
- Scores 7–8 require at least ONE nice-to-have to be satisfied.
- If ZERO nice-to-haves are satisfied → the score must NOT exceed 6.


====================================================================
RISK & REWARD FACTORS
====================================================================
Risk factor examples:
- Skill gaps  
- Limited experience  
- Job hopping  
- Lack of clarity  
- Weak technical grounding  

Reward factor examples:
- Strong foundation  
- High-quality experience  
- Leadership  
- Strong technical achievement  

Risk and reward scores must be:
- "Low", "Medium", or "High"  
Each with a short explanation.

Risk can reduce the score (within allowed ranges).  
Reward can increase the score (within allowed ranges).  
Reward **cannot** override missing must-haves.

====================================================================
REQUIRED OUTPUT FIELDS
====================================================================
You MUST explicitly list:

- satisfied_must_haves  
- missing_must_haves  
- satisfied_nice_to_haves  
- missing_nice_to_haves  

Return JSON ONLY in this structure:

{{
  "candidate_strengths": [],
  "candidate_weaknesses": [],
  "risk_factor": {{
    "score": "",
    "explanation": ""
  }},
  "reward_factor": {{
    "score": "",
    "explanation": ""
  }},
  "satisfied_must_haves": [],
  "missing_must_haves": [],
  "satisfied_nice_to_haves": [],
  "missing_nice_to_haves": [],
  "overall_fit_rating": 0,
  "justification_for_rating": ""
}}

Job Description: {job_description}
Explicit MUST-HAVEs: {must_have_requirements}
Explicit NICE-TO-HAVEs: {nice_to_have_requirements}
Candidate Resume: {resume_text}
""")

CONTACT_PROMPT = ChatPromptTemplate.from_template("""
Extract:
{{
 "First Name": "",
 "Last Name": "",
 "Email Address": ""
}}
Resume:
{resume_text}
""")

POSITION_EXTRACT_PROMPT = ChatPromptTemplate.from_template("""
Extract the job title/position from this job description. Return only the job title, nothing else.

Job Description:
{jd_text}
""")

class RiskReward(BaseModel):
    score: str
    explanation: str

    @field_validator("score", mode="before")
    @classmethod
    def normalize_score(cls, v):
        v = str(v).title()
        if "Low" in v:
            return "Low"
        if "High" in v:
            return "High"
        return "Medium"

class ScreeningResult(BaseModel):
    candidate_strengths: List[str]
    candidate_weaknesses: List[str]
    risk_factor: RiskReward
    reward_factor: RiskReward
    overall_fit_rating: int
    justification_for_rating: str
    Date: str
    Resume: str
    First_Name: str = ""
    Last_Name: str = ""
    Email: str = ""
    Full_Resume: str = ""

    Satisfied_Must_Haves: List[str] = []
    Missing_Must_Haves: List[str] = []
    Satisfied_Nice_Haves: List[str] = []
    Missing_Nice_Haves: List[str] = []

@retry(stop=stop_after_attempt(3), wait=wait_exponential())
def llm_json(llm, prompt, **kwargs):
    raw = (prompt | llm | StrOutputParser()).invoke(kwargs)
    match = re.search(r"\{.*\}", raw, flags=re.S)
    if not match:
        raise ValueError("Invalid JSON from LLM:\n" + raw)
    return json.loads(match.group(0))

def llm_text(llm, prompt, **kwargs):
    return (prompt | llm | StrOutputParser()).invoke(kwargs).strip()

def evaluate_single_resume(llm, fixed_jd, must_haves, nice_to_haves, resume_text, resume_link):
    """Evaluate one resume against the improved JD + explicit must/nice requirements."""
    must_haves_text = (
        "\n".join([f"- {x}" for x in must_haves]) if isinstance(must_haves, list) else str(must_haves or "")
    )
    nice_to_haves_text = (
        "\n".join([f"- {x}" for x in nice_to_haves]) if isinstance(nice_to_haves, list) else str(nice_to_haves or "")
    )

    data = llm_json(
        llm,
        RECRUITER_PROMPT,
        job_description=fixed_jd,
        must_have_requirements=must_haves_text,
        nice_to_have_requirements=nice_to_haves_text,
        resume_text=resume_text
    )

    sr = ScreeningResult(
        candidate_strengths=data.get("candidate_strengths", []),
        candidate_weaknesses=data.get("candidate_weaknesses", []),
        risk_factor=RiskReward(**data.get("risk_factor", {})),
        reward_factor=RiskReward(**data.get("reward_factor", {})),
        overall_fit_rating=int(data.get("overall_fit_rating", 0)),
        justification_for_rating=data.get("justification_for_rating", ""),
        Date=time.strftime("%Y-%m-%d %I:%M %p"),
        Resume=resume_link,
        Full_Resume=resume_text,  # Store full resume text
        Satisfied_Must_Haves=data.get("satisfied_must_haves", []),
        Missing_Must_Haves=data.get("missing_must_haves", []),
        Satisfied_Nice_Haves=data.get("satisfied_nice_to_haves", []),
        Missing_Nice_Haves=data.get("missing_nice_to_haves", []),
    )

    # Contact info
    try:
        ci = llm_json(llm, CONTACT_PROMPT, resume_text=resume_text)
        sr.First_Name = ci.get("First Name", "")
        sr.Last_Name = ci.get("Last Name", "")
        sr.Email = ci.get("Email Address", "")
    except Exception:
        emails = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z.-]+\.[A-Za-z]{2,}", resume_text)
        sr.Email = emails[0] if emails else ""

    return sr
def normalize_improved_jd(improved):
    """Convert the improved_job_description field into a readable text string."""
    if isinstance(improved, str):
        return improved
    if isinstance(improved, dict):
        parts = []
        for section, content in improved.items():
            parts.append(str(section).strip().upper())
            if isinstance(content, list):
                for item in content:
                    parts.append(f"- {str(item).strip()}")
            else:
                parts.append(str(content).strip())
            parts.append("")
        return "\n".join(parts).strip()
    return str(improved)

def results_to_dataframe(results: List[ScreeningResult]):
    rows = []
    for r in results:
        rows.append({
            "Date": r.Date,
            "Resume Link": r.Resume,
            "First Name": r.First_Name,
            "Last Name": r.Last_Name,
            "Email": r.Email,
            "Satisfied Must-Haves": "\n\n".join(r.Satisfied_Must_Haves or []),
            "Missing Must-Haves": "\n\n".join(r.Missing_Must_Haves or []),
            "Satisfied Nice-to-Haves": "\n\n".join(r.Satisfied_Nice_Haves or []),
            "Missing Nice-to-Haves": "\n\n".join(r.Missing_Nice_Haves or []),
            "Strengths": "\n\n".join(r.candidate_strengths or []),
            "Weaknesses": "\n\n".join(r.candidate_weaknesses or []),
            "Risk Factor": f"{r.risk_factor.score} - {r.risk_factor.explanation}",
            "Reward Factor": f"{r.reward_factor.score} - {r.reward_factor.explanation}",
            "Overall Fit": r.overall_fit_rating,
            "Justification": r.justification_for_rating,
            "Full Resume": r.Full_Resume,
        })
    return pd.DataFrame(rows)
def save_excels(df, out_dir, threshold):
    ensure_dir(out_dir)
    all_path = os.path.join(out_dir, "Resume_Screening_Results.xlsx")
    high_path = os.path.join(out_dir, "High_Scoring_Candidates.xlsx")

    df.to_excel(all_path, index=False)

    # Stricter high-scoring criteria:
    # - Overall fit above threshold
    # - AND no missing MUST-HAVEs (when that column exists)
    high_df = df[df["Overall Fit"] >= threshold].copy()
    if "Missing Must-Haves" in high_df.columns:
        high_df = high_df[high_df["Missing Must-Haves"].fillna("").str.strip().eq("")]

    high_df.to_excel(high_path, index=False)
    return all_path, high_path
def save_jd_pdf(fixed_jd: str, must_haves, nice_haves, improved_jd_dir: str) -> str:
    """Save the improved JD as a readable PDF + extracted Must/Nice requirements."""
    ensure_dir(improved_jd_dir)
    path = os.path.join(improved_jd_dir, "Improved_Job_Description.pdf")

    c = canvas.Canvas(path, pagesize=A4)
    width, height = A4
    x = 50
    y = height - 50

    def new_page():
        nonlocal y
        c.showPage()
        y = height - 50
        c.setFont("Helvetica", 11)

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(x, y, "Improved Job Description")
    y -= 30

    # Body JD
    c.setFont("Helvetica", 11)
    for line in fixed_jd.split("\n"):
        # Wrap long lines crudely by splitting
        chunks = []
        if len(line) <= 110:
            chunks = [line]
        else:
            # Simple wrap at ~110 chars
            buf = line
            while len(buf) > 110:
                chunks.append(buf[:110])
                buf = buf[110:]
            if buf:
                chunks.append(buf)

        for ch in chunks:
            if y < 60:
                new_page()
            c.drawString(x, y, ch)
            y -= 14

    # MUST-HAVES
    y -= 20
    if y < 80:
        new_page()
    c.setFont("Helvetica-Bold", 13)
    c.drawString(x, y, "MUST-HAVE REQUIREMENTS:")
    y -= 18
    c.setFont("Helvetica", 11)

    for req in (must_haves or []):
        if y < 60:
            new_page()
        c.drawString(x, y, f"- {req}")
        y -= 14

    # NICE-HAVES
    y -= 20
    if y < 80:
        new_page()
    c.setFont("Helvetica-Bold", 13)
    c.drawString(x, y, "NICE-TO-HAVE REQUIREMENTS:")
    y -= 18
    c.setFont("Helvetica", 11)

    for req in (nice_haves or []):
        if y < 60:
            new_page()
        c.drawString(x, y, f"- {req}")
        y -= 14

    c.save()
    return path
def extract_position_from_jd(llm, jd_text: str) -> str:
    """Extract the job position/title from the job description."""
    try:
        position = llm_text(llm, POSITION_EXTRACT_PROMPT, jd_text=jd_text)
        return position.strip()
    except:
        # Fallback: try to extract from common patterns
        patterns = [
            r"Job Title[:\s]+([^\n]+)",
            r"Position[:\s]+([^\n]+)",
            r"Role[:\s]+([^\n]+)",
            r"Title[:\s]+([^\n]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, jd_text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return "AI Specialist"  # Default fallback

# ============================================================================
# END CV FILTERING FUNCTIONS
# ============================================================================


def load_candidates_from_excel(file_path):
    """Load candidates from Excel file (High_Scoring_Candidates.xlsx)"""
    try:
        df = pd.read_excel(file_path)
        required_cols = ['First Name', 'Last Name', 'Full Resume']
        if not all(col in df.columns for col in required_cols):
            st.error(f"Excel file must contain columns: {', '.join(required_cols)}")
            return None
        
        # Ensure Full Resume is not empty or NaN
        # Fill NaN values with empty string and convert to string
        df['Full Resume'] = df['Full Resume'].fillna('').astype(str)
        
        # Check if any resumes are missing or too short
        missing_resumes = df[df['Full Resume'].str.len() < 50]
        if len(missing_resumes) > 0:
            st.warning(f"⚠️ {len(missing_resumes)} candidate(s) have missing or very short resume text in the Excel file.")
        
        return df
    except Exception as e:
        st.error(f"Failed to read Excel: {e}")
        return None


def sort_candidates(df):
    if 'Overall Fit' in df.columns:
        df = df.sort_values(
            by=['Overall Fit', 'First Name', 'Last Name'],
            ascending=[False, True, True],
            na_position='last'
        ).reset_index(drop=True)
    else:
        df = df.sort_values(
            by=['First Name', 'Last Name'],
            ascending=[True, True]
        ).reset_index(drop=True)
    return df


def get_audio_duration(audio_path):
    """Get exact audio duration using pydub"""
    if not PYDUB_AVAILABLE:
        # Fallback: estimate from file size
        file_size_kb = os.path.getsize(audio_path) / 1024
        return max(file_size_kb / 50, 2)

    try:
        audio = AudioSegment.from_mp3(audio_path)
        duration_seconds = len(audio) / 1000
        return duration_seconds
    except Exception as e:
        file_size_kb = os.path.getsize(audio_path) / 1024
        return max(file_size_kb / 50, 2)


def play_audio_html5(audio_path, wait=True):
    """Play audio using Streamlit's native audio player with autoplay"""
    print(f"[DEBUG play_audio_html5] Called with path: {audio_path}")
    print(f"[DEBUG play_audio_html5] Wait: {wait}")
    
    if not os.path.exists(audio_path):
        st.error("Audio file not found")
        print(f"[ERROR play_audio_html5] File does not exist: {audio_path}")
        return 0.0

    try:
        # Get exact duration
        duration = get_audio_duration(audio_path)
        print(f"[DEBUG play_audio_html5] Duration: {duration}s")

        # Read audio file
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        
        print(f"[DEBUG play_audio_html5] Read {len(audio_bytes)} bytes")

        # Detect audio format
        file_ext = os.path.splitext(audio_path)[1].lower()
        audio_format = "audio/mp3" if file_ext == ".mp3" else "audio/wav"
        print(f"[DEBUG play_audio_html5] Audio format: {audio_format}, extension: {file_ext}")

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
        # Note: Streamlit's autoplay parameter name
        st.audio(audio_bytes, format=audio_format, autoplay=True)
        print(f"[DEBUG play_audio_html5] Streamlit audio player rendered with autoplay")

        # Wait for audio to finish playing if requested
        if wait:
            time.sleep(duration + 1)
            print(f"[DEBUG play_audio_html5] Waited {duration + 1}s for playback")
            
        return duration

    except Exception as e:
        st.error(f"Error playing audio: {str(e)}")
        print(f"[EXCEPTION play_audio_html5] {str(e)}")
        import traceback
        traceback.print_exc()
        return 0.0


st.set_page_config(page_title="IntiqAI", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for beautiful UI
st.markdown("""
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
""", unsafe_allow_html=True)

# Header with gradient
st.markdown("""
<div class="main-header">
    <h1>IntiqAI</h1>
    <p>AI-powered voice-based technical interviews</p>
</div>
""", unsafe_allow_html=True)

if "app_page" not in st.session_state:
    st.session_state.app_page = "cv_filtering_setup"

if "candidates_df" not in st.session_state:
    st.session_state.candidates_df = None

if "selected_candidate_index" not in st.session_state:
    st.session_state.selected_candidate_index = None

if "state" not in st.session_state:
    st.session_state.state = None

if "last_processed_audio" not in st.session_state:
    st.session_state.last_processed_audio = None

if "job_description" not in st.session_state:
    st.session_state.job_description = ""

if "position" not in st.session_state:
    st.session_state.position = "AI Specialist"

if "filtering_in_progress" not in st.session_state:
    st.session_state.filtering_in_progress = False

if "filtering_status" not in st.session_state:
    st.session_state.filtering_status = ""

if "excel_output_path" not in st.session_state:
    st.session_state.excel_output_path = None

if "all_results_path" not in st.session_state:
    st.session_state.all_results_path = None

# TIMING ANALYSIS: Track if time analysis PDF was generated (NEW)
if "time_analysis_generated" not in st.session_state:
    st.session_state.time_analysis_generated = False

if "stt_model" not in st.session_state:
    st.session_state.stt_model = "whisper_base"

if "stt_model_label" not in st.session_state:
    st.session_state.stt_model_label = "Whisper Base (balanced)"

# Performance optimization settings - ALWAYS ON
st.session_state.use_tts_cache = True  # Always use cache


# --- CACHE WORKFLOW TO PREVENT RE-INITIALIZATION ISSUES ---
@st.cache_resource
def get_workflow():
    return build_workflow()

workflow = get_workflow()
# -----------------------------------------------------------

# TIMING ANALYSIS: Initialize timing state (NEW)
ensure_session(st.session_state)


# Page 0: CV Filtering Setup
if st.session_state.app_page == "cv_filtering_setup":
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
            
            # Create a hidden root window
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            
            # Open directory picker
            directory = filedialog.askdirectory(
                title="Select Directory with CV PDF Files",
                initialdir=st.session_state.get("resumes_dir", os.getcwd())
            )
            
            root.destroy()
            return directory
        except Exception as e:
            st.error(f"Error opening directory browser: {str(e)}")
            return None

    # --- CALLBACK FUNCTION (The Fix) ---
    def on_browse_click():
        """Callback to handle directory selection before UI reruns"""
        selected_dir = browse_directory()
        if selected_dir:
            # Update both the internal variable and the widget's key
            st.session_state.resumes_dir = selected_dir
            st.session_state.resumes_dir_input = selected_dir
    # -----------------------------------
    
    # Initialize resumes_dir in session state if not present
    if "resumes_dir" not in st.session_state:
        st.session_state.resumes_dir = ""
    
    col1, col2 = st.columns([4, 1])
    with col1:
        # The text input widget
        resumes_dir = st.text_input(
            "📁 Directory Path with CV PDF Files:",
            value=st.session_state.resumes_dir,
            placeholder=r"e.g., C:\Users\96658\Desktop\TalentTalk\uploaded_resumes",
            help="Enter the full path to the directory containing the CV PDF files, or click 'Browse' to select a directory",
            key="resumes_dir_input"
        )
        
        # Sync manual typing
        if resumes_dir != st.session_state.resumes_dir:
            st.session_state.resumes_dir = resumes_dir

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        # Use on_click to trigger the update safely
        st.button("📂 Browse", on_click=on_browse_click, use_container_width=True, help="Open file browser", key="browse_dir_btn")
    
    # --- Logic below handles validation and next steps ---
    
    # Path validation and directory listing
    # (Use st.session_state.resumes_dir to ensure we are checking the latest value)
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
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(jd_file.read())
                tmp_path = tmp_file.name
            
            # Extract text from PDF
            job_description = read_pdf_text_from_path(tmp_path)
            job_description = clean_whitespace(job_description)
            
            # Clean up temporary file
            os.unlink(tmp_path)
            
            if job_description:
                st.success(f"✅ Job description loaded successfully ({len(job_description)} characters)")
                # Show preview
                with st.expander("📄 Preview Job Description (first 500 characters)"):
                    st.text(job_description[:500] + "..." if len(job_description) > 500 else job_description)
        except Exception as e:
            st.error(f"❌ Error reading PDF file: {str(e)}")
            job_description = ""
    elif st.session_state.get("job_description"):
        # Use previously loaded job description if available
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

# Page 0.5: Filtering Process
elif st.session_state.app_page == "filtering_process":
    st.markdown("""
    <div style="background: white; padding: 2rem; border-radius: 15px; margin-bottom: 2rem; box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
        <h2 style="color: #333; margin-bottom: 1rem;">🔄 CV Filtering in Progress</h2>
    </div>
    """, unsafe_allow_html=True)
    
    status_placeholder = st.empty()
    progress_placeholder = st.empty()
    
    # Run the filtering process
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
            must_haves = [x.strip("-• 	") for x in must_haves.splitlines() if x.strip()]
        if isinstance(nice_haves, str):
            nice_haves = [x.strip("-• 	") for x in nice_haves.splitlines() if x.strip()]
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
        st.session_state.candidates_df = load_candidates_from_excel(high_path)
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

# Page 1: Candidate List View
elif st.session_state.app_page == "candidate_list":

    if st.session_state.candidates_df is not None:
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
            # Create a card for each candidate
            st.markdown(f"""
            <div style="background: white; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem; 
                        box-shadow: 0 2px 5px rgba(0,0,0,0.05); border: 1px solid #f0f0f0;">
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([3, 1, 1])

            with col1:
                st.markdown(f"### 👤 {row['First Name']} {row['Last Name']}")

            with col2:
                if 'Overall Fit' in st.session_state.candidates_df.columns:
                    score = row.get('Overall Fit', 'N/A')
                    if score != 'N/A':
                        st.metric("Score", f"{score}/10")
                    else:
                        st.write("Score: N/A")

            with col3:
                if st.button(f"🎤 Start Interview", key=f"interview_{index}", use_container_width=True):
                    st.session_state.selected_candidate_index = index
                    candidate_info = st.session_state.candidates_df.loc[index]

                    # Get resume text and validate
                    resume_text = candidate_info.get('Full Resume', '')
                    if not resume_text or len(str(resume_text).strip()) < 50:
                        st.error(f"❌ Resume text is missing or too short for candidate {candidate_info.get('First Name', '')} {candidate_info.get('Last Name', '')}. Please check the Excel file.")
                        st.stop()
                    
                    # Get position with fallback
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
                    # TIMING ANALYSIS: Reset timing data (NEW)
                    st.session_state.timings = []
                    st.session_state.turn_idx = 0
                    st.session_state.time_analysis_generated = False
                    
                    # Set flags for auto-start
                    st.session_state.auto_start_interview = True
                    st.session_state.interview_stage = "switching"  # switching -> generating -> playing
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
                # Load all results
                all_results_df = pd.read_excel(st.session_state.all_results_path)
                
                # Get qualified candidate names (First Name + Last Name combination)
                qualified_names = set()
                if st.session_state.candidates_df is not None:
                    for _, row in st.session_state.candidates_df.iterrows():
                        first_name = str(row.get('First Name', '')).strip()
                        last_name = str(row.get('Last Name', '')).strip()
                        qualified_names.add((first_name.lower(), last_name.lower()))
                
                # Filter non-qualified candidates
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
                
                # Display non-qualified candidates section
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
                        
                        # Justification in expander
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


# Page 2: Interview View
elif st.session_state.app_page == "interview":
    candidate_info = st.session_state.candidates_df.loc[st.session_state.selected_candidate_index]

    # Beautiful sidebar
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

    def process_message(user_input):
        if not st.session_state.state.get("position"):
            st.error("Please enter a position in the sidebar before starting the interview.")
            return
        
        # Validate input is not empty
        if not user_input or len(user_input.strip()) < 3:
            st.error("Invalid input: message is too short or empty.")
            return

        st.session_state.state["messages"].append(HumanMessage(content=user_input.strip()))
        
        # TIMING ANALYSIS: Increment turn counter (NEW)
        st.session_state.turn_idx += 1
        turn = st.session_state.turn_idx

        try:
            # Validate state before invoking
            if not st.session_state.state.get("resume_text"):
                st.error("❌ Resume text is missing. Cannot proceed with interview.")
                return
            
            with st.spinner("Recruiter..."):
                # TIMING ANALYSIS: Time AI generation (NEW)
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
                    
                    # Check for error messages from the workflow
                    if ai_text and ("error" in ai_text.lower() or "apologize" in ai_text.lower() or "encountered" in ai_text.lower()):
                        st.warning("⚠️ The AI workflow may have encountered an error. Check the console for details.")
                        import traceback
                        st.code(f"AI Response: {ai_text}\n\nState: {st.session_state.state.get('position', 'N/A')}\nResume length: {len(str(st.session_state.state.get('resume_text', '')))}")

                    # Debug: Check if this is a tool-calling message with no content
                    if not ai_text or ai_text.strip() == "":
                        # This might be a tool call message - check if there are tool_calls
                        if hasattr(ai_message, 'tool_calls') and ai_message.tool_calls:
                            st.info("🔧 AI is retrieving information from knowledge base...")
                            # The workflow will continue automatically, just don't generate TTS
                        else:
                            st.warning("Recruiter Response is empty.")
                    else:
                        st.subheader("Recruiter Response")
                        st.write(ai_text)

                        # Generate voice using Kokoro TTS
                        with st.spinner("Generating voice..."):
                            try:
                                # TIMING ANALYSIS: Time TTS generation (NEW)
                                tts_tok = timing_start("tts_generate", {"chars": len(ai_text), "use_cache": st.session_state.use_tts_cache})
                                
                                if USE_OPTIMIZED:
                                    audio_path, tts_error = elevenlabs_tts(ai_text, use_cache=st.session_state.use_tts_cache)
                                else:
                                    audio_path, tts_error = elevenlabs_tts(ai_text)
                                
                                st.session_state.timings.append(timing_end(tts_tok, turn))

                                if tts_error:
                                    st.error(f"Voice generation failed: {tts_error}")
                                elif audio_path and os.path.exists(audio_path):
                                    # TIMING ANALYSIS: Time audio playback (NEW)
                                    pb_tok = timing_start("playback_wait", {"audio_path": audio_path})
                                    waited = play_audio_html5(audio_path)  # Default wait=True for regular messages
                                    pb_timing = timing_end(pb_tok, turn)
                                    pb_timing.meta["waited_s"] = waited
                                    st.session_state.timings.append(pb_timing)
                                else:
                                    st.error("Audio file not found")

                            except Exception as e:
                                st.error(f"Voice generation error: {str(e)}")
                                import traceback
                                traceback.print_exc()

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
        
        # Add an initial placeholder message to start the conversation
        st.session_state.state["messages"].append(HumanMessage(content="[Interview Started]"))
        
        # Process the first message to get recruiter's introduction
        with st.spinner("Starting interview..."):
            # TIMING ANALYSIS: Increment turn counter
            st.session_state.turn_idx += 1
            turn = st.session_state.turn_idx
            
            try:
                # Validate state before invoking workflow
                if not st.session_state.state.get("resume_text"):
                    raise ValueError("Resume text is missing. Please check the candidate data.")
                
                if not st.session_state.state.get("position"):
                    st.session_state.state["position"] = "AI Specialist"
                
                # TIMING ANALYSIS: Time AI generation
                ai_tok = timing_start("ai_generation", {"len_messages": len(st.session_state.state["messages"])})
                result = workflow.invoke(st.session_state.state)
                st.session_state.timings.append(timing_end(ai_tok, turn))
                
                for key, value in result.items():
                    if key == "messages":
                        st.session_state.state["messages"] = value
                    else:
                        st.session_state.state[key] = value
                
                # Remove the [Interview Started] placeholder message
                if st.session_state.state["messages"] and st.session_state.state["messages"][0].content == "[Interview Started]":
                    st.session_state.state["messages"].pop(0)
                
                # Get the AI's introduction message and generate audio
                if st.session_state.state["messages"]:
                    ai_message = st.session_state.state["messages"][-1]
                    if isinstance(ai_message, AIMessage):
                        ai_text = ai_message.content
                        
                        # Check if AI returned an error message
                        if ai_text and ("error" in ai_text.lower() or "apologize" in ai_text.lower()):
                            # This might indicate a workflow error - log it
                            st.warning(f"⚠️ AI returned a potential error message. Check the workflow configuration.")
                        
                        # Generate audio (but don't play yet)
                        if ai_text and ai_text.strip():
                            tts_tok = timing_start("tts_generate", {"chars": len(ai_text)})
                            
                            if USE_OPTIMIZED:
                                audio_path, tts_error = elevenlabs_tts(ai_text, use_cache=True)
                            else:
                                audio_path, tts_error = elevenlabs_tts(ai_text)
                            
                            st.session_state.timings.append(timing_end(tts_tok, turn))
                            
                            if not tts_error and audio_path and os.path.exists(audio_path):
                                # Store audio to play after UI updates
                                st.session_state.intro_audio_path = audio_path
                                st.session_state.intro_audio_turn = turn
                
                # Force rerun to display the message
                st.rerun()
                
            except Exception as e:
                error_msg = f"Failed to start interview: {str(e)}"
                st.error(error_msg)
                import traceback
                st.code(traceback.format_exc())
                # Remove the placeholder message if error occurs
                if st.session_state.state.get("messages") and st.session_state.state["messages"][0].content == "[Interview Started]":
                    st.session_state.state["messages"].pop(0)

    # Display transcript in a beautiful container
    if st.session_state.state["messages"]:
        st.markdown("""
        <div class="transcript-container">
            <h3 style="color: #333; margin-bottom: 1.5rem;">💬 Interview Transcript</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Create a scrollable container for messages
        transcript_container = st.container()
        with transcript_container:
            for m in st.session_state.state["messages"]:
                if isinstance(m, HumanMessage):
                    # Skip placeholder messages
                    if m.content != "[Interview Started]":
                        st.markdown(f"""
                        <div class="candidate-message" style="color: #333333 !important;">
                            <strong style="color: #667eea !important; font-size: 1.1em;">👤 Candidate:</strong><br>
                            <span style="color: #333333 !important; font-size: 1em; line-height: 1.6;">{m.content}</span>
                        </div>
                        """, unsafe_allow_html=True)
                elif isinstance(m, AIMessage):
                    # Only display AIMessages that have actual content
                    # Skip messages that have both content AND tool_calls (these are just tool announcements)
                    if m.content and m.content.strip():
                        # Check if this message also has tool_calls - if so, skip it (it's a duplicate)
                        has_tool_calls = hasattr(m, 'tool_calls') and m.tool_calls
                        if not has_tool_calls:
                            st.markdown(f"""
                            <div class="recruiter-message" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white !important; padding: 1rem 1.5rem; border-radius: 15px; margin: 1rem 0; border-left: 4px solid #764ba2;">
                                <strong style="color: white !important; font-size: 1.1em;">🤖 Recruiter:</strong><br>
                                <span style="color: white !important; font-size: 1em; line-height: 1.6;">{m.content}</span>
                            </div>
                            """, unsafe_allow_html=True)
    
    # Play intro audio if available using Streamlit's native audio
    if hasattr(st.session_state, 'intro_audio_path') and st.session_state.intro_audio_path:
        audio_path = st.session_state.intro_audio_path
        turn = st.session_state.intro_audio_turn
        
        # Display audio player (non-blocking)
        with open(audio_path, 'rb') as audio_file:
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/mp3', autoplay=True)
        
        # Track timing
        duration = get_audio_duration(audio_path)
        pb_tok = timing_start("playback_wait", {"audio_path": audio_path})
        pb_timing = timing_end(pb_tok, turn)
        pb_timing.meta["waited_s"] = duration
        st.session_state.timings.append(pb_timing)
        
        # Clean up
        del st.session_state.intro_audio_path
        del st.session_state.intro_audio_turn
    
    # Beautiful fixed-style recording bar at the bottom
    # Check if interview has ended BEFORE showing microphone
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
        
        # Create columns for centered microphone
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            # Place the audio recorder
            audio_bytes = audio_recorder(
                text="Click to start/stop recording",
                recording_color="#764ba2",
                neutral_color="#667eea",
                icon_name="microphone",
                pause_threshold=10.0,   #allow 10 seconds of silence
                sample_rate=44100,     #better quality, more stable
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
                    # TIMING ANALYSIS: Time STT transcription (NEW)
                    stt_tok = timing_start("stt_transcribe", {"format": "wav", "model": st.session_state.stt_model})
                    
                    if USE_OPTIMIZED:
                        transcribed_text, stt_error = transcribe_audio_file(audio_path, stt_model=st.session_state.stt_model, max_wait=180)
                    else:
                        transcribed_text, stt_error = transcribe_audio_file(audio_path, max_wait=180)
                        
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

    # TIMING ANALYSIS: Auto-generate Time Analysis PDF silently when interview ends (NEW)
    if interview_ended and not st.session_state.time_analysis_generated and st.session_state.get("timings"):
        try:
            tz = pytz.timezone('Asia/Kuwait')
            dt_display = datetime.now(tz).strftime("%Y-%m-%d %H-%M-%S")
            candidate_name = f"{candidate_info['First Name']} {candidate_info['Last Name']}"
            position = st.session_state.state.get("position", "AI Specialist")
            
            # Generate PDF silently in background
            pdf_path = generate_time_analysis_pdf(
                candidate_name=candidate_name,
                position=position,
                timings=st.session_state.timings,
                dt_display_local=dt_display,
                tz_label="UTC+3"
            )
            st.session_state.time_analysis_generated = True
            # Note: PDF generated silently, saved to: C:\Users\96658\Desktop\TalentTalk\Time Ananlysis Reports
        except Exception as e:
            # Silent failure - don't show errors to user
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

    # Display evaluation result with better styling
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
        # Show placeholder if interview ended but no evaluation yet
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
                # TIMING ANALYSIS: Clear timing data (NEW)
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