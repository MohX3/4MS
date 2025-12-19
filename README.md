## 4MSHire AI – Voice‑Based Technical Interview & CV Screening

4MSHire AI is a **Streamlit** web app that:
- **Filters CVs** against a Job Description using Google Gemini and LangChain
- **Runs voice‑based technical interviews** with candidates, including STT (Whisper / AssemblyAI) and TTS (Edge)
- **Generates HR reports and timing analysis PDFs** for each interview

This README explains how any team member can **set up, run, and use** the project step‑by‑step.

---

## 1. Prerequisites

- **Python**: 3.10 or 3.11 (recommended; Whisper/AI tooling can be fragile on other versions)
- **Git**
- A GitHub account (for pulling/pushing code)
- Recommended OS: Windows 10/11, macOS, or Linux

Optional but recommended:
- **Google AI Studio / Google Cloud** account for a **Gemini API key**
- **AssemblyAI** account for cloud speech‑to‑text (if you want best STT quality)

---

## 2. Clone the Repository

On each team member’s machine:

```bash
git clone https://github.com/MohX3/4MS.git
cd 4MS
```

If you’re already in the project folder (because you copied it manually), just make sure your terminal path is the project root (where `interview.py` and `requirements.txt` live).

---

## 3. Create and Activate a Virtual Environment

### Windows (PowerShell)

```powershell
cd 4MS
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
cd 4MS
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt after activation.

---

## 4. Install Python Dependencies

With the virtual environment **activated**:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:
- `streamlit` – web UI framework
- `langchain`, `langgraph`, `google-genai`, `langchain_google_genai` – LLM orchestration & Gemini
- `openai-whisper`, `assemblyai`, `audio-recorder-streamlit`, `pydub` – audio/STT
- `reportlab`, `fpdf` – PDF report generation
and other utilities listed in `requirements.txt`.

---

## 5. Environment Variables (.env Setup)

The app loads environment variables via `python-dotenv` at the top of `interview.py`:


### 5.1 Create `.env`

In the project root (`4MS`), create a file named `.env`:

```text
GOOGLE_API_KEY=your_google_api_key_here
```

Notes:
- `GOOGLE_API_KEY` **must** be set; otherwise the app raises an error on startup.

If you have an `.env.example` file, you can copy it:

```bash
cp .env.example .env   # macOS/Linux
copy .env.example .env # Windows PowerShell
```

Then fill in your own keys.

---

## 6. Running the App

From the project root, with the virtual environment **activated** and `.env` configured:

```bash
streamlit run interview.py
```

Streamlit will print a local URL such as:

```text
Local URL: http://localhost:8501
```

Open that URL in your browser to use the 4MSHire AI app.

---

## 7. Using the App – CV Filtering Workflow

When you open the app (`4MSHire AI`):

### 7.1 CV Filtering Setup Page

- **Directory Path with CV PDF Files**
  - Choose or type the folder path that contains **candidate CVs in PDF format**.
  - On Windows this might look like:
    - `C:\Users\YOUR_NAME\Desktop\4MS\uploaded_resumes`
  - You can click **“📂 Browse”** in the UI to select the folder.

- **Job Description (PDF File)**
  - Upload a **Job Description PDF** (e.g. `Job Descriptions/Front_End_Developer_General_JD.pdf`).
  - The app extracts text, cleans it, and uses it for AI‑based screening.

Then click:

- **“🚀 Start CV Filtering Process”**

The app will:
- Improve/structure the Job Description (Gemini)
- Scan all CV PDFs in the directory
- Score candidates and generate:
  - `Filtered Resumes/Resume Screening Results/Resume_Screening_Results.xlsx`
  - `Filtered Resumes/High Scoring Candidates/High_Scoring_Candidates.xlsx`
  - An improved JD PDF in `Improved Job Descriptions/`

After processing, it navigates to the **Candidate List** page.

---

## 8. Candidate List & Starting Interviews

On the **Candidate List** page:

- You’ll see a card per candidate (name and overall fit score).
- For any candidate, click **“🎤 Start Interview”**.

The app will:
- Load the candidate’s **full resume text** from the Excel file.
- Use the improved Job Description to set the **position** (e.g. “AI Specialist”).
- Initialize the interview workflow (LangGraph / LLM agents).

You’ll then be taken to the **Interview** page.

---

## 9. Interview Page – Voice‑Based Technical Interview

### 9.1 Sidebar Settings

- **Interviewer Mode**: `friendly`, `formal`, or `technical`
- **Number of Technical Questions** and **Follow‑up Questions**
- **Speech‑to‑Text Engine**:
  - Whisper Tiny/Base/Small/Medium/Large (local, via `openai-whisper`)
  - AssemblyAI (cloud; requires `ASSEMBLYAI_API_KEY`)
- **Optional Questions PDF**:
  - Upload a PDF with custom interview questions if desired.

### 9.2 Interview Flow

- The AI recruiter introduces the interview (audio + text).
- At the bottom, use the **microphone widget** to record your spoken answers.
- The app:
  - Records audio
  - Transcribes it using the selected STT model
  - Sends it to the LLM workflow
  - Plays back the recruiter’s next question/response via TTS

The app automatically logs **timings** (generation, TTS, playback, STT) and creates **time analysis PDFs** under `Time Ananlysis Reports/`.

When the interview ends (the AI says something like “that’s it for today”), you’ll see an option to **generate evaluation and HR report**.

---

## 10. Evaluation, Reports, and Exports

After the interview finishes:

- Click **“📊 Generate Evaluation and Report”**.
- The workflow will:
  - Analyze the candidate’s performance
  - Generate a detailed **evaluation text**
  - Create an **HR report PDF** saved in `generated_reports/`
- If available, a **Download PDF Report** button will appear in the UI.
- Time analysis PDFs are created automatically in `Time Ananlysis Reports/` (for before/after or per‑interview timing).

---

## 11. Team Git Workflow (Collaboration)

For collaboration on GitHub, see `GITHUB_SETUP.md` in this repo. In summary:

- **Get the latest changes** on your existing clone:
  - `git checkout main`                # switch to main branch
  - `git pull origin main`             # pull latest from GitHub


- **Make changes and push**:
  - `git add .`
  - `git commit -m "Clear description of your changes"`
  - `git push -u origin feature/your-feature-name`

---

## 12. Quick Troubleshooting

- **`GOOGLE_API_KEY environment variable is not set`**
  - Check that `.env` exists in the project root and contains `GOOGLE_API_KEY=...`.
  - Restart the terminal after editing `.env` if necessary.

- **Streamlit app doesn’t open**
  - Confirm virtual env is active.
  - Run `streamlit run interview.py` from the project root.

- **No CVs found**
  - Check that the directory path is correct and contains `.pdf` files.

- **Whisper / audio errors**
  - Make sure `ffmpeg` is installed on your system if Whisper or `pydub` complain.
  - On Windows, you may need to add `ffmpeg` to your PATH.

If your team runs into any other setup issues, capture the exact error message and command you ran, and you can update this README with additional hints as needed.


