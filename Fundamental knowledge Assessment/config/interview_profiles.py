INTERVIEWER_PROFILES = {
    "professional": {
        "name": "Technical Interviewer",
        "style": "Thorough evaluation focused on fundamental knowledge",
        "temperature": 0.3,
        "scoring_style": "strict",
        "tone": "formal"
    }
}

# Number of fundamental questions to generate
# Can be overridden via FUNDAMENTAL_QUESTIONS_COUNT environment variable
DEFAULT_FUNDAMENTAL_QUESTIONS_COUNT = 5