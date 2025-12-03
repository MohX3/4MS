import json
from groq import Groq

class QuestionGenerator:
    def __init__(self, api_key, model=None):
        self.client = Groq(api_key=api_key)
        # Use the provided model or default to llama-3.1-8b-instant
        self.model = model if model else "llama-3.1-8b-instant"
    
    def generate_fundamental_questions(self, role: str, job_description: str):
        """Generate fundamental coding/knowledge questions based on JD"""
        
        prompt = f"""Generate 5 fundamental questions for a {role} position.

Job Description: {job_description[:1500]}

Focus on essential knowledge that filters unqualified candidates.
Keep questions concise (1-2 sentences each).

For technical/engineering roles, include:
- 1-2 coding problems
- 2-3 concept questions  
- 1 tool/library question

For non-technical roles, focus on:
- Core concept questions
- Practical scenario questions
- Tool/process questions

Return JSON format with exactly 5 questions:
{{
  "questions": [
    {{
      "question": "question text",
      "type": "coding/concept/library",
      "difficulty": "basic/intermediate",
      "expected_keywords": ["keyword1", "keyword2"]
    }}
  ]
}}

Make questions specific to the role and JD."""
        
        try:
            print(f"Generating fundamental questions from JD using {self.model}...")
            
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a technical interviewer generating fundamental knowledge questions. Be concise and practical."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model,
                temperature=0.7,
                max_tokens=800,
                response_format={"type": "json_object"}
            )
            
            response_text = chat_completion.choices[0].message.content
            
            # Parse JSON response
            questions_data = json.loads(response_text.strip())
            questions = questions_data.get("questions", [])
            
            # Ensure we have exactly 5 questions
            if len(questions) > 5:
                questions = questions[:5]
            elif len(questions) < 5:
                # Add fallback questions if needed
                fallback = self._get_fallback_questions(role)
                questions.extend(fallback[:5 - len(questions)])
            
            print(f"Successfully generated {len(questions)} questions.")
            return questions
            
        except Exception as e:
            print(f"Question generation error: {e}")
            print("Using fallback questions instead.")
            return self._get_fallback_questions(role)[:5]
    
    def _get_fallback_questions(self, role):
        """Fallback fundamental questions"""
        role_lower = role.lower()
        
        if any(keyword in role_lower for keyword in ['engineer', 'developer', 'programmer', 'software']):
            return [
                {
                    "question": "Write a function to find the maximum value in a list/array.",
                    "type": "coding",
                    "difficulty": "basic",
                    "expected_keywords": ["function", "loop", "max", "array", "iterate"]
                },
                {
                    "question": "What is the difference between a list and a tuple in Python?",
                    "type": "concept",
                    "difficulty": "basic",
                    "expected_keywords": ["mutable", "immutable", "modifiable", "fixed"]
                },
                {
                    "question": "Explain what a REST API is and give an example HTTP method.",
                    "type": "concept",
                    "difficulty": "intermediate",
                    "expected_keywords": ["representational", "state", "transfer", "http", "get", "post"]
                },
                {
                    "question": "What is SQL injection and how can it be prevented?",
                    "type": "concept",
                    "difficulty": "basic",
                    "expected_keywords": ["parameterized", "queries", "input", "validation", "sanitize"]
                },
                {
                    "question": "Name three HTTP status codes and their meanings.",
                    "type": "concept",
                    "difficulty": "basic",
                    "expected_keywords": ["200", "404", "500", "ok", "not found", "error"]
                }
            ]
        elif any(keyword in role_lower for keyword in ['data', 'analyst', 'scientist', 'ml', 'ai']):
            return [
                {
                    "question": "What is the difference between supervised and unsupervised learning?",
                    "type": "concept",
                    "difficulty": "basic",
                    "expected_keywords": ["labeled", "unlabeled", "classification", "clustering"]
                },
                {
                    "question": "Explain what overfitting is and one way to prevent it.",
                    "type": "concept",
                    "difficulty": "intermediate",
                    "expected_keywords": ["regularization", "cross-validation", "training", "test"]
                },
                {
                    "question": "What is a JOIN in SQL and explain INNER JOIN vs LEFT JOIN.",
                    "type": "concept",
                    "difficulty": "basic",
                    "expected_keywords": ["combine", "tables", "matching", "all rows"]
                },
                {
                    "question": "What is the difference between classification and regression?",
                    "type": "concept",
                    "difficulty": "basic",
                    "expected_keywords": ["categorical", "continuous", "discrete", "predict"]
                },
                {
                    "question": "Why is feature scaling important in machine learning?",
                    "type": "concept",
                    "difficulty": "intermediate",
                    "expected_keywords": ["normalization", "standardization", "scale", "algorithms"]
                }
            ]
        elif any(keyword in role_lower for keyword in ['frontend', 'ui', 'ux', 'web']):
            return [
                {
                    "question": "What is the difference between HTML and HTML5?",
                    "type": "concept",
                    "difficulty": "basic",
                    "expected_keywords": ["semantic", "elements", "audio", "video", "canvas"]
                },
                {
                    "question": "Explain the CSS box model.",
                    "type": "concept",
                    "difficulty": "basic",
                    "expected_keywords": ["margin", "border", "padding", "content"]
                },
                {
                    "question": "What is the difference between == and === in JavaScript?",
                    "type": "concept",
                    "difficulty": "basic",
                    "expected_keywords": ["equality", "strict", "type", "coercion"]
                },
                {
                    "question": "What is React and why is it popular?",
                    "type": "concept",
                    "difficulty": "basic",
                    "expected_keywords": ["components", "virtual dom", "state", "props"]
                },
                {
                    "question": "What are CSS media queries used for?",
                    "type": "concept",
                    "difficulty": "basic",
                    "expected_keywords": ["responsive", "design", "breakpoints", "screen size"]
                }
            ]
        else:
            return [
                {
                    "question": "Based on the job description, what are the most important skills for this role?",
                    "type": "concept",
                    "difficulty": "basic",
                    "expected_keywords": []
                },
                {
                    "question": "What tools or technologies mentioned are you familiar with?",
                    "type": "concept",
                    "difficulty": "basic",
                    "expected_keywords": []
                },
                {
                    "question": "Describe a time you had to learn a new technology quickly.",
                    "type": "concept",
                    "difficulty": "intermediate",
                    "expected_keywords": ["learning", "adaptation", "project", "research"]
                },
                {
                    "question": "How do you stay updated with industry trends?",
                    "type": "concept",
                    "difficulty": "basic",
                    "expected_keywords": ["blogs", "courses", "conferences", "networking"]
                },
                {
                    "question": "What is your approach to debugging or problem-solving?",
                    "type": "concept",
                    "difficulty": "intermediate",
                    "expected_keywords": ["methodical", "testing", "isolate", "document"]
                }
            ]
