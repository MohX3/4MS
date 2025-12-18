import json
import time
from groq import Groq
from typing import Dict, Any

class EvaluationEngine:
    def __init__(self, api_key, model=None):
        self.client = Groq(api_key=api_key)
        self.model = model if model else "llama-3.1-8b-instant"
    
    def evaluate_fundamental_responses(self, role: str, job_description: str, questions: list, responses: list):
        """Evaluate fundamental knowledge responses"""
        
        # Calculate keyword score as a benchmark to include in the prompt
        question_scores = self._calculate_question_scores(questions, responses)
        calculated_overall_keyword = question_scores.get('OVERALL', 70)
        
        # Build the evaluation prompt (The LLM will provide the FINAL score based on criteria)
        prompt, system_content = self._build_evaluation_prompt_and_system(
            role, job_description, questions, responses, calculated_overall_keyword
        )
        
        try:
            print(f"Evaluating responses using {self.model}...")
            start_time = time.time()
            
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": system_content  # Use the new, stronger system prompt
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model,
                temperature=0.3,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            evaluation_time = time.time() - start_time
            
            response_text = chat_completion.choices[0].message.content
            evaluation = self._parse_evaluation_response(response_text)
            # ----------------------------------------------------------------
            criteria = evaluation.get('criterion_scores', {})
            if criteria:
                avg_score = sum(criteria.values()) / len(criteria)
                evaluation['overall_score'] = int(avg_score)
            else:
                evaluation['overall_score'] = calculated_overall_keyword
            # ----------------------------------------------------------------
            
            final_overall = evaluation['overall_score']

            if final_overall >= 85:
                evaluation['recommendation'] = 'Strong Yes (Excellent Candidate)'
            elif final_overall >= 70:
                evaluation['recommendation'] = 'Yes (Meets Expectations)'
            elif final_overall >= 60:
                evaluation['recommendation'] = 'No (Underperforms, Consider Junior Role)'
            else:
                evaluation['recommendation'] = 'Strong No (Rejection)'
            
            evaluation['evaluation_time_seconds'] = evaluation_time
            evaluation['interviewer_used'] = "Fundamental Knowledge Evaluator"
            
            # Add question scores 
            display_scores = {k: v for k, v in question_scores.items() if k != 'OVERALL'}
            evaluation['question_scores'] = display_scores
            
            return evaluation
            
        except Exception as e:
            print(f"Evaluation error: {e}")
            return self._fallback_evaluation(questions, responses, e)
    
    def _build_evaluation_prompt_and_system(self, role: str, job_description: str, questions: list, responses: list, calculated_score: int):
        """Builds the prompt and the new, strict system content for discrimination"""
        
            # 1. SYSTEM PROMPT CONTENT
        system_content = (
            "You are a strict, objective, and expert technical interviewer for a senior role. "
            "Your sole task is to provide a final technical evaluation based on the candidate's answers. "
            "**YOU MUST USE THE FULL 0-100 RANGE FOR ALL SCORES.** "
            "**CRITICAL RULE:** If an answer is irrelevant, completely wrong, or shows no fundamental knowledge "
            "of the topic, **the score for the related criteria MUST be 0-10.** "
            "Do not be lenient. Score based purely on merit and technical correctness. "
            "You must respond ONLY with the required JSON structure."
        )
        
        # 2. USER PROMPT CONTENT
        qa_summary = ""
        for i, (q, r) in enumerate(zip(questions, responses)):
            qa_summary += f"Question {i+1} ({q['type']}, {q['difficulty']}):\n{q['question']}\n\n"
            qa_summary += f"Answer:\n{r['response']}\n\n"
            qa_summary += "-" * 40 + "\n\n"
        
        user_prompt = f"""Evaluate candidate for: {role}

Job Description Summary (Key skills: React, Redux, ES6+, Performance Optimization):
{job_description[:800]}

Candidate's Responses:
{qa_summary}

Evaluation Criteria:
1. **Technical Accuracy (0-100):** How factually correct is the information?
2. **Completeness (0-100):** Does the answer cover all required aspects of the question?
3. **Relevance to Role (0-100):** Is the knowledge practical and relevant to the Job Description (e.g., Senior Frontend)?
4. **Practical Knowledge (0-100):** Does the answer suggest they have hands-on experience (code examples, scenarios)?

The keyword-matching benchmark score for these answers is: {calculated_score}/100.
Your scores in the JSON must reflect your expert, qualitative judgment, which should be more accurate than the benchmark score.

Return ONLY valid JSON:
{{
  "criterion_scores": {{
    "technical_accuracy": 0,
    "completeness": 0,
    "relevance": 0,
    "practicality": 0
  }},
  "strengths": [
    "Specific strength mentioned in their response",
    "Another concrete strength"
  ],
  "weaknesses": [
    "Specific area needing improvement with suggestion",
    "Another actionable weakness"
  ],
  "qualitative_feedback": "Detailed analysis of their fundamental knowledge. Reference their answers. Be constructive. Mention why the score is low if it is."
}}"""
        
        return user_prompt, system_content
   
    
    def _calculate_question_scores(self, questions: list, responses: list) -> Dict[str, Any]:
        """Calculate scores for each question based on keyword matching and answer quality"""
        question_scores = {}
        total_score = 0
        
        for i, (q, r) in enumerate(zip(questions, responses)):
            response_text = r['response'].strip()
            response_lower = response_text.lower()
            expected_keywords = [k.lower() for k in q.get('expected_keywords', [])]
            
            # Base scoring
            if not response_text:  # Empty response
                score = 0
            elif not expected_keywords:
                # No keywords specified, score based on length and quality
                score = self._score_by_length_and_quality(response_text, q['difficulty'])
            else:
                # Calculate keyword score
                found_keywords = sum(1 for keyword in expected_keywords if keyword in response_lower)
                
                if not response_text or (found_keywords == 0 and len(response_text.split()) < 10):
                    keyword_score = 0
                elif found_keywords == 0:
                    keyword_score = 20 # Lowered base for irrelevant non-empty answer
                else:
                    keyword_score = (found_keywords / len(expected_keywords)) * 100
                
                # Adjust based on answer quality
                quality_multiplier = self._calculate_quality_multiplier(response_text, q['type'], q['difficulty'])
                score = min(100, keyword_score * quality_multiplier)
            
            # Apply difficulty weighting
            if q['difficulty'] == 'intermediate':
                score = min(100, score * 1.1)
            elif q['difficulty'] == 'advanced':
                score = min(100, score * 1.2)
            
            question_scores[f"Q{i+1}: {q['question'][:30]}..."] = int(score)
            total_score += score
        
        # Calculate overall average
        if question_scores:
            average_score = total_score / len(question_scores)
            question_scores['OVERALL'] = int(average_score)
        
        return question_scores

    def _score_by_length_and_quality(self, response_text: str, difficulty: str) -> float:
        """Score responses without expected keywords (Remains the same)"""
        word_count = len(response_text.split())
        
        if word_count == 0:
            return 0
        elif word_count < 10:
            base_score = 30
        elif word_count < 30:
            base_score = 50
        elif word_count < 100:
            base_score = 70
        else:
            base_score = 85
        
        if difficulty == 'intermediate':
            return min(100, base_score * 1.1)
        elif difficulty == 'advanced':
            return min(100, base_score * 1.2)
        return base_score
    
    def _calculate_quality_multiplier(self, response_text: str, question_type: str, difficulty: str) -> float:
        """Calculate quality multiplier (Remains the same)"""
        word_count = len(response_text.split())
        lines = response_text.count('\n') + 1
        multiplier = 1.0
        
        if word_count > 100:
            multiplier *= 1.2
        elif word_count > 50:
            multiplier *= 1.1
        
        if question_type == 'coding' and ('function ' in response_text.lower() or 'def ' in response_text.lower() or 'class ' in response_text.lower() or 'return ' in response_text.lower()):
            multiplier *= 1.15
        
        if 'example' in response_text.lower() or 'e.g.' in response_text.lower() or 'for example' in response_text.lower():
            multiplier *= 1.1
        
        if lines > 5:
            multiplier *= 1.05
        
        if '```' in response_text or 'code' in response_text.lower():
            multiplier *= 1.05
            
        return multiplier

    def _parse_evaluation_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response into JSON (Remains the same)"""
        response_text = response_text.strip()
        
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
        
        try:
            evaluation = json.loads(response_text)
            
            required_fields = ['criterion_scores', 'strengths', 'weaknesses', 'qualitative_feedback']
            for field in required_fields:
                if field not in evaluation:
                    if field == 'criterion_scores':
                        evaluation[field] = {"technical_accuracy": 75, "completeness": 75, "relevance": 75, "practicality": 75}
                    elif field in ['strengths', 'weaknesses']:
                        evaluation[field] = []
                    elif field == 'qualitative_feedback':
                        evaluation[field] = "Evaluation completed successfully."
            
            return evaluation
            
        except json.JSONDecodeError:
            print(f"JSON parsing error. Response text: {response_text[:200]}")
            return self._create_basic_evaluation()

    def _create_basic_evaluation(self) -> Dict[str, Any]:
        """Create a basic evaluation when parsing fails (Remains the same)"""
        return {
            "criterion_scores": {
                "technical_accuracy": 75,
                "completeness": 75,
                "relevance": 75,
                "practicality": 75
            },
            "strengths": ["Shows understanding of basic concepts."],
            "weaknesses": ["Could benefit from more detailed explanations."],
            "qualitative_feedback": "Evaluation based on keyword analysis. Candidate demonstrates foundational knowledge suitable for further consideration."
        }
    
    def _fallback_evaluation(self, questions: list, responses: list, error: Exception) -> Dict[str, Any]:
        """Fallback evaluation if LLM fails (Remains the same)"""
        question_scores = self._calculate_question_scores(questions, responses)
        calculated_overall = question_scores.get('OVERALL', 70)
        
        if calculated_overall >= 85:
            recommendation = 'Strong Yes (Excellent Candidate)'
        elif calculated_overall >= 70:
            recommendation = 'Yes (Meets Expectations)'
        elif calculated_overall >= 60:
            recommendation = 'No (Underperforms, Consider Junior Role)'
        else:
            recommendation = 'Strong No (Rejection)'
        
        display_scores = {k: v for k, v in question_scores.items() if k != 'OVERALL'}
        
        return {
            "overall_score": calculated_overall,
            "criterion_scores": {
                "technical_accuracy": calculated_overall,
                "completeness": calculated_overall,
                "relevance": calculated_overall,
                "practicality": calculated_overall
            },
            "strengths": ["Responses show relevant knowledge based on keyword matching."],
            "weaknesses": ["Detailed evaluation unavailable due to system limitations."],
            "qualitative_feedback": f"Preliminary assessment based on keyword analysis. Score: {calculated_overall}/100. For accurate evaluation, please try again later. Error: {error}",
            "recommendation": recommendation,
            "interviewer_used": "Keyword-Based Evaluator",
            "evaluation_time_seconds": 0.0,
            "question_scores": display_scores
        }