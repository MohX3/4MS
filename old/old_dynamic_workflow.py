# src/dynamic_workflow.py - FULL FIXED VERSION

import os
from typing import TypedDict, Annotated, Sequence
from operator import add
import streamlit as st
from datetime import datetime, timedelta
import pytz


# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages


# LangChain imports
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, ToolMessage
from langchain_core.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
from langchain.tools.retriever import create_retriever_tool
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document


# PDF generation
from src.pdf_utils import generate_pdf


# --- Helper function to format retriever outputs ---
def format_retriever_output(result):
    """
    Convert retriever tool outputs (which may be Document objects or lists)
    into plain text strings that Gemini can accept.
    """
    if result is None:
        return "No results found."
    
    if isinstance(result, str):
        return result
    
    # Handle Document objects
    if isinstance(result, Document):
        return result.page_content
    
    # Handle lists of Documents
    if isinstance(result, list):
        if not result:
            return "No results found."
        
        formatted_parts = []
        for idx, item in enumerate(result, 1):
            if isinstance(item, Document):
                formatted_parts.append(f"Result {idx}:\n{item.page_content}")
            else:
                formatted_parts.append(f"Result {idx}:\n{str(item)}")
        
        return "\n\n".join(formatted_parts)
    
    # Fallback for any other type
    return str(result)


class AgentState(TypedDict):
    mode: str
    num_of_q: int
    num_of_follow_up: int
    position: str
    evaluation_result: Annotated[str, add]
    company_name: str
    messages: Annotated[Sequence[BaseMessage], add_messages]
    report: Annotated[str, add]
    pdf_path: str | None
    resume_path: str | None
    questions_path: str | None
    resume_text: str | None


# --- TIMEZONE-AWARE DATE FUNCTIONS (UTC+3) ---
def get_current_date_4ms():
    """Get current date in UTC+3 timezone (IntiqAI timezone)"""
    try:
        tz = pytz.timezone('Asia/Kuwait')  # UTC+3
        current = datetime.now(tz)
        return current.strftime("%d %B %Y")
    except:
        utc_now = datetime.utcnow()
        local_time = utc_now + timedelta(hours=3)
        return local_time.strftime("%d %B %Y")


def get_current_datetime_4ms():
    """Get current date and time in UTC+3 timezone"""
    try:
        tz = pytz.timezone('Asia/Kuwait')  # UTC+3
        current = datetime.now(tz)
        return current.strftime("%d %B %Y, %H:%M")
    except:
        utc_now = datetime.utcnow()
        local_time = utc_now + timedelta(hours=3)
        return local_time.strftime("%d %B %Y, %H:%M")


# --- LLM and Embeddings ---
llm = init_chat_model("google_genai:gemini-2.5-flash-lite")
evaluator_llm = init_chat_model("google_genai:gemini-2.5-flash-lite", temperature=0.0)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# --- Prompts ---
interviewer_prompt = PromptTemplate(
    input_variables=["mode", "company_name", "position", "number_of_questions", "number_of_followup"],
    template="""
You are an {mode} AI interviewer for a leading tech company called {company_name}, conducting an interview for a {position} position.
Your goal is to assess the candidate's technical skills, problem-solving abilities, communication skills, and experience relevant to data science roles.
Maintain a professional yet approachable tone.


You have access to two tools:
1. `retrieve_questions`: This tool can search a knowledge base of interview questions related to the {position} position. Use this tool to find relevant questions to ask the candidate.
2. `retrieve_resume`: This tool can search the candidate's resume to find information about their past projects and experience. Use this tool to ask relevant projects from their resume like {position} projects.

CRITICAL RULES - FOLLOW EXACTLY:
1. When you call a tool, DO NOT include any question text in that same message. Just call the tool silently.
2. WAIT for the tool results to come back.
3. THEN, in your NEXT message, use the tool results to ask your question.
4. NEVER ask the same question twice.

Interview Flow:
- Step 1: Introduce yourself and ask candidate to introduce themselves (no tools needed)
- Step 2: After their intro, SILENTLY call retrieve_resume tool (no question yet)
- Step 3: Use the resume results to ask ONE question about their project
- Step 4: For technical questions, SILENTLY call retrieve_questions tool first
- Step 5: Use the retrieved questions to ask the candidate

Interview Structure:
1. ONE Introduction question at the beginning
2. ONE question about a project from their resume (retrieve_resume → then ask)
3. {number_of_questions} technical questions from the knowledge base (retrieve_questions → then ask)
4. Up to {number_of_followup} follow-up questions ONLY if their answer is too vague or incomplete


If asked any irrelevant question, respond with: "Sorry, this is out of scope."


After the interview is finished you output this sentance exacly: "Thank you, that's it for today."


Begin the interview now.
"""
)


evaluator_prompt = PromptTemplate(
    input_variables=["num_of_q", "num_of_follow_up", "position"],
    template="""You are an AI evaluator for a job interview. Your ONLY task is to evaluate the candidate's responses and provide scores.

DO NOT continue the conversation. DO NOT act as the recruiter. ONLY provide the evaluation.

Interview Structure:
- 1 Introduction question
- 1 Project question
- {num_of_q} Technical questions (with up to {num_of_follow_up} follow-up questions each)

Position: {position}

Evaluation Criteria:
- Score each response from 1 to 10 (1 = poor, 10 = excellent)
- Consider: relevance, clarity, depth, technical accuracy
- Ignore any irrelevant questions or answers

REQUIRED OUTPUT FORMAT (nothing else):

Evaluation:
1. Introduction question: [score: X/10] - [brief reasoning]
2. Project question: [score: X/10] - [brief reasoning]
3. Technical question one: [score: X/10] - [brief reasoning]
4. Technical question two: [score: X/10] - [brief reasoning]

IMPORTANT: Output ONLY the evaluation in the format above. Do not include any other text, questions, or conversation.
"""
)


report_writer_prompt = PromptTemplate(
    input_variables=["position", "company_name", "interview_transcript", "evaluation_report", "interview_date"],
    template="""You are an AI HR Report Writer. Your task is to synthesize information from a job interview transcript and its evaluation into a concise, professional report for Human Resources at {company_name}.


The interview was for a **{position}** position.
Interview Date: {interview_date}


Your report should focus on key takeaways relevant to HR's decision-making, including but not limited to:


- **Candidate's Overall Suitability:** A brief summary of whether the candidate seems suitable for the role based on their performance.
- **Strengths:** Specific areas where the candidate performed well, supported by examples from the transcript if clear.
- **Areas for Development/Weaknesses:** Specific areas where the candidate struggled or showed gaps, supported by examples from the transcript if clear.
- **Key Technical Skills Demonstrated:** List any core technical skills explicitly mentioned or clearly demonstrated by the candidate's answers.
- **Communication Skills:** Assessment of clarity, conciseness, and overall effectiveness of their communication during the interview.
- **Recommendations (Optional):** A high-level recommendation (e.g., "Proceed to next round," "Consider for a different role," "Not a good fit at this time").


---


**Interview Transcript:**


{interview_transcript}


---


**Evaluation Report:**


{evaluation_report}


"""
)


# --- Vector Store and Retriever Setup ---
DEFAULT_QUESTIONS_PDF = "data/default_questions.pdf"
DEFAULT_RESUME_PDF = "data/default_resume.pdf"


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


def initialize_questions_retriever(questions_path=None):
    questions_file = questions_path if questions_path and os.path.exists(questions_path) else DEFAULT_QUESTIONS_PDF


    import uuid
    import time
    session_id = f"{uuid.uuid4().hex}_{int(time.time())}"
    collection_name = f"questions_{session_id}"


    loader = PyPDFLoader(questions_file)
    pages = loader.load()
    pages_split = text_splitter.split_documents(pages)


    questions_vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        collection_name=collection_name
    )


    questions_retriever = questions_vectorstore.as_retriever(search_kwargs={"k": 3})


    questions_retriever_tool = create_retriever_tool(
        questions_retriever,
        "retrieve_questions",
        "Search and return interview questions related to the position from the knowledge base.",
    )


    return questions_retriever_tool


def initialize_resume_retriever(resume_path=None, resume_text=None):
    import uuid
    import time
    session_id = f"{uuid.uuid4().hex}_{int(time.time())}"
    collection_name = f"resume_{session_id}"


    docs = []
    if resume_text:
        docs = [Document(page_content=resume_text)]
    else:
        resume_file = resume_path if resume_path and os.path.exists(resume_path) else DEFAULT_RESUME_PDF
        resume_loader = PyPDFLoader(resume_file)
        docs = resume_loader.load()


    resume_split = text_splitter.split_documents(docs)


    resume_vectorstore = Chroma.from_documents(
        documents=resume_split,
        embedding=embeddings,
        collection_name=collection_name
    )


    resume_retriever = resume_vectorstore.as_retriever(search_kwargs={"k": 3})


    resume_retriever_tool = create_retriever_tool(
        resume_retriever,
        "retrieve_resume",
        "Search the candidate's resume to find specific projects, skills, and experiences.",
    )


    return resume_retriever_tool


# --- Graph Nodes ---
def recruiter(state: AgentState) -> AgentState:
    """
    Main recruiter node - generates AI responses.
    Does NOT handle tool calls - those are handled by the tools node.
    """
    resume_retriever_tool = initialize_resume_retriever(
        resume_path=state.get("resume_path"),
        resume_text=state.get("resume_text")
    )
    questions_retriever_tool = initialize_questions_retriever(state.get("questions_path"))

    sys_prompt = SystemMessage(content=interviewer_prompt.format(
        mode=state['mode'],
        company_name=state['company_name'],
        position=state['position'],
        number_of_questions=state['num_of_q'],
        number_of_followup=state['num_of_follow_up']
    ))

    # Build message history for LLM
    # Strategy: Include the full conversation but handle tool results specially
    conversation_messages = []
    
    i = 0
    while i < len(state["messages"]):
        msg = state["messages"][i]
        
        if isinstance(msg, HumanMessage):
            # Always include human messages
            conversation_messages.append(msg)
            i += 1
            
        elif isinstance(msg, AIMessage):
            # Check if this AI message has tool_calls
            if getattr(msg, 'tool_calls', None):
                # This AI message called tools - look for the following ToolMessages
                tool_results = []
                j = i + 1
                while j < len(state["messages"]) and isinstance(state["messages"][j], ToolMessage):
                    tool_results.append(state["messages"][j])
                    j += 1
                
                # If we found tool results, include them in the conversation
                if tool_results:
                    # Include the AI message with tool_calls
                    conversation_messages.append(msg)
                    # Include all the tool result messages
                    conversation_messages.extend(tool_results)
                    i = j  # Skip past the tool messages we just processed
                else:
                    # No tool results found, skip this AI message
                    i += 1
            else:
                # Regular AI message with content
                if msg.content and msg.content.strip():
                    conversation_messages.append(msg)
                i += 1
                
        elif isinstance(msg, ToolMessage):
            # Standalone ToolMessage (shouldn't happen, but skip it)
            i += 1
        else:
            i += 1

    all_messages = [sys_prompt] + conversation_messages
    
    try:
        response = llm.bind_tools([questions_retriever_tool, resume_retriever_tool]).invoke(all_messages)
        print(f"[DEBUG] Recruiter response - has tool_calls: {bool(getattr(response, 'tool_calls', None))}, has content: {bool(response.content)}")
        return {"messages": [response]}
    except Exception as e:
        print(f"[ERROR] LLM invocation failed: {str(e)}")
        print(f"[DEBUG] Message count: {len(all_messages)}")
        print(f"[DEBUG] Last message: {conversation_messages[-1] if conversation_messages else 'None'}")
        # Return a fallback message
        return {"messages": [AIMessage(content="I apologize, but I encountered an error. Could you please repeat your last response?")]}


def evaluator(state: AgentState) -> AgentState:
    sys_prompt = evaluator_prompt.format(
        num_of_q=state['num_of_q'],
        num_of_follow_up=state['num_of_follow_up'],
        position=state['position']
    )

    # Build the interview transcript for evaluation
    interview_transcript = []
    for m in state["messages"]:
        if isinstance(m, HumanMessage):
            if m.content and m.content.strip():
                interview_transcript.append(f"Candidate: {m.content}")
        elif isinstance(m, AIMessage):
            # Only include AIMessages with actual content
            if m.content and m.content.strip():
                # Exclude the final "that's it for today" message from evaluation
                if "that's it for today" not in m.content.lower():
                    interview_transcript.append(f"AI Recruiter: {m.content}")

    # Validate we have content to evaluate
    if not interview_transcript or len(interview_transcript) < 2:
        print("[ERROR] Not enough interview content to evaluate")
        return {"evaluation_result": "Error: Insufficient interview content for evaluation."}

    # Create a single prompt with the full transcript
    full_prompt = f"""{sys_prompt}

Here is the complete interview transcript to evaluate:

{chr(10).join(interview_transcript)}

Please provide the evaluation now in the specified format."""

    print(f"[DEBUG] Evaluator - transcript has {len(interview_transcript)} messages")
    print(f"[DEBUG] Evaluator - prompt length: {len(full_prompt)} chars")
    print(f"[DEBUG] Evaluator - first 200 chars of prompt: {full_prompt[:200]}")
    
    # Validate the prompt is not empty
    if not full_prompt or len(full_prompt.strip()) < 50:
        print(f"[ERROR] Evaluator prompt is too short or empty")
        return {"evaluation_result": "Error: Generated prompt is invalid or empty."}
    
    try:
        print(f"[DEBUG] Invoking evaluator LLM...")
        # Use HumanMessage instead of SystemMessage for Gemini compatibility
        # Gemini expects at least one user message
        human_message = HumanMessage(content=full_prompt)
        results = evaluator_llm.invoke([human_message])
        print(f"[DEBUG] Evaluator completed successfully")
        return {"evaluation_result": results.content}
    except Exception as e:
        print(f"[ERROR] Evaluator failed: {str(e)}")
        print(f"[DEBUG] Full transcript:")
        for i, line in enumerate(interview_transcript[:10], 1):  # Print first 10 lines
            print(f"  {i}. {line[:100]}")
        return {"evaluation_result": f"Error during evaluation: {str(e)}"}


def report_writer(state: AgentState) -> AgentState:
    # GET CURRENT DATE WITH CORRECT TIMEZONE (UTC+3)
    interview_date = get_current_date_4ms()
    
    interviewer_transcript = []
    for m in state["messages"]:
        if isinstance(m, HumanMessage):
            interviewer_transcript.append('Candidate: ' + str(m.content))
        elif isinstance(m, AIMessage):
            if 'Evaluation:\n1. Introduction question' not in m.content:
                interviewer_transcript.append('AI Recruiter: ' + str(m.content))


    sys_prompt = report_writer_prompt.format(
        position=state['position'],
        company_name=state['company_name'],
        interview_transcript='\n'.join(interviewer_transcript),
        evaluation_report=state["evaluation_result"],
        interview_date=interview_date
    )


    sys_message = SystemMessage(content=sys_prompt)
    all_messages = [sys_message, HumanMessage(content="Generate the HR report")]


    result = llm.invoke(all_messages)
    return {"report": result.content}


def pdf_generator_node(state: AgentState) -> AgentState:
    if not state.get("report"):
        return {"pdf_path": None}


    candidate_info = st.session_state.candidates_df.loc[st.session_state.selected_candidate_index]
    candidate_name = f"{candidate_info['First Name']} {candidate_info['Last Name']}"
    filename = f"HR_Report_{candidate_name}_{state['position']}.pdf".replace(" ", "_")


    try:
        pdf_path = generate_pdf(state["report"], filename=filename)
        return {"pdf_path": pdf_path}
    except Exception as e:
        print(f"Error generating PDF: {str(e)}")
        return {"pdf_path": None}


def custom_tools_condition(state):
    """
    Determine where to route after the recruiter node.
    """
    if not state['messages']:
        print("[DEBUG] No messages, routing to WAIT_FOR_HUMAN")
        return "WAIT_FOR_HUMAN"

    last_message = state['messages'][-1]
    
    # Check if AI wants to call tools
    if isinstance(last_message, AIMessage) and getattr(last_message, 'tool_calls', None):
        # Safety check: count recent consecutive tool calls to prevent infinite loops
        consecutive_tool_calls = 0
        for msg in reversed(state['messages'][-10:]):  # Check last 10 messages
            if isinstance(msg, AIMessage) and getattr(msg, 'tool_calls', None):
                consecutive_tool_calls += 1
            elif isinstance(msg, HumanMessage):
                break  # Stop at the last human message
        
        if consecutive_tool_calls > 3:
            print(f"[WARNING] Too many consecutive tool calls ({consecutive_tool_calls}), forcing response")
            # Force the AI to respond with text by returning a message
            return "WAIT_FOR_HUMAN"
        
        print(f"[DEBUG] AI wants to call {len(last_message.tool_calls)} tool(s), routing to tools")
        return "tools"
    
    # Check if interview is ending
    elif isinstance(last_message, AIMessage) and last_message.content and "that's it for today" in last_message.content.lower():
        print("[DEBUG] Interview ending, routing to evaluator")
        return "END_CONVERSATION"
    
    # Normal response - wait for human
    else:
        print(f"[DEBUG] Normal response (type: {type(last_message).__name__}), routing to WAIT_FOR_HUMAN")
        return "WAIT_FOR_HUMAN"


def tools_node(state: AgentState) -> AgentState:
    """
    Handle tool calls by executing them and formatting the results.
    This node is called when the AI wants to use a tool.
    """
    resume_retriever_tool = initialize_resume_retriever(
        resume_path=state.get("resume_path"),
        resume_text=state.get("resume_text")
    )
    questions_retriever_tool = initialize_questions_retriever(state.get("questions_path"))
    
    # Get the last AI message which should have tool_calls
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage) or not getattr(last_message, "tool_calls", None):
        print("[WARNING] tools_node called but no tool_calls found")
        return {}
    
    tool_messages = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call.get("name")
        tool_id = tool_call.get("id", "")
        query = tool_call.get("args", {}).get("query", "")
        
        print(f"[DEBUG] Executing tool: {tool_name} with query: {query[:50]}...")
        
        try:
            if tool_name == "retrieve_resume":
                result = resume_retriever_tool.invoke({"query": query})
            elif tool_name == "retrieve_questions":
                result = questions_retriever_tool.invoke({"query": query})
            else:
                result = f"Unknown tool: {tool_name}"
            
            # Format the result to plain text
            formatted_result = format_retriever_output(result)
            
            tool_message = ToolMessage(
                content=formatted_result,
                tool_call_id=tool_id,
                name=tool_name
            )
            tool_messages.append(tool_message)
            print(f"[DEBUG] Tool {tool_name} returned {len(formatted_result)} chars")
            
        except Exception as e:
            print(f"[ERROR] Tool execution failed: {str(e)}")
            tool_message = ToolMessage(
                content=f"Error executing tool: {str(e)}",
                tool_call_id=tool_id,
                name=tool_name
            )
            tool_messages.append(tool_message)
    
    return {"messages": tool_messages}


def build_workflow():
    workflow = StateGraph(AgentState)

    # Add all nodes
    workflow.add_node("recruiter", recruiter)
    workflow.add_node("tools", tools_node)
    workflow.add_node("evaluator", evaluator)
    workflow.add_node("report_writer", report_writer)
    workflow.add_node("pdf_generator", pdf_generator_node)

    # Set entry point
    workflow.set_entry_point("recruiter")

    # Conditional routing from recruiter
    workflow.add_conditional_edges(
        "recruiter",
        custom_tools_condition,
        {
            "tools": "tools",
            "END_CONVERSATION": END,  # End interview, let user trigger evaluation manually
            "WAIT_FOR_HUMAN": END
        }
    )

    # After tools execute, go back to recruiter to generate response with tool results
    workflow.add_edge("tools", "recruiter")
    
    # Evaluator goes directly to report writer (no tools needed)
    workflow.add_edge("evaluator", "report_writer")

    # Report writer to PDF generator
    workflow.add_edge("report_writer", "pdf_generator")
    
    # PDF generator to END
    workflow.add_edge("pdf_generator", END)

    return workflow.compile()
