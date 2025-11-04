# routers/intake_router.py
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
import uuid
import openai
import json
import os
from dotenv import load_dotenv

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

router = APIRouter(prefix="/intake", tags=["Intake Bot"])

# In-memory session store
sessions = {}


# -------------------------------
# Data Models
# -------------------------------
class IntakeRequest(BaseModel):
    session_id: Optional[str] = None
    user_message: str


class IntakeResponse(BaseModel):
    session_id: str
    bot_reply: str
    finished: bool = False
    summary: Optional[dict] = None


# -------------------------------
# Helper Functions
# -------------------------------
def analyze_incident_with_gpt(conversation: list):
    """
    Send conversation history to GPT to analyze and decide:
    - Next question
    - Case or no case
    - Summary
    """
    prompt = f"""
    You are a legal intake assistant for MedChrono.
    Based on the following conversation:
    {conversation}

    Extract the following:
    - incident_type
    - location
    - injury
    - negligence
    - evidence
    - incident_date (if given)
    - case_likelihood (High, Medium, Low)
    - summary (2-3 lines)
    - next_question (if intake not finished)
    - finished (true if you have enough info)

    Return valid JSON.
    """

    try:
        completion = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # or "gpt-5"
            messages=[{"role": "system", "content": prompt}],
            temperature=0.4
        )
        return completion.choices[0].message["content"]
    except Exception as e:
        return json.dumps({"error": str(e)})


# -------------------------------
# API Endpoints
# -------------------------------
@router.post("/respond", response_model=IntakeResponse)
async def respond_to_user(req: IntakeRequest):
    session_id = req.session_id or str(uuid.uuid4())
    history = sessions.get(session_id, [])
    history.append({"role": "user", "content": req.user_message})

    gpt_result = analyze_incident_with_gpt(history)

    try:
        result_data = json.loads(gpt_result)
    except json.JSONDecodeError:
        result_data = {
            "incident_type": None,
            "next_question": "Can you tell me more about what happened?",
            "case_likelihood": "Unknown",
            "summary": None,
            "finished": False
        }

    sessions[session_id] = history
    history.append({"role": "assistant", "content": result_data.get("next_question", "")})

    if result_data.get("finished"):
        return IntakeResponse(
            session_id=session_id,
            bot_reply="Thank you. Our attorney will call you shortly.",
            finished=True,
            summary=result_data
        )
    else:
        return IntakeResponse(
            session_id=session_id,
            bot_reply=result_data.get("next_question", "Can you tell me more?"),
            finished=False
        )


@router.get("/summary/{session_id}")
async def get_summary(session_id: str):
    """Return the conversation or summary for a session."""
    return sessions.get(session_id, [])