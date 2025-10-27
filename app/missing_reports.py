import os
import json
from typing import List, Dict
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import re

# ---- Load environment variables ----
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("❌ OPENAI_API_KEY not found. Please set it in your .env file.")

# ---- FastAPI Router ----
router = APIRouter(prefix="/missing_reports", tags=["Missing Reports"])

# ---- LLM ----
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_api_key)

# ---- Config ----
REPORTS_FOLDER = os.path.join(os.path.dirname(__file__), "documents")

# ---- Helper: Load PDF texts ----
def load_pdf_texts(folder_path: str) -> Dict[str, str]:
    pdf_texts = {}
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, file))
            docs = loader.load()
            text = "\n".join([doc.page_content for doc in docs])
            pdf_texts[file] = text
    return pdf_texts

# ---- Request & Response Models ----
class CaseAnalysisRequest(BaseModel):
    case_type: str
    sub_category: str

class MissingReportResponse(BaseModel):
    detected_reports: List[str]
    required_reports: List[str]
    missing_reports: List[str]
    report_summary: str

# ---- Helper: Detect mentioned reports in text ----
def find_mentions(text: str, report_list: List[str]) -> List[str]:
    mentions = []
    text_lower = text.lower()
    for report in report_list:
        if re.search(r"\b" + re.escape(report.lower()) + r"\b", text_lower):
            mentions.append(report)
    return mentions

# ---- Helper: Extract JSON from code block ----
def extract_json_from_response(response_text: str) -> str:
    """
    Extract JSON from markdown code blocks if present.
    """
    match = re.search(r"```json(.*?)```", response_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return response_text

# ---- Endpoint ----
@router.post("/", response_model=MissingReportResponse)
async def analyze_missing_reports(request: CaseAnalysisRequest):
    case_type = request.case_type.strip()
    sub_category = request.sub_category.strip()

    if not case_type or not sub_category:
        raise HTTPException(status_code=400, detail="Case type and sub-category cannot be empty.")

    # Load PDFs
    pdf_texts = load_pdf_texts(REPORTS_FOLDER)
    if not pdf_texts:
        raise HTTPException(status_code=404, detail="No PDF reports found in folder.")

    # Combine texts for LLM analysis
    combined_text = "\n\n".join([f"{name}:\n{text}" for name, text in pdf_texts.items()])

    # ---- LLM Prompt ----
    prompt = f"""
You are an expert legal and case report assistant.

Case Type: {case_type}
Sub-category: {sub_category}

You have access to several PDF reports (with text content).

Tasks:
1. Identify which report type each PDF represents (e.g., ambulance report, medical report, police report, insurance report, incident report, forensic report, witness statement).
2. Based on the case type and sub-category, list which reports are typically required.
3. Compare the available reports and determine which reports are missing.
4. If any PDF mentions another report (e.g., "see attached old medical report"), check if it exists; if not, include it as missing.
5. Return only **JSON** with this structure:

{{
  "detected_reports": [list of found report types],
  "required_reports": [list of expected report types],
  "missing_reports": [list of missing reports],
  "report_summary": "short summary of findings"
}}

PDF Contents:
{combined_text}
"""

    # ---- Call LLM ----
    response = llm.invoke(prompt)
    response_text = response.content.strip()

    # ---- Extract JSON from code block if needed ----
    json_text = extract_json_from_response(response_text)

    # ---- Parse JSON safely ----
    try:
        result = json.loads(json_text)
    except Exception:
        # Fallback: return raw LLM response in report_summary
        result = {
            "detected_reports": [],
            "required_reports": [],
            "missing_reports": [],
            "report_summary": response_text,
        }

    return MissingReportResponse(**result)