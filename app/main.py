# app/main.py
import os
import json
from fastapi import FastAPI
from .subcategory import get_subcategories_from_ai
from .document_processor import process_folder
from .chatassistant import router as chat_router
from .missing_reports import router as missing_router  # <-- Import missing_report router

app = FastAPI(title="Dynamic Case Subcategory API", version="1.0")

# ---- Include Routers ----
app.include_router(chat_router)        # /chat
app.include_router(missing_router)     # /missing_reports

@app.get("/")
def root():
    return {
        "message": "Welcome to the API! Available endpoints: /chat, /missing_reports, /subcategories/{case_type}, /summarize"
    }

@app.get("/subcategories/{case_type}")
def fetch_subcategories(case_type: str):
    result = get_subcategories_from_ai(case_type)
    return json.loads(result)

@app.get("/summarize")
def summarize_documents():
    folder_path = os.path.join(os.path.dirname(__file__), "documents")
    result = process_folder(folder_path)
    return {"data": result}