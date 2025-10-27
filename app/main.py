# app/main.py
import os
import json
import tempfile
import shutil
from pathlib import Path
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .subcategory import get_subcategories_from_ai
from .document_processor import process_folder, process_file
from .chatassistant import router as chat_router
from .missing_reports import router as missing_router  # <-- Import missing_report router

app = FastAPI(title="Dynamic Case Subcategory API", version="1.0")

# Add CORS middleware to allow frontend to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://medchrono.io", "http://localhost:3000"],  # Add your frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Include Routers ----
app.include_router(chat_router)        # /chat
app.include_router(missing_router)     # /missing_reports

@app.get("/")
def root():
    return {
        "message": "Welcome to the API! Available endpoints: /chat, /missing_reports, /subcategories/{case_type}, /summarize, /process-document"
    }

@app.get("/subcategories/{case_type}")
def fetch_subcategories(case_type: str):
    result = get_subcategories_from_ai(case_type)
    return json.loads(result)

@app.post("/process-document")
async def process_single_document(file: UploadFile = File(...)):
    """
    Process a single uploaded document and return AI analysis.
    Accepts PDF, images (PNG, JPG, JPEG), and text files.
    """
    # Validate file type
    allowed_extensions = {".pdf", ".png", ".jpg", ".jpeg", ".txt"}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Supported types: {', '.join(allowed_extensions)}"
        )

    # Create a temporary file to store the upload
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            # Write uploaded content to temporary file
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        # Process the file using existing document_processor
        result = process_file(tmp_file_path)

        # Clean up temporary file
        os.unlink(tmp_file_path)

        if result is None:
            raise HTTPException(
                status_code=400,
                detail="Failed to process document. The file may be corrupted or unsupported."
            )

        return {
            "success": True,
            "data": result
        }

    except Exception as e:
        # Clean up temporary file if it exists
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )

@app.post("/process-documents-batch")
async def process_multiple_documents(files: List[UploadFile] = File(...)):
    """
    Process multiple uploaded documents and return AI analysis for each.
    Accepts PDF, images (PNG, JPG, JPEG), and text files.
    """
    if len(files) > 50:
        raise HTTPException(
            status_code=400,
            detail="Maximum 50 files allowed per batch"
        )

    results = []
    temp_dir = None

    try:
        # Create a temporary directory for batch processing
        temp_dir = tempfile.mkdtemp()

        # Save all uploaded files to temp directory
        for file in files:
            file_ext = Path(file.filename).suffix.lower()
            allowed_extensions = {".pdf", ".png", ".jpg", ".jpeg", ".txt"}

            if file_ext not in allowed_extensions:
                results.append({
                    "file_name": file.filename,
                    "success": False,
                    "error": f"Unsupported file type: {file_ext}"
                })
                continue

            # Save file to temp directory
            file_path = os.path.join(temp_dir, file.filename)
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)

        # Process the entire folder
        batch_result = process_folder(temp_dir)

        # Clean up temporary directory
        shutil.rmtree(temp_dir)

        return {
            "success": True,
            "data": batch_result
        }

    except Exception as e:
        # Clean up temporary directory if it exists
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

        raise HTTPException(
            status_code=500,
            detail=f"Error processing documents: {str(e)}"
        )

@app.get("/summarize")
def summarize_documents():
    """
    DEPRECATED: This endpoint is deprecated. Use /process-document or /process-documents-batch instead.

    This endpoint tries to read from a local 'documents' folder which doesn't exist in production.
    Use the file upload endpoints instead.
    """
    raise HTTPException(
        status_code=410,
        detail="This endpoint is deprecated. Please use POST /process-document or POST /process-documents-batch with file uploads instead."
    )