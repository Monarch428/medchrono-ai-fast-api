# ===========================
# Import necessary libraries
# ===========================
import os                          # For file and folder path management
import json                        # For JSON serialization/deserialization
import pdfplumber                  # For extracting text from PDF files
import pytesseract                 # For extracting text from images (OCR)
from PIL import Image               # To open image files for OCR
from openai import OpenAI           # OpenAI API client
from dotenv import load_dotenv      # Load environment variables from a .env file

# ======================================
# Load environment variables and OpenAI client
# ======================================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ======================================
# Function: Extract text from PDF files
# ======================================
def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

# ======================================
# Function: Extract text from image files
# ======================================
def extract_text_from_image(file_path):
    image = Image.open(file_path)
    text = pytesseract.image_to_string(image)
    return text

# ======================================
# Function: Extract text from TXT files
# ======================================
def extract_text_from_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# ======================================
# Function: Process a single file
# ======================================
def process_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    # Extract text based on file type
    if ext == ".pdf":
        text = extract_text_from_pdf(file_path)
    elif ext in [".png", ".jpg", ".jpeg"]:
        text = extract_text_from_image(file_path)
    elif ext == ".txt":
        text = extract_text_from_txt(file_path)
    else:
        return None  # Unsupported file type

    # Fallback for empty documents
    if not text.strip():
        return {
            "file_name": os.path.basename(file_path),
            "report_type": "Unknown",
            "report_author": "Unknown",
            "issued_by": "Unknown",
            "issued_date": "Unknown"
        }

    # ======================================
    # OpenAI Prompt
    # ======================================
    prompt = f"""
You are a professional legal AI assistant. I will provide you with a document related to a case. 
The document can be in PDF, TXT, or image format, and may include Medical, EMS/Ambulance, Accident, Hospital, Legal, or any other type of report. 

Your task is to **read the document and extract all relevant details** that a lawyer would need for case analysis. 
The JSON object you return should include:

1. **file_name**: Name of the file.
2. **report_type**: Classify the document (Medical, EMS/Ambulance, Accident, Hospital, Legal, or Other).
3. **report_author**: Author or issuing entity, or "Unknown" if not stated.
4. **issued_by**: Person or organization issuing the document, or "Unknown".
5. **issued_date**: Date of document or service in YYYY-MM-DD format, or "Unknown".
6. **client_or_subject_info**: Any identifiable information about the client, patient, or subject (age, sex, identifiers), or "Unknown".
7. **key_issues_or_complaints**: Primary and secondary complaints, issues, or reasons for the report.
8. **observations_or_data**: Any observations, measurements, or relevant data (e.g., vitals, test results, incident details, notes).
9. **actions_taken**: Any actions, treatments, interventions, procedures, or responses documented.
10. **case_summary_points**: **Detailed, structured, bullet-point summary** capturing all key facts:
    - Dates and times of events (service, admissions, transport, assessments)
    - Individuals involved (authors, staff, witnesses, parties)
    - Complaints, symptoms, or issues raised
    - Observations, measurements, or physical findings
    - Actions, treatments, or interventions applied
    - Work-related, trauma, accident, or incident notes
    - Narrative highlights, sequences, or important events
    - Present **5–15 bullet points**, concise and fact-focused.
11. **case_overview**: A short paragraph summarizing the document for the lawyer’s quick understanding, highlighting the most relevant facts and insights for the case.
12. **full_summary**: A comprehensive narrative paragraph summarizing the **entire content** of the file in clear, natural language. 
    - Include all important details, context, and flow of events.
    - Should read like a full-document summary (as if explaining the whole file to someone who hasn’t read it).
    - Keep it factual and neutral, without assumptions or interpretations.

**Guidelines:**
- Extract **all details**, even if implicit or scattered in the document.
- If a field is missing, write "Unknown".
- Keep **case_summary_points** factual, chronological, and structured.
- **case_overview** is a short lawyer-focused insight paragraph.
- **full_summary** is a longer narrative capturing the entire document’s content cohesively.
- Include any relevant details for case understanding, regardless of document type.
- Do not omit any major events, observations, or actions important for the case.

Document:
{text[:20000]}  # Provide the first 20,000 characters for long documents
"""


    try:
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt + "\n\nDocument:\n" + text[:15000]}],
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)

        # Ensure report_summary is always a list
        if "report_summary" not in result:
            result["report_summary"] = ["Summary not available"]
        elif isinstance(result["report_summary"], str):
            # Convert string to list of bullet points
            result["report_summary"] = [
                line.strip() for line in result["report_summary"].split("\n") if line.strip()
            ]

        # Always include file_name
        result["file_name"] = os.path.basename(file_path)
        return result

    except Exception as e:
        return {
            "file_name": os.path.basename(file_path),
            "report_type": "Unknown",
            "report_author": "Unknown",
            "issued_by": "Unknown",
            "issued_date": "Unknown"
        }

# ======================================
# Function: Process entire folder
# ======================================
def process_folder(folder_path):
    results = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            print(f"Processing: {file}")
            summary = process_file(file_path)
            if summary:
                results.append(summary)
    return {"documents": results}