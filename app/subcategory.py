# subcategory.py
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_subcategories_from_ai(case_type: str):
    """
    Uses OpenAI API to generate subcategories for a given case type.
    """
    prompt = f"""
    You are a legal assistant AI.
    The user has given a case type: "{case_type}".
    Generate a JSON response with:
    - "case_type": the given case type
    - "subcategories": a list of 4–6 possible subcategories under this case
    - "other_case_types": a list of 3–4 related case types
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}   # ✅ FIXED
    )

    return response.choices[0].message.content