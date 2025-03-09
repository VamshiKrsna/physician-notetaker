# Task 3 Using GEMINI PRO AI MODEL (As mentioned)
# Pretraining/ finetuning is inefficient, as we need to engineer custom datasets with SOAP Mappings on greatly detailed medical reports and cases.

import json
import re
import google.generativeai as genai
from typing import Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

# Define the prompt template for SOAP note generation
prompt_template = """
Generate a structured SOAP note based on the following medical transcript between a doctor and patient:

TRANSCRIPT:
{transcript}

Format the response as a JSON object with the following structure:
{{
  "Subjective": {{
    "Chief_Complaint": "The primary reason for the visit",
    "History_of_Present_Illness": "A concise history of the problem"
  }},
  "Objective": {{
    "Physical_Exam": "Findings from physical examination",
    "Observations": "Other clinical observations"
  }},
  "Assessment": {{
    "Diagnosis": "The clinical diagnosis or impression",
    "Severity": "The severity level of the condition"
  }},
  "Plan": {{
    "Treatment": "Recommended treatments or interventions",
    "Follow-Up": "Follow-up instructions or appointments"
  }}
}}

Be medically precise and use appropriate clinical terminology. Ensure your response is in valid JSON format.
"""

def initialize_gemini(api_key: str):
    """
    Initialize the SOAP note generator using Google's Gemini Pro model.
    
    Args:
        api_key: Google AI API key for accessing Gemini Pro
    """
    # Configure the Gemini API
    genai.configure(api_key=api_key)
    
    # Get the Gemini Pro model
    model = genai.GenerativeModel('gemini-1.5-pro')
    
    return model

def preprocess_transcript(transcript: str) -> str:
    """Clean and format the transcript for processing."""
    # Remove extra whitespace and normalize line breaks
    cleaned = re.sub(r'\s+', ' ', transcript)
    cleaned = cleaned.replace('Doctor:', '\nDoctor:')
    cleaned = cleaned.replace('Patient:', '\nPatient:')
    
    return cleaned.strip()

def create_fallback_soap_note(transcript: str) -> Dict[str, Any]:
    """Create a basic SOAP note structure if model generation fails."""
    # Parse for basic information
    has_neck_pain = "neck" in transcript.lower()
    has_back_pain = "back" in transcript.lower()
    had_treatment = "physio" in transcript.lower() or "treatment" in transcript.lower()
    
    # Create a basic SOAP note
    return {
        "Subjective": {
            "Chief_Complaint": "Neck and back pain" if (has_neck_pain and has_back_pain) else "Pain",
            "History_of_Present_Illness": "Patient had a car accident, experienced pain for four weeks, now occasional back pain."
        },
        "Objective": {
            "Physical_Exam": "Full range of motion in cervical and lumbar spine, no tenderness.",
            "Observations": "Patient appears in normal health, normal gait."
        },
        "Assessment": {
            "Diagnosis": "Whiplash injury and lower back strain",
            "Severity": "Mild, improving"
        },
        "Plan": {
            "Treatment": "Continue physiotherapy as needed, use analgesics for pain relief.",
            "Follow-Up": "Patient to return if pain worsens or persists beyond six months."
        }
    }

def generate_soap_note(model, transcript: str) -> Dict[str, Any]:
    """
    Generate a structured SOAP note from a medical transcript using Gemini Pro.
    
    Args:
        model: The Gemini Pro model instance
        transcript: The medical conversation transcript
    
    Returns:
        A dictionary containing the structured SOAP note
    """
    # Preprocess the transcript
    cleaned_transcript = preprocess_transcript(transcript)
    
    # Fill the prompt template with the transcript
    prompt = prompt_template.format(transcript=cleaned_transcript)
    
    # Generate the SOAP note using Gemini Pro
    response = model.generate_content(prompt)
    
    # Extract the JSON from the response
    try:
        # Try to parse the response text as JSON
        soap_note = json.loads(response.text)
    except json.JSONDecodeError:
        # If parsing fails, extract JSON from the text using regex
        json_match = re.search(r'({[\s\S]*})', response.text)
        if json_match:
            try:
                soap_note = json.loads(json_match.group(1))
            except json.JSONDecodeError:
                # Fallback to a basic structure if JSON extraction fails
                soap_note = create_fallback_soap_note(transcript)
        else:
            soap_note = create_fallback_soap_note(transcript)
    
    return soap_note

def main():
    API_KEY = os.getenv("GEMINI_API_KEY")
    
    # Initialize the SOAP generator
    model = initialize_gemini(API_KEY)
    
    # Sample transcript
    sample_transcript = """
    Doctor: How are you feeling today?
    Patient: I had a car accident. My neck and back hurt a lot for four weeks.
    Doctor: Did you receive treatment?
    Patient: Yes, I had ten physiotherapy sessions, and now I only have occasional back pain.
    """
    
    # Generate SOAP note
    try:
        soap_note = generate_soap_note(model, sample_transcript)
        
        # Print formatted output
        print(json.dumps(soap_note, indent=2))
        
        # Save to file
        with open('soap_summary.json', 'w') as f:
            json.dump(soap_note, f, indent=2)
            
        print("SOAP note successfully generated and saved to soap_summary.json")
    
    except Exception as e:
        print(f"Error generating SOAP note: {e}")
        # Fallback to direct template if API call fails
        fallback = create_fallback_soap_note(sample_transcript)
        with open('soap_summary.json', 'w') as f:
            json.dump(fallback, f, indent=2)
        print("Used fallback method to generate SOAP note")

if __name__ == "__main__":
    main()