# Using Gemini Generative AI, lets solve Task 1
import json
import os
from google.generativeai import GenerativeModel, configure
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv() # Loads google api key from .env file

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

configure(api_key=GEMINI_API_KEY) # Configuring gemini models with our api key

model = GenerativeModel("gemini-1.5-pro") # using gemini 1.5 pro model

prompt_template = """
        You are a specialized medical AI assistant trained to extract structured medical information from physician-patient conversations.

        Analyze the following physician-patient conversation transcript carefully, and extract the following information in JSON format:
        
        1. Patient_Name: The patient's name or "Unknown" if not mentioned.
        2. Symptoms: List all symptoms mentioned by the patient.
        3. Diagnosis: Any diagnoses mentioned in the conversation.
        4. Treatment: All treatments, medications, or procedures mentioned or recommended.
        5. Current_Status: The patient's current health status as described in the conversation.
        6. Prognosis: Any mentioned expected outcomes, recovery timelines, or future outlooks.
        
        IMPORTANT INSTRUCTIONS:
        - Be comprehensive and exact when extracting information.
        - Include only information explicitly mentioned in the transcript.
        - If information for a category is not mentioned, use ["Not mentioned in conversation"] for that field.
        - Return ONLY valid JSON with no additional text, explanations, or preamble.
        - Ensure your response is parseable by Python's json.loads() function.
        
        Here's the physician-patient conversation transcript:
        
        {conversation}
        """

# Function to process convo : 
def process_convo(conversation):
    prompt = prompt_template.format(conversation=conversation)
    response = model.generate_content(prompt)
            
    # extract and parse the JSON response
    json_text = response.text
            
    # cleaning the response (sometimes models may add markdown code blocks)
    if "```json" in json_text:
        json_text = json_text.split("```json")[1].split("```")[0].strip()
    elif "```" in json_text:
        json_text = json_text.split("```")[1].split("```")[0].strip()
                
    # Parse JSON
    result = json.loads(json_text)
            
    return result
        
if __name__ == "__main__":
    with open("cleaned_convo.txt","r") as f:
        convo = f.read()

    result = process_convo(convo)
    print(json.dumps(result,indent=2))
    
    # Save the summary to transformer_summary.json
    with open("transformer_summary.json", "w") as f:
        json.dump(result, f, indent=2)
