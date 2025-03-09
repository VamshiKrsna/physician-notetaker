import spacy
from spacy.matcher import PhraseMatcher
import json
from typing import Dict, List, Set

# loading the small English model
# nlp = spacy.load("en_core_web_sm")
nlp = spacy.load("en_core_sci_scibert")

# init custom medical term categories, extend as we grow the usecase
symptoms = ["pain", "discomfort", "ache", "stiffness", "shock", "anxiety", 
            "difficulty", "trouble sleeping", "backache", "neck pain", "back pain", "headache", "spine pain", "muscle pain",
            "flu", "cough", "cold", "sneezing", "coughing", "fever", "fatigue", "nausea", "vomiting", "diarrhea", "mood swings", "weight loss", "muscle cramps"]
treatments = ["physiotherapy", "painkillers", "x-ray", "examination", 
             "medical attention", "treatment", "therapy", "medication", "meds", "over-the-counter", "homeopathy", "homeopathic", "homeopathic medicine", "homeopathic treatment", "homeopathic care"]
diagnoses = ["whiplash", "injury", "damage", "trauma", "condition","surgery","psiatica", "whiplash injury", "whiplash pain", "whiplash symptoms"]
body_parts = ["neck", "back", "head", "spine", "muscles", "limbs", "bones", "joints", "skin", "eyes", "ears", "nose", "tongue", "stomach", "heart", "lungs", "kidneys", "bones", "joints", "skin", "eyes", "ears", "nose", "tongue", "stomach", "heart", "lungs", "kidneys"]

# creating custom phrase matchers
def create_matcher(terms: List[str]) -> PhraseMatcher:
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(text) for text in terms]
    matcher.add("Terms", patterns)
    return matcher

symptom_matcher = create_matcher(symptoms)
treatment_matcher = create_matcher(treatments)
diagnosis_matcher = create_matcher(diagnoses)
body_part_matcher = create_matcher(body_parts)

def extract_matches(doc, matcher: PhraseMatcher) -> Set[str]:
    matches = matcher(doc)
    return {doc[start:end].text.lower() for _, start, end in matches}

def extract_patient_name(text: str) -> str:
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON" and "Ms." in text or "Mr." in text:
            return ent.text
    return "Unknown"

def analyze_conversation(text: str) -> Dict:
    doc = nlp(text)
    
    # extracting all information from convo
    symptoms_found = extract_matches(doc, symptom_matcher)
    treatments_found = extract_matches(doc, treatment_matcher)
    diagnoses_found = extract_matches(doc, diagnosis_matcher)
    
    # extracting current status and prognosis with improved context understanding
    current_status = ""
    prognosis = ""
    
    for sent in doc.sents:
        sent_text = sent.text.lower()
        if "current" in sent_text or "now" in sent_text or "still" in sent_text:
            current_status = sent.text.encode('ascii', 'ignore').decode()
        if "future" in sent_text or "expect" in sent_text or "recovery" in sent_text:
            prognosis = sent.text.encode('ascii', 'ignore').decode()

    # Create structured output
    medical_summary = {
        "Patient_Name": extract_patient_name(text),
        "Symptoms": list(symptoms_found),
        "Diagnosis": list(diagnoses_found),
        "Treatment": list(treatments_found),
        "Current_Status": current_status,
        "Prognosis": prognosis
    }
    
    return medical_summary

def main():
    # Read the conversation from file
    with open("cleaned_convo.txt", "r") as f:
        conversation = f.read()
    
    # Analyze the conversation
    summary = analyze_conversation(conversation)
    
    # Print formatted JSON output
    print(json.dumps(summary, indent=2))
    
    # Optionally save to file
    with open("medical_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Example 2 from the assignment doc : 
    convo2 = """
    Doctor: How are you feeling today?
    Patient: I had a car accident. My neck and back hurt a lot for four weeks.
    Doctor: Did you receive treatment?
    Patient: Yes, I had ten physiotherapy sessions, and now I only have occasional back pain.
    """
    summary2 = analyze_conversation(convo2)
    print(json.dumps(summary2, indent=2))
    with open("medical_summary2.json", "w") as f:
        json.dump(summary2, f, indent=2)

    # Figured out why convo2 output is not as desired, the convo in itself doesn't contain all the info, 
    # It basically looks like an extension to initial convo. So, We should be good with this implementation alone

if __name__ == "__main__":
    main()