import spacy
from spacy.matcher import PhraseMatcher
import json
from typing import Dict, List, Set

class MedicalNLP:
    def __init__(self):
        # loading the small English model
        # self.nlp = spacy.load("en_core_web_sm")
        self.nlp = spacy.load("en_core_sci_scibert")
        
        # init custom medical term categories, extend as we grow the usecase
        self.symptoms = ["pain", "discomfort", "ache", "stiffness", "shock", "anxiety", 
                        "difficulty", "trouble sleeping", "backache", "neck pain", "back pain", "headache", "spine pain", "muscle pain",
                        "flu", "cough", "cold", "sneezing", "coughing", "fever", "fatigue", "nausea", "vomiting", "diarrhea", "mood swings", "weight loss", "muscle cramps"]
        self.treatments = ["physiotherapy", "painkillers", "x-ray", "examination", 
                         "medical attention", "treatment", "therapy", "medication", "meds", "over-the-counter", "homeopathy", "homeopathic", "homeopathic medicine", "homeopathic treatment", "homeopathic care"]
        self.diagnoses = ["whiplash", "injury", "damage", "trauma", "condition","surgery","psiatica", "whiplash injury", "whiplash pain", "whiplash symptoms"]
        self.body_parts = ["neck", "back", "head", "spine", "muscles", "limbs", "bones", "joints", "skin", "eyes", "ears", "nose", "tongue", "stomach", "heart", "lungs", "kidneys", "bones", "joints", "skin", "eyes", "ears", "nose", "tongue", "stomach", "heart", "lungs", "kidneys"]
        
        # creating custom phrase matchers
        self.symptom_matcher = self._create_matcher(self.symptoms)
        self.treatment_matcher = self._create_matcher(self.treatments)
        self.diagnosis_matcher = self._create_matcher(self.diagnoses)
        self.body_part_matcher = self._create_matcher(self.body_parts)

    def _create_matcher(self, terms: List[str]) -> PhraseMatcher:
        matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        patterns = [self.nlp.make_doc(text) for text in terms]
        matcher.add("Terms", patterns)
        return matcher

    def _extract_matches(self, doc, matcher: PhraseMatcher) -> Set[str]:
        matches = matcher(doc)
        return {doc[start:end].text.lower() for _, start, end in matches}

    def _extract_patient_name(self, text: str) -> str:
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ == "PERSON" and "Ms." in text or "Mr." in text:
                return ent.text
        return "Unknown"

    def analyze_conversation(self, text: str) -> Dict:
        doc = self.nlp(text)
        
        # extracting all information from convo
        symptoms = self._extract_matches(doc, self.symptom_matcher)
        treatments = self._extract_matches(doc, self.treatment_matcher)
        diagnoses = self._extract_matches(doc, self.diagnosis_matcher)
        
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
            "Patient_Name": self._extract_patient_name(text),
            "Symptoms": list(symptoms),
            "Diagnosis": list(diagnoses),
            "Treatment": list(treatments),
            "Current_Status": current_status,
            "Prognosis": prognosis
        }
        
        return medical_summary

def main():
    # Initialize the medical NLP pipeline
    medical_nlp = MedicalNLP()
    
    # Read the conversation from file
    with open("cleaned_convo.txt", "r") as f:
        conversation = f.read()
    
    # Analyze the conversation
    summary = medical_nlp.analyze_conversation(conversation)
    
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
    summary2 = medical_nlp.analyze_conversation(convo2)
    print(json.dumps(summary2, indent=2))
    with open("medical_summary2.json","w") as f:
      json.dump(summary2, f, indent=2)

    # Figured out why convo2 output is not as desired, the convo in itself doesn't contain all the info, 
    # It basically looks like an extension to initial convo. So, We should be good with this implementation alone

if __name__ == "__main__":
    main()