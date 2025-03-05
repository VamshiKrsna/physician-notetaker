import spacy
import json
from typing import Dict, List, Any

# loading the spacy model in english 
# nlp = spacy.load('en_core_web_sm')


class MedicalNERExtractor:
    def __init__(self, model_name="en_core_sci_md"):
        """
        Initialize the Medical NER Extractor
        
        Args:
            model_name (str): spaCy model to use for NER
        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"Downloading {model_name} model...")
            spacy.cli.download(model_name)
            self.nlp = spacy.load(model_name)
        
        # Custom medical entity patterns
        self.medical_keywords = {
            "symptoms": [
                "pain", "ache", "injury", "discomfort", "impact", 
                "stiffness", "inflammation", "swelling"
            ],
            "treatments": [
                "physiotherapy", "medication", "painkillers", 
                "therapy", "examination", "x-ray", "check-up"
            ],
            "medical_conditions": [
                "whiplash", "strain", "sprain", "concussion", 
                "trauma", "bruise", "fracture"
            ]
        }
    
    def extract_medical_entities(self, text: str) -> Dict[str, Any]:
        """
        Extract medical entities from the given text
        
        Args:
            text (str): Input medical text/transcript
        
        Returns:
            Dict of extracted medical entities
        """
        doc = self.nlp(text)
        
        # Initialize results dictionary
        entities = {
            "named_entities": [],
            "symptoms": [],
            "treatments": [],
            "medical_conditions": [],
            "locations": []
        }
        
        # Extract spaCy named entities
        for ent in doc.ents:
            entities["named_entities"].append({
                "text": ent.text,
                "label": ent.label_
            })
            
            # Extract location entities
            if ent.label_ in ["GPE", "LOC"]:
                entities["locations"].append(ent.text)
        
        # Custom keyword-based extraction
        lowered_text = text.lower()
        
        for category, keywords in self.medical_keywords.items():
            matches = [
                keyword for keyword in keywords 
                if keyword in lowered_text
            ]
            entities[category].extend(matches)
        
        return entities
    
    def extract_medical_details(self, text: str) -> Dict[str, Any]:
        """
        Generate a structured medical summary
        
        Args:
            text (str): Full medical conversation/transcript
        
        Returns:
            Structured medical summary
        """
        entities = self.extract_medical_entities(text)
        
        # Structured medical summary
        summary = {
            "Patient_Name": self._extract_patient_name(text),
            "Symptoms": list(set(entities["symptoms"])),
            "Treatments": list(set(entities["treatments"])),
            "Medical_Conditions": list(set(entities["medical_conditions"])),
            "Locations_Mentioned": list(set(entities["locations"]))
        }
        
        return summary
    
    def _extract_patient_name(self, text: str) -> str:
        """
        Extract patient name from text (basic implementation)
        
        Args:
            text (str): Input text
        
        Returns:
            str: Extracted patient name or placeholder
        """
        doc = self.nlp(text)
        
        # Look for person names
        person_names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        
        return person_names[0] if person_names else "Unknown Patient"

def main():
    # Example usage
    transcript = """
    Physician: Good morning, Ms. Jones. How are you feeling today?
    Patient: Good morning, doctor. I'm doing better, but I still have some discomfort.
    I was in a car accident and experienced neck pain and back stiffness.
    I went through physiotherapy and took some painkillers.
    """
    
    ner_extractor = MedicalNERExtractor()
    
    # Demonstrate entity extraction
    print("Raw Medical Entities:")
    print(json.dumps(ner_extractor.extract_medical_entities(transcript), indent=2))
    
    print("\nStructured Medical Summary:")
    print(json.dumps(ner_extractor.extract_medical_details(transcript), indent=2))

if __name__ == "__main__":
    main()