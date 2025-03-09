from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification
import json
from typing import Dict, List, Set

class MedicalNLPTransformer:
    def __init__(self):
        # Initialize NER pipeline with medical NER model
        self.ner_model_name = "samrawal/bert-base-uncased_clinical-ner"
        self.ner = pipeline("ner", model=self.ner_model_name, aggregation_strategy="simple")
        
        # Initialize zero-shot classification for improved entity categorization
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        
        # Medical term categories
        self.categories = {
            "symptoms": ["pain", "discomfort", "ache", "stiffness", "anxiety", "difficulty"],
            "treatments": ["physiotherapy", "medication", "therapy", "examination"],
            "diagnoses": ["whiplash", "injury", "trauma", "condition"],
            "body_parts": ["neck", "back", "head", "spine", "muscles"]
        }
    
    def _classify_entity(self, entity: str) -> str:
        # Classify the entity into one of our medical categories
        candidate_labels = list(self.categories.keys())
        result = self.classifier(entity, candidate_labels)
        return result['labels'][0] if result['scores'][0] > 0.5 else "other"
    
    def _extract_patient_name(self, text: str) -> str:
        # Use NER to find person names
        entities = self.ner(text)
        for entity in entities:
            if entity['entity_group'] == 'B-NAME' and ('Ms.' in text or 'Mr.' in text):
                return entity['word']
        return "Unknown"
    
    def _extract_medical_entities(self, text: str) -> Dict[str, Set[str]]:
        # Extract entities using transformer NER
        entities = self.ner(text)
        
        # Initialize categories
        medical_entities = {
            "symptoms": set(),
            "treatments": set(),
            "diagnoses": set(),
            "body_parts": set()
        }
        
        # Categorize entities
        for entity in entities:
            if entity['score'] > 0.5:  # Confidence threshold
                category = self._classify_entity(entity['word'])
                if category in medical_entities:
                    medical_entities[category].add(entity['word'].lower())
        
        return medical_entities
    
    def _extract_status_and_prognosis(self, text: str) -> tuple:
        # Use zero-shot classification for better context understanding
        sentences = text.split('.')
        current_status = ""
        prognosis = ""
        
        status_labels = ["current condition", "present state"]
        prognosis_labels = ["future outlook", "recovery expectation"]
        
        for sentence in sentences:
            if sentence.strip():
                # Classify for current status
                status_result = self.classifier(sentence, status_labels)
                if status_result['scores'][0] > 0.6:
                    current_status = sentence.strip()
                
                # Classify for prognosis
                prognosis_result = self.classifier(sentence, prognosis_labels)
                if prognosis_result['scores'][0] > 0.6:
                    prognosis = sentence.strip()
        
        return current_status, prognosis
    
    def analyze_conversation(self, text: str) -> Dict:
        # Extract medical entities
        medical_entities = self._extract_medical_entities(text)
        
        # Extract current status and prognosis
        current_status, prognosis = self._extract_status_and_prognosis(text)
        
        # Create structured output
        medical_summary = {
            "Patient_Name": self._extract_patient_name(text),
            "Symptoms": list(medical_entities["symptoms"]),
            "Diagnosis": list(medical_entities["diagnoses"]),
            "Treatment": list(medical_entities["treatments"]),
            "Current_Status": current_status,
            "Prognosis": prognosis
        }
        
        return medical_summary

def main():
    # Initialize the transformer-based medical NLP pipeline
    medical_nlp = MedicalNLPTransformer()
    
    # Read the conversation from file
    with open("cleaned_convo.txt", "r") as f:
        conversation = f.read()
    
    # Analyze the conversation
    summary = medical_nlp.analyze_conversation(conversation)
    
    # Print formatted JSON output
    print(json.dumps(summary, indent=2))
    
    # Save to file
    with open("medical_summary_transformer.json", "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()