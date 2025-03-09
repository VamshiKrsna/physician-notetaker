import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List, Any
import json

class MedicalSentimentAnalyzer:
    def __init__(self):
        # Initialize tokenizer and model from clinical-bert
        self.tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        self.model = AutoModelForSequenceClassification.from_pretrained(
            'emilyalsentzer/Bio_ClinicalBERT',
            num_labels=3  # Anxious, Neutral, Reassured
        )
        
        # Define sentiment and intent labels
        self.sentiment_labels = ['Anxious', 'Neutral', 'Reassured']
        self.intent_labels = [
            'Seeking reassurance',
            'Reporting symptoms',
            'Expressing concern',
            'Requesting information',
            'Acknowledging treatment'
        ]
        
        # Intent keywords mapping
        self.intent_keywords = {
            'Seeking reassurance': ['hope', 'worried', 'concerned', 'will it', 'better'],
            'Reporting symptoms': ['pain', 'hurts', 'feeling', 'symptoms', 'experiencing'],
            'Expressing concern': ['worried', 'anxious', 'scared', 'concerned', 'afraid'],
            'Requesting information': ['what', 'how', 'when', 'why', 'should I'],
            'Acknowledging treatment': ['treatment', 'medication', 'therapy', 'helps', 'working']
        }
    
    def _preprocess_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize and prepare text for model input"""
        return self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
    
    def _detect_intent(self, text: str) -> str:
        """Rule-based intent detection using keyword matching"""
        text_lower = text.lower()
        matched_intents = []
        
        for intent, keywords in self.intent_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                matched_intents.append(intent)
        
        # Return the most relevant intent or default
        return matched_intents[0] if matched_intents else 'Neutral'
    
    def analyze_sentiment(self, text: str) -> Dict[str, str]:
        """Analyze both sentiment and intent of the input text"""
        # Tokenize input
        inputs = self._preprocess_text(text)
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=1)
            sentiment_idx = torch.argmax(predictions).item()
        
        # Get sentiment and intent
        sentiment = self.sentiment_labels[sentiment_idx]
        intent = self._detect_intent(text)
        
        return {
            'Sentiment': sentiment,
            'Intent': intent
        }

def main():
    # Initialize analyzer
    analyzer = MedicalSentimentAnalyzer()
    
    # Test with sample input
    sample_text = "I'm a bit worried about my back pain, but I hope it gets better soon."
    sample_text2 = "I am scared that I might die, Please help me out of this"
    result = analyzer.analyze_sentiment(sample_text)
    result2 = analyzer.analyze_sentiment(sample_text2)
    
    # Print formatted output
    print(json.dumps(result, indent = 2))
    print(json.dumps(result2, indent = 2))

    # Save results to file
    with open('sentiment_analysis.json', 'w') as f:
        json.dump(result, f, indent=2)

    with open('sentiment_analysis2.json', 'w') as f:
        json.dump(result2, f, indent=2)

if __name__ == '__main__':
    main()