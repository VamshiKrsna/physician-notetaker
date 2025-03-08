import spacy
import re

def sanitize_conversation(text:str)->str:
    """
    Sanitizes and cleans the raw conversation text (removes * > symbols)
    """
    text = text.replace("*","")
    text = text.replace(">","")
    return text.strip()


if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")

    with open("convo.txt","r") as f:
      text = f.read()
    cleaned_text = sanitize_conversation(text)
    print(cleaned_text)

    # Save cleaned text to a text file.
    with open("cleaned_convo.txt","w") as f:
      f.write(cleaned_text)
    # Run the above snippet once to generate a neat conversation file.

    doc = nlp(cleaned_text)
    print(doc)