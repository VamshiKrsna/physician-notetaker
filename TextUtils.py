# Utils/ methods for cleaning conversations, etc.

def sanitize_conversation(text:str)->str:
    """
    Sanitizes and cleans the raw conversation text (removes * > symbols)
    """
    text = text.replace("*","")
    text = text.replace(">","")
    return text.strip()