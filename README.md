# physician-notetaker
NLP based medical transcript and artifact generator

### 1 . NER With Spacy/ Transformers


**Approach :**  Tried SciSpacy's en_core_sci_scibert model as it is trained specifically for BioClinical and Medical Usecases. Had to write my own categories/ entities for better NER as per our conversation. It worked good enough but lacked better summarization in prognosis and current_status. To compensate for this shortcomings, I wrote similar script but this time, I had to use Cannon to kill a mosquito, I used Gemini pro for better summarization of Prognosis and Current_State


You can find Spacy Implementation for Task 1 <a href= "https://github.com/VamshiKrsna/physician-notetaker/blob/main/Task1Spacy.py"> here</a> 


Spacy Generated outputs like : <a href= "https://github.com/VamshiKrsna/physician-notetaker/blob/main/medical_summary.json"> medical_summary.json </a>


Spacy Implementation was good enough, but it lacked better summarization for prognosis and current_state


Tried Task 1 with Gemini Model, you can find it <a href= "https://github.com/VamshiKrsna/physician-notetaker/blob/main/Task1Gemini.py"> here, Task1Gemini </a>



Gemini Model Generated outputs like : <a href= "https://github.com/VamshiKrsna/physician-notetaker/blob/main/medical_summary.json"> medical_summary2.json </a>



This one obviously performs the best as it is using a SOTA Model.



### 2 . Sentiment Analysis


**Approach :**  In this task, I tried to finetune an instance of BERT, but found it inefficient, so I found a pretrained model called Bio_ClinicalBERT on huggingface, which was pretrained and finetuned on MMIC and other medical datasets. This one worked well for me, hence went forward with this one. It performed well in sentiment and intent analysis.


Sentiment & Intent Analysis -> <a href= "https://github.com/VamshiKrsna/physician-notetaker/blob/main/Task2SentimentAnalysis.py"> Task2SentimentAnalysis </a>



Sentiment and Intent Analyzer Generated outputs like : <a href= "https://github.com/VamshiKrsna/physician-notetaker/blob/main/sentiment_analysis.json"> medical_summary.json </a>



### 3 . SOAP Note Generation


**Approach :**  Researched how to finetune a model for usecase like this, couldn't find any predefined datasets, came up with a custom dataset, unfortunately couldn't build/ finetune the model, so, tried to build this with better efficiency using GEMINI itself, which gave me some decent outputs.



Find SOAP Note Generation <a href= "https://github.com/VamshiKrsna/physician-notetaker/blob/main/Task3SOAP.py"> here, Task3SOAP </a>



SOAP Note Generator powered by GEMINI Generated outputs like : <a href= "https://github.com/VamshiKrsna/physician-notetaker/blob/main/soap_summary.json"> medical_summary.json </a>




#### Questions : 



- How would you handle **ambiguous or missing medical data** in the transcript?


    By Logical Inferencing/ Rule based mapping, for example, if a car accident is mentioned, we have possible injuries like whiplash injury, neck strain, etc. 
    To ensure better accuracy, I'd employ Human reviews and extend the entities manually as the usecase/ application grows.  

  
- What **pre-trained NLP models** would you use for medical summarization?


    I have used SciSpacy here and I have found several Bio Clinical variations of NLP Models already pretrained on medical datasets on huggingface, I'd move forward with the ones that fit best for my task/usecase like Bio_ClinicalBERT, etc.



- How would you fine-tune **BERT** for medical sentiment detection?


  Prepare Medical/ Clinical Variations of BERT or other NLP Models like Clinical/ Bio_ClinicalBERT,


  Prepare a labeled dataset with labels Anxious, Neutral, Reassured and Intents as well


  Create a Trainer, train it and evaluate it using a portion of train data (test data)

  

- What datasets would you use for training a **healthcare-specific** sentiment model?


MIMIC III, IV or MedNLI datasets are healthcare specific datasets I would use for finetuning/ training.



- How would you train an NLP model to **map medical transcripts into SOAP format**?


Prepare a large training and testing datasets with predefined examples of our required schema (SOAP Format)


Choose a model like BERT and train it on our training dataset.


Evaluate it based on metrics like ROUGE.


- What **rule-based or deep-learning** techniques would improve the accuracy of SOAP note generation?


Entity Recognition, Entity Mapping to corresponding sections of SOAP are the techniques that can potentially improve accuracy of SOAP.
