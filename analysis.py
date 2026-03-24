import spacy
import scispacy
from groq import Groq
import os
from typing import List, Dict
from rapidfuzz import process, fuzz

class MedicalAnalysisService:
    def __init__(self, api_key: str = None):
        # Load scispaCy models
        try:
            self.nlp_bc5cdr = spacy.load("en_ner_bc5cdr_md")
            self.nlp_bionlp = spacy.load("en_ner_bionlp13cg_md")
        except Exception as e:
            print(f"Models loading failed: {e}. Ensure they are installed.")
            self.nlp_bc5cdr = None
            self.nlp_bionlp = None
        
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if self.api_key:
            self.groq_client = Groq(api_key=self.api_key)
        else:
            self.groq_client = None
            
        # Common medical terms for fuzzy matching (medication emphasis)
        self.medical_dictionary = [
            "Metformin", "Lisinopril", "Diabetes", "Hypertension", "Glucose", 
            "Cholesterol", "Amoxicillin", "Ibuprofen", "Acetaminophen", "Asthma",
            "Atorvastatin", "Levothyroxine", "Amlodipine", "Metoprolol", "Omeprazole",
            "Insulin", "Prednisone", "Albuterol", "Warfarin", "Hydrochlorothiazide"
        ]

    def fuzzy_correct(self, text: str) -> str:
        """Corrects potential misspellings in the text using fuzzy matching."""
        words = text.split()
        corrected_words = []
        for word in words:
            if len(word) > 4:
                match = process.extractOne(word, self.medical_dictionary, scorer=fuzz.WRatio)
                if match and match[1] > 85:
                    corrected_words.append(match[0])
                    continue
            corrected_words.append(word)
        return " ".join(corrected_words)

    def extract_entities(self, text: str) -> List[Dict]:
        if not self.nlp_bc5cdr:
            return []
        
        # Apply fuzzy correction
        corrected_text = self.fuzzy_correct(text)
        
        doc_bc5cdr = self.nlp_bc5cdr(corrected_text)
        doc_bionlp = self.nlp_bionlp(corrected_text)
        
        entities = []
        seen = set()
        
        for doc in [doc_bc5cdr, doc_bionlp]:
            for ent in doc.ents:
                key = (ent.text.lower(), ent.label_)
                if key not in seen:
                    entities.append({
                        "text": ent.text,
                        "label": ent.label_,
                    })
                    seen.add(key)
        return entities

    def generate_patient_summary(self, transcript: str, entities: List[Dict], api_key: str = None) -> str:
        client = self.groq_client
        if api_key:
            client = Groq(api_key=api_key)
        
        if not client:
            return "Error: Groq API Key missing. Please provide it in the settings."
            
        entity_str = ", ".join([f"{e['text']} ({e['label']})" for e in entities])
        
        prompt = f"""
        You are a highly skilled and compassionate medical assistant. Your goal is to help a patient understand their consultation.
        
        TASK:
        Based on the transcript and identified medical entities, create a structured patient-friendly summary.
        
        PRIORITIZE:
        1. **Medication Instructions**: Clearly list any new or existing medications, including dosages, frequencies, and specific instructions (e.g., 'Take with food').
        2. **Doctor's Advice**: Summarize the key medical advice and explanations given by the doctor.
        3. **Next Steps**: Explicitly list what the patient needs to do next (appointments, labs, monitoring).
        
        STYLE:
        - Use simple, everyday language. Avoid complex jargon unless explaining it simply.
        - Be empathetic and clear.
        - Use bullet points for readability.
        
        CONSULTATION CONTEXT:
        Transcript: {transcript}
        Medical Entities: {entity_str}
        
        PATIENT SUMMARY & MEDICATION GUIDE:
        """
        
        completion = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a professional medical scribe dedicated to patient clarity and medication adherence."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3, # Lower temperature for medical accuracy
            max_tokens=2048,
        )
        
        return completion.choices[0].message.content
