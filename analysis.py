import spacy
import scispacy
from groq import Groq
import os
import json
from typing import List, Dict, Optional
from rapidfuzz import process, fuzz

class MedicalAnalysisService:
    def __init__(self, api_key: Optional[str] = None):
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

    def extract_entities(self, text: str, api_key: Optional[str] = None) -> List[Dict]:
        entities = []
        seen = set()

        # 1. ScispaCy NER (existing)
        if self.nlp_bc5cdr:
            # Apply fuzzy correction
            corrected_text = self.fuzzy_correct(text)
            doc_bc5cdr = self.nlp_bc5cdr(corrected_text)
            doc_bionlp = self.nlp_bionlp(corrected_text)
            
            for doc in [doc_bc5cdr, doc_bionlp]:
                for ent in doc.ents:
                    key = (ent.text.lower(), ent.label_)
                    if key not in seen:
                        entities.append({"text": ent.text, "label": ent.label_})
                        seen.add(key)
        
        # 2. LLM-based Detailed Extraction (New)
        detailed_entities = self.extract_detailed_entities(text, api_key)
        for ent in detailed_entities:
            key = (ent['text'].lower(), ent['label'])
            if key not in seen:
                entities.append(ent)
                seen.add(key)
                
        return entities

    def extract_detailed_entities(self, text: str, api_key: Optional[str] = None) -> List[Dict]:
        client = self.groq_client
        if api_key:
            client = Groq(api_key=api_key)
        
        if not client:
            return []

        prompt = f"""
        Extract detailed medical information from the following consultation transcript.
        Focus on:
        - Patient Details (Age, Gender, ID)
        - Patient Surgery (Existing, upcoming, pre-op, post-op)
        - Medications (Name, Dosage, Frequency, Time Slot)
        - Injuries (Details, location, pre/post-surgery state)
        - Pain levels and locations
        
        TRANSCRIPT: {text}
        
        RETURN ONLY A JSON ARRAY of objects: [{{"text": "ENTITY", "label": "LABEL"}}]
        Valid Labels: PATIENT_DETAIL, SURGERY, MEDICATION, DOSAGE, TIME_SLOT, INJURY, PAIN.
        """
        
        try:
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            content = completion.choices[0].message.content
            # Handle potential JSON wrapping if not using response_format correctly
            data = json.loads(content)
            if isinstance(data, dict) and "entities" in data:
                 return data["entities"]
            if isinstance(data, list):
                return data
            return []
        except Exception as e:
            print(f"Detailed extraction failed: {e}")
            return []

    def generate_patient_summary(self, transcript: str, entities: List[Dict], api_key: Optional[str] = None) -> str:
        client = self.groq_client
        if api_key:
            client = Groq(api_key=api_key)
        
        if not client:
            return "Error: Groq API Key missing. Please provide it in the settings."
            
        entity_str = ", ".join([f"{e['text']} ({e['label']})" for e in entities])
        
        prompt = f"""
        You are a highly skilled medical scribe. Create a patient-friendly summary from this consultation.
        
        CONSULTATION DATA:
        Transcript: {transcript}
        Key Medical Details (extracted): {entity_str}
        
        REQUIREMENTS:
        1. **Patient Context**: Briefly mention relevant details like current state or reason for visit.
        2. **Surgery Info**: Detail any surgery mentioned (pre-op instructions or post-op care/injuries).
        3. **Medication Schedule**: List medications with exact dosages and *specific time slots* (e.g., Morning/Night).
        4. **Pain/Symptom Management**: Address any pain or injuries discussed.
        5. **Empathetic Tone**: Use clear, reassuring, and simple language.
        
        PATIENT SUMMARY:
        """
        
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You provide clear, accurate, and empathetic medical summaries for patients."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2048,
        )
        
        return completion.choices[0].message.content
