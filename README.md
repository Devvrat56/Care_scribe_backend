# AI Scribe Backend

A state-of-the-art medical scribe backend powered by Groq and scispaCy.

## Features
- **Fast Transcription**: Leverages Groq's Whisper API (`whisper-large-v3`) for near-instant speech-to-text.
- **Biomedical Entity Extraction**: Uses `scispaCy` to identify diseases, chemicals, and anatomical terms.
- **Phonetic/Spelling Correction**: Integrated fuzzy matching with `rapidfuzz` to correct medical terminology.
- **Patient-Centric Summarization**: Uses Groq's Llama 3.1 to generate empathetic, medication-focused summaries.

## API Endpoints

### 1. `POST /transcribe`
Transcribes an audio file using Groq Whisper.
- **Query Parameter**: `api_key` (Optional if SET in environment)
- **Body**: `FormData` containing the file.
- **Response**: `{ "transcript": "...", "file_id": "..." }`

### 2. `POST /analyze`
Analyzes medical text to extract entities and generate a summary.
- **Body**: `{ "text": "...", "api_key": "..." }`
- **Response**: `{ "entities": [...], "summary": "..." }`

## Setup & Run

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Environment Variables**:
   Create a `.env` file with:
   ```env
   GROQ_API_KEY=your_key_here
   ```
3. **Start the Server**:
   ```bash
   python main.py
   ```
