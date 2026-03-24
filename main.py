from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import shutil
import os
import uuid
from transcription import transcription_service
from analysis import MedicalAnalysisService
import storage_utils

app = FastAPI(title="AI Scribe Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global analysis service (init on startup)
analysis_service = None

@app.on_event("startup")
async def startup_event():
    global analysis_service
    # We'll try to load models. If they aren't downloaded yet, this might fail initially
    try:
        analysis_service = MedicalAnalysisService()
    except Exception as e:
        print(f"Warning: Models not yet available: {e}")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class AnalysisRequest(BaseModel):
    text: str
    api_key: Optional[str] = None
    file_id: Optional[str] = None

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...), api_key: Optional[str] = None):
    file_id = str(uuid.uuid4())
    file_extension = file.filename.split(".")[-1]
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}.{file_extension}")
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Pass API key to transcription service
        transcript = transcription_service.transcribe(file_path, api_key=api_key)
        
        # Save transcript to local storage
        storage_utils.save_transcript(file_id, transcript)
        
        return {"transcript": transcript, "file_id": file_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_path):
             os.remove(file_path)

@app.post("/analyze")
async def analyze_text(request: AnalysisRequest):
    if not analysis_service:
        raise HTTPException(status_code=503, detail="Analysis service not initialized")
    
    entities = analysis_service.extract_entities(request.text, api_key=request.api_key)
    summary = analysis_service.generate_patient_summary(request.text, entities, api_key=request.api_key)
    
    # Save entities and summary to local storage if file_id is provided
    if request.file_id:
        storage_utils.save_entities(request.file_id, entities)
        storage_utils.save_summary(request.file_id, summary)
    
    return {
        "entities": entities,
        "summary": summary
    }

# Serve frontend (Commented out as Streamlit is used instead)
# app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
