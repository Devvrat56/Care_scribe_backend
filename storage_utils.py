import os
import json
from typing import Optional

STORAGE_DIR = "storage"

def ensure_storage_dir(file_id: Optional[str] = None):
    """Ensures that the base storage directory and the specific file_id directory exist."""
    os.makedirs(STORAGE_DIR, exist_ok=True)
    if file_id:
        path = os.path.join(STORAGE_DIR, file_id)
        os.makedirs(path, exist_ok=True)
        return path
    return STORAGE_DIR

def save_transcript(file_id: str, transcript: str):
    """Saves the transcription text to a file."""
    dir_path = ensure_storage_dir(file_id)
    file_path = os.path.join(dir_path, "transcript.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(transcript)
    return file_path

def save_entities(file_id: str, entities: list):
    """Saves the extracted entities to a JSON file."""
    dir_path = ensure_storage_dir(file_id)
    file_path = os.path.join(dir_path, "entities.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(entities, f, indent=4)
    return file_path

def save_summary(file_id: str, summary: str):
    """Saves the generated summary to a Markdown file."""
    dir_path = ensure_storage_dir(file_id)
    file_path = os.path.join(dir_path, "summary.md")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(summary)
    return file_path
