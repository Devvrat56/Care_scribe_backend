import streamlit as st
import os
import json
import uuid
import shutil
from transcription import transcription_service
from analysis import MedicalAnalysisService
import storage_utils

# Initialize services
analysis_service = MedicalAnalysisService()

# Directory for temporary file storage
TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

st.set_page_config(page_title="AI Medical Scribe Tester", layout="wide")

st.title("🩺 AI Medical Scribe - Dedicated Tester")
st.markdown("""
This app processes medical consultations using the Groq API.
1. **Transcribe**: Converts audio to text using Whisper.
2. **Analyze**: Extracts clinical details and generates a patient summary using Llama 3.3.
""")

# Sidebar for API Key
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Groq API Key (Optional)", type="password", 
                           help="If provided, this will override the system defaults.")
    st.info("Direct Integration Mode: No separate backend required.")

# Tabs for different stages
tab1, tab2 = st.tabs(["Transcription", "Medical Analysis"])

with tab1:
    st.header("1. Transcribe Consultation")
    audio_file = st.file_uploader("Upload Audio File", type=["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm", "ogg", "flac"])
    
    if st.button("Start Transcription") and audio_file:
        with st.spinner("Processing Audio..."):
            # Generate unique ID and save file temporarily
            file_id = str(uuid.uuid4())
            file_ext = audio_file.name.split(".")[-1]
            temp_path = os.path.join(TEMP_DIR, f"{file_id}.{file_ext}")
            
            with open(temp_path, "wb") as f:
                f.write(audio_file.getvalue())
            
            try:
                # Direct call to transcription service
                transcript = transcription_service.transcribe(temp_path, api_key=api_key)
                
                # Save to local storage
                storage_utils.save_transcript(file_id, transcript)
                
                st.session_state['transcript'] = transcript
                st.session_state['file_id'] = file_id
                
                st.success("Transcription Complete!")
                st.text_area("Transcript", value=transcript, height=300)
                st.info(f"Session ID (Saved locally): {file_id}")
                
            except Exception as e:
                st.error(f"Transcription Error: {str(e)}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

with tab2:
    st.header("2. AI Medical Analysis")
    
    # Use transcript from session state if available
    input_text = st.text_area("Transcript for Analysis", 
                               value=st.session_state.get('transcript', ""), 
                               height=300, 
                               help="You can edit the transcript before analysis.")
    
    curr_file_id = st.text_input("Session ID (Optional)", 
                                value=st.session_state.get('file_id', ""), 
                                help="Used to save results in local storage.")

    if st.button("Run Analysis") and input_text:
        with st.spinner("Extracting Clinical Data & Generating Summary..."):
            try:
                # Direct calls to analysis service
                entities = analysis_service.extract_entities(input_text, api_key=api_key)
                summary = analysis_service.generate_patient_summary(input_text, entities, api_key=api_key)
                
                # Save results if file_id is available
                if curr_file_id:
                    storage_utils.save_entities(curr_file_id, entities)
                    storage_utils.save_summary(curr_file_id, summary)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Structured Clinical Entities")
                    if entities:
                        for ent in entities:
                            st.markdown(f"**{ent['label']}**: {ent['text']}")
                    else:
                        st.write("No specific medical entities extracted.")
                
                with col2:
                    st.subheader("Final Patient Summary")
                    st.markdown(summary)
                
                if curr_file_id:
                    st.success(f"Analysis complete! Data indexed under: `{curr_file_id}`")
                else:
                    st.success("Analysis complete!")
                    
            except Exception as e:
                st.error(f"Analysis Error: {str(e)}")

# Footer
st.markdown("---")
st.caption("AI Medical Scribe - Self-Contained Deployment")
