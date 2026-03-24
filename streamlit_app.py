import streamlit as st
import requests
import os
import json

# Backend URL (Assumes FastAPI is running on port 8000)
BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="AI Medical Scribe Tester", layout="wide")

st.title("🩺 AI Medical Scribe - Backend Tester")
st.markdown("""
This app allows you to test the AI Medical Scribe backend. 
1. Upload an audio recording.
2. Get the transcript.
3. Automatically extract medical entities and generate a patient-friendly summary.
""")

# Sidebar for API Key
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Groq API Key (Optional)", type="password", help="If not provided, the backend's environment variables will be used.")

# Tabs for different stages
tab1, tab2 = st.tabs(["Transcription", "Medical Analysis"])

with tab1:
    st.header("1. Transcribe Consultation")
    audio_file = st.file_uploader("Upload Audio File", type=["mp3", "wav", "m4a"])
    
    if st.button("Start Transcription") and audio_file:
        with st.spinner("Transcribing..."):
            files = {"file": (audio_file.name, audio_file.getvalue(), audio_file.type)}
            params = {"api_key": api_key} if api_key else {}
            
            try:
                response = requests.post(f"{BACKEND_URL}/transcribe", files=files, params=params)
                response.raise_for_status()
                data = response.json()
                
                st.session_state['transcript'] = data.get("transcript")
                st.session_state['file_id'] = data.get("file_id")
                
                st.success("Transcription Complete!")
                st.text_area("Transcript", value=st.session_state['transcript'], height=300)
                st.info(f"Session ID: {st.session_state['file_id']}")
                
            except Exception as e:
                st.error(f"Error: {e}")

with tab2:
    st.header("2. AI Medical Analysis")
    
    # Use transcript from session state if available
    input_text = st.text_area("Transcript for Analysis", 
                               value=st.session_state.get('transcript', ""), 
                               height=300, 
                               help="You can edit the transcript before analysis.")
    
    file_id = st.text_input("Session ID (Optional)", value=st.session_state.get('file_id', ""), help="Used to group results in local storage.")

    if st.button("Run Analysis") and input_text:
        with st.spinner("Analyzing text..."):
            payload = {
                "text": input_text,
                "api_key": api_key if api_key else None,
                "file_id": file_id if file_id else None
            }
            
            try:
                response = requests.post(f"{BACKEND_URL}/analyze", json=payload)
                response.raise_for_status()
                analysis_data = response.json()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Medical Entities")
                    if analysis_data.get("entities"):
                        st.json(analysis_data["entities"])
                    else:
                        st.write("No entities found.")
                
                with col2:
                    st.subheader("Patient Summary")
                    st.markdown(analysis_data.get("summary", "No summary generated."))
                
                st.success(f"Analysis complete! Results saved in `storage/{file_id}/`")
                
            except Exception as e:
                st.error(f"Error: {e}")

# Footer
st.markdown("---")
st.caption("AI Medical Scribe Backend - Local Storage Enabled")
