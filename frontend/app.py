import streamlit as st
import asyncio
import websockets
import json
import requests
import time
from datetime import datetime

# Constants
BACKEND_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws"

# Initialize session state
if 'transcription' not in st.session_state:
    st.session_state.transcription = ""
if 'answer' not in st.session_state:
    st.session_state.answer = ""
if 'is_listening' not in st.session_state:
    st.session_state.is_listening = False

# Page config
st.set_page_config(
    page_title="Real-Time Q&A",
    page_icon="🎙️",
    layout="wide"
)

# Title and description
st.title("🎙️ Real-Time Q&A")
st.markdown("""
This application provides real-time transcription and answers to your questions.
The transcription and answers will appear automatically as you speak.
""")

# Create two columns for transcription and answer
col1, col2 = st.columns(2)

# Transcription display
with col1:
    st.subheader("🎯 Transcription")
    transcription_placeholder = st.empty()
    transcription_placeholder.markdown(f"*{st.session_state.transcription}*")

# Answer display
with col2:
    st.subheader("💡 Answer")
    answer_placeholder = st.empty()
    answer_placeholder.markdown(f"*{st.session_state.answer}*")

# WebSocket connection function
async def connect_websocket():
    try:
        async with websockets.connect(WS_URL) as websocket:
            while True:
                try:
                    data = await websocket.recv()
                    message = json.loads(data)
                    
                    # Update session state
                    st.session_state.transcription = message.get("transcription", "")
                    st.session_state.answer = message.get("answer", "")
                    
                    # Update display
                    transcription_placeholder.markdown(f"*{st.session_state.transcription}*")
                    answer_placeholder.markdown(f"*{st.session_state.answer}*")
                    
                except websockets.exceptions.ConnectionClosed:
                    break
    except Exception as e:
        st.error(f"WebSocket connection error: {str(e)}")

# Start/Stop button
if st.button("Start/Stop Listening"):
    st.session_state.is_listening = not st.session_state.is_listening
    
    if st.session_state.is_listening:
        st.success("Listening started...")
        # Start WebSocket connection in a separate thread
        asyncio.run(connect_websocket())
    else:
        st.info("Listening stopped.")

# Status indicator
status_color = "green" if st.session_state.is_listening else "red"
st.markdown(f"""
<div style='text-align: center; margin-top: 20px;'>
    <span style='color: {status_color}; font-size: 24px;'>●</span>
    <span style='margin-left: 10px;'>{'Listening...' if st.session_state.is_listening else 'Not listening'}</span>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    Powered by Streamlit, FastAPI, Whisper, and ChatGPT
</div>
""", unsafe_allow_html=True) 