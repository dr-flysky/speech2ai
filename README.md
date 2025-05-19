# Real-Time Q&A Application

This application provides real-time transcription and answers to your questions using Streamlit (frontend) and FastAPI (backend). It uses Whisper for speech-to-text conversion and ChatGPT for generating answers.

## Features

- Real-time audio transcription using Whisper
- Real-time answers using ChatGPT
- WebSocket-based real-time updates
- Modern and responsive UI using Streamlit

## Prerequisites

- Python 3.8 or higher
- OpenAI API key
- FFmpeg (required for Whisper)

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install backend dependencies:
```bash
cd backend
pip install -r requirements.txt
```

4. Install frontend dependencies:
```bash
cd ../frontend
pip install -r requirements.txt
```

5. Create a `.env` file in the backend directory:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## Running the Application

1. Start the backend server:
```bash
cd backend
python main.py
```

2. In a new terminal, start the frontend:
```bash
cd frontend
streamlit run app.py
```

3. Open your browser and navigate to `http://localhost:8501`

## Usage

1. Click the "Start/Stop Listening" button to begin/stop the audio processing
2. Speak your question
3. The transcription and answer will appear in real-time

## Architecture

- **Frontend (Streamlit)**: Handles the user interface and WebSocket connection
- **Backend (FastAPI)**: Processes audio, performs transcription, and generates answers
- **WebSocket**: Enables real-time communication between frontend and backend
- **Whisper**: Handles speech-to-text conversion
- **ChatGPT**: Generates answers based on the transcription

## Notes

- Make sure you have a stable internet connection
- The application requires an OpenAI API key
- Audio processing is done in real-time
- The application supports WebSocket connections for real-time updates

## License

MIT License 