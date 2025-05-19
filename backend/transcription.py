from fastapi import FastAPI, WebSocket, UploadFile, File, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import json
import os
from typing import List
import openai
from dotenv import load_dotenv
import io
from io import BytesIO
import threading
import wave
import subprocess
import numpy as np
from pydub import AudioSegment
import time
from queue import Queue
from datetime import datetime, timedelta
import logging
import signal
import sys
import webrtcvad
import math
import asyncio
from fastapi import WebSocket

# Constants
RATE = 16000
CHANNELS = 1
WHISPER_MODEL = "./whisper.cpp/models/ggml-base.en.bin"  # Path to the Whisper model

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

vad = webrtcvad.Vad(3)  # 0–3: aggressiveness level

class ThreadSafeVariable:
    def __init__(self, initial_value=None):
        self._value = initial_value
        self._lock = threading.Lock()

    def get(self):
        with self._lock:
            return self._value

    def set(self, value):
        with self._lock:
            self._value = value

    def update(self, func):
        with self._lock:
            self._value = func(self._value)

    def add(self, value):
        with self._lock:
            self._value += value
            return self._value
    def chat_completion(self, **kwargs):
        with self._lock:
            return self._value.chat.completions.create(**kwargs)
    def audio_transcriptions(self, **kwargs):
        with self._lock:
            return self._value.audio.transcriptions.create(**kwargs)
        
class SpeechProcessor:
    def __init__(self, websocket: WebSocket):
        self.data_queue = Queue()
        self._task = None
        self._running = False
        self.silent_time = 0
        self.chat_role = "You are a helpful assistant. Answer the following question concisely."
        self.paragraph_blob = None
        self.socket = websocket

    def start(self):
        self._running = True
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        threading.Thread(target=self._loop.run_forever, daemon=True).start()
        asyncio.run_coroutine_threadsafe(self._worker(), self._loop)

    async def stop(self):
        self._running = False
        if self._task:
            await self._task
            self._task = None

    async def _worker(self):
        try:
            await self.handle_transcription()
        finally:
            pass

    async def set_chat_role(self, role):
        self.chat_role = role

    async def process_audio(self, webm_blob):
        logger.debug(f"Received blob: {len(webm_blob)} bytes")
        web_io = BytesIO(webm_blob)
        audio = AudioSegment.from_file(web_io, format="webm")
        # with open("temp_audio.webm", "wb") as temp_file:
            # temp_file.write(webm_blob)

        result = 'speech'
        if (audio.dBFS == float('-inf')):
            result = 'empty'
        elif (not is_speaking_audio(audio)):
            result = 'silent'

        logger.debug(f"Audio is {result}")
        if result == 'speech':
            self.data_queue.put(audio)
            self.silent_time = 0
        else:
            self.silent_time += 1
            if self.silent_time > 3:
                self.silent_time = 0
                if self.paragraph_blob:
                    logger.info(f"Paragraph is complete.")
                self.paragraph_blob = None
        
        return result

    async def handle_transcription(self):
        convert_format = 'mp3'  # or 'wav'
        while self._running:
            try:
                now = datetime.utcnow()
                if not self.data_queue.empty():        # Check if there is data in the queue
                    # Combine audio data from queue
                    audio_segments = []
                    for audio in self.data_queue.queue:  # `data_queue.queue` is a list of bytes
                        audio_segments.append(audio)
                    logger.debug(f"Received {len(audio_segments)} audio segments from queue.")
                    self.data_queue.queue.clear()
                    is_new_pragraph = False
                    if (self.paragraph_blob is None):
                        self.paragraph_blob = audio_segments[0]
                        is_new_pragraph = True
                    else:
                        self.paragraph_blob = self.paragraph_blob + audio_segments[0]
                    for segment in audio_segments[1:]:
                        self.paragraph_blob = self.paragraph_blob + segment
                    audio_blob = convert_webm_blob_to_audio_blob(self.paragraph_blob, format=convert_format)
                    # logger.debug(f"Converted file to {convert_format} format.")

                    # Save the wav blob to a temporary file
                    now_str = datetime.now().strftime("%H_%M_%S__%f")
                    audio_path = f"audio_{now_str}.{convert_format}"
                    with open(audio_path, "wb") as temp_file:
                        temp_file.write(audio_blob)
                    logger.debug(f"Saved {convert_format} to temporary file: {audio_path}")

                    # Transcribe audio using Whisper
                    paragraph = transcribe_offline(audio_blob, audio_path, input_mode = 'file').strip()
                    # paragraph = transcribe_online(audio_blob)
                    logger.info(f"transcription: {paragraph}")
                    # Broadcast to all connected clients

                    from connection import manager
                    asyncio.create_task(manager.send(self, { "opcode": "transcribe", "transcription": paragraph, "new": is_new_pragraph }))

                    # Clean up temporary files
                    os.remove(audio_path)
                else:
                    # Wait for a new audio file to be uploaded
                    await asyncio.sleep(0.25)
            except Exception as e:
                logger.error(f"Error in transcription thread: {e}")
                self.data_queue.queue.clear()
                # break

    async def get_answer_from_ai(self, question: str):
        logger.info(f"Question: {question}")
        # Get answer from ChatGPT
        response = openai_client.chat_completion(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": self.chat_role},
                {"role": "user", "content": question}
            ]
        )
        answer = response.choices[0].message.content
        logger.info(f"Answer: {answer}")

        # Broadcast to all connected clients
        from connection import manager
        asyncio.create_task(manager.send(self, { "opcode": "answer", "question": question, "answer": answer }))

def convert_webm_blob_to_audio_blob(audio, format: str) -> bytes:
        # Convert to MP3 into another BytesIO buffer
        # logger.debug(f"Converting audio to {format} format...")
        audio_io = BytesIO()
        audio.export(audio_io, format=format)

        return audio_io.getvalue()  # Return audio as byte blob

def is_speaking_audio(audio: AudioSegment):
    # Convert to the format required by webrtcvad
    audio = audio.set_channels(1)       # Mono
    audio = audio.set_frame_rate(16000) # 16 kHz
    audio = audio.set_sample_width(2)   # 16-bit (2 bytes)
    # Convert AudioSegment to raw PCM bytes
    pcm_audio = audio.raw_data
    # Frame size in ms (10, 20, or 30 only)
    frame_duration = 30
    sample_rate = 16000
    frame_bytes = int(sample_rate * frame_duration / 1000) * 2  # 2 bytes/sample

    # Split the raw audio into frames
    def frame_generator(audio_bytes, frame_size):
        for i in range(0, len(audio_bytes), frame_size):
            yield audio_bytes[i:i + frame_size]

    # Run VAD on each frame
    frames = list(frame_generator(pcm_audio, frame_bytes))

    speech_frame_count = 0
    for i, frame in enumerate(frames):
        if len(frame) != frame_bytes:
            continue  # Ignore incomplete frames at the end
        is_speech = vad.is_speech(frame, sample_rate)
        if (is_speech):
            speech_frame_count += 1

    logger.debug(f"Frame {speech_frame_count}/{len(frames)}: Speech detected")
    return speech_frame_count > 10 or speech_frame_count > len(frames) / 3

def transcribe_offline(blob_bytes, audio_path, input_mode =  'file'):
        if (input_mode == 'blob'):
            # Create in-memory WAV file
            logger.debug("Creating in-memory audio file...")
            wav_io = io.BytesIO(blob_bytes)
            wf = wave.open(wav_io, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # paInt16 = 2 bytes
            wf.setframerate(RATE)
            wf.writeframes(blob_bytes)
            wf.close()
            wav_io.seek(0)
            logger.debug("In-memory WAV file created.")

        # Run whisper.cpp with blob as stdin
        whisper_cmd = None
        if (input_mode == 'blob'):
            whisper_cmd = [
                "./whisper.cpp/build/bin/Release/whisper-cli", "-m", WHISPER_MODEL, "-nt", "-",  # "-" = read WAV from stdin
            ]
        else:
            whisper_cmd = [
                "./whisper.cpp/build/bin/Release/whisper-cli", "-m", WHISPER_MODEL, "-nt", "-f", audio_path,
            ]

        # logger.debug(f"Running command: {' '.join(whisper_cmd)}")
        # Set up the subprocess to run whisper.cpp
        proc = subprocess.Popen(
            whisper_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        stderr, stdout = None, None
        if (input_mode == 'blob'):
            stdout, stderr = proc.communicate(input=wav_io.read())
        else:
            stdout, stderr = proc.communicate()
        # logger.debug(f"Command output: {stdout.decode('utf-8')}")
        # logger.debug(f"Command error: {stderr.decode('utf-8')}")
        # logger.debug(f"Command return code: {proc.returncode}")
        if proc.returncode != 0:
            raise Exception(f"Error in transcription: {stderr.decode('utf-8')}")

        return stdout.decode("utf-8")

def transcribe_online(audio_blob):
    # Assuming audio_blob is your binary blob (e.g., from an uploaded file)
    audio_file = io.BytesIO(audio_blob)
    audio_file.name = "audio.mp3"  # Give it a name and extension

    # Transcribe
    response = openai_client.audio_transcriptions(
        model="whisper-1",
        file=audio_file,
        language="en"
    )
    return response.text

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")
openai_client = ThreadSafeVariable(openai.OpenAI(api_key=openai.api_key))


def signal_handler(sig, frame):
    print(f"Received signal: {sig}. Exiting gracefully.")

    # Optional: wait for threads to clean up
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # kill command