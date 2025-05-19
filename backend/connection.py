from fastapi import WebSocket, WebSocketDisconnect
from typing import List, Dict
from  transcription import SpeechProcessor
import asyncio
import json
import base64
import logging
import threading

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[WebSocket, SpeechProcessor] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        proc = SpeechProcessor(websocket)
        proc.start()
        self.active_connections[websocket] = proc
        logger.debug(f"connect: active connection {len(self.active_connections)}")

    async def disconnect(self, socket: WebSocket):
        logger.debug(f"disconnect: active connection {len(self.active_connections)}")
        if socket in self.active_connections:
            await self.active_connections[socket].stop()
            del self.active_connections[socket]

    async def broadcast(self, message: dict):
        for connection in list(self.active_connections.keys()):
            try:
                # logger.debug(f"send message {connection},  {message}")
                await connection.send_json(message)
            except WebSocketDisconnect:
                self.disconnect(connection)

    async def send(self, processor: SpeechProcessor, message: dict):
        try:
            # logger.debug(f"send message {connection},  {message}")
            await processor.socket.send_json(message)
        except WebSocketDisconnect:
            await self.disconnect(self.socket)

manager = ConnectionManager()

async def on_connect_websocket(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            try:
                message = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
            except asyncio.TimeoutError:
                continue  # Allows checking for cancellation every second
            except asyncio.CancelledError:
                logger.info("WebSocket handler cancelled.")
                break
            try:
                sp = manager.active_connections[websocket]
                data = json.loads(message)
                opcode = data.get("opcode")
                if opcode == 'recording_audio':
                    audio_blob = base64.b64decode(data.get("blob"))
                    asyncio.run_coroutine_threadsafe(sp.process_audio(audio_blob), sp._loop)
                elif opcode == 'get_answer':
                    question = data.get("question")
                    logger.debug(f"question: {question}")
                    asyncio.create_task(sp.get_answer_from_ai(question))
                elif opcode == 'set_chat_role':
                    role = data.get("role")
                    logger.debug(f"Chat Role: {role}")
                    asyncio.run_coroutine_threadsafe(sp.set_chat_role(role), sp._loop)
                    
            except json.JSONDecodeError:
                logger.error("Received invalid JSON data.")
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    finally:
        # await manager.disconnect(websocket)
        pass
