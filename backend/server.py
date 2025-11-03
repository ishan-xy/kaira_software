import asyncio
import websockets
import json
import logging
import sys
import aiohttp
import threading
import zmq
import numpy as np

# --- Global State for Identity ---
current_known_identity = "Unknown"
last_known_identity = "Unknown"
IDENTITY_SUB_URL = "ipc:///tmp/identity_stream"
AI_TRANSCRIPTION_SUB_URL = "ipc:///tmp/ai_transcription_stream"
AI_PROMPT_PUSH_URL = "ipc:///tmp/ai_prompt_stream" 

# --- NEW: Global state for tracking AI response stream ---
IS_STREAMING_AI_RESPONSE = False

# --- Basic Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logging.getLogger('websockets').setLevel(logging.WARNING)

# --- Connection Management ---
browser_websocket = None
transcriptor_websocket = None
prompt_publisher = None 
zmq_context = None 
TRANSCRIPTOR_URI = "ws://localhost:8002"

# --- ZMQ: Identity Worker (No changes) ---
def identity_subscriber_worker(loop, identity_queue):
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    
    try:
        logging.info(f"ZMQ Worker connecting to identity stream at {IDENTITY_SUB_URL}")
        socket.connect(IDENTITY_SUB_URL)
        socket.subscribe(b"current_identity")
        
        while True:
            topic, identity_json = socket.recv_multipart()
            loop.call_soon_threadsafe(identity_queue.put_nowait, identity_json.decode())

    except Exception as e:
        logging.error(f"Error in identity worker: {e}")
    finally:
        socket.close()
        context.term()
        logging.info("ZMQ Identity Worker shut down.")

# --- ZMQ: AI Transcription Worker (No changes) ---
def transcription_subscriber_worker(loop, transcription_queue):
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    
    try:
        logging.info(f"ZMQ Worker connecting to AI transcription stream at {AI_TRANSCRIPTION_SUB_URL}")
        socket.connect(AI_TRANSCRIPTION_SUB_URL)
        socket.subscribe(b"ai_transcription")
        
        while True:
            topic, transcription_json = socket.recv_multipart()
            loop.call_soon_threadsafe(transcription_queue.put_nowait, transcription_json.decode())

    except Exception as e:
        logging.error(f"Error in transcription worker: {e}")
    finally:
        socket.close()
        context.term()
        logging.info("ZMQ Transcription Worker shut down.")

# --- Async: Identity Consumer (No changes) ---
async def consume_identity_updates(identity_queue):
    global current_known_identity, last_known_identity
    while True:
        identity_json = await identity_queue.get()
        
        try:
            data = json.loads(identity_json)
            new_identity = data.get("identity")
            
            if new_identity and new_identity != current_known_identity:
                logging.info(f"👥 Identity changed: {current_known_identity} -> {new_identity}")
                if current_known_identity != "Unknown":
                     last_known_identity = current_known_identity
                current_known_identity = new_identity
            
            elif new_identity == "Unknown" and current_known_identity != "Unknown":
                last_known_identity = current_known_identity
                current_known_identity = "Unknown"
                logging.info("👥 Identity changed: Unknown")

        except Exception as e:
            logging.error(f"Error processing identity update: {e}")

# --- Async: AI Transcription Consumer (MODIFIED) ---
async def consume_transcription_updates(transcription_queue):
    """
    Pulls transcriptions from ZMQ and relays them to the browser
    in streaming chunks.
    """
    global IS_STREAMING_AI_RESPONSE
    while True:
        transcription_json = await transcription_queue.get()
        
        try:
            data = json.loads(transcription_json)
            msg_type = data.get("type")
            text = data.get("text")

            if msg_type == "chunk":
                if not IS_STREAMING_AI_RESPONSE:
                    # This is the first chunk of a new response
                    browser_msg_type = "aiResponseStart"
                    IS_STREAMING_AI_RESPONSE = True
                else:
                    # This is a subsequent chunk
                    browser_msg_type = "aiResponseChunk"
                
                # Relay the chunk to the browser
                await send_to_browser(json.dumps({
                    'type': browser_msg_type,
                    'text': text
                }))
            
            elif msg_type == "final":
                # The stream for this response is done.
                IS_STREAMING_AI_RESPONSE = False
                
                # Send a final, complete message to the browser
                # This ensures the browser has the full text and can reset.
                await send_to_browser(json.dumps({
                    'type': 'aiResponseFinal', 
                    'text': text
                }))
                logging.info(f"[AI Processing] Relayed final transcription: {text[:50]}...")
        
        except Exception as e:
            logging.error(f"Error processing transcription update: {e}")

# --- Main Server Functions ---

async def send_to_browser(message):
    global browser_websocket
    if browser_websocket:
        try:
            await browser_websocket.send(message)
        except websockets.exceptions.ConnectionClosed:
            logging.warning("Browser connection closed mid-send.")

# --- process_ai_prompt (No changes) ---
async def process_ai_prompt(prompt_text):
    global current_known_identity, last_known_identity, prompt_publisher
    logging.info(f"\n[AI Processing] Received prompt: {prompt_text}")
    name = current_known_identity
    
    if name != "Unknown":
        prompt_text = f"The person speaking is {name}. {prompt_text}"
    elif last_known_identity != "Unknown":
         prompt_text = f"The previous person was {last_known_identity}. {prompt_text}"
    
    logging.info(f"[AI Processing] Sending prompt to liveapi: {prompt_text}")

    if prompt_publisher:
        try:
            prompt_publisher.send_json({"prompt": prompt_text})
        except Exception as e:
            logging.error(f"Failed to send prompt via ZMQ: {e}")
    else:
        logging.error("ZMQ Prompt Publisher not initialized.")

# --- connect_to_transcriptor (Robust version from previous answer) ---
async def connect_to_transcriptor():
    global transcriptor_websocket
    while True:
        try:
            async with websockets.connect(TRANSCRIPTOR_URI) as websocket:
                logging.info("Connected to Audio Transcriptor")
                transcriptor_websocket = websocket
                
                await send_to_browser(json.dumps({'type': 'resumeAudio'}))
                
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        
                        if data['type'] == 'realtime':
                            await send_to_browser(message)
                            
                        elif data['type'] == 'fullSentence':
                            await send_to_browser(message)
                            asyncio.create_task(process_ai_prompt(data['text']))
                    
                    except json.JSONDecodeError:
                        logging.error(f"Received malformed JSON from transcriptor: {message}")
                    except Exception as e:
                        logging.error(f"Error processing transcriptor message: {e}")
                        
        except (websockets.exceptions.ConnectionClosed, ConnectionRefusedError) as e:
            logging.warning(f"Transcriptor connection lost ({e}). Reconnecting in 2s...")
        except Exception as e:
            logging.error(f"Unexpected error in transcriptor loop: {e}. Reconnecting in 2s...")
        
        transcriptor_websocket = None
        await send_to_browser(json.dumps({'type': 'pauseAudio'}))
        await asyncio.sleep(2)

# --- browser_handler (Robust version from previous answer) ---
async def browser_handler(websocket):
    global browser_websocket
    
    if browser_websocket:
        logging.warning("New client connected, disconnecting the old one.")
        try:
            await browser_websocket.send(json.dumps({'type': 'pauseAudio'}))
            await browser_websocket.close(code=1000, reason="Replaced by new connection")
        except websockets.exceptions.ConnectionClosed:
            pass
            
    browser_websocket = websocket
    logging.info(f"Browser client connected.")
    
    await send_to_browser(json.dumps({'type': 'pauseAudio'}))
    
    if transcriptor_websocket:
        logging.info("Transcriptor is already connected, sending resume.")
        await send_to_browser(json.dumps({'type': 'resumeAudio'}))
        
    try:
        async for message in websocket:
            if transcriptor_websocket:
                try:
                    await transcriptor_websocket.send(message)
                except websockets.exceptions.ConnectionClosed:
                    logging.warning("Transcriptor connection closed mid-send. Skipping.")
            else:
                pass
                
    except websockets.exceptions.ConnectionClosed:
        logging.info("Browser client disconnected")
    except Exception as e:
        logging.error(f"UNCAUGHT ERROR in browser_handler: {e}", exc_info=True)
    finally:
        if browser_websocket == websocket:
             browser_websocket = None
             logging.info(f"Active client removed.")

# --- main (No changes) ---
async def main():
    global prompt_publisher, zmq_context
    loop = asyncio.get_running_loop()
    identity_queue = asyncio.Queue()
    transcription_queue = asyncio.Queue()
    
    try:
        zmq_context = zmq.Context()
        prompt_publisher = zmq_context.socket(zmq.PUSH)
        prompt_publisher.connect(AI_PROMPT_PUSH_URL)
        logging.info(f"ZMQ PUSH socket connected to {AI_PROMPT_PUSH_URL}")
    except Exception as e:
        logging.fatal(f"Could not connect ZMQ PUSH socket: {e}")
        return

    identity_thread = threading.Thread(
        target=identity_subscriber_worker,
        args=(loop, identity_queue),
        daemon=True
    )
    identity_thread.start()
    
    transcription_thread = threading.Thread(
        target=transcription_subscriber_worker,
        args=(loop, transcription_queue),
        daemon=True
    )
    transcription_thread.start()
    
    asyncio.create_task(consume_identity_updates(identity_queue))
    asyncio.create_task(consume_transcription_updates(transcription_queue)) # This will run the modified version
    asyncio.create_task(connect_to_transcriptor())
    
    logging.info("Main server started on ws://localhost:8001")
    async with websockets.serve(browser_handler, "localhost", 8001):
        await asyncio.Future()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("\nShutting down main server...")
    finally:
        if prompt_publisher:
            prompt_publisher.close()
        if zmq_context:
            zmq_context.term()
        logging.info("ZMQ resources cleaned up.")