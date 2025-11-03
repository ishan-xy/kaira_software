# liveserver.py
import asyncio
import json
import logging
import os
import time  # <<< NEW IMPORT
from pathlib import Path
import zmq
import threading

import aiohttp_cors
from dotenv import load_dotenv
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription
from google import genai

# --- 0. Configuration & Setup (No changes) ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WebRTC_Server")

load_dotenv()
API_KEY = os.getenv("GENAI_API_KEY")
if not API_KEY:
    print("FATAL: Missing GENAI_API_KEY in environment.")
    exit(1)

client = genai.Client(api_key=API_KEY)
model = "gemini-2.0-flash-live-001"
GENAI_CONFIG = {
  "response_modalities": ["AUDIO"],
  "system_instruction": "You are a helpful assistant and answer in a friendly tone. every prompt will contain The person speaking is <name> and you should address them by their name and remember it if someone asks about it",
  "output_audio_transcription": {}
}

pc_set = set()
active_data_channel = None

# --- ZMQ Setup (No changes) ---
AI_TRANSCRIPTION_PUB_URL = "ipc:///tmp/ai_transcription_stream"
AI_PROMPT_PULL_URL = "ipc:///tmp/ai_prompt_stream" 

zmq_context = zmq.Context()
transcription_publisher = zmq_context.socket(zmq.PUB)
transcription_publisher.bind(AI_TRANSCRIPTION_PUB_URL)
logger.info(f"ZMQ Publisher bound to {AI_TRANSCRIPTION_PUB_URL}")


# --- ZMQ Prompt Receiver Thread (MODIFIED) ---
def prompt_receiver_worker(loop, publisher):
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.bind(AI_PROMPT_PULL_URL)
    logger.info(f"ZMQ PULL socket bound to {AI_PROMPT_PULL_URL}")

    while True:
        try:
            data = socket.recv_json()
            prompt = data.get("prompt")
            
            global active_data_channel
            
            if prompt:
                # --- START OF FIX ---
                # Wait for the data channel to be established
                while not active_data_channel:
                    logger.warning("Received prompt via ZMQ, waiting for active data channel...")
                    # This is a worker thread, so time.sleep is safe and correct
                    time.sleep(0.1)
                # --- END OF FIX ---

                logger.info(f"Received prompt via ZMQ: {prompt[:50]}...")
                asyncio.run_coroutine_threadsafe(
                    run_gemini_session(prompt, active_data_channel, publisher), 
                    loop
                )

        except Exception as e:
            logger.error(f"Error in prompt_receiver_worker: {e}")
            
# --- Async Gemini Session Handler (No changes) ---
async def run_gemini_session(prompt, channel, publisher):
    logger.info("Connecting to Gemini for new prompt...") 
    try:
        async with client.aio.live.connect(model=model, config=GENAI_CONFIG) as session: #type: ignore
            logger.info("Gemini connected. Sending prompt.")
            
            await session.send_client_content(
                turns={"role": "user", "parts": [{"text": prompt}]},
                turn_complete=True
            )
            
            await stream_gemini_audio(session, channel, publisher)
    
    except Exception as e:
        logger.error(f"Error in run_gemini_session: {e}")

# --- 1. Gemini Live API Handler (No changes, streaming text chunks) ---
async def stream_gemini_audio(session, data_channel, publisher):
    logger.info("Streaming response to WebRTC Data Channel...")
    
    start_time = time.monotonic()
    first_chunk_time = 0.0
    end_time = 0.0
    chunk_counter = 0
    full_transcription = "" 

    try:
        async for response in session.receive():
            if response.data is not None:
                if chunk_counter == 0:
                    first_chunk_time = time.monotonic()
                    logger.info("First chunk received from Gemini.")
                
                chunk_counter += 1
                data_channel.send(response.data)
                
            if response.server_content.output_transcription:
                chunk_text = response.server_content.output_transcription.text
                full_transcription += chunk_text  
                print(chunk_text, end='', flush=True)
                
                payload = json.dumps({"type": "chunk", "text": chunk_text})
                publisher.send_multipart([b"ai_transcription", payload.encode()])
        
        end_time = time.monotonic()
        logger.info("Gemini audio stream closed successfully.")
        
        if full_transcription:
            payload = json.dumps({"type": "final", "text": full_transcription})
            publisher.send_multipart([b"ai_transcription", payload.encode()])
            logger.info(f"Published final transcription to ZMQ: {full_transcription[:50]}...")
        
    except Exception as e:
        logger.error(f"Error in Gemini streaming: {e}")
    finally:
        logger.info("Gemini audio stream finished.")

        if chunk_counter > 0:
            response_time = first_chunk_time - start_time
            all_chunks_receive_time = end_time - first_chunk_time
            total_time = end_time - start_time
            
            print("\n📊 **PERFORMANCE ANALYSIS**")
            print(f"  - **Time to First Chunk (from prompt):** {response_time:.4f} seconds")
            print(f"  - **All Chunks Receive Time:** {all_chunks_receive_time:.4f} seconds")
            print(f"  - **Total Stream Time:** {total_time:.4f} seconds")
        else:
            print("\n❌ No audio chunks were received for timing analysis.")


# --- 2. WebRTC Signaling Handler (No changes) ---
async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_set.add(pc)

    @pc.on("datachannel")
    async def on_datachannel(channel):
        global active_data_channel
        logger.info(f"Data Channel '{channel.label}' received. Storing as active channel.")
        active_data_channel = channel

        @channel.on("close")
        def on_close():
            global active_data_channel
            logger.info("Active Data Channel closed.")
            active_data_channel = None

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        logger.info("ICE connection state is %s", pc.iceConnectionState)
        if pc.iceConnectionState == "failed" or pc.iceConnectionState == "closed":
            await pc.close()
            pc_set.discard(pc)
            logger.info("PeerConnection closed.")

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


# --- 3. Application Lifecycle (No changes) ---
async def start_prompt_listener(app):
    logger.info("Application started, starting ZMQ prompt listener thread...")
    loop = asyncio.get_running_loop() 
    
    prompt_thread = threading.Thread(
        target=prompt_receiver_worker,
        args=(loop, transcription_publisher),
        daemon=True
    )
    prompt_thread.start()
    app['prompt_thread'] = prompt_thread

async def on_shutdown(app):
    coros = [pc.close() for pc in list(pc_set)]
    await asyncio.gather(*coros)
    pc_set.clear()
    
    logger.profo("Shutting down ZMQ publisher.")
    transcription_publisher.close()
    zmq_context.term()

# --- 4. Main Execution (No changes) ---
if __name__ == "__main__":
    app = web.Application()
    
    app.on_startup.append(start_prompt_listener)
    app.on_shutdown.append(on_shutdown)
    
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods="*",
        )
    })
    
    offer_route = app.router.add_post("/offer", offer)
    cors.add(offer_route)

    print("-" * 50)
    print("✅ Python WebRTC (DataChannel) Backend Ready!")
    print("    Listening for signaling on port 8081")
    print(f"    Publishing AI transcriptions to {AI_TRANSCRIPTION_PUB_URL}")
    print(f"    Listening for AI prompts on {AI_PROMPT_PULL_URL}")
    print("-" * 50)
    
    web.run_app(app, host="0.0.0.0", port=8081)