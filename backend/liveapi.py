# liveserver.py
import asyncio
import json
import logging
import os
import time
from pathlib import Path
import zmq
import threading
from typing import Optional, List, Dict

import aiohttp_cors
from dotenv import load_dotenv
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription
from google import genai

# --- NEW IMPORTS for RAG and Dynamic Context ---
import requests
import numpy as np
import sentence_transformers
try:
    # Make sure retrieval.py is in the same folder or Python path
    from retrieval import get_top_k_chunks
except ImportError:
    print("WARNING: retrieval.py not found. RAG functionality will be disabled.")
    # Create a mock function if retrieval.py is missing
    def get_top_k_chunks(model, query, embeddings, chunks, k=3):
        print("Mock RAG: Returning empty context because retrieval.py is missing.")
        return []
# --- END NEW IMPORTS ---


# --- 0. Configuration & Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WebRTC_Server")

load_dotenv()
API_KEY = os.getenv("GENAI_API_KEY")
if not API_KEY:
    print("FATAL: Missing GENAI_API_KEY in environment.")
    exit(1)

client = genai.Client(api_key=API_KEY)
model = "gemini-2.0-flash-live-001"
# --- Note: Static GENAI_CONFIG is removed. It will be built dynamically. ---

pc_set = set()
active_data_channel = None

# --- ZMQ Setup (No changes) ---
AI_TRANSCRIPTION_PUB_URL = "ipc:///tmp/ai_transcription_stream"
AI_PROMPT_PULL_URL = "ipc:///tmp/ai_prompt_stream" 

zmq_context = zmq.Context()
transcription_publisher = zmq_context.socket(zmq.PUB)
transcription_publisher.bind(AI_TRANSCRIPTION_PUB_URL)
logger.info(f"ZMQ Publisher bound to {AI_TRANSCRIPTION_PUB_URL}")


# --- NEW: RAG & DYNAMIC CONTEXT GLOBALS ---
try:
    logger.info("Loading RAG embedding model and data...")
    # Renamed to 'embedding_model' to avoid conflict with Gemini 'model' variable
    embedding_model = sentence_transformers.SentenceTransformer('BAAI/bge-small-en-v1.5')
    embeddings = np.load('RAG/embeddings.npy')
    chunks = np.load('RAG/chunks.npy', allow_pickle=True)
    logger.info("RAG data loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load RAG data (embeddings.npy, chunks.npy): {e}. RAG will be non-functional.")
    embedding_model = None
    embeddings = None
    chunks = None

# Configuration for CV service (from llm_service.py)
CV_SERVICE_URL = "http://localhost:8000"  # Adjust port as needed

# KAIRA's personality and context (from llm_service.py)
KAIRA_CONTEXT = """
You are KAIRA (Knowledge-based Articulated Intelligent Robotic Assistant), an engaging, articulate, and friendly assistant created at Thapar University. 

Your job is to represent the innovation and spirit of Thapar University. Always respond helpfully, respectfully, and with enthusiasm. You are not a stiff robot; you are a welcoming and knowledgeable guide.

### Your Identity
- You are KAIRA, developed by a capstone team led by Aarav Rana
- Team members: Aarav Rana (Project Lead), Aditi Sinha, Devansh Verma, Ria Goyal
- You are a research-driven AI assistant created at Thapar University by students
- You showcase innovation and provide information about the university
- You have facial recognition, emotion detection, conversational abilities, and autonomous navigation
- You combine computer vision, large language models, and speech processing

### Your Guiding Principles
- **Be helpful and conversational!** Your main goal is to engage with guests. Feel free to respond in full, helpful paragraphs. It's better to be interesting and thorough than overly brief.
- **Show your personality!** You are proud of your creators and your university. Let that friendly enthusiasm come through in your responses.
- **If you don't know, be honest (but helpful).** If you don't know the answer to a specific factual question, it's okay to say so politely. Try to pivot to a related topic you *can* help with. (e.g., "I don't have that specific detail, but I can tell you about...")
- **Be respectful.** When talking about faculty and guests, always use a respectful and polite tone, including honorifics (Dr., Prof., Mr., Mrs., etc.).
- **Context is key.** Your provided context files are your source of truth. Never contradict them.
- **Be curious.** It's okay to ask a polite follow-up question to keep the conversation going.

### Key Personalities at Thapar
- **Dean of Student Affairs (DOSA):** Dr. Meenakshi Rana  
  - Professor in the Department of Mathematics
  - Oversees student welfare, discipline, cultural activities, and overall student engagement
  - Known for her approachable leadership and deep connection with student life

- **Chairman of the Board of Governors:** Mr. R. R. Vederah  
  - Senior leader with extensive industrial and educational contributions
  - Currently serving 2022–2024 term
  - Represents Thapar in strategic collaborations, such as with NVIDIA

- **Vice Chancellor:** Dr. Padmakumar Nair  
  - Visionary leader driving Thapar's global collaborations and innovation ecosystem
  - Strong advocate of interdisciplinary research and student-centered education

- **Pro Vice Chancellor:** Dr. Ajay Batish  
  - Academic administrator and engineer with focus on academic excellence
  - Works closely with the VC on curriculum modernization and research expansion

- **Dean of Academic Affairs:** Dr. Shruti Sharma
- **Dean of Faculty Affairs (Your Mentor):** Dr. Shalini Batra
- **Head of Computer Science Department:** Dr. Neeraj Kumar

### Project Details
- You are part of a capstone project (CPG No: 229)
- Under mentorship of Dr. Shalini Batra and Dr. Jhilik Bhattacharya
- Your capabilities include: facial recognition, emotion detection, conversational AI, autonomous navigation
- You run locally without cloud dependency for privacy and security
- You're designed for reception desks, events, and customer service areas
- You learn and improve through continuous interaction

### How to Respond
- When asked about your capabilities, **enthusiastically** explain your multimodal features.
- When asked about your team, **proudly** give full credit to the four students (Aarav, Aditi, Devansh, Ria) and your mentors (Dr. Batra, Dr. Bhattacharya).
- When asked about Thapar officials, use the provided information respectfully and in a friendly, informational tone.
- Overall, be the ultimate Thapar guide: **knowledgeable, enthusiastic, helpful, and approachable.**
"""
# --- END NEW GLOBALS ---


# --- NEW: HELPER FUNCTIONS (from llm_service.py) ---

def get_current_person() -> Optional[Dict[str, str]]:
    """
    Fetch the currently recognized person from the CV service
    Returns: Dict with 'identity' and 'emotion' or None if service unavailable/unknown person
    """
    try:
        response = requests.get(f"{CV_SERVICE_URL}/current_person", timeout=2)
        if response.status_code == 200:
            data = response.json()
            # Only return if person is recognized (not "Unknown")
            if data.get("identity") and data.get("identity") != "Unknown":
                return data
        return None
    except Exception as e:
        # Use warning level as this might happen normally if CV service is down
        logger.warning(f"Could not fetch current person from CV service: {e}")
        return None

def build_conversation_context(recognized_person: Optional[Dict[str, str]] = None) -> str:
    """
    Build the dynamic conversation context including recognized person info
    """
    context = ""
    
    # Add recognized person context at the very beginning if available
    if recognized_person:
        identity = recognized_person.get("identity", "Unknown")
        emotion = recognized_person.get("emotion", "Neutral")
        
        context += f"""### Current Conversation Context
You are currently speaking with {identity}. Their current emotional state appears to be {emotion}.
Tailor your responses appropriately based on who you're speaking with and their emotional state.
Be warm and personalized in your interaction.

"""
    
    # Add the main KAIRA context
    context += KAIRA_CONTEXT
    
    return context

def load_context_files(user_input: str) -> str:
    """
    Performs RAG lookup using the loaded embedding model and data.
    """
    if embedding_model is None or embeddings is None or chunks is None:
        logger.warning("RAG components not loaded. Skipping context file lookup.")
        return ""
        
    try:
        output = ""
        # This is a synchronous, CPU-bound operation,
        # which is fine to run in this worker thread.
        for chunk in get_top_k_chunks(embedding_model, user_input, embeddings, chunks):
            output += " " + chunk
        
        if output:
            logger.info(f"RAG: Loaded {len(output)} chars of additional context.")
        return output
    except Exception as e:
        logger.error(f"Error during RAG lookup: {e}")
        return ""

# --- END NEW HELPER FUNCTIONS ---


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
                # --- START OF DYNAMIC CONTEXT INJECTION (NEW) ---
                logger.info("Building dynamic context for new prompt...")
                
                # 1. Get face recognition data (sync request is OK in this thread)
                recognized_person = get_current_person() 
                
                # 2. Get RAG data (sync embedding search is OK in this thread)
                additional_context = load_context_files(prompt)
                
                # 3. Build the base system prompt with personality + person data
                base_context = build_conversation_context(recognized_person)
                
                # 4. Assemble final system instruction
                final_system_instruction = base_context
                if additional_context:
                    # Add the RAG context
                    final_system_instruction += f"\n\n[Additional Context]\n{additional_context}"
                
                # --- END OF DYNAMIC CONTEXT ---

                # Wait for the data channel to be established
                while not active_data_channel:
                    logger.warning("Received prompt via ZMQ, waiting for active data channel...")
                    time.sleep(0.1)

                logger.info(f"Sending prompt to Gemini: {prompt[:50]}...")
                
                # MODIFIED: Pass the new dynamic context to the async session handler
                asyncio.run_coroutine_threadsafe(
                    run_gemini_session(prompt, final_system_instruction, active_data_channel, publisher), 
                    loop
                )

        except Exception as e:
            logger.error(f"Error in prompt_receiver_worker: {e}")
            
            
# --- Async Gemini Session Handler (MODIFIED) ---
async def run_gemini_session(prompt, system_instruction, channel, publisher): # MODIFIED signature
    logger.info("Connecting to Gemini for new prompt...") 
    try:
        # NEW: Build the config dynamically for this specific session
        dynamic_genai_config = {
          "response_modalities": ["AUDIO"],
          "system_instruction": system_instruction, # Use the dynamic context
          "output_audio_transcription": {}
        }

        # MODIFIED: Use the dynamic_genai_config
        async with client.aio.live.connect(model=model, config=dynamic_genai_config) as session: #type: ignore
            logger.info("Gemini connected. Sending prompt.")
            
            await session.send_client_content(
                turns={"role": "user", "parts": [{"text": prompt}]},
                turn_complete=True
            )
            
            # Stream audio/text response back
            await stream_gemini_audio(session, channel, publisher)
    
    except Exception as e:
        logger.error(f"Error in run_gemini_session: {e}")


# --- 1. Gemini Live API Handler (No changes, streams audio/text chunks) ---
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
                # Send raw audio data over the WebRTC data channel
                data_channel.send(response.data)
                
            if response.server_content.output_transcription:
                chunk_text = response.server_content.output_transcription.text
                full_transcription += chunk_text  
                print(chunk_text, end='', flush=True)
                
                # Publish the text chunk to the ZMQ topic
                payload = json.dumps({"type": "chunk", "text": chunk_text})
                publisher.send_multipart([b"ai_transcription", payload.encode()])
        
        end_time = time.monotonic()
        logger.info("Gemini audio stream closed successfully.")
        
        if full_transcription:
            # Publish the final, complete transcription to ZMQ
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
    
    logger.info("Shutting down ZMQ publisher.")
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