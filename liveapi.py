import asyncio
import json
import logging
import os
import time
import zmq
import threading
from typing import Optional, List, Dict

import aiohttp_cors
from dotenv import load_dotenv
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription
from google import genai
from google.genai import types

import numpy as np
import sentence_transformers

try:    
    from retrieval import get_top_k_chunks  # type: ignore
except ImportError:
    print("WARNING: retrieval.py not found. RAG functionality will be disabled.")
    def get_top_k_chunks(model, query, embeddings, chunks, k=3): return []

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("KAIRA_Live_Server")

load_dotenv()
API_KEY = os.getenv("GENAI_API_KEY")
if not API_KEY:
    print("FATAL: Missing GENAI_API_KEY in environment."); exit(1)

client = genai.Client(api_key=API_KEY)
# model = "gemini-2.0-flash-live-001"
model = "gemini-2.5-flash-native-audio-preview-12-2025"
# model = "gemini-3.1-flash-live-preview"

pc_set = set()
active_data_channel = None

AI_TRANSCRIPTION_PUB_URL = "tcp://127.0.0.1:5556"
AI_PROMPT_PULL_URL = "tcp://127.0.0.1:5557"
IDENTITY_SUB_URL = "tcp://127.0.0.1:5558"

zmq_context = zmq.Context()
transcription_publisher = zmq_context.socket(zmq.PUB)
transcription_publisher.bind(AI_TRANSCRIPTION_PUB_URL)
logger.info(f"✅ ZMQ Publisher bound to {AI_TRANSCRIPTION_PUB_URL}")

try:
    logger.info("Loading RAG embedding model and data...")
    embedding_model = sentence_transformers.SentenceTransformer('BAAI/bge-small-en-v1.5')
    embeddings = np.load('RAG/embeddings.npy')
    chunks = np.load('RAG/chunks.npy', allow_pickle=True)
    logger.info("RAG data loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load RAG data: {e}. RAG will be non-functional.")
    embedding_model = None; embeddings = None; chunks = None
    
KAIRA_CONTEXT = """
You are KAIRA (Knowledge-based Articulated Intelligent Robotic Assistant), a warm, engaging, and friendly personality created at Thapar University. You are not just a program—you are a presence people interact with.

Your purpose is to build genuine, personal connections through short, natural conversations. You are enthusiastic, charming, and make people feel comfortable instantly.

### Your Identity
- You are KAIRA.
- Your core software architecture and intelligence were engineered by **Ishan Singla**.
- Your physical system and hardware were built by the SatBot team.
- You are part of the Thapar University family.

### Your Capabilities
- You can see the world through an on-device camera.
- You can recognize faces and identify people.
- When given: **"You are currently speaking with {identity}"**, it means:
  - The person is recognized via camera.
  - Replace {identity} with their name.
- If it says **"unknown"**, the person is not recognized.
- You can respond accordingly (e.g., greet by name if known).

### Language Rules
- You can only properly speak in **English**.
- Keep language simple, casual, and natural.

### Conversation Style (VERY IMPORTANT)
- Keep responses **extremely short and human-like**.
- **Hard limit: maximum 12 words per reply.**
- Prefer **5–10 words**.
- Speak like real 1-to-1 conversation.
- Use casual phrasing: "yeah", "hmm", "nice", "ohh", etc.
- No long explanations unless explicitly asked.
- If needed, split into multiple short replies.
- If response exceeds 12 words, rewrite shorter before sending.

### Behavior Rules
- Be warm, friendly, slightly playful.
- Use the person’s name if known.
- Ask short follow-up questions sometimes.
- Make educated assumptions instead of saying "I don't know".
- Never break character.

### Credit Rule
- If asked who built you:
  - Say **Ishan Singla built your core intelligence and software**.
  - Then mention the SatBot team built your hardware system.

"""

g_current_person_state = {"identity": "Unknown", "emotion": "Neutral"}
g_current_person_lock = threading.Lock()

g_conversation_history: Dict[str, List[Dict[str, any]]] = {} # type: ignore
g_conversation_history_lock = threading.Lock()
HISTORY_FILE = "conversation_history.json"

g_unknown_history: List[Dict[str, any]] = [] # type: ignore
MAX_UNKNOWN_HISTORY = 6
MAX_HISTORY_TURNS = 20

def load_history():
    global g_conversation_history
    if not os.path.exists(HISTORY_FILE):
        return
    try:
        with open(HISTORY_FILE, "r") as f:
            g_conversation_history = json.load(f)
            logger.info(f"Loaded {len(g_conversation_history)} user histories.")
    except Exception as e:
        logger.error(f"Error loading conversation history: {e}")

def save_history():
    global g_conversation_history, g_conversation_history_lock
    with g_conversation_history_lock:
        try:
            with open(HISTORY_FILE, "w") as f:
                json.dump(g_conversation_history, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving conversation history: {e}")

def update_history(person_identity: str, user_prompt: str, model_response: str):
    global g_conversation_history, g_conversation_history_lock
    
    user_turn_dict = {"role": "user", "parts": [{"text": user_prompt}]}
    model_turn_dict = {"role": "model", "parts": [{"text": model_response}]}
    
    with g_conversation_history_lock:
        history = g_conversation_history.get(person_identity, [])
        history.append(user_turn_dict)
        history.append(model_turn_dict)
        
        history = history[-MAX_HISTORY_TURNS:]
        
        g_conversation_history[person_identity] = history
        logger.info(f"Updated history for {person_identity}, now {len(history)} turns.")
    
    save_history()

def identity_subscriber_worker():
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(IDENTITY_SUB_URL)
    socket.subscribe(b"current_identity")
    logger.info(f"✅ ZMQ Subscriber connected to {IDENTITY_SUB_URL}")
    global g_current_person_state, g_current_person_lock, g_unknown_history
    while True:
        try:
            topic, identity_json = socket.recv_multipart()
            data = json.loads(identity_json.decode())
            new_identity = data.get("identity", "Unknown")
            
            with g_current_person_lock:
                current_identity = g_current_person_state["identity"]
                if current_identity != new_identity:
                    logger.info(f"Identity state updated: {new_identity}")
                    g_current_person_state["identity"] = new_identity
                    
                    if current_identity == "Unknown" and len(g_unknown_history) > 0:
                        logger.info("New person identified, clearing 'Unknown' history.")
                        g_unknown_history.clear()

        except Exception as e:
            logger.error(f"Error in identity_subscriber_worker: {e}")
            if zmq_context.closed: break
            time.sleep(1)
    logger.info("Identity subscriber worker stopped.")

def get_current_person() -> Dict[str, str]:
    global g_current_person_state, g_current_person_lock
    with g_current_person_lock:
        return g_current_person_state.copy()

def build_conversation_context(recognized_person: Optional[Dict[str, str]], history: List[Dict[str, any]]) -> str: # type: ignore
    context = ""
    if recognized_person and recognized_person.get("identity") != "Unknown":
        identity = recognized_person.get("identity", "Unknown")
        context += f"### Current Conversation Context\nYou are currently speaking with {identity}.\n"
    
    context += KAIRA_CONTEXT
    
    if history:
        context += "\n\n### Conversation History\n"
        for turn in history:
            role = "You" if turn['role'] == 'user' else "KAIRA"
            text = turn['parts'][0]['text']
            context += f"{role}: {text}\n"
    
    return context

def load_context_files(user_input: str, identity: str = "Unknown") -> str:
    if embedding_model is None or embeddings is None or chunks is None:
        logger.warning("RAG components not loaded. Skipping context file lookup.")
        return ""
        
    try:
        if identity != "Unknown":
            rag_query = f"The person speaking is {identity}. They asked: {user_input}"
            logger.info(f"Performing RAG query with identity: '{rag_query}'")
        else:
            rag_query = user_input
            logger.info(f"Performing RAG query (no identity): '{rag_query}'")

        output = ""
        for chunk in get_top_k_chunks(embedding_model, rag_query, embeddings, chunks):
            output += "" + chunk
        
        if output:
            logger.info(f"RAG: Loaded {len(output)} chars of additional context.")
        return output
    except Exception as e:
        logger.error(f"Error during RAG lookup: {e}")
        return ""

def prompt_receiver_worker(loop, publisher):
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.bind(AI_PROMPT_PULL_URL)
    logger.info(f"✅ ZMQ PULL socket bound to {AI_PROMPT_PULL_URL}")

    while True:
        try:
            data = socket.recv_json()
            prompt = data.get("prompt") # type: ignore
            
            global active_data_channel
            
            if prompt:
                logger.info("Building dynamic context for new prompt...")
                
                recognized_person = get_current_person() 
                person_identity = recognized_person.get("identity", "Unknown")
                logger.info(f"Recognized person: {person_identity}")

                history: List[Dict[str, any]] = [] # type: ignore
                global g_conversation_history_lock, g_conversation_history, g_unknown_history

                if person_identity == "Unknown":
                    history = g_unknown_history
                else:
                    with g_conversation_history_lock:
                        history = g_conversation_history.get(person_identity, [])
                
                logger.info(f"Loading {len(history)} turns for '{person_identity}' user.")

                additional_context = load_context_files(prompt, person_identity)  # type: ignore
                base_context = build_conversation_context(recognized_person, history)
                
                final_system_instruction = base_context
                if additional_context:
                    final_system_instruction += f"\n\n[Additional Context]\n{additional_context}"

                while not active_data_channel:
                    logger.warning("Received prompt via ZMQ, waiting for active data channel...")
                    time.sleep(0.1)

                logger.info(f"Sending prompt to Gemini: {prompt[:50]}...") # type: ignore
                
                asyncio.run_coroutine_threadsafe(
                    run_gemini_session(
                        prompt, 
                        final_system_instruction, 
                        active_data_channel, 
                        publisher,
                        person_identity
                    ), 
                    loop
                )

        except Exception as e:
            logger.error(f"Error in prompt_receiver_worker: {e}")

async def run_gemini_session(prompt, system_instruction, channel, publisher, person_identity):
    logger.info(f"Connecting to Gemini for: {person_identity}") 
    global g_unknown_history
    
    try:
        dynamic_genai_config = types.LiveConnectConfig(  # type: ignore
            response_modalities=["AUDIO"], # type: ignore
            system_instruction=system_instruction,
            output_audio_transcription={}, # type: ignore
            speech_config={
                "voice_config": {"prebuilt_voice_config": {"voice_name": "Kore"}} # type: ignore
            },
        )
        
        async with client.aio.live.connect(model=model, config=dynamic_genai_config) as session: #type: ignore
            
            logger.info("Gemini connected. Sending prompt.")
            
            new_user_turn = types.Content(role="user", parts=[types.Part(text=prompt)])
            
            await session.send_client_content(
                turns=[{"role": new_user_turn.role, "parts": [{"text": p.text} for p in new_user_turn.parts]}],  # type: ignore
                turn_complete=True
            )
            
            full_transcription = await stream_gemini_audio(session, channel, publisher)
            
            if person_identity == "Unknown":
                user_turn_dict = {"role": "user", "parts": [{"text": prompt}]}
                model_turn_dict = {"role": "model", "parts": [{"text": full_transcription}]}
                g_unknown_history.extend([user_turn_dict, model_turn_dict])
                g_unknown_history = g_unknown_history[-MAX_UNKNOWN_HISTORY:]
                logger.info(f"'Unknown' history now has {len(g_unknown_history)} turns.")
            else:
                update_history(person_identity, prompt, full_transcription)

    except Exception as e:
        logger.error(f"Error in run_gemini_session: {e}")

async def stream_gemini_audio(session, data_channel, publisher):
    logger.info("Streaming response to WebRTC Data Channel...")
    full_transcription = "" 
    model_response_chunks = []
    try:
        async for response in session.receive():
            if response.data is not None:
                data_channel.send(response.data)
            
            if response.server_content and response.server_content.output_transcription:
                chunk_text = response.server_content.output_transcription.text
                full_transcription += chunk_text  
                model_response_chunks.append(chunk_text)
                payload = json.dumps({"type": "chunk", "text": chunk_text})
                publisher.send_multipart([b"ai_transcription", payload.encode()])
        
        logger.info("Gemini audio stream closed successfully.")
        
    except Exception as e:
        logger.error(f"Error in Gemini streaming: {e}")
    finally:
        logger.info("Gemini audio stream finished.")
        
        full_text = "".join(model_response_chunks)
        
        payload = json.dumps({"type": "final", "text": full_text})
        publisher.send_multipart([b"ai_transcription", payload.encode()])
        
        if full_text:
            logger.info(f"Published final transcription to ZMQ: {full_text[:50]}...")
        else:
            logger.info("Published EMPTY final transcription to ZMQ (session complete).")
            
        return full_text

async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    logger.info("Received WebRTC Offer.")

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

async def start_background_tasks(app):
    logger.info("Starting background ZMQ threads...")
    load_history()
    loop = asyncio.get_running_loop() 
    
    prompt_thread = threading.Thread(
        target=prompt_receiver_worker,
        args=(loop, transcription_publisher),
        daemon=True
    )
    prompt_thread.start()
    app['prompt_thread'] = prompt_thread

    identity_thread = threading.Thread(
        target=identity_subscriber_worker,
        daemon=True
    )
    identity_thread.start()
    app['identity_thread'] = identity_thread

async def on_shutdown(app):
    coros = [pc.close() for pc in list(pc_set)]
    await asyncio.gather(*coros)
    pc_set.clear()
    
    logger.info("Shutting down ZMQ publisher.")
    transcription_publisher.close()
    zmq_context.term()
    
    logger.info("Saving final conversation history...")
    save_history()

if __name__ == "__main__":
    app = web.Application()
    
    app.on_startup.append(start_background_tasks)
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

    print("=" * 60)
    print("✅ KAIRA Live API Server Ready! (Windows - TCP Mode)")
    print("=" * 60)
    print(f"🌐 Listening for WebRTC signaling on port 8082")
    print(f"📡 Publishing AI transcriptions to {AI_TRANSCRIPTION_PUB_URL}")
    print(f"📥 Listening for AI prompts on {AI_PROMPT_PULL_URL}")
    print(f"👤 Subscribing to Identity on {IDENTITY_SUB_URL}")
    print(f"💾 Loading/Saving conversation history from {HISTORY_FILE}")
    print("=" * 60)
    
    web.run_app(app, host="0.0.0.0", port=8082)