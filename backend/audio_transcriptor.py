import asyncio
import websockets
import threading
import numpy as np
from scipy.signal import resample
import json
import logging
import sys
from RealtimeSTT import AudioToTextRecorder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logging.getLogger('websockets').setLevel(logging.WARNING)
logging.getLogger('faster_whisper').setLevel(logging.WARNING)
logging.getLogger('RealtimeSTT').setLevel(logging.WARNING)

is_running = True
recorder = None
recorder_ready = threading.Event()
main_server_websocket = None 
main_loop = None

async def send_to_main_server(message):
    global main_server_websocket
    if main_server_websocket:
        try:
            await main_server_websocket.send(message)
        except websockets.exceptions.ConnectionClosed:
            main_server_websocket = None
            logging.warning("Main server disconnected")

def text_detected(text):
    global main_loop
    if main_loop:
        asyncio.run_coroutine_threadsafe(
            send_to_main_server(json.dumps({
                'type': 'realtime',
                'text': text
            })), main_loop)

def run_recorder():
    global recorder, main_loop, is_running
    logging.info("Initializing RealtimeSTT...")
    recorder = AudioToTextRecorder(**recorder_config)
    logging.info("RealtimeSTT initialized")
    recorder_ready.set()

    while is_running:
        try:
            full_sentence = recorder.text()
            if full_sentence:
                logging.info(f"Detected sentence: {full_sentence}")
                if main_loop:
                    asyncio.run_coroutine_threadsafe(
                        send_to_main_server(json.dumps({
                            'type': 'fullSentence',
                            'text': full_sentence
                        })), main_loop)
        except Exception as e:
            logging.error(f"Error in recorder thread: {e}")
            continue

recorder_config = {
    'spinner': False,
    'use_microphone': False,
    'model': 'small',
    'language': 'en',
    'silero_sensitivity': 0.4,
    'webrtc_sensitivity': 2,
    'post_speech_silence_duration': 0.7,
    'min_length_of_recording': 0,
    'min_gap_between_recordings': 0,
    'enable_realtime_transcription': True,
    'realtime_processing_pause': 0,
    'realtime_model_type': 'tiny.en',
    'on_realtime_transcription_stabilized': text_detected,
}

def decode_and_resample(audio_data, original_sample_rate, target_sample_rate):
    try:
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        num_original_samples = len(audio_np)
        num_target_samples = int(num_original_samples * target_sample_rate / original_sample_rate)
        resampled_audio = resample(audio_np, num_target_samples)
        return resampled_audio.astype(np.int16).tobytes()  # type: ignore
    except Exception as e:
        logging.error(f"Error in resampling: {e}")
        return audio_data

async def transcriptor_handler(websocket):
    global main_server_websocket
    logging.info("Main server connected")
    main_server_websocket = websocket

    try:
        async for message in websocket:
            if not recorder_ready.is_set():
                logging.warning("Recorder not ready, skipping audio")
                continue

            try:
                metadata_length = int.from_bytes(message[:4], byteorder='little')
                metadata_json = message[4:4+metadata_length].decode('utf-8')
                metadata = json.loads(metadata_json)
                sample_rate = metadata['sampleRate']
                chunk = message[4+metadata_length:]
                
                resampled_chunk = decode_and_resample(chunk, sample_rate, 16000)
                recorder.feed_audio(resampled_chunk) # type: ignore
            except Exception as e:
                logging.error(f"Error processing message: {e}")
                continue
    except websockets.exceptions.ConnectionClosed:
        logging.info("Main server disconnected")
    finally:
        if main_server_websocket == websocket:
            main_server_websocket = None

async def main():
    global main_loop
    main_loop = asyncio.get_running_loop()

    recorder_thread = threading.Thread(target=run_recorder)
    recorder_thread.daemon = True
    recorder_thread.start()
    recorder_ready.wait()

    logging.info("Audio Transcriptor server started on ws://localhost:8002")
    async with websockets.serve(transcriptor_handler, "localhost", 8002):
        await asyncio.Future()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        is_running = False
        if recorder:
            recorder.stop()
            recorder.shutdown()
    finally:
        if recorder:
            del recorder
        logging.info("Transcriptor shutting down.")