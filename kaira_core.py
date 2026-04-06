import pyaudio
import numpy as np
import threading
import time
import logging
import zmq
import json
import queue
from stt_processor import STTProcessor
from webrtc_client import WebRTCClient 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

AI_PROMPT_PUSH_URL = "tcp://127.0.0.1:5557"
AI_TRANSCRIPTION_SUB_URL = "tcp://127.0.0.1:5556"

class KAIRACore:
    def __init__(self):
        self.state = {
            'display_text': "",
            'kaira_response_text': "",
            'normalized_amplitude': 0.0,
            'is_final_sentence': False,
            'is_kaira_speaking': False,
            'last_sentence_time': 0,
            'listening_state': 'WAITING',
            'in_conversation_mode': False,
        }
        self.state_lock = threading.RLock()
        self.cached_state = self.state.copy()
        self.last_cache_time = 0
        self.ai_response_timeout = 3.0
        self.ai_response_start_time = 0

        self.conversation_mode_timeout = 10.0
        self.last_activity_time = 0
        self.conversation_mode_timer = None

        self.chunk = 2048
        self.sample_rate = 16000
        self.channels = 1
        self.is_listening = False
        self.is_recording = False
        self.p_audio = pyaudio.PyAudio()
        self.input_stream = None

        self.recording_timeout = 12.0
        self.recording_timer = None

        self.stt_processor = STTProcessor(
            on_full_sentence_text=self._on_stt_full_sentence
        )
        
        self.zmq_context = zmq.Context()
        self.prompt_pusher = self.zmq_context.socket(zmq.PUSH)
        self.prompt_pusher.connect(AI_PROMPT_PUSH_URL)
        self.transcription_sub = self.zmq_context.socket(zmq.SUB)
        self.transcription_sub.connect(AI_TRANSCRIPTION_SUB_URL)
        self.transcription_sub.subscribe(b"ai_transcription")
        
        self.transcription_thread = threading.Thread(
            target=self._transcription_subscriber_worker, 
            daemon=True
        )
        
        self.audio_playback_queue = queue.Queue()
        self.playback_stream = None
        self.audio_playback_thread = threading.Thread(
            target=self._audio_playback_worker,
            daemon=True
        )
        self.webrtc_client = WebRTCClient(self.audio_playback_queue)

        self.stt_results_queue = queue.Queue()
        self.stt_processing_thread = threading.Thread(
            target=self._process_stt_results,
            daemon=True
        )
        
        print("KAIRA Core initialized.")

    def _audio_playback_worker(self):
        logger.info("Audio playback worker started.")
        self.playback_stream = self.p_audio.open(
            format=pyaudio.paInt16, 
            channels=1, 
            rate=24000, 
            output=True
        )
        while self.is_listening:
            try:
                chunk = self.audio_playback_queue.get(timeout=1.0)
                if chunk is None:
                    break
                if self.playback_stream:
                    self.playback_stream.write(chunk)
            except queue.Empty:
                continue
            except Exception as e:
                if self.is_listening:
                    logger.error(f"Audio playback error: {e}")
        
        logger.info("Audio playback worker stopping...")
        if self.playback_stream:
            self.playback_stream.stop_stream()
            self.playback_stream.close()
        logger.info("Audio playback worker stopped.")

    def _transcription_subscriber_worker(self):
        logger.info(f"Listening for AI transcriptions on {AI_TRANSCRIPTION_SUB_URL}")
        while self.is_listening:
            try:
                topic, payload = self.transcription_sub.recv_multipart()
                data = json.loads(payload.decode())

                with self.state_lock:
                    if data['type'] == 'chunk':
                        text_chunk = data['text']
                        if not self.state['is_kaira_speaking']:
                            self.state['is_kaira_speaking'] = True
                            self.state['display_text'] = ""
                            self.state['is_final_sentence'] = False
                            self.state['listening_state'] = 'SPEAKING'
                        self.state['kaira_response_text'] += text_chunk
                    elif data['type'] == 'final':
                        self.state['is_kaira_speaking'] = False
                        self.state['listening_state'] = 'WAITING'
                        self.state['last_sentence_time'] = time.time()

                        logger.info("AI response finished")

                        if self.state['in_conversation_mode']:
                            logger.info("Conversation mode active: Auto-start recording in 2 seconds...")
                            threading.Timer(2.0, self._auto_start_recording).start()
                        else:
                            logger.info("Not in conversation mode")

            except zmq.ZMQError as e:
                if not self.is_listening:
                    break
                logger.error(f"ZMQ Error in transcription worker: {e}")
            except Exception as e:
                logger.error(f"Error in transcription worker: {e}")

        logger.info("Transcription subscriber worker stopped.")

    def start_audio_input_stream(self):
        if not self.p_audio:
            logger.error("PyAudio not initialized.")
            return
        
        try:
            self.input_stream = self.p_audio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk,
                stream_callback=self.audio_input_callback
            )
            self.input_stream.start_stream()
            logger.info(f"Audio input stream started (Rate: {self.sample_rate}Hz)")
        except Exception as e:
            logger.error(f"Error starting audio input stream: {e}")

    def audio_input_callback(self, in_data, frame_count, time_info, status):
        try:
            if self.is_recording and self.stt_processor:
                audio_array = np.frombuffer(in_data, dtype=np.int16)
                self.stt_processor.feed_audio(audio_array.tobytes())

        except Exception as e:
            logger.error(f"Audio input callback error: {e}")

        return (in_data, pyaudio.paContinue)

    def _enter_conversation_mode(self):
        with self.state_lock:
            self.state['in_conversation_mode'] = True
            self.last_activity_time = time.time()
        logger.info("Entered conversation mode - auto-listen enabled")
        self._start_inactivity_timer() 

    def _exit_conversation_mode(self):
        with self.state_lock:
            self.state['in_conversation_mode'] = False
            self.state['listening_state'] = 'WAITING'
            self.state['display_text'] = ""
        logger.info("Exited conversation mode")
        self._cancel_inactivity_timer()

    def _start_inactivity_timer(self):
        self._cancel_inactivity_timer()
        self.conversation_mode_timer = threading.Timer(
            self.conversation_mode_timeout,
            self._on_conversation_timeout
        )
        self.conversation_mode_timer.daemon = True
        self.conversation_mode_timer.start()
        logger.debug(f"Inactivity timer started ({self.conversation_mode_timeout}s)")

    def _cancel_inactivity_timer(self):
        if self.conversation_mode_timer:
            self.conversation_mode_timer.cancel()
            self.conversation_mode_timer = None
            logger.info("Inactivity timer cancelled.")

    def _reset_inactivity_timer(self):
        with self.state_lock:
            in_conversation_mode = self.state['in_conversation_mode']
            if in_conversation_mode:
                self.last_activity_time = time.time()

        if in_conversation_mode:
            self._start_inactivity_timer()
            logger.info("Activity detected - resetting inactivity timer")

    def _on_conversation_timeout(self):
        logger.info(f"Conversation timeout - {self.conversation_mode_timeout} seconds of inactivity")
        
        try:
            if self.stt_processor and self.stt_processor.recorder and self.is_recording:
                logger.info("Feeding silence to STT to finalize any buffered audio...")
                silence = b'\x00\x00' * 16000
                chunk_size = 2048 * 2
                for i in range(0, len(silence), chunk_size):
                    chunk = silence[i:i+chunk_size]
                    self.stt_processor.feed_audio(chunk)
        except Exception as e:
            logger.error(f"Error feeding silence to STT on conversation timeout: {e}")
        
        self._exit_conversation_mode()
        self.stop_recording()

    def _on_recording_timeout(self):
        logger.info(f"Recording timeout - {self.recording_timeout} seconds reached, forcing finalization...")

        try:
            if self.stt_processor and self.stt_processor.recorder:
                silence = b'\x00\x00' * 16000

                chunk_size = 2048 * 2
                for i in range(0, len(silence), chunk_size):
                    chunk = silence[i:i+chunk_size]
                    self.stt_processor.feed_audio(chunk)

                logger.info("Fed silence to STT to trigger finalization")
        except Exception as e:
            logger.error(f"Error feeding silence to STT: {e}")

        self.stop_recording()

    def _auto_start_recording(self):
        with self.state_lock:
            if not self.state['in_conversation_mode']:
                logger.info("Auto-start cancelled: Not in conversation mode")
                return
            if self.is_recording:
                logger.info("Auto-start cancelled: Already recording")
                return

        logger.info("Auto-starting recording in conversation mode...")
        self.start_recording()
        self._reset_inactivity_timer()

    def _process_stt_results(self):
        logger.info("STT results processing thread started.")
        while self.is_listening:
            try:
                text = self.stt_results_queue.get(timeout=1.0)

                logger.info(f"Processing STT result: '{text}'")

                normalized = text.lower().strip()
                if "stop kaira" in normalized or "stop kira" in normalized:
                    logger.info("'Stop Kaira' detected - halting and exiting conversation mode")
                    self.stop_recording()
                    self._exit_conversation_mode()
                    with self.state_lock:
                        self.state['display_text'] = ""
                        self.state['is_final_sentence'] = True
                        self.state['listening_state'] = "WAITING"
                    continue

                lock_acquired = self.state_lock.acquire(timeout=5.0)

                if not lock_acquired:
                    logger.error("CRITICAL: Could not acquire state_lock after 5 seconds")
                    continue

                try:
                    if not self.state['in_conversation_mode']:
                        logger.info("STT result arrived after timeout, rescuing conversation mode.")
                        self.state['in_conversation_mode'] = True
                    
                    self.state['display_text'] = text
                    self.state['last_sentence_time'] = time.time()
                    self.state['is_final_sentence'] = True
                    self.state['is_kaira_speaking'] = False
                    self.state['kaira_response_text'] = ""
                    self.state['listening_state'] = 'THINKING'
                    self.ai_response_start_time = time.time()
                    logger.info(f"State updated successfully")
                finally:
                    self.state_lock.release()

                logger.info("Full sentence detected - cancelling inactivity timer...")
                self._cancel_inactivity_timer()

                try:
                    logger.info(f"Sending prompt to AI: '{text}'")
                    payload = {"prompt": text, "timestamp": time.time()}
                    self.prompt_pusher.send_json(payload)
                except Exception as e:
                    logger.error(f"Failed to send prompt via ZMQ: {e}")
                    if self.state_lock.acquire(timeout=1.0):
                        try:
                            self.state['is_kaira_speaking'] = False
                            self.state['kaira_response_text'] = "Error: Could not connect to AI."
                        finally:
                            self.state_lock.release()

                self.stop_recording(new_state='THINKING')

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing STT result: {e}")
                import traceback
                logger.error(traceback.format_exc())

        logger.info("STT results processing thread stopped.")

    def _on_stt_full_sentence(self, text):
        logger.info(f"Full sentence received: '{text}'")
        self.stt_results_queue.put(text)

    def start_recording(self):
        with self.state_lock:
            if self.state['is_kaira_speaking']:
                logger.warning("Recording blocked: KAIRA is still speaking")
                return
            if self.is_recording:
                logger.warning("Recording blocked: Already recording")
                return

            logger.info("Recording START")
            self.is_recording = True
            self.state['listening_state'] = 'LISTENING'
            self.state['display_text'] = "Listening..."
            self.state['is_final_sentence'] = False
            self.state['kaira_response_text'] = ""

        if self.recording_timer:
            self.recording_timer.cancel()
        self.recording_timer = threading.Timer(self.recording_timeout, self._on_recording_timeout)
        self.recording_timer.daemon = True
        self.recording_timer.start()
        logger.info(f"Recording timeout set for {self.recording_timeout} seconds")

    def stop_recording(self, new_state='WAITING'):
        if not self.is_recording:
            return

        logger.info("Recording STOP")
        self.is_recording = False
        with self.state_lock:
            self.state['listening_state'] = new_state

        if self.recording_timer:
            self.recording_timer.cancel()
            self.recording_timer = None

    def enable_conversation_mode(self):
        with self.state_lock:
            already_in_convo = self.state['in_conversation_mode']

        if not already_in_convo:
            self._enter_conversation_mode()
            self.start_recording() 
            self._reset_inactivity_timer()
            logger.info("Conversation mode manually enabled, starting recording.")
        else:
            logger.info("Already in conversation mode")

    def get_state(self):
        current_time = time.time()

        if current_time - self.last_cache_time > 0.016:
            if self.state_lock.acquire(blocking=False):
                try:
                    self.cached_state = self.state.copy()
                    self.last_cache_time = current_time
                finally:
                    self.state_lock.release()

        return self.cached_state.copy()

    def start(self):
        print("Starting KAIRA Core services...")
        self.is_listening = True
        self.stt_processor.start()
        self.transcription_thread.start()
        self.audio_playback_thread.start()
        self.stt_processing_thread.start()
        self.webrtc_client.start()
        self.start_audio_input_stream()
        print("KAIRA Core is running. Press SPACE to start recording.")

    def stop(self):
        print("\nStopping KAIRA Core services...")
        self.is_listening = False

        self._cancel_inactivity_timer()

        if self.recording_timer:
            self.recording_timer.cancel()
            self.recording_timer = None

        if self.stt_processor:
            self.stt_processor.stop()

        self.webrtc_client.stop()
        self.audio_playback_queue.put(None)
        self.audio_playback_thread.join(timeout=1.0)

        if self.stt_processing_thread:
            self.stt_processing_thread.join(timeout=1.0)

        if self.transcription_thread:
            self.transcription_thread.join(timeout=1.0)

        self.prompt_pusher.close()
        self.transcription_sub.close()
        self.zmq_context.term()

        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()

        if self.p_audio:
            self.p_audio.terminate()

        print("KAIRA Core stopped.")