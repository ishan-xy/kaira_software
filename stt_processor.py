import threading
import logging
import sys
from RealtimeSTT import AudioToTextRecorder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logging.getLogger('faster_whisper').setLevel(logging.WARNING)
logging.getLogger('RealtimeSTT').setLevel(logging.WARNING)

class STTProcessor:
    
    recorder_config = {
        'spinner': False,
        'use_microphone': False,
        'model': 'small.en',
        'language': 'en',
        'silero_sensitivity': 0.4,
        'webrtc_sensitivity': 3,
        'post_speech_silence_duration': 0.7,
        'min_length_of_recording': 0.3,
        'min_gap_between_recordings': 0,
        'enable_realtime_transcription': False,
    }

    def __init__(self, on_full_sentence_text):
        self.on_full_sentence_text = on_full_sentence_text
        
        logging.info("Initializing RealtimeSTT (manual audio feed)...")
        self.recorder = AudioToTextRecorder(**self.recorder_config)
        logging.info("RealtimeSTT initialized.")
        
        self.is_running = False
        self.processing_thread = None

    def _processing_loop(self):
        logging.info("STT processing thread started.")
        while self.is_running:
            try:
                full_sentence = self.recorder.text()

                if full_sentence and self.is_running:
                    logging.info(f"Detected full sentence: {full_sentence}")
                    if self.on_full_sentence_text:
                        try:
                            self.on_full_sentence_text(full_sentence)
                        except Exception as e:
                            logging.error(f"Callback error: {e}")
                
            except Exception as e:
                if self.is_running:
                    logging.error(f"Error in STT processing loop: {e}")
                else:
                    logging.info("STT loop interrupted by stop().")
                
        logging.info("STT processing thread stopped.")

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        logging.info("RealtimeSTT recorder is waiting for audio...")

    def stop(self):
        if not self.is_running:
            return
        logging.info("Stopping STT processor...")
        self.is_running = False
        if self.recorder:
            self.recorder.stop() 
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        logging.info("STT processor stopped.")

    def feed_audio(self, audio_chunk):
        if self.is_running and audio_chunk:
            try:
                self.recorder.feed_audio(audio_chunk)
            except Exception as e:
                logging.error(f"Error feeding audio to STT: {e}")