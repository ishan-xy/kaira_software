import asyncio
import logging
import os
import wave
from dotenv import load_dotenv
from google import genai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Gemini_Echo_Client")

load_dotenv()
API_KEY = os.getenv("GENAI_API_KEY")
if not API_KEY:
    print("FATAL: Missing GENAI_API_KEY in environment.")
    exit(1)

client = genai.Client(api_key=API_KEY)
model = "gemini-2.0-flash-live-001"
OUTPUT_FILENAME = "gemini_echo_output.wav"

PROMPT_TO_ECHO = '''Good evening, Ajay Baatish,Rajeev Ranjan Vederah, Shaalini Batra
'''

# --- Audio parameters based on Gemini Live API standard output ---
SAMPLE_RATE = 24000  # 24kHz
SAMPLE_WIDTH = 2     # 16-bit (2 bytes)
CHANNELS = 1         # Mono 

async def main():
    logger.info("Starting Gemini Echo Client...")

    if not PROMPT_TO_ECHO:
        logger.error("The hard-coded PROMPT_TO_ECHO is empty. Exiting.")
        return

    logger.info(f"Configuring Gemini to echo: '{PROMPT_TO_ECHO[:50]}...'")

    echo_genai_config = {
      "response_modalities": ["AUDIO"],
      "system_instruction": "You are a simple echo. Repeat the user's message back to them verbatim. Add no other words or conversation.",
      "speech_config": {
        "voice_config": {"prebuilt_voice_config": {"voice_name": "Kore"}}
      },
    }

    chunk_counter = 0
    wf = None  # Initialize wave file object

    try:
        logger.info(f"Connecting to Gemini... Output will be saved to '{OUTPUT_FILENAME}'")

        # Open the wave file before connecting
        wf = wave.open(OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)

        async with client.aio.live.connect(model=model, config=echo_genai_config) as session: #type: ignore
            logger.info("Gemini connected. Sending prompt.")

            await session.send_client_content(
                turns={"role": "user", "parts": [{"text": PROMPT_TO_ECHO}]},
                turn_complete=True
            )

            logger.info("Streaming audio response from Gemini...")

            async for response in session.receive():
                if response.data is not None:
                    # Write audio data directly to the file
                    wf.writeframes(response.data)
                    chunk_counter += 1
                    if chunk_counter % 20 == 0:
                        print(f"Received {chunk_counter} audio chunks...", end='\r')

        print() 
        logger.info("Audio stream finished.")
        
        if chunk_counter == 0:
            logger.warning("No audio data was received.")
        else:
            logger.info(f"Successfully saved {chunk_counter} audio chunks to '{OUTPUT_FILENAME}'.")

    except Exception as e:
        logger.error(f"An error occurred during the Gemini session: {e}")
    finally:
        # Ensure the file is closed no matter what
        if wf:
            wf.close()
            logger.info(f"File '{OUTPUT_FILENAME}' closed.")
        logger.info("Exiting.")


if __name__ == "__main__":
    asyncio.run(main())