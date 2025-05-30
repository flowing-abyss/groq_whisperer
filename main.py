from os import path, unlink, getenv
from tempfile import NamedTemporaryFile
import wave
from pyaudio import PyAudio, paInt16
from keyboard import wait, on_press_key, unhook_all, press_and_release
from pyperclip import copy
from playsound3 import playsound
from time import sleep
from groq import Groq
from contextlib import contextmanager
from dotenv import load_dotenv
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Set up Groq client
client = Groq(api_key=getenv("GROQ_API_KEY"))

# Global PyAudio instance
p = PyAudio()

# Constants
CHUNK_SIZE = 8192
SAMPLE_RATE = 16000
CHANNELS = 1

@contextmanager
def audio_stream(sample_rate=SAMPLE_RATE, channels=CHANNELS, chunk=CHUNK_SIZE):
    """Context manager for audio stream handling"""
    try:
        stream = p.open(
            format=paInt16,
            channels=channels,
            rate=sample_rate,
            input=True,
            frames_per_buffer=chunk,
        )
        yield stream
    finally:
        stream.stop_stream()
        stream.close()

async def play_sound_async(sound_file):
    """Play sound asynchronously"""
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        await loop.run_in_executor(executor, playsound, sound_file)

async def record_audio():
    """Record audio from the microphone between two PAUSE button presses."""
    frames = []
    logger.info("Press PAUSE to start recording...")

    wait("pause")
    logger.info("Recording... (Press PAUSE again to stop)")
    await play_sound_async("start.mp3")

    stop_recording = False
    def on_pause_press(e):
        nonlocal stop_recording
        stop_recording = True
    
    on_press_key("pause", on_pause_press)

    try:
        with audio_stream() as stream:
            while not stop_recording:
                try:
                    data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                    frames.append(data)
                except IOError as e:
                    logger.warning(f"Audio buffer overflow: {e}")
                    continue
    except Exception as e:
        logger.error(f"Error during recording: {e}")
        return None, None
    finally:
        unhook_all()
        await play_sound_async("stop.mp3")

    logger.info("Recording finished.")
    return frames, SAMPLE_RATE

async def save_audio(frames, sample_rate):
    """Save recorded audio to a temporary WAV file."""
    if not frames:
        return None
        
    try:
        with NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            with wave.open(temp_audio.name, "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(paInt16))
                wf.setframerate(sample_rate)
                wf.writeframes(b"".join(frames))
            return temp_audio.name
    except Exception as e:
        logger.error(f"Error saving audio: {e}")
        return None

async def transcribe_audio(audio_file_path):
    """Transcribe audio using Groq's Whisper implementation."""
    if not audio_file_path:
        return None
        
    try:
        with open(audio_file_path, "rb") as file:
            transcription = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.audio.transcriptions.create(
                    file=(path.basename(audio_file_path), file.read()),
                    model="whisper-large-v3",
                    prompt="""The audio is by a programmer discussing programming issues, the programmer mostly uses python and might mention python libraries or reference code in his speech.""",
                    response_format="text",
                    language="ru",
                )
            )
        return transcription
    except Exception as e:
        logger.error(f"An error occurred during transcription: {str(e)}")
        return None

async def copy_transcription_to_clipboard(text):
    """Copy the transcribed text to clipboard and paste it using keyboard simulation."""
    if not text:
        return
        
    try:
        copy(text)
        await asyncio.sleep(0.1)  # Small delay to ensure text is copied
        press_and_release('ctrl+v')
    except Exception as e:
        logger.error(f"Error during copy/paste: {str(e)}")

async def process_recording():
    """Process a single recording cycle."""
    frames, sample_rate = await record_audio()
    if not frames:
        logger.warning("Recording failed. Trying again...")
        return

    temp_audio_file = await save_audio(frames, sample_rate)
    if not temp_audio_file:
        logger.warning("Failed to save audio. Trying again...")
        return

    logger.info("Transcribing...")
    transcription = await transcribe_audio(temp_audio_file)

    if transcription:
        logger.info("\nTranscription:")
        logger.info(transcription)
        logger.info("Copying transcription to clipboard...")
        await copy_transcription_to_clipboard(transcription)
        logger.info("Transcription copied and pasted.")
    else:
        logger.warning("Transcription failed.")

    try:
        if temp_audio_file and path.exists(temp_audio_file):
            unlink(temp_audio_file)
    except Exception as e:
        logger.error(f"Error cleaning up temporary file: {e}")

async def main():
    try:
        while True:
            await process_recording()
            logger.info("\nReady for next recording. Press PAUSE to start.")

    except KeyboardInterrupt:
        logger.info("\nProgram terminated by user.")
    finally:
        p.terminate()

if __name__ == "__main__":
    asyncio.run(main())