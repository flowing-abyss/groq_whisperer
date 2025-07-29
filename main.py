import asyncio
import logging
import signal
import sys
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from os import getenv, path, unlink
from tempfile import NamedTemporaryFile

from dotenv import load_dotenv
from groq import Groq
from pynput import keyboard
from playsound3 import playsound
from pyaudio import PyAudio, paInt16
from pyperclip import copy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('whisperer.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
CHUNK_SIZE = 8192
SAMPLE_RATE = 16000
CHANNELS = 1
MAX_RESTART_ATTEMPTS = 5
RESTART_DELAY = 5

# Global variables and event for shutdown coordination
should_exit = False
shutdown_event = asyncio.Event()
p = None
client = None

def setup_globals():
    """Initialize global variables"""
    global p, client
    try:
        load_dotenv()
        api_key = getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")
        client = Groq(api_key=api_key)
        p = PyAudio()
    except Exception as e:
        logger.error(f"Failed to initialize globals: {e}")
        raise

def cleanup():
    """Cleanup global resources"""
    global p, should_exit, client
    if p:
        try:
            p.terminate()
            logger.info("Audio resources cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    # Reset global objects
    p = None
    client = None
    should_exit = True
    logger.info("Cleanup completed, exiting...")

def signal_handler(signum, frame):
    """Handle system signals for graceful shutdown"""
    global should_exit
    signal_name = signal.Signals(signum).name
    logger.info(f"\nReceived signal {signal_name}")
    logger.info("Initiating graceful shutdown...")
    should_exit = True
    sys.exit(0)  # Немедленное завершение программы
    
    # Force exit after 5 seconds if graceful shutdown fails
    def force_exit():
        logger.warning("Forcing exit due to timeout...")
        cleanup()
        sys.exit(1)
    
    signal.signal(signum, signal.SIG_IGN)  # Ignore subsequent signals
    signal.alarm(5)  # Set timeout for force exit
    
    try:
        cleanup()
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
    finally:
        sys.exit(0)

def setup_signal_handlers():
    """Set up handlers for system signals"""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    if hasattr(signal, 'SIGALRM'):  # Not available on Windows
        signal.signal(signal.SIGALRM, lambda s, f: sys.exit(1))
    logger.info("Signal handlers set up")

# Global PyAudio instance
p = PyAudio()

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
        if not should_exit:  # Only try to close if we're not in exit process
            try:
                stream.stop_stream()
                stream.close()
            except Exception as e:
                logger.error(f"Error closing audio stream: {e}")

async def play_sound_async(sound_file):
    """Play sound asynchronously"""
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        await loop.run_in_executor(executor, playsound, sound_file)

async def record_audio():
    """Record audio from the microphone between two PAUSE button presses."""
    global should_exit
    loop = asyncio.get_running_loop()
    max_retries = 3
    for attempt in range(max_retries):
        try:
            frames = []
            logger.info("Press PAUSE to start recording...")

            # Wait for the first PAUSE press to start
            start_event = asyncio.Event()
            def on_press_start(key):
                if key == keyboard.Key.pause:
                    loop.call_soon_threadsafe(start_event.set)
                    return False  # Stop this listener

            start_listener = keyboard.Listener(on_press=on_press_start)
            start_listener.start()
            await start_event.wait()
            start_listener.join() # Ensure listener is stopped

            logger.info("Recording... (Press PAUSE again to stop)")
            await play_sound_async("start.mp3")

            # Record and wait for the second PAUSE press to stop
            stop_event = asyncio.Event()
            def on_press_stop(key):
                if key == keyboard.Key.pause:
                    loop.call_soon_threadsafe(stop_event.set)
                    return False # Stop this listener

            stop_listener = keyboard.Listener(on_press=on_press_stop)
            stop_listener.start()

            try:
                with audio_stream() as stream:
                    no_data_count = 0
                    while not stop_event.is_set() and not should_exit:
                        try:
                            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                            if any(byte != 0 for byte in data):
                                frames.append(data)
                                no_data_count = 0
                            else:
                                no_data_count += 1
                                if no_data_count > 100:
                                    logger.warning("Prolonged silence detected, stopping.")
                                    break
                        except IOError as e:
                            logger.warning(f"Audio buffer issue: {e}")
                            await asyncio.sleep(0.1)
                        await asyncio.sleep(0.01) # Yield control
                
                stop_listener.join()
                return frames, SAMPLE_RATE
            finally:
                if stop_listener.is_alive():
                    stop_listener.stop()
                await play_sound_async("stop.mp3")
                
        except Exception as e:
            logger.error(f"Recording attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                logger.info("Retrying recording...")
                reset_audio()
                await asyncio.sleep(1)
            else:
                logger.error("All recording attempts failed")
                return None, None

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
        await asyncio.to_thread(_press_and_release_ctrl_v)
    except Exception as e:
        logger.error(f"Error during copy/paste: {str(e)}")

def _press_and_release_ctrl_v():
    """Helper function to press Ctrl+V, to be run in a separate thread."""
    controller = keyboard.Controller()
    with controller.pressed(keyboard.Key.ctrl):
        controller.press('v')
        controller.release('v')

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

def reset_audio():
    global p
    try:
        p.terminate()
    except Exception:
        pass
    p = PyAudio()

async def main():
    """Main program loop"""
    retries = 0
    max_retries = 3
    retry_delay = 5

    try:
        while not should_exit:
            try:
                await process_recording()
                if should_exit:
                    logger.info("Shutdown requested, stopping recording cycle")
                    break
                logger.info("\nReady for next recording. Press PAUSE to start.")
                retries = 0
            except KeyboardInterrupt:
                logger.info("\nKeyboard interrupt received in main loop")
                break
            except Exception as e:
                logger.error(f"Error in recording process: {e}")
                if should_exit:
                    break
                retries += 1
                if retries >= max_retries:
                    logger.error(f"Max retries ({max_retries}) reached. Resetting systems...")
                    reset_audio()
                    retries = 0
                await asyncio.sleep(retry_delay)
    except asyncio.CancelledError:
        logger.info("Main loop cancelled")
    finally:
        cleanup()
        logger.info("Main loop completed")

def run_with_auto_restart():
    """Run the main loop with automatic restart on failure"""
    global should_exit
    restart_count = 0
    while restart_count < MAX_RESTART_ATTEMPTS and not should_exit:
        try:
            logger.info(f"Starting whisperer (attempt {restart_count + 1})")
            setup_globals()
            setup_signal_handlers()
            asyncio.run(main())
            if should_exit:
                logger.info("Clean shutdown requested")
                break
        except KeyboardInterrupt:
            logger.info("\nKeyboard interrupt received in restart loop")
            should_exit = True
            break
        except Exception as e:
            if should_exit:
                logger.info("Shutdown during error recovery")
                break
            restart_count += 1
            logger.error(f"Critical error, restarting: {e}")
            cleanup()
            if restart_count < MAX_RESTART_ATTEMPTS:
                logger.info(f"Restarting in {RESTART_DELAY} seconds...")
                time.sleep(RESTART_DELAY)
            else:
                logger.error("Max restart attempts reached. Exiting.")
                break
    
    logger.info("Program terminated")

if __name__ == "__main__":
    try:
        setup_signal_handlers()
        run_with_auto_restart()
    except KeyboardInterrupt:
        logger.info("\nFinal keyboard interrupt caught")
        should_exit = True
    finally:
        cleanup()
        sys.exit(0)