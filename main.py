import asyncio
import logging
import signal
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from os import getenv

import numpy as np
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
MAX_RECORDING_DURATION = 300  # 5 minutes maximum recording
MAX_FRAMES = int(SAMPLE_RATE * MAX_RECORDING_DURATION / CHUNK_SIZE)
SILENCE_THRESHOLD = 500  # Amplitude threshold for silence detection
MAX_SILENCE_CHUNKS = 100  # Maximum consecutive silent chunks before stopping

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

def detect_silence(data, threshold=SILENCE_THRESHOLD):
    """Detect silence in audio data using amplitude analysis"""
    try:
        audio_data = np.frombuffer(data, dtype=np.int16)
        max_amplitude = np.max(np.abs(audio_data))
        return max_amplitude < threshold
    except Exception:
        # Fallback to simple check if numpy fails
        return all(byte == 0 for byte in data[:100])  # Check first 100 bytes only

def create_wav_data(frames, sample_rate):
    """Create WAV file data in memory without using temporary files"""
    if not frames:
        return None
    
    # Combine all frames
    audio_data = b"".join(frames)
    
    # WAV file header
    sample_width = p.get_sample_size(paInt16)
    num_frames = len(audio_data) // sample_width
    
    # Create WAV header
    wav_header = struct.pack('<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        36 + len(audio_data),  # File size
        b'WAVE',
        b'fmt ',
        16,  # PCM header size
        1,   # PCM format
        CHANNELS,
        sample_rate,
        sample_rate * CHANNELS * sample_width,  # Byte rate
        CHANNELS * sample_width,  # Block align
        sample_width * 8,  # Bits per sample
        b'data',
        len(audio_data)
    )
    
    return wav_header + audio_data

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
    
    # Ignore subsequent signals to prevent recursive calls
    signal.signal(signum, signal.SIG_IGN)
    
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

# Global PyAudio instance (initialized in setup_globals)
p = None

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
                    # Play stop sound immediately when PAUSE is pressed (synchronously for instant response)
                    try:
                        playsound("stop.mp3", block=False)
                    except Exception as e:
                        logger.warning(f"Could not play stop sound: {e}")
                    loop.call_soon_threadsafe(stop_event.set)
                    return False # Stop this listener

            stop_listener = keyboard.Listener(on_press=on_press_stop)
            stop_listener.start()

            try:
                with audio_stream() as stream:
                    silence_count = 0
                    frame_count = 0
                    while not stop_event.is_set() and not should_exit and frame_count < MAX_FRAMES:
                        try:
                            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                            frame_count += 1
                            
                            if detect_silence(data):
                                silence_count += 1
                                if silence_count > MAX_SILENCE_CHUNKS:
                                    logger.info("Prolonged silence detected, stopping recording.")
                                    break
                            else:
                                frames.append(data)
                                silence_count = 0
                                
                        except IOError as e:
                            logger.warning(f"Audio buffer issue: {e}")
                            await asyncio.sleep(0.1)
                        
                        # Reduced sleep for better responsiveness
                        if frame_count % 10 == 0:  # Only yield every 10 frames
                            await asyncio.sleep(0.001)
                
                stop_listener.join()
                
                # Log recording statistics
                if frame_count >= MAX_FRAMES:
                    logger.info(f"Recording stopped: maximum duration reached ({MAX_RECORDING_DURATION}s)")
                elif silence_count > MAX_SILENCE_CHUNKS:
                    logger.info("Recording stopped: prolonged silence detected")
                else:
                    logger.info(f"Recording completed: {len(frames)} frames captured")
                
                return frames, SAMPLE_RATE
            finally:
                if stop_listener.is_alive():
                    stop_listener.stop()
                # Stop sound is now played immediately when PAUSE is pressed
                
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

async def transcribe_audio_from_memory(wav_data):
    """Transcribe audio using Groq's Whisper implementation from memory data."""
    if not wav_data:
        return None
        
    try:
        transcription = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.audio.transcriptions.create(
                file=("audio.wav", wav_data),
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
    """Process a single recording cycle using in-memory operations."""
    frames, sample_rate = await record_audio()
    if not frames:
        logger.warning("Recording failed. Trying again...")
        return

    logger.info("Creating WAV data in memory...")
    wav_data = create_wav_data(frames, sample_rate)
    if not wav_data:
        logger.warning("Failed to create WAV data. Trying again...")
        return

    logger.info(f"Audio processed: {len(frames)} frames, {len(frames) * CHUNK_SIZE / SAMPLE_RATE:.1f}s duration")
    logger.info("Transcribing...")
    transcription = await transcribe_audio_from_memory(wav_data)

    if transcription:
        logger.info("\nTranscription:")
        logger.info(transcription)
        logger.info("Copying transcription to clipboard...")
        await copy_transcription_to_clipboard(transcription)
        logger.info("Transcription copied and pasted.")
    else:
        logger.warning("Transcription failed.")
    
    # No temporary file cleanup needed - everything was in memory!

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