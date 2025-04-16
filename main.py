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

# Load environment variables from .env file
load_dotenv()

# Set up Groq client
client = Groq(api_key=getenv("GROQ_API_KEY"))

# Global PyAudio instance
p = PyAudio()

@contextmanager
def audio_stream(sample_rate=16000, channels=1, chunk=2048):
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
    await loop.run_in_executor(None, playsound, sound_file)

def record_audio(sample_rate=16000, channels=1, chunk=2048):
    """
    Record audio from the microphone between two PAUSE button presses.
    """
    frames = []
    print("Press PAUSE to start recording...")

    wait("pause")  # Wait for first PAUSE press
    print("Recording... (Press PAUSE again to stop)")
    asyncio.run(play_sound_async("start.mp3"))  # Play start sound asynchronously

    stop_recording = False
    def on_pause_press(e):
        nonlocal stop_recording
        stop_recording = True
    
    on_press_key("pause", on_pause_press)

    try:
        with audio_stream(sample_rate, channels, chunk) as stream:
            while not stop_recording:
                try:
                    data = stream.read(chunk, exception_on_overflow=False)
                    frames.append(data)
                except IOError as e:
                    print(f"Audio buffer overflow: {e}")
                    continue
    except Exception as e:
        print(f"Error during recording: {e}")
        return None, None
    finally:
        unhook_all()  # Remove the event handler
        asyncio.run(play_sound_async("stop.mp3"))  # Play stop sound asynchronously

    print("Recording finished.")
    return frames, sample_rate

def save_audio(frames, sample_rate):
    """
    Save recorded audio to a temporary WAV file.
    """
    if not frames:
        return None
        
    try:
        with NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            wf = wave.open(temp_audio.name, "wb")
            wf.setnchannels(1)
            wf.setsampwidth(p.get_sample_size(paInt16))
            wf.setframerate(sample_rate)
            wf.writeframes(b"".join(frames))
            wf.close()
            return temp_audio.name
    except Exception as e:
        print(f"Error saving audio: {e}")
        return None

def transcribe_audio(audio_file_path):
    """
    Transcribe audio using Groq's Whisper implementation.
    """
    if not audio_file_path:
        return None
        
    try:
        with open(audio_file_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(path.basename(audio_file_path), file.read()),
                model="whisper-large-v3-turbo",
                prompt="""The audio is by a programmer discussing programming issues, the programmer mostly uses python and might mention python libraries or reference code in his speech.""",
                response_format="text",
                language="ru",
            )
        return transcription
    except Exception as e:
        print(f"An error occurred during transcription: {str(e)}")
        return None

def copy_transcription_to_clipboard(text):
    """
    Copy the transcribed text to clipboard and paste it using keyboard simulation.
    """
    if not text:
        return
        
    try:
        copy(text)
        sleep(0.1)  # Small delay to ensure text is copied
        press_and_release('ctrl+v')
    except Exception as e:
        print(f"Error during copy/paste: {str(e)}")

def main():
    try:
        while True:
            # Record audio
            frames, sample_rate = record_audio()
            if not frames:
                print("Recording failed. Trying again...")
                continue

            # Save audio to temporary file
            temp_audio_file = save_audio(frames, sample_rate)
            if not temp_audio_file:
                print("Failed to save audio. Trying again...")
                continue

            # Transcribe audio
            print("Transcribing...")
            transcription = transcribe_audio(temp_audio_file)

            # Copy transcription to clipboard
            if transcription:
                print("\nTranscription:")
                print(transcription)
                print("Copying transcription to clipboard...")
                copy_transcription_to_clipboard(transcription)
                print("Transcription copied and pasted.")
            else:
                print("Transcription failed.")

            # Clean up temporary file
            try:
                if temp_audio_file and path.exists(temp_audio_file):
                    unlink(temp_audio_file)
            except Exception as e:
                print(f"Error cleaning up temporary file: {e}")

            print("\nReady for next recording. Press PAUSE to start.")

    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
    finally:
        p.terminate()

if __name__ == "__main__":
    main()