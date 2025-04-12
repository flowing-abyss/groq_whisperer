import os
import tempfile
import wave
import pyaudio
import keyboard
import pyperclip
import winsound
import time
from groq import Groq

# Set up Groq client
client = Groq(api_key="")

def record_audio(sample_rate=16000, channels=1, chunk=1024):
    """
    Record audio from the microphone between two PAUSE button presses.
    """
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=channels,
        rate=sample_rate,
        input=True,
        frames_per_buffer=chunk,
    )

    print("Press PAUSE to start recording...")
    frames = []

    keyboard.wait("pause")  # Wait for first PAUSE press
    print("Recording... (Press PAUSE again to stop)")
    winsound.Beep(1000, 200)  # Play start sound (1000Hz for 200ms)

    stop_recording = False
    def on_pause_press(e):
        nonlocal stop_recording
        stop_recording = True
    
    keyboard.on_press_key("pause", on_pause_press)

    while not stop_recording:
        data = stream.read(chunk)
        frames.append(data)

    keyboard.unhook_all()  # Remove the event handler
    winsound.Beep(2500, 200)  # Play stop sound (2500Hz for 200ms)

    print("Recording finished.")
    stream.stop_stream()
    stream.close()
    p.terminate()

    return frames, sample_rate


def save_audio(frames, sample_rate):
    """
    Save recorded audio to a temporary WAV file.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        wf = wave.open(temp_audio.name, "wb")
        wf.setnchannels(1)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(frames))
        wf.close()
        return temp_audio.name


def transcribe_audio(audio_file_path):
    """
    Transcribe audio using Groq's Whisper implementation.
    """
    try:
        with open(audio_file_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename(audio_file_path), file.read()),
                model="whisper-large-v3",
                prompt="""The audio is by a programmer discussing programming issues, the programmer mostly uses python and might mention python libraries or reference code in his speech.""",
                response_format="text",
                language="ru",
            )
        return transcription  # This is now directly the transcription text
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None


def copy_transcription_to_clipboard(text):
    """
    Copy the transcribed text to clipboard and paste it using keyboard simulation.
    """
    try:
        pyperclip.copy(text)
        time.sleep(0.1)  # Small delay to ensure text is copied
        keyboard.press_and_release('ctrl+v')
    except Exception as e:
        print(f"Error during copy/paste: {str(e)}")


def main():
    while True:
        # Record audio
        frames, sample_rate = record_audio()

        # Save audio to temporary file
        temp_audio_file = save_audio(frames, sample_rate)

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
        os.unlink(temp_audio_file)

        print("\nReady for next recording. Press PAUSE to start.")


if __name__ == "__main__":
    main()