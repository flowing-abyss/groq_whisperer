# Groq Whisperer: Voice-to-Text Transcription Tool

**Groq Whisperer** is a Python-based application that allows users to record audio and transcribe it to text using Groq's Whisper implementation. The transcribed text is automatically copied to the clipboard for easy pasting into other applications.

## Features

- Record audio by holding down the **PAUSE** key
- Transcribe recorded audio to text using Groq's API
- Automatically copy transcription to clipboard
- Continuous operation for multiple recordings

## Prerequisites

- Python 3.7 or higher
- A Groq API key (set as an environment variable)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/flowing-abyss/groq_whisperer
   cd groq_whisperer
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS and Linux:
     ```
     source venv/bin/activate
     ```

4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

5. Set up your Groq API key (main.py):
     ```
     client = Groq(api_key="API key")
     ```

## Usage

1. Run the script:
   ```
   python main.py
   ```
   or
   ```
   run_background_admin.bat
   ```

2. Press the "Pause" button to start recording.
3. Press the "Pause" button to stop recording.
4. The transcript will automatically be copied to the clipboard and can be pasted using Ctrl+V.
5. Repeat steps 2-4 for additional recordings.

## Dependencies

For a complete list of dependencies, see the `requirements.txt` file.

## Notes

- Make sure your microphone is properly configured and working before running the script.
- The transcription quality may vary depending on the audio quality and background noise.
- Ensure you have a stable internet connection for the transcription process.

## License

[MIT License](LICENSE)
