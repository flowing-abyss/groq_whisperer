# Groq Whisperer: Voice-to-Text Transcription Tool

**Groq Whisperer** is a Python-based application that allows users to record audio and transcribe it to text using Groq's Whisper implementation. The transcribed text is automatically copied to the clipboard for easy pasting into other applications.

## Features

- Record audio by pressing the **PAUSE** key.
- Transcribe recorded audio to text using Groq's API.
- Automatically copy transcription to clipboard and paste it.
- Continuous operation for multiple recordings.
- Optimized for faster transcription.

## Prerequisites

- Python 3.7 or higher
- A Groq API key

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/flowing-abyss/groq_whisperer
    cd groq_whisperer
    ```

2.  Create a virtual environment:
    ```bash
    python -m venv .venv
    ```

3.  Activate the virtual environment:
    - On Windows:
      ```
      .venv\Scripts\activate
      ```
    - On macOS and Linux:
      ```
      source .venv/bin/activate
      ```

4.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

5.  Create a `.env` file and set the Groq API key in it:
    ```
    GROQ_API_KEY=your_api_key_here
    ```

## Usage

1.  Run the script using the background launchers:
    - On Windows:
      ```
      run_background.bat
      ```
    - On macOS and Linux:
      ```
      ./run_background.sh
      ```
    *Note: You may need to make the script executable on macOS/Linux: `chmod +x run_background.sh`*

2.  Press the **PAUSE** key to start recording.
3.  Press the **PAUSE** key again to stop recording.
4.  The transcript will automatically be copied to the clipboard and pasted into the active window.
5.  Repeat steps 2-4 for additional recordings.

## Dependencies

For a complete list of dependencies, see the `requirements.txt` file.

## Notes

- Make sure your microphone is properly configured and working before running the script.
- The transcription quality may vary depending on the audio quality and background noise.
- Ensure you have a stable internet connection for the transcription process.

## License

[MIT License](LICENSE)
