# AudioTranscriptionApp

AudioTranscriptionApp is a powerful tool for transcribing audio files into text. This project provides an easy-to-use interface for converting spoken words in audio recordings into written text.

## Features

- **Audio file transcription**: Convert spoken words in audio recordings into written text.
- **Support for various audio formats**: Handles different audio formats for transcription.
- **User-friendly command-line interface**: Easy-to-use CLI for interacting with the application.
- **High accuracy transcription using Google Speech Recognition**: Utilizes Google Speech Recognition for accurate transcription.

## How It Works

The `app.py` file contains the core functionalities of the AudioTranscriptionApp. Here's a brief overview of how it works:

1. **Audio Recording**: The application records audio using the microphone.
2. **Audio Processing**: The recorded audio is processed and converted into a format suitable for transcription.
3. **Transcription**: The processed audio is transcribed using the Whisper model.
4. **Output**: The transcription result is displayed and can be typed directly using the keyboard controller.

### Key Components

- **transcribe_audio**: This function transcribes the audio data using the specified Whisper model.
- **record_audio**: This function records audio from the microphone and stores it in a queue for processing.
- **on_activate**: This function handles the activation of audio recording and transcription.
- **main**: The main function initializes the application, sets up the Whisper model, and handles user interactions.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/AudioTranscriptionApp.git
   ```
2. Navigate to the project directory:
   ```
   cd AudioTranscriptionApp
   ```
3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```
   sudo python src/app.py
   ```
2. When pressing the hotkey, it will start recording and when pressing again, it still stop recording and the transcribed tax will automatically be typed to the text box of the current window.

## Roadmap

- [x] Support custom vocabulary through initial prompt.
- [x] Customizable hotkey.
- [ ] Implement robust error handling and exception management.
- [ ] Create a separate configuration file for better settings management.
- [ ] Modularize the codebase for improved organization and maintainability.
- [ ] Add progress indicators for better user feedback during transcription.
- [ ] Expand customization options through additional command-line arguments.
- [ ] Optimize performance, especially for audio processing and transcription.
- [ ] Refactor code to reduce global variables and implement proper state management.
- [ ] Add comprehensive unit tests for core functionalities.
- [ ] Implement a proper logging system for better debugging and monitoring.
- [ ] Enhance input validation for user inputs and configuration settings.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [SpeechRecognition](https://pypi.org/project/SpeechRecognition/) library for providing speech recognition capabilities.
