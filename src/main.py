"""
Main entry point for the AudioTranscriptionApp.
"""

import os
import speech_recognition as sr

def transcribe_audio(file_path):
    """
    Transcribe the given audio file.
    """
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Speech recognition could not understand the audio"
    except sr.RequestError as e:
        return f"Could not request results from speech recognition service; {e}"

def main():
    """
    Main function to run the AudioTranscriptionApp.
    """
    print("Welcome to AudioTranscriptionApp!")
    
    audio_file = input("Please enter the path to your audio file: ")
    
    if not os.path.exists(audio_file):
        print(f"Error: The file {audio_file} does not exist.")
        return

    print("Transcribing audio...")
    transcription = transcribe_audio(audio_file)
    print("\nTranscription:")
    print(transcription)

if __name__ == "__main__":
    main()
