"""
Main entry point for the AudioTranscriptionApp.
"""

import os
import speech_recognition as sr

def transcribe_audio(audio):
    """
    Transcribe the given audio data.
    """
    recognizer = sr.Recognizer()
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
    
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Adjusting for ambient noise. Please wait...")
        recognizer.adjust_for_ambient_noise(source)
        print("Recording audio. Please speak now...")
        audio = recognizer.listen(source)

    print("Transcribing audio...")
    transcription = transcribe_audio(audio)
    print("\nTranscription:")
    print(transcription)

if __name__ == "__main__":
    main()
