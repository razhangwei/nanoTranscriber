"""
Main entry point for the AudioTranscriptionApp.
"""

import os
import speech_recognition as sr
import mlx_whisper
import numpy as np
from speech_recognition import AudioData


def transcribe_audio(audio: AudioData, model_name: str = "whisper-base-mlx") -> str:
    """
    Transcribe the given audio data using mlx_whisper.

    Args:
        audio (AudioData): The audio data to transcribe.
        model_name (str): The name of the Whisper model to use.

    Returns:
        str: The transcribed text or an error message.
    """
    try:
        # Convert audio to numpy array
        audio_data = audio.get_raw_data(convert_rate=16000, convert_width=2)
        audio_array = np.frombuffer(audio_data, dtype=np.int16)

        # Normalize the audio to float32 in the range [-1.0, 1.0]
        audio_float32 = audio_array.astype(np.float32) / 32768.0

        result = mlx_whisper.transcribe(
            audio_float32, path_or_hf_repo="mlx-community/" + model_name
        )
        return result["text"]
    except Exception as e:
        return str(e)


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
