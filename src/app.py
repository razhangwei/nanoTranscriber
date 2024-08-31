import os
import threading
import time
import speech_recognition as sr
import mlx_whisper
import numpy as np
import click
from speech_recognition import AudioData
from pynput import keyboard
from pynput.keyboard import Key, Controller
import sys

# Global variables
is_recording = False
keyboard_controller = Controller()


def transcribe_audio(audio: AudioData, model_name: str) -> str:
    try:
        # Convert audio to numpy array
        audio_data = audio.get_raw_data(convert_rate=16000, convert_width=2)
        audio_array = np.frombuffer(audio_data, dtype=np.int16)

        # Normalize the audio to float32 in the range [-1.0, 1.0]
        audio_float32 = audio_array.astype(np.float32) / 32768.0

        result = mlx_whisper.transcribe(
            audio_float32, path_or_hf_repo=f"mlx-community/whisper-{model_name}-mlx"
        )
        return result["text"]
    except Exception as e:
        return str(e)


def record_and_transcribe(model_name: str) -> str:
    global is_recording
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Recording audio. Please speak now...")
        try:
            audio = recognizer.listen(source, timeout=3, phrase_time_limit=5)
            print("Finished recording.")
        except sr.WaitTimeoutError:
            print("No speech detected. Timeout reached.")
            return ""

    print("Transcribing audio...")
    try:
        transcription = transcribe_audio(audio, model_name)
        print("Transcription completed.")
        return transcription
    except Exception as e:
        print(f"Error during transcription: {e}")
        return ""


def on_activate(model_name: str):
    global is_recording
    if not is_recording:
        is_recording = True
        print("Started recording...")
        transcription = record_and_transcribe(model_name)
        is_recording = False

        if transcription:
            print("Typing transcription...")
            keyboard_controller.type(transcription)
            print("Transcription typed:", transcription)
        else:
            print("No transcription to type.")
    else:
        print("Already recording...")


@click.command()
@click.option(
    "--model-name",
    default="base",
    type=click.Choice(["base", "small", "medium", "large"]),
    help="The name of the Whisper model to use.",
)
def main(model_name):
    print("Welcome to AudioTranscriptionApp!")
    print(f"Using Whisper model: {model_name}")
    print("Press Ctrl + v to start recording and transcribe.")
    print("If hotkey doesn't work, press Enter to start recording manually.")

    def on_press(key):
        if key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
            return True
        if key == keyboard.KeyCode.from_char("v"):
            on_activate(model_name)

    def on_release(key):
        if key == keyboard.Key.esc:
            return False

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    while True:
        user_input = input("Press Enter to start recording manually (or 'q' to quit): ")
        if user_input.lower() == "q":
            print("Exiting the application...")
            listener.stop()
            sys.exit(0)
        print("Manual activation triggered.")
        on_activate(model_name)


if __name__ == "__main__":
    main()
