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
from mlx_whisper.transcribe import ModelHolder
import mlx.core as mx
import queue

# Global variables
is_recording = False
keyboard_controller = Controller()
recording_queue = queue.Queue()
stop_recording = threading.Event()


# TODO: to improve the performance
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


def record_audio():
    global is_recording, audio_data
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Recording audio. Please speak now...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio_data = recognizer.listen(source, timeout=1000, phrase_time_limit=1000)
            recording_queue.put(audio_data)
        except sr.WaitTimeoutError:
            print("No speech detected. Timeout reached.")
        finally:
            is_recording = False


def on_activate(model_name):
    global is_recording, recording_thread
    if not is_recording:
        is_recording = True
        stop_recording.clear()
        print("Started recording...")
        recording_thread = threading.Thread(target=record_audio)
        recording_thread.start()
    else:
        stop_recording.set()
        is_recording = False
        print("Stopped recording.")
        if recording_thread:
            recording_thread.join()
        try:
            audio_data = recording_queue.get_nowait()
            print("Transcribing audio...")
            transcription = transcribe_audio(audio_data, model_name)
            if transcription:
                keyboard_controller.type(transcription)
            else:
                print("No transcription to type.")
        except queue.Empty:
            print("No audio data to transcribe.")


@click.command()
@click.option(
    "--model-name",
    default="base",
    type=click.Choice(["base", "small", "medium", "large-v3"]),
    help="The name of the Whisper model to use.",
)
@click.option(
    "--timeout",
    default=3,
    type=int,
    help="Timeout for speech recognition in seconds.",
)
def main(model_name, timeout):
    print("Welcome to AudioTranscriptionApp!")
    print(f"Using Whisper model: {model_name}")
    print(f"Timeout set to: {timeout} seconds")
    print("Press Ctrl + v to start recording and transcribe.")
    print("If hotkey doesn't work, press Enter to start recording manually.")

    # pre-load model

    print("Pre-loading model...")
    _ = ModelHolder.get_model(f"mlx-community/whisper-{model_name}-mlx", mx.float16)

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
