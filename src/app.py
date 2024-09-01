import os
import threading
import time
import speech_recognition as sr
import mlx_whisper
import numpy as np
import click
from speech_recognition import AudioData
from pynput import keyboard
import sys
from mlx_whisper.transcribe import ModelHolder
import mlx.core as mx
import queue
from dotenv import load_dotenv

load_dotenv()

# Global variables
is_recording = False
keyboard_controller = keyboard.Controller()
pressed_keys = set()
recording_queue = queue.Queue()
stop_recording = threading.Event()
CUSTOM_VOCAB = os.getenv("CUSTOM_VOCAB")


def get_hf_repo(model_name: str, language: str = None) -> str:
    if language == "en" and model_name in ["large-v3"]:
        return "mlx-community/distil-whisper-large-v3"
    elif language == "en" and model_name == "medium":
        return "mlx-community/distil-whisper-medium.en"
    else:
        return f"mlx-community/whisper-{model_name}-mlx"


def parse_hotkey(hotkey_str):
    """Parse hotkey string into a set of Key objects"""
    keys = set()
    for key in hotkey_str.split("+"):
        key = key.strip().lower()
        try:
            # Try to get the key from the Key enum
            keys.add(keyboard.Key[key])
        except KeyError:
            # If it's not in the Key enum, treat it as a character
            keys.add(keyboard.KeyCode.from_char(key))
    return keys


def transcribe_audio(audio: AudioData, model_name: str, language: str) -> str:
    # Convert audio to numpy array
    audio_data = audio.get_raw_data(convert_rate=16000, convert_width=2)
    audio_array = np.frombuffer(audio_data, dtype=np.int16)

    # Normalize the audio to float32 in the range [-1.0, 1.0]
    audio_float32 = audio_array.astype(np.float32) / 32768.0

    start_time = time.time()
    result = mlx_whisper.transcribe(
        audio_float32,
        path_or_hf_repo=get_hf_repo(model_name, language),
        language=language,
        initial_prompt=", ".join(CUSTOM_VOCAB.split(",")) if CUSTOM_VOCAB else None,
    )
    end_time = time.time()
    transcription_time = end_time - start_time
    print(
        f"Transcription time: {transcription_time:.2f} s. Speed: {len(result['text'].split()) / transcription_time:.2f} words per second."
    )

    return result["text"]


def record_audio():
    global is_recording
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Recording audio. Please speak now...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)

        audio_chunks: list[AudioData] = []
        while is_recording:
            audio_chunk: AudioData = recognizer.listen(source)
            audio_chunks.append(audio_chunk)

    # Combine all audio chunks
    if audio_chunks:
        # Concatenate raw audio data
        raw_data = b"".join(chunk.get_raw_data() for chunk in audio_chunks)

        # Create a new AudioData object with the combined raw data
        combined_audio = sr.AudioData(
            raw_data, audio_chunks[0].sample_rate, audio_chunks[0].sample_width
        )

        recording_queue.put(combined_audio)
    else:
        print("No audio recorded.")

    is_recording = False


def on_activate(model_name, language):
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
            transcription = transcribe_audio(audio_data, model_name, language)
            if transcription:
                print(f"Transcription text: {transcription}")
                keyboard_controller.type(transcription)
            else:
                print("No transcription to type.")
        except queue.Empty:
            print("No audio data to transcribe.")


@click.command()
@click.option(
    "-m",
    "--model-name",
    default="base",
    type=click.Choice(["base", "small", "medium", "large-v3"]),
    help="The name of the Whisper model to use.",
)
@click.option(
    "-l",
    "--language",
    default=None,
    help="Language for transcription. Use ISO 639-1 codes (e.g., 'en' for English). "
    "If not specified, it defaults to multilingual model and language will be auto detected.",
)
def main(model_name, language):
    print("Welcome to AudioTranscriptionApp!")
    print(f"Using Whisper model: {model_name}")
    print(f"Language set to: {language if language else 'auto-detect'}")
    print(f"Press {os.getenv('HOTKEY')} to start recording and transcribe.")
    print("If hotkey doesn't work, press Enter to start recording manually.")

    # pre-load model
    print("Pre-loading model...")
    _ = ModelHolder.get_model(get_hf_repo(model_name, language), mx.float16)

    hotkey = parse_hotkey(os.getenv("HOTKEY"))

    def on_press(key):
        global pressed_keys
        pressed_keys.add(key)
        if all(k in pressed_keys for k in hotkey):
            on_activate(model_name, language)
            pressed_keys.clear()  # Clear the set after activation

    def on_release(key):
        global pressed_keys
        if key in pressed_keys:
            pressed_keys.remove(key)
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
        on_activate(model_name, language)


if __name__ == "__main__":
    main()
