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
    """
    Determine the appropriate Hugging Face repository for the Whisper model.

    Args:
        model_name (str): The name of the Whisper model.
        language (str, optional): The target language for transcription.

    Returns:
        str: The Hugging Face repository name for the specified model and language.
    """
    if language == "en" and model_name in ["large-v3"]:
        return "mlx-community/distil-whisper-large-v3"
    elif language == "en" and model_name == "medium":
        return "mlx-community/distil-whisper-medium.en"
    else:
        return f"mlx-community/whisper-{model_name}-mlx"


def parse_hotkey(hotkey_str: str) -> set[keyboard.Key]:
    """
    Parse a hotkey string into a set of Key objects.

    Args:
        hotkey_str (str): A string representation of the hotkey (e.g., "ctrl+shift+a").

    Returns:
        set: A set of pynput.keyboard.Key objects representing the hotkey.
    """
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
    """
    Transcribe the given audio data using the specified Whisper model and language.

    Args:
        audio (AudioData): The audio data to transcribe.
        model_name (str): The name of the Whisper model to use.
        language (str): The language of the audio for transcription.

    Returns:
        str: The transcribed text.
    """
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
        f"Transcription time: {transcription_time:.2f} s. "
        f"Speed: {len(result['text'].split()) / transcription_time:.2f} words per second."
    )

    return result["text"]


def record_audio(chunk_duration=0.5):
    """
    Record audio from the microphone and add it to the recording queue.

    This function runs in a separate thread and continues recording
    as long as the global is_recording flag is True.
    """
    global is_recording
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Recording audio. Please speak now...")
        recognizer.adjust_for_ambient_noise(source)

        audio_chunks: list[AudioData] = []
        while is_recording:
            if stop_recording.is_set():
                print("Recording stopped by 'esc' key.")
                is_recording = False
                audio_chunks.clear()
                break
            audio_chunk = recognizer.record(source, duration=chunk_duration)
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
    """
    Handle the activation of audio recording and transcription.

    This function is called when the hotkey is pressed or manual activation is triggered.
    It starts or stops the recording process and initiates transcription when recording stops.

    Args:
        model_name (str): The name of the Whisper model to use for transcription.
        language (str): The language to use for transcription.
    """
    global is_recording, recording_thread

    if not is_recording:
        is_recording = True
        stop_recording.clear()
        print(
            "Started recording... Press 'esc' to stop recording without transcribing."
        )
        recording_thread = threading.Thread(target=record_audio)
        recording_thread.start()
        return

    stop_recording.set()
    is_recording = False
    print("Stopped recording.")
    if recording_thread:
        recording_thread.join()

    print("Transcribing audio...")
    try:
        audio_data = recording_queue.get_nowait()
        transcription = transcribe_audio(audio_data, model_name, language)
        if transcription:
            print(f"Transcription text: {transcription}")
            keyboard_controller.type(
                transcription
            )  # type the transcription to the current active window
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
    print(f"Using Whisper model: {get_hf_repo(model_name, language)}")
    print(f"Language set to: {language if language else 'auto-detect'}")
    print(f"Press {os.getenv('HOTKEY')} to start recording and transcribe.")
    print("If hotkey doesn't work, press Enter to start recording manually.")

    # Pre-load model
    print("Pre-loading model...")
    _ = ModelHolder.get_model(get_hf_repo(model_name, language), mx.float16)

    hotkey = parse_hotkey(os.getenv("HOTKEY"))

    def on_press(key):
        """Handle key press events."""
        global pressed_keys
        pressed_keys.add(key)
        if all(k in pressed_keys for k in hotkey):
            on_activate(model_name, language)
            pressed_keys.clear()  # Clear the set after activation

        if key == keyboard.Key.esc:
            stop_recording.set()

    def on_release(key):
        """Handle key release events."""
        global pressed_keys
        if key in pressed_keys:
            pressed_keys.remove(key)

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
