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
import logging

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

CUSTOM_VOCAB = os.getenv("CUSTOM_VOCAB")

class AudioRecorder:
    def __init__(self):
        self.is_recording = False
        self.recording_queue = queue.Queue()
        self.stop_recording = threading.Event()
        self.recording_thread = None

    def start_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.stop_recording.clear()
            logger.info("Started recording... Press 'esc' to stop recording without transcribing.")
            self.recording_thread = threading.Thread(target=self._record_audio)
            self.recording_thread.start()

    def stop_recording_process(self):
        self.is_recording = False
        self.stop_recording.set()
        if self.recording_thread:
            self.recording_thread.join()

    def _record_audio(self, chunk_duration=0.5):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            logger.info("Recording audio. Please speak now...")
            recognizer.adjust_for_ambient_noise(source)
            audio_chunks = []
            while self.is_recording:
                if self.stop_recording.is_set():
                    logger.info("Recording stopped by 'esc' key.")
                    audio_chunks.clear()
                    break
                audio_chunk = recognizer.record(source, duration=chunk_duration)
                audio_chunks.append(audio_chunk)

        if audio_chunks:
            raw_data = b"".join(chunk.get_raw_data() for chunk in audio_chunks)
            combined_audio = sr.AudioData(
                raw_data, audio_chunks[0].sample_rate, audio_chunks[0].sample_width
            )
            self.recording_queue.put(combined_audio)

    def get_recorded_audio(self):
        try:
            return self.recording_queue.get_nowait()
        except queue.Empty:
            return None

# Global variables
keyboard_controller = keyboard.Controller()
pressed_keys = set()


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
    logger.info(
        f"Transcription time: {transcription_time:.2f} s. "
        f"Speed: {len(result['text'].split()) / transcription_time:.2f} words per second."
    )

    return result["text"].strip()



def print_dots(dot_count, lock, every=2):
    while getattr(threading.current_thread(), "do_run", True):
        keyboard_controller.type(".")
        with lock:
            dot_count[0] += 1
        time.sleep(every)


def provide_feedback(message):
    keyboard_controller.type(message)
    dot_count = [0]
    dot_lock = threading.Lock()

    dot_thread = threading.Thread(target=print_dots, args=(dot_count, dot_lock))
    dot_thread.daemon = True
    dot_thread.start()

    return dot_thread, dot_count, message


def clear_feedback(dot_thread, dot_count, message):
    dot_thread.do_run = False
    dot_thread.join()

    for _ in range(len(message) + dot_count[0]):
        keyboard_controller.tap(keyboard.Key.backspace)
        time.sleep(0.001)


def on_activate(model_name, language, audio_recorder):
    """
    Handle the activation of audio recording and transcription.

    This function is called when the hotkey is pressed or manual activation is triggered.
    It starts or stops the recording process and initiates transcription when recording stops.

    Args:
        model_name (str): The name of the Whisper model to use for transcription.
        language (str): The language to use for transcription.
        audio_recorder (AudioRecorder): The AudioRecorder instance to use for recording.
    """
    global dot_thread, dot_count, message

    if not audio_recorder.is_recording:
        audio_recorder.start_recording()
        dot_thread, dot_count, message = provide_feedback("Recording")
        return

    audio_recorder.stop_recording_process()
    if 'dot_thread' in globals() and dot_thread:
        clear_feedback(dot_thread, dot_count, message)
        dot_thread, dot_count, message = None, None, None

    logger.info("Transcribing audio...")
    audio_data = audio_recorder.get_recorded_audio()
    if audio_data:
        dot_thread, dot_count, message = provide_feedback("Transcribing")

        transcription = transcribe_audio(audio_data, model_name, language)

        if 'dot_thread' in globals() and dot_thread:
            clear_feedback(dot_thread, dot_count, message)
            dot_thread, dot_count, message = None, None, None

        if transcription:
            logger.info(f"Transcription text: {transcription}")
            keyboard_controller.type(transcription)
        else:
            logger.info("No transcription to type.")

        logger.info("Press hotkey to start recording again.")
    else:
        logger.info("No audio transcribed.")


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
    logger.info("Welcome to AudioTranscriptionApp!")
    logger.info(f"Using Whisper model: {get_hf_repo(model_name, language)}")
    logger.info(f"Language set to: {language if language else 'auto-detect'}")
    logger.info(f"Press {os.getenv('HOTKEY')} to start recording and transcribe.")
    # Pre-load model
    logger.info("Pre-loading model...")
    _ = ModelHolder.get_model(get_hf_repo(model_name, language), mx.float16)

    hotkey = parse_hotkey(os.getenv("HOTKEY"))
    audio_recorder = AudioRecorder()

    def on_press(key):
        """Handle key press events."""
        global pressed_keys
        pressed_keys.add(key)
        if all(k in pressed_keys for k in hotkey):
            on_activate(model_name, language, audio_recorder)
            pressed_keys.clear()  # Clear the set after activation

        if key == keyboard.Key.esc:
            audio_recorder.stop_recording_process()

    def on_release(key):
        """Handle key release events."""
        global pressed_keys
        if key in pressed_keys:
            pressed_keys.remove(key)

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        logger.info("Application is running. Press the hotkey to start recording.")
        listener.join()


if __name__ == "__main__":
    main()
