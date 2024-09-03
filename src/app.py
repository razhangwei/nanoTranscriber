import logging
import os
import time

import click
import mlx.core as mx
import mlx_whisper
import numpy as np
from dotenv import load_dotenv
from mlx_whisper.transcribe import ModelHolder
from pynput import keyboard
from speech_recognition import AudioData

from utils import AudioRecorder, FeedbackManager

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

CUSTOM_VOCAB = os.getenv("CUSTOM_VOCAB")

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


# TODO: remove and use keyboard.HotKey.parse instead.
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


def on_activate(model_name, language, audio_recorder, feedback_manager):
    """
    Handle the activation of audio recording and transcription.

    This function is called when the hotkey is pressed or manual activation is triggered.
    It starts or stops the recording process and initiates transcription when recording stops.

    Args:
        model_name (str): The name of the Whisper model to use for transcription.
        language (str): The language to use for transcription.
        audio_recorder (AudioRecorder): The AudioRecorder instance to use for recording.
        feedback_manager (FeedbackManager): The FeedbackManager instance to handle feedback.
    """
    if not audio_recorder.is_recording:
        audio_recorder.start_recording()
        feedback_manager.provide_feedback("Recording")
        return

    audio_recorder.stop_recording_process()
    feedback_manager.clear_feedback()

    logger.info("Transcribing audio...")
    audio_data = audio_recorder.get_recorded_audio()
    if audio_data:
        feedback_manager.provide_feedback("Transcribing")

        transcription = transcribe_audio(audio_data, model_name, language)

        feedback_manager.clear_feedback()

        if transcription:
            logger.info(f"Transcription text: {transcription}")
            feedback_manager.keyboard_controller.type(transcription)
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
    feedback_manager = FeedbackManager()

    def on_press(key):
        """Handle key press events."""
        global pressed_keys
        pressed_keys.add(key)
        if all(k in pressed_keys for k in hotkey):
            on_activate(model_name, language, audio_recorder, feedback_manager)
            pressed_keys.clear()  # Clear the set after activation

        if key == keyboard.Key.esc:
            audio_recorder.stop_recording_process()
            feedback_manager.clear_feedback()

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
