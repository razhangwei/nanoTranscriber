import logging
import queue
import threading
import time

import speech_recognition as sr
from pynput import keyboard

logger = logging.getLogger(__name__)


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
            logger.info(
                "Started recording... Press 'esc' to stop recording without transcribing."
            )
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


class FeedbackManager:
    """
    Manage user feedback by printing a message and printing dots after it.

    This class takes care of printing a message and then appending dots to it
    until told to clear. It also handles clearing the feedback by deleting the
    message and dots.

    The printing of dots is done in a separate thread to avoid blocking.
    """

    def __init__(self):
        self.dot_thread = None
        self.message = None
        self.keyboard_controller = keyboard.Controller()

    def provide_feedback(self, message):
        """
        Print a message and then print dots after it until clear_feedback is called.

        This function is non-blocking and runs in a separate thread.

        Args:
            message (str): The message to print before the dots.
        """
        self.message = message
        self.keyboard_controller = keyboard.Controller()
        self.keyboard_controller.type(self.message)
        dot_lock = threading.Lock()

        self.dot_thread = threading.Thread(target=self._print_dots, args=(dot_lock,))
        self.dot_thread.daemon = True
        self.dot_thread.start()

    def _print_dots(self, lock, every=2):
        while getattr(self.dot_thread, "do_run", True):
            self.keyboard_controller.type(".")
            with lock:
                self.message += "."
            time.sleep(every)

    def clear_feedback(self):
        """
        Clear the feedback by deleting the message and dots.

        This function stops the thread printing dots and deletes the message
        from the console.
        """
        if self.dot_thread:
            self.dot_thread.do_run = False
            self.dot_thread.join()

            for _ in range(len(self.message)):
                self.keyboard_controller.tap(keyboard.Key.backspace)
                time.sleep(0.001)

        self.dot_thread = None
        self.message = None
