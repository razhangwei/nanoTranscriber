import rumps
import threading
from pynput import keyboard
import os
from dotenv import load_dotenv
from utils import AudioRecorder, FeedbackManager
from app import transcribe_audio, get_hf_repo, ModelHolder
import mlx.core as mx

load_dotenv()


class AudioTranscriptionApp(rumps.App):
    def __init__(self):
        super(AudioTranscriptionApp, self).__init__("🎙️")
        self.menu = ["Start Recording", "Stop Recording", "Settings"]
        self.hotkey = keyboard.HotKey(
            keyboard.HotKey.parse("<shift>+<ctrl>+<cmd>+r"), self.toggle_recording
        )
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

        self.audio_recorder = AudioRecorder()
        self.feedback_manager = FeedbackManager()
        self.model_name = os.getenv("MODEL_NAME", "base")
        self.language = os.getenv("LANGUAGE", None)

        # Pre-load model
        ModelHolder.get_model(get_hf_repo(self.model_name, self.language), mx.float16)

    def on_press(self, key):
        if key == keyboard.Key.esc and self.audio_recorder.is_recording:
            self.stop_recording(None)
        else:
            self.hotkey.press(key)

    def toggle_recording(self):
        if not self.audio_recorder.is_recording:
            self.start_recording(None)
        else:
            self.stop_recording(None)

    @rumps.clicked("Start Recording")
    def start_recording(self, _):
        if not self.audio_recorder.is_recording:
            self.audio_recorder.start_recording()
            # self.title = "🔴"  # Change icon to indicate recording

            rumps.notification("Audio Transcription", "Recording Started", "")

    @rumps.clicked("Stop Recording")
    def stop_recording(self, _):
        if self.audio_recorder.is_recording:
            self.audio_recorder.stop_recording_process()
            self.title = "🎙️"
            rumps.notification("Audio Transcription", "Recording Stopped", "")
            self.transcribe_audio()

    def transcribe_audio(self):
        audio_data = self.audio_recorder.get_recorded_audio()
        if audio_data:
            self.feedback_manager.provide_feedback("Transcribing")
            transcription = transcribe_audio(audio_data, self.model_name, self.language)
            self.feedback_manager.clear_feedback()

            if transcription:
                rumps.notification(
                    "Audio Transcription",
                    "Transcription Complete",
                    transcription[:50] + "...",
                )
                # type the transcription via keyboard
                self.feedback_manager.keyboard_controller.type(transcription)
            else:
                rumps.notification(
                    "Audio Transcription", "No transcription available", ""
                )
        else:
            rumps.notification("Audio Transcription", "No audio recorded", "")

    @rumps.clicked("Settings")
    def settings(self, _):
        # Implement settings functionality
        # You could use rumps.Window to create a simple settings window
        window = rumps.Window(
            message="",
            title="Settings",
            default_text="",
            ok="Save",
            dimensions=(320, 160),
        )
        response = window.run()
        # Process settings here


if __name__ == "__main__":
    AudioTranscriptionApp().run()
