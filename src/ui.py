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
        self.is_recording = False
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
        self.hotkey.press(key)

    def toggle_recording(self):
        if not self.audio_recorder.is_recording:
            self.start_recording(None)
        else:
            self.stop_recording(None)
            self.transcribe_audio()

    @rumps.clicked("Start Recording")
    def start_recording(self, _):
        if not self.is_recording:
            self.is_recording = True
            self.title = "🔴"  # Change icon to indicate recording
            # Start your recording process here
            # You might want to run this in a separate thread
            threading.Thread(target=self.record_and_transcribe).start()
            rumps.notification("Audio Transcription", "Recording Started", "")

    @rumps.clicked("Stop Recording")
    def stop_recording(self, _):
        if self.is_recording:
            self.is_recording = False
            self.title = "🎙️"  # Change icon back to default
            # Stop your recording process here
            rumps.notification(
                "Audio Transcription", "Recording Stopped", "Transcribing..."
            )

    def record_and_transcribe(self):
        self.audio_recorder.start_recording()
        self.feedback_manager.provide_feedback("Recording")

    def transcribe_audio(self):
        self.audio_recorder.stop_recording_process()
        self.feedback_manager.clear_feedback()

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
                # Here you might want to save the transcription or copy it to clipboard
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
