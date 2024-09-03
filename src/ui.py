import rumps
import threading
from pynput import keyboard
import json
import logging
from utils import AudioRecorder, FeedbackManager
from app import transcribe_audio, get_hf_repo, ModelHolder
import mlx.core as mx

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class AudioTranscriptionApp(rumps.App):
    def __init__(self):
        super(AudioTranscriptionApp, self).__init__("🎙️")
        self.menu = ["Start Recording", "Stop Recording", "Settings"]
        self.hotkey = keyboard.HotKey(
            keyboard.HotKey.parse("<shift>+<ctrl>+<cmd>+r"), self.toggle_recording
        )
        self.listener = keyboard.Listener(
            on_press=self.on_press, on_release=self.on_release
        )
        self.listener.start()

        self.audio_recorder = AudioRecorder()
        self.feedback_manager = FeedbackManager()
        
        self.default_configs = {
            'model_name': 'large-v3',
            'language': 'en'
        }
        self.configs = {}
        self.load_settings()
        
        # Pre-load model
        self.load_model()

    def on_press(self, key):
        if key == keyboard.Key.esc and self.audio_recorder.is_recording:
            self.stop_recording(transcribe=False)
        else:
            self.hotkey.press(key)

    def on_release(self, key):
        self.hotkey.release(key)

    def toggle_recording(self):
        if not self.audio_recorder.is_recording:
            self.start_recording(None)
        else:
            self.stop_recording(transcribe=True)

    @rumps.clicked("Start Recording")
    def start_recording(self, _):
        if not self.audio_recorder.is_recording:
            self.audio_recorder.start_recording()
            self.title = "🔴"  # Change icon to indicate recording
            self.menu["Start Recording"].set_callback(None)
            self.menu["Stop Recording"].set_callback(
                lambda _: self.stop_recording(transcribe=True)
            )
            rumps.notification("Audio Transcription", "Recording Started", "")

    @rumps.clicked("Stop Recording")
    def stop_recording(self, transcribe=True):
        if self.audio_recorder.is_recording:
            self.audio_recorder.stop_recording_process()
            self.title = "🎙️"
            self.menu["Start Recording"].set_callback(self.start_recording)
            self.menu["Stop Recording"].set_callback(None)
            rumps.notification("Audio Transcription", "Recording Stopped", "")
            if transcribe:
                self.transcribe_audio()

    def transcribe_audio(self):
        audio_data = self.audio_recorder.get_recorded_audio()
        if audio_data:
            self.feedback_manager.provide_feedback("Transcribing")
            transcription = transcribe_audio(audio_data, self.configs['model_name'], self.configs['language'])
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

    def load_settings(self):
        try:
            with open('settings.json', 'r') as f:
                settings = json.load(f)
            self.configs = {**self.default_configs, **settings}
        except FileNotFoundError:
            self.configs = self.default_configs.copy()
            self.save_settings()

    def save_settings(self):
        with open('settings.json', 'w') as f:
            json.dump(self.configs, f)

    def load_model(self):
        ModelHolder.get_model(get_hf_repo(self.configs['model_name'], self.configs['language']), mx.float16)

    @rumps.clicked("Settings")
    def settings(self, _):
        logger.debug("Opening settings window")
        window = rumps.Window(
            message="Enter settings (JSON format):",
            title="Settings",
            default_text=json.dumps(self.configs, indent=2),
            ok="Save",
            dimensions=(320, 160),
        )
        response = window.run()
        if response.clicked:
            logger.debug(f"Settings window response: {response.text}")
            try:
                new_settings = json.loads(response.text)
                self.configs.update(new_settings)
                self.save_settings()
                self.load_model()
                logger.info("Settings updated successfully")
                rumps.alert(title="Settings", message="Settings updated. Model will be reloaded.")
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in settings: {e}")
                rumps.alert(title="Settings Error", message="Invalid JSON. Settings were not updated.")


if __name__ == "__main__":
    AudioTranscriptionApp().run()
