import rumps
import threading
from pynput import keyboard

# Import your existing audio transcription functionality
# from your_transcription_module import start_recording, stop_recording, transcribe_audio


class AudioTranscriptionApp(rumps.App):
    def __init__(self):
        super(AudioTranscriptionApp, self).__init__("🎙️")
        self.menu = ["Start Recording", "Stop Recording", "Settings"]
        self.is_recording = False
        self.hotkey = keyboard.HotKey(
            keyboard.HotKey.parse("<cmd>+<shift>+r"), self.toggle_recording
        )
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

    def on_press(self, key):
        self.hotkey.press(key)

    def toggle_recording(self):
        if self.is_recording:
            self.stop_recording(None)
        else:
            self.start_recording(None)

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
        # This is where you'd implement your recording and transcription logic
        # For example:
        # audio_data = start_recording()
        # transcription = transcribe_audio(audio_data)
        # Here, you might want to save the transcription or copy it to clipboard
        pass

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
