from setuptools import setup, find_packages

setup(
    name="AudioTranscriptionApp",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # List your project dependencies here
    ],
    entry_points={
        "console_scripts": [
            "audio-transcription=src.main:main",
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A powerful tool for transcribing audio files into text",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/AudioTranscriptionApp",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
