from setuptools import setup, find_packages

setup(
    name="batchtool",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "faster-whisper",
        "ffmpeg-python",
        "torch",
        "numpy",
        "PyQt6",
    ],
) 