import os
import time
from typing import BinaryIO, Union

import ffmpeg
import jiwer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import whisper
from datasets import load_dataset
from loguru import logger
from tqdm import tqdm
from whisper.normalizers import BasicTextNormalizer, EnglishTextNormalizer

from datasets_loaders.Common_voice import Common_voice
from datasets_loaders.Common_voice_5_1 import Common_voice_5_1
from datasets_loaders.Common_voice_13 import Common_voice_13
from datasets_loaders.Fleurs import Fleurs
from datasets_loaders.FTSpeech import FTSpeech
from datasets_loaders.LibriSpeech import LibriSpeech
from datasets_loaders.NST_dk import NST_dk
from datasets_loaders.Rev16 import Rev16

SAMPLE_RATE=16000

def get_WER_MultipleTexts(transcription:list, reference:list, normalizer=EnglishTextNormalizer()) -> float: 
    """
    Calculate WER between transcription and reference.
    Transcription and reference are lists of strings.
    """
    if normalizer is not None:
        transcription = [normalizer(text) for text in transcription]
        reference = [normalizer(text) for text in reference]
    wer = jiwer.wer(reference, transcription)
    return wer

def get_WER_SingleText(transcription:str, reference:str, normalizer=EnglishTextNormalizer()) -> float:
    """Calculate WER between transcription and reference.
    Transcription and reference are strings."""
    if normalizer is not None:
        transcription = normalizer(transcription)
        reference = normalizer(reference)
    wer = jiwer.wer(reference, transcription)
    return wer

def find_audio_files(input) -> list:
    """Find all audio files in a directory."""
    audio_files = []
    for root, dirs, files in os.walk(input):
        for file in files:
            if file.endswith(".wav") or file.endswith(".mp3"):
                audio_files.append(os.path.join(root, file))
    return audio_files


def input_files_list(input):
    """Check if provided input is a directory or file path."""
    if os.path.isdir(input):
        return find_audio_files(input)
    else:
        return [input]

def load_audio_file(file: BinaryIO, sr: int = SAMPLE_RATE):
    """
    Open an audio file object and read as mono waveform, resampling as necessary.
    Modified from https://github.com/openai/whisper/blob/main/whisper/audio.py to accept a file object
    Parameters
    ----------
    file: BinaryIO
        The audio file like object
    sr: int
        The sample rate to resample the audio if necessary
    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    try:
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        out, _ = (
            ffmpeg.input("pipe:", threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .run(cmd="ffmpeg", capture_stdout=True, capture_stderr=True, input=file.read())
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0