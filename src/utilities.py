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
from datasets_loaders.Common_voice_11 import Common_voice_11
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

    """
    Benchmark a Whisper model on a longform audio file.
    It will do batching of the file, to see the speed up of batching.

    Parameters
    ----------
    cfg : Config object
        The configuration object.
    """


    file_path = cfg.benchmark.file
    filename = file_path.split('/')[-1].split('.')[0]

    if cfg.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU instead.")
        cfg.device = 'cpu'
    
    if cfg.model in cfg.available_models:
        model = whisper.load_model(cfg.model, device=torch.device(cfg.device))

    model = model.to(cfg.device)
    logger.info(f"Model is {'multilingual' if model.is_multilingual else 'English-only'} \
        and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters.")
    
    logger.info(f"Running initial run on {file_path}.")
    # we do one initial run to warmup gpu and cache
    model.transcribe(file_path, **{"language" : cfg.benchmark.language})

    # double batch size for each run
    results_batched = {}
    results_linear = {}

    def run(model, file_path, batch_size):
        start = time()
        model.transcribe([file_path]*batch_size, **{"language" : cfg.benchmark.language})
        end = time()
        return (end - start)/60

    logger.info(f"Batch size: 1")
    results_linear[1] = run(model, file_path, 1)
    results_batched[1] = results_linear[1]

    i = 2
    while i <= cfg.batch_size:
        logger.info(f"Batch size: {i}")
        results_linear[i] = results_linear[i//2]*2
        results_batched[i] = run(model, file_path, i)

        logger.info(f"Batc  nr. {i}: {results_batched[i]} min")
        i *= 2

        
    # save as csv
    with open(f'{"../benchmarks/batching/{filename}_time"}.csv', 'w') as f:
        for key in results.keys():
            f.write("%s,%s" % (key, results[key]))
            f.write("\n")
    
    # plot results
    fig , ax = plt.subplots(figsize=(12, 12))
    ax.plot(list(results_batched.keys()), list(results_batched.values()), label='Batched')
    ax.plot(list(results_linear.keys()), list(results_linear.values()), label='Linear')
    ax.set_xlabel('Batch size')
    ax.set_ylabel('Time (min)')
    ax.set_title(f'Batching time for {filename}')
    ax.legend()
    plt.savefig(f'{"../benchmarks/batching/{filename}_time"}.png')
