import jiwer
from whisper.normalizers import EnglishTextNormalizer
import os
from loguru import logger
import whisper
import torch
<<<<<<< HEAD
from datasets.LibriSpeech import LibriSpeech
=======
from datasets import LibriSpeech
>>>>>>> a9c608d0f279982b6bd5315ae95ce83e28037124
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch
import time

def get_WER_MultipleTexts(transcription:list, reference:list, normalizer=EnglishTextNormalizer()) -> float: 
    """
    Calculate WER between transcription and reference.
    Transcription and reference are lists of strings.
    """
    transcription = [normalizer(text) for text in transcription]
    reference = [normalizer(text) for text in reference]
    wer = jiwer.wer(reference, transcription)
    return wer

def get_WER_SingleText(transcription:str, reference:str, normalizer=EnglishTextNormalizer()) -> float:
    """Calculate WER between transcription and reference.
    Transcription and reference are strings."""
    normalizer = EnglishTextNormalizer()
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


<<<<<<< HEAD
def benchmark_model(cfg, options:whisper.DecodingOptions ,normalizer=EnglishTextNormalizer()):
=======
def benchmark_model(cfg, model:str, dataset_str:str, device:str, decode_options:dict, without_timestamps = True ,normalizer=EnglishTextNormalizer()) -> pd.DataFrame:
>>>>>>> a9c608d0f279982b6bd5315ae95ce83e28037124
    """
    Benchmark a Whisper model on a dataset.
    
    Parameters
    ----------
    model: Whisper model
    dataset: path to directory with audio files and reference transcriptions
    """
    # We can then add more benchmarking datasets
<<<<<<< HEAD
    if cfg.dataset_str == 'LibriSpeech':
=======
    if dataset_str == 'LibriSpeech':
>>>>>>> a9c608d0f279982b6bd5315ae95ce83e28037124
        dataset = LibriSpeech("test-clean")
    else:
        logger.error("Dataset not supported.")
        return
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=16)
<<<<<<< HEAD
    logger.info(f"Loaded {cfg.dataset_str} dataset with {len(dataset)} utterances.")
=======
    logger.info(f"Loaded {dataset_str} dataset with {len(dataset)} utterances.")
>>>>>>> a9c608d0f279982b6bd5315ae95ce83e28037124
    
    if cfg.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU instead.")
        cfg.device = 'cpu'
    
<<<<<<< HEAD
    if cfg.model in cfg.available_models:
        model = whisper.load_model(cfg.model, device=torch.device(cfg.device))
=======
    if model in cfg.available_models:
        model = whisper.load_model(model, device=torch.device(cfg.device))
>>>>>>> a9c608d0f279982b6bd5315ae95ce83e28037124
        logger.info(f"Loaded {model} model.")
    else:
        logger.error("Model not supported.")
        return
    
    logger.info(f"Model is {'multilingual' if model.is_multilingual else 'English-only'} \
        and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters.")
    
<<<<<<< HEAD
    model = model.to(cfg.device)

    hypotheses = []
    references = []
    # measure time
    start = time.time()
    for mels, texts in tqdm(loader):
        results = model.decode(mels, options)
        hypotheses.extend([result.text for result in results])
        references.extend(texts)
    end = time.time()

    wer = get_WER_MultipleTexts(hypotheses, references, normalizer=normalizer)
    logger.info(f"Decoding time: {end - start:.2f} seconds")
    logger.info(f"WER: {wer:.2%}")
=======
    model = model.to(device)

    # predict without timestamps for short-form transcription
    options = whisper.DecodingOptions(language="en", without_timestamps=without_timestamps)
    
    hypotheses = []
    references = []

    for mels, texts in tqdm(loader):
        results = model.decode(mels, options)
        hypotheses.extend([result.text for result in results])
        references.extend(texts)

    data = pd.DataFrame(dict(hypothesis=hypotheses, reference=references))

    wer = get_WER_MultipleTexts(hypotheses, references, normalizer=normalizer)
    logger.info(f"WER: {wer:.2%}")
    return data
>>>>>>> a9c608d0f279982b6bd5315ae95ce83e28037124




