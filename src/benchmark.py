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
from pyannote.audio import Pipeline
from pyannote.audio import Inference
from huggingface_hub import login

from datasets_loaders.Common_voice import Common_voice
from datasets_loaders.Common_voice_5_1 import Common_voice_5_1
from datasets_loaders.Common_voice_11 import Common_voice_11
from datasets_loaders.Fleurs import Fleurs
from datasets_loaders.FTSpeech import FTSpeech
from datasets_loaders.LibriSpeech import LibriSpeech
from datasets_loaders.NST_dk import NST_dk
from datasets_loaders.Rev16 import Rev16
from datasets_loaders.TEDLIUM import TEDLIUM
from utilities import *


def benchmark_model(cfg, options:whisper.DecodingOptions):
    """
    Benchmark a Whisper model on a dataset.
    
    Parameters
    ----------
    cfg : Config object
        The configuration object.
    options : whisper.DecodingOptions
        The decoding options.    
    """
    if cfg.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU instead.")
        cfg.device = 'cpu'
    # We can then add more benchmarking datasets
    if cfg.benchmark.dataset == 'LibriSpeech':
        dataset = LibriSpeech("test-clean", device='cpu')
    elif cfg.benchmark.dataset == 'fleurs':
        dataset = Fleurs(split='test', device='cpu', language = cfg.benchmark.dataset_language)
    elif cfg.benchmark.dataset == 'FTSpeech':
        dataset = FTSpeech(split='ft-speech_test-balanced')
    elif cfg.benchmark.dataset == 'NST_dk':
        dataset = NST_dk(split='test', device='cpu')
    elif cfg.benchmark.dataset == 'Common_voice':
        dataset = Common_voice_11(split='test', device='cpu', language = cfg.benchmark.language)
    elif cfg.benchmark.dataset == 'Common_voice_5_1':
        dataset = Common_voice_5_1(split='test', device='cpu', language = cfg.benchmark.language)
    else:
        logger.error("Dataset not supported.")
        return
    
    if cfg.benchmark.language == 'en':
        normalizer=EnglishTextNormalizer()
    else:
        normalizer=BasicTextNormalizer()

    if cfg.batch_size == -1:
        cfg.batch_size = None 
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    logger.info(f"Loaded {cfg.benchmark.dataset} dataset with {len(dataset)} utterances. Language {cfg.benchmark.language}")
    
    
    
    if cfg.model in cfg.available_models:
        model = whisper.load_model(cfg.model, device=torch.device(cfg.device))
        logger.info(f"Loaded {model} model.")
    else:
        logger.error("Model not supported.")
        return
    
    logger.info(f"Model is {'multilingual' if model.is_multilingual else 'English-only'} \
        and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters.")
    
    model = model.to(cfg.device)

    hypotheses = []
    references = []
    # measure time
    start = time.time()
    for mels, texts in tqdm(loader):
        results = model.decode(mels.to(cfg.device).to(torch.float16), options)
        hypotheses.extend([result.text for result in results])
        references.extend(texts)
    end = time.time()

    wer = get_WER_MultipleTexts(hypotheses, references, normalizer=normalizer)
    logger.info(f"Time: {end - start:.5f} seconds, WER: {wer:.5%}, Model: {cfg.model}, Dataset: {cfg.benchmark.dataset} CATCHME")
    # for i,j in zip(hypotheses, references):
    #     print(f'{j}     <>      {i}')



def benchmark_longform_time(cfg):
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


def benchmark_longform_wer(cfg, options:whisper.DecodingOptions):
    """
    Benchmark a Whisper model on a longforms.
    
    Parameters
    ----------
    cfg : Config object
        The configuration object.
    options : whisper.DecodingOptions
        The decoding options.    
    """

    login(cfg.api_token)

    if cfg.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU instead.")
        cfg.device = 'cpu'

    if cfg.benchmark.dataset == 'rev16':
        dataset = Rev16()
    if cfg.benchmark.dataset == 'ted':
        dataset = TEDLIUM()
    else:
        logger.error("Dataset not supported.")
        return
    
    if cfg.benchmark.language == 'en':
        normalizer=EnglishTextNormalizer()
    else:
        normalizer=BasicTextNormalizer()
    
    if cfg.batch_size == -1:
        cfg.batch_size = None

    loader = torch.utils.data.DataLoader(dataset, num_workers=cfg.num_workers, batch_size=None)
    logger.info(f"Loaded {cfg.benchmark.dataset} dataset with {len(dataset)} utterances.")
    

    
    if cfg.model in cfg.available_models:
        if cfg.whisper_version == 'whisper':
            model = whisper.load_model(cfg.model, device=torch.device(cfg.device))
        elif cfg.whisper_version == 'whisperx':
            import whisperx
            model = whisperx.load_model(cfg.model, device=torch.device(cfg.device))
            vad_pipeline = Inference(
                    "pyannote/segmentation",
                    pre_aggregation_hook=lambda segmentation: segmentation,
                    use_auth_token=True,
                    device=torch.device(cfg.device),
                )
    else:
        logger.error("Model not supported.")
        return
    logger.info(f"Loaded {cfg.model} model - {cfg.whisper_version}.")

    logger.info(f"Model is {'multilingual' if model.is_multilingual else 'English-only'} \
        and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters.")
    
    model = model.to(cfg.device)
    
    hypotheses = []
    references = []
    start_total = time.time()
    for audios, texts in tqdm(loader):
        start_batch = time.time()
        # print(audio_paths)
        if cfg.whisper_version == 'whisper':
            results = model.transcribe(audios, **{"language" : cfg.benchmark.language})
        elif cfg.whisper_version == 'whisperx':
            results = whisperx.transcribe_with_vad_parallel(model, audios, vad_pipeline, batch_size=cfg.batch_size, **{"language" : cfg.benchmark.language, "task" : "transcribe"})
            
            results['text'] = ''.join([x['text'] for x in results['segments']])
        if isinstance(results, list):            
            hypotheses.extend([result['text'] for result in results])
            references.extend(texts)
        else:
            hypotheses.extend([results['text']])
            references.extend([texts])
        end_batch = time.time()
    end_total = time.time()

    wer = get_WER_MultipleTexts(hypotheses, references, normalizer=normalizer)
    logger.info(f"Time: {end_total - start_total:.5f} seconds, WER: {wer:.5%}, Model: {cfg.model}, Dataset: {cfg.benchmark.dataset} CATCHME")