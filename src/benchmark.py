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
import json
from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoProcessor, WhisperTokenizer, pipeline

from datasets_loaders.Common_voice import Common_voice
from datasets_loaders.Common_voice_5_1 import Common_voice_5_1
from datasets_loaders.Common_voice_13 import Common_voice_13
from datasets_loaders.Fleurs import Fleurs
from datasets_loaders.FTSpeech import FTSpeech
from datasets_loaders.LibriSpeech import LibriSpeech
from datasets_loaders.NST_dk import NST_dk
from datasets_loaders.Rev16 import Rev16
from datasets_loaders.TEDLIUM import TEDLIUM
from datasets_loaders.HCAndersen import HCAndersen
from utilities import *

import json

language_dict = {
    'da': 'danish',
    'en': 'english',
    'fr': 'french',
}


whisper_cache_dir = "/work3/s183954/whisper"


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

    def run_huggingface_benchmark(cfg, options, loader):
        logger.info("Running benchmark with HuggingFace models.")
        processor = WhisperProcessor.from_pretrained(f"openai/whisper-{cfg.model}")
        model = WhisperForConditionalGeneration.from_pretrained(f"openai/whisper-{cfg.model}").to(cfg.device)
        # convert model to fp16
        model = model.half()

        logger.info(f"Loaded model.")
        forced_decoder_ids = processor.get_decoder_prompt_ids(language=language_dict[cfg.benchmark.language], task="transcribe")


        hypotheses = []
        references = []

        for mels, texts in tqdm(loader):
            predicted_ids = model.generate(mels.to(cfg.device).to(torch.float16), forced_decoder_ids=forced_decoder_ids)
            # decode token ids to text
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
            hypotheses.extend(transcription)
            references.extend(texts)
        
        return hypotheses, references


    
    def run_whisper_benchmark(cfg, options, loader):
        logger.info("Running benchmark with Whisper models.")
        if cfg.model in cfg.available_models and cfg.finetuned_model == False:
            model = whisper.load_model(cfg.model, device=torch.device(cfg.device), download_root=whisper_cache_dir)
        elif cfg.finetuned_model:
            model = whisper.load_model(cfg.finetuned_model_path, device=torch.device(cfg.device))
            logger.info(f"Loaded finetuned model") 
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

        return hypotheses, references


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
        dataset = Common_voice_13(split='test', device='cpu', language = cfg.benchmark.language)
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
    
    
    if cfg.huggingface == True:
        hypotheses, references = run_huggingface_benchmark(cfg, options, loader)
    else:
        hypotheses, references = run_whisper_benchmark(cfg, options, loader)

    

    wer = get_WER_MultipleTexts(hypotheses, references, normalizer=normalizer)
    logger.info(f"WER: {wer:.5%}, Model: {cfg.model}, Dataset: {cfg.benchmark.dataset}, Huggingface: {cfg.huggingface} CATCHME")
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

    # check if exist 
    if not os.path.exists(file_path):
        logger.error(f"File {file_path} does not exist.")
        return

    if cfg.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU instead.")
        cfg.device = 'cpu'
    
    if cfg.batch_size == -1:
        cfg.batch_size = None

    def run_whisper_time_benchmark(cgf, file_path):
        if cfg.model in cfg.available_models:
            model = whisper.load_model(cfg.model, device=torch.device(cfg.device), download_root=whisper_cache_dir)
        else:
            logger.error("Model not supported.")
            return
        logger.info(f"Loaded {cfg.model}")

        logger.info(f"Model is {'multilingual' if model.is_multilingual else 'English-only'}")
        logger.info(f"Running benchmark on {cfg.device}")
        # warmup 
        options = whisper.DecodingOptions(
            fp16=cfg.decode_options.fp16, 
            language=cfg.benchmark.language)

        model.transcribe(file_path, **{"fp16" : True, "language" : cfg.benchmark.language})
        # measure time
        start = time.time()
        for i in tqdm(range(10)):
            model.transcribe(file_path, **{"fp16" : True, "language" : cfg.benchmark.language})
        end = time.time()
        time_taken = (end - start) / 10
        return time_taken


    def run_huggingface_time_benchmark(cfg, file_path):
        logger.info(f"Running benchmark {cfg.whisper_version} on {cfg.device}")
        pipe = pipeline(
            "automatic-speech-recognition",
            model=f"openai/whisper-{cfg.model}",
            device=f"{cfg.device}:0",
            torch_dtype=torch.float16,
        )
        
        hypotheses = []
        references = []
        language_token = f"<|{cfg.benchmark.language}|>"
        if "en" in cfg.model:
            generate_kwargs = {"no_repeat_ngram_size":5}
        else:
            generate_kwargs = {"task":"transcribe", "language":language_token, "no_repeat_ngram_size":5}

        # warmup
        pipe(file_path, chunk_length_s=30,
                            generate_kwargs = generate_kwargs)
        # measure time
        start = time.time()
        for i in tqdm(range(10)):
            if cfg.batch_size == None:
                pipe(file_path, chunk_length_s=30,
                            generate_kwargs = generate_kwargs)
            else:    
                pipe(
                    file_path, return_timestamps=True, chunk_length_s=30, stride_length_s=[6,3], batch_size=cfg.batch_size,
                    generate_kwargs = generate_kwargs
                )
        end = time.time()
        time_taken = (end - start) / 10
        return time_taken

    if cfg.whisper_version == 'whisper':
        time_taken = run_whisper_time_benchmark(cfg, file_path)
    elif cfg.whisper_version == 'huggingface':
        time_taken = run_huggingface_time_benchmark(cfg, file_path)
    else:
        logger.error("Model not supported.")
        return
    
    logger.info(f"Time: {time_taken:.5f} seconds, model: {cfg.model}, device: {cfg.device}, language: {cfg.benchmark.language}, whisper_version: {cfg.whisper_version}")




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

    # login(cfg.hf_auth_token)

    if cfg.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU instead.")
        cfg.device = 'cpu'


    if cfg.benchmark.dataset == 'rev16':
        dataset = Rev16()
    elif cfg.benchmark.dataset == 'ted':
        dataset = TEDLIUM()
    elif cfg.benchmark.dataset == 'HCAndersen':
        dataset = HCAndersen()
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
    

    def run_whisper_longform_benchmark(cfg,loader):
        if cfg.model in cfg.available_models:
            model = whisper.load_model(cfg.model, device=torch.device(cfg.device), download_root=whisper_cache_dir)
        else:
            logger.error("Model not supported.")
            return
        logger.info(f"Loaded {cfg.model}")

        logger.info(f"Model is {'multilingual' if model.is_multilingual else 'English-only'}")

        model.to(torch.device(cfg.device))
        logger.info(f"Running benchmark on {cfg.device}")

        hypotheses = []
        references = []

        for audio, text in tqdm(loader):
            results = model.transcribe(audio,
                                    condition_on_previous_text = cfg.decode_options.condition_on_previous_text,
                                    **{"language" : cfg.benchmark.language})
            hypotheses.extend([results['text']])
            references.extend([text])

        return hypotheses, references

    def run_whsiperx_longform_benchmark(cfg,loader):
        import whisperx
        model = whisperx.load_model(cfg.model, device=torch.device(cfg.device))
        vad_pipeline = Inference(
                "pyannote/segmentation",
                pre_aggregation_hook=lambda segmentation: segmentation,
                use_auth_token=True,
                device=torch.device(cfg.device),
            )
        
        logger.info(f"Loaded {cfg.model}")
        model = model.to(cfg.device)

        hypotheses = []
        references = []
        for audio, text in tqdm(loader):
            results = whisperx.transcribe_with_vad_parallel(model,audio, vad_pipeline,
                                                            batch_size=cfg.batch_size,
                                                            **{"language" : cfg.benchmark.language, "task" : "transcribe"})
            
            results['text'] = ''.join([x['text'] for x in results['segments']])
            hypotheses.extend([result['text'] for result in results])
            references.extend(text)

        return hypotheses, references
    

    def run_huggingface_longform_benchmark(cfg,loader):
        logger.info(f"Running benchmark {cfg.whisper_version} on {cfg.device}")
        pipe = pipeline(
            "automatic-speech-recognition",
            model=f"openai/whisper-{cfg.model}",
            device=f"{cfg.device}:0",
            torch_dtype=torch.float16,
        )
        
        hypotheses = []
        references = []
        language_token = f"<|{cfg.benchmark.language}|>"
        if "en" in cfg.model:
            generate_kwargs = {"no_repeat_ngram_size":5}
        else:
            generate_kwargs = {"task":"transcribe", "language":language_token, "no_repeat_ngram_size":5}


        for audio, text in tqdm(loader):
            if cfg.batch_size == None:
                results = pipe(audio, chunk_length_s=30,
                            generate_kwargs = generate_kwargs)
            else:    
                results = pipe(
                    audio, return_timestamps=True, chunk_length_s=30, stride_length_s=[6,3], batch_size=cfg.batch_size,
                    generate_kwargs = generate_kwargs
                )
            hypotheses.extend([results['text']])
            references.extend([text])
        
        return hypotheses, references


    
    if cfg.whisper_version == 'whisper':
        hypotheses, references = run_whisper_longform_benchmark(cfg,loader)
    elif cfg.whisper_version == 'whisperx':
        hypotheses, references = run_whsiperx_longform_benchmark(cfg,loader)
    elif cfg.whisper_version == 'huggingface':
        hypotheses, references = run_huggingface_longform_benchmark(cfg,loader)
    else:
        logger.error("Model not supported.")
        return

    wer = get_WER_MultipleTexts(hypotheses, references, normalizer=normalizer)
    #logger.info(f"Time: {end_total - start_total:.5f} seconds, WER: {wer:.5%}, Model: {cfg.model}, Dataset: {cfg.benchmark.dataset} CATCHME")
    logger.info(f"WER: {wer:.5%}, Model: {cfg.model}, Dataset: {cfg.benchmark.dataset}, whisper_version: {cfg.whisper_version}, batch_size: {cfg.batch_size} CATCHME")