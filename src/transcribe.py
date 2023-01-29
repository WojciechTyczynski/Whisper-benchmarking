from loguru import logger
import whisper
import pandas as pd
import torch
import utilities as ut
import platform

def transcribe(cfg, files):

    # Check if device is available
    if cfg.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU instead.")
        cfg.device = 'cpu'

    # load Whisper model
    model = whisper.load_model(
        cfg.model,
        device=torch.device(cfg.device),
    )

    result_dict = {}

    # load audio file
    for file in files:
        logger.info(f"Transcribing {file}")
        result = model.transcribe(file, **cfg.decode_options)
        if platform.system() == 'Windows': # I need this fix for my pc at home LOL
            input = input.replace('\\', '/')
        output_name = input.split('/')[-1].split('.')[0]
        result_dict[output_name] = result['segments']


    return result_dict

    