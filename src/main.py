import os

import hydra
import pandas as pd
import torch
import whisper
from loguru import logger
import benchmark as bm

@torch.inference_mode()
@hydra.main(version_base=None, config_path="../conf", config_name="conf.yaml")
def run(cfg) -> None :
    os.chdir(hydra.utils.get_original_cwd())
    options=whisper.DecodingOptions(fp16=cfg.decode_options.fp16,
            language=cfg.benchmark.language,
            beam_size=cfg.decode_options.beam_size
            # temperature=0,
        )
    
    try:
        benchmark_type = cfg.benchmark.type
    except:
        benchmark_type = None
    
    logger.info(f"Running benchmark {benchmark_type}")
    if benchmark_type == None:
        bm.benchmark_model(cfg, options)
    elif benchmark_type == 'longform_wer':
        bm.benchmark_longform_wer(cfg, options)
    elif benchmark_type == 'longform_time':
        bm.benchmark_longform_time(cfg)
    else:
        logger.error("Benchmark type not supported")


if __name__ == "__main__":
    print("Running main.py")
    run()