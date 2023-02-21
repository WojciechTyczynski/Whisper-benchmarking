import os

import hydra
import pandas as pd
import torch
import whisper
from loguru import logger
from transcribe import transcribe
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
    
    if benchmark_type == None:
        bm.benchmark_model(cfg, options)
    elif benchmark_type == 'longform_wer':
        bm.benchmark_longform_wer(cfg, options)
    elif benchmark_type == 'longform_time':
        bm.benchmark_longform_time(cfg)
    else:
        logger.error("Benchmark type not supported")

    # check input
    # inputs = ut.input_files_list(cfg.input)

    # results = transcribe(cfg, inputs)

    # # check if direrctory with transcriptions exists and create it if not
    # if not os.path.exists(cfg.output):
    #     os.mkdir(cfg.output)

    # for transcribed_file in results:            
        
    #     # print(file)
    #     # print(cfg.output)
    #     print(f'{cfg.output}/{transcribed_file}_transcribtion.csv')
    #     df = pd.DataFrame(results[transcribed_file])
    #     df[['start', 'end', 'text']].to_csv(f'{cfg.output}/{transcribed_file}_transcribtion.csv', index=True)


if __name__ == "__main__":
    run()