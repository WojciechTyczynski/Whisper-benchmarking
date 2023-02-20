from loguru import logger
import whisper
import pandas as pd
import hydra
import torch
import os
import utilities as ut
from transcribe import transcribe


@torch.inference_mode()
@hydra.main(version_base=None, config_path="../conf", config_name="conf.yaml")
def run(cfg) -> None :
    os.chdir(hydra.utils.get_original_cwd())
    print(cfg.benchmark.dataset)    
    options=whisper.DecodingOptions(fp16=cfg.decode_options.fp16,
            language=cfg.benchmark.language,
            beam_size=cfg.decode_options.beam_size
            # temperature=0,
        )
    if cfg.benchmark_type == 'default':
        ut.benchmark_model(cfg, options)
    elif cfg.benchmark_type == 'longform_wer':
        ut.benchmark_longform_wer(cfg, options)
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