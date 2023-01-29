from loguru import logger
import whisper
import pandas as pd
import hydra
import torch
import os
import utilities as ut
import platform


@torch.inference_mode()
@hydra.main(version_base=None, config_path="../conf", config_name="conf.yaml")
def transcribe(cfg) -> None :
    
    os.chdir(hydra.utils.get_original_cwd())

    # check input
    inputs = ut.input_files_list(cfg.input)

    # Check if device is available
    if cfg.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU instead.")
        cfg.device = 'cpu'


    # load Whisper model
    model = whisper.load_model(
        cfg.model,
        device=torch.device(cfg.device),
    )

    # check if direrctory with transcriptions exists and create it if not
    if not os.path.exists(cfg.output):
        os.mkdir(cfg.output)

    for input in inputs:            
        # load audio file
        logger.info(f"Transcribing {input}")
        result = model.transcribe(input, **cfg.decode_options)
        print(result.keys())
        #save result to csv
        if platform.system() == 'Windows': # I need this fix for my pc at home LOL
            input = input.replace('\\', '/')
        output_name = input.split('/')[-1].split('.')[0]
        df = pd.DataFrame(result['segments'])
        df[['start', 'end', 'text']].to_csv(f'{cfg.output}/{output_name}_transcribtion.csv', index=True)


if __name__ == "__main__":
    transcribe()


    