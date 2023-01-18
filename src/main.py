from loguru import logger
import whisper
import pandas as pd
import hydra
import torch
import os

@torch.inference_mode()
@hydra.main(version_base=None, config_path="../conf", config_name="conf.yaml")
def transcribe(cfg) -> None :
    
    os.chdir(hydra.utils.get_original_cwd())

    # check input
    inputs = input_files_list(cfg.input)

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

        #save result to csv
        output_name = input.split('/')[-1].split('.')[0]
        df = pd.DataFrame(result['segments'])
        df[['start', 'end', 'text']].to_csv(f'{cfg.output}/{output_name}_transcribtion.csv', index=True)

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
        

if __name__ == "__main__":
    transcribe()


    