import glob
import os

import pandas as pd
import torch
import torchaudio
import whisper


class HCAndersen(torch.utils.data.Dataset):
    """
    A simple class to wrap HCAndersen data
    """
    def __init__(self, device='cpu', path = '/work3/s183954/longform_danish'):
        self.df = pd.read_csv(f"{path}/dataset.csv", sep=";")
        self.device = device
        self.path = path

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, item):
        row = self.df.iloc[item]
        audio_file = row['audio']
        text_file = row['text']
        audio_file_path = f"{self.path}/audio/{audio_file}"
        audio = whisper.load_audio(audio_file_path)
        
        with open(f"{self.path}/cleaned_text/{text_file}", "r") as f:
            text = f.read()
        return (audio_file_path, text)
