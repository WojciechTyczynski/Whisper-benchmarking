import glob
import os

import pandas as pd
import torch
import torchaudio
import whisper


class Rev16(torch.utils.data.Dataset):
    """
    A simple class to wrap Rev16 dataset.
    """
    def __init__(self, device='cpu', path = '/work3/s212373/rev16'):
        self.file_ids = ['26', '29', '21', '9',
                         '18', '4', '17','11',
                         '20', '3', '24', '10',
                         '14', '32', '27', '23']
        self.audio_files = {file_id:f'{file_id}.mp3' for file_id in self.file_ids}
        self.text_files = {file_id:f'{file_id}.txt' for file_id in self.file_ids}
        self.device = device
        self.path = path

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, item):
        file_id = self.file_ids[item]
        audio_file = self.audio_files[file_id]
        text_file = self.text_files[file_id]
        audio_file_path = f"{self.path}/{audio_file}"
        audio = whisper.load_audio(audio_file_path)
        # try:
        #     assert sample_rate == 16000
        # except:
        #      audio = torchaudio.functional.resample(audio, sample_rate, 16000)
        with open(f"{self.path}/{text_file}") as f:
            text = f.readlines()
        return (audio, text)
