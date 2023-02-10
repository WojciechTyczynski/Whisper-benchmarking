import os

import pandas as pd
import torch
import torchaudio
import whisper


class NST_dk(torch.utils.data.Dataset):
    """
    A simple class to wrap NST_dk dataset and trim/pad the audio to 30 seconds.
    It will drop the last few seconds of a very small portion of the utterances.
    """
    def __init__(self, device='cpu', path = '/work3/s184954/NST_dk/'):
        self.device = device
        self.path = path
        df = pd.read_csv(path + "NST_dk.csv", sep=",")
        self.text_list = df['text']
        self.file_names = df['filename_both_channels']

    def __len__(self):
        return len(self.text_list)

    def __getitem__(self, item):
        text = self.text_list[item]
        file_name = self.file_names[item]
        folder= file_name.split("_")[0]
        file_path = f"{self.path}/dk/{folder}/{file_name}"
        audio, sample_rate = torchaudio.load(file_path)
        assert sample_rate == 16000
        audio = whisper.pad_or_trim(audio.flatten()).to(self.device)
        mel = whisper.log_mel_spectrogram(audio)
        return (mel, text)


