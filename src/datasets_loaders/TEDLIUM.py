import glob
import os

import pandas as pd
import torch
import torchaudio
import whisper


class TEDLIUM(torch.utils.data.Dataset):
    """
    A simple class to wrap TEDLIUM dataset.
    """
    def __init__(self, device='cpu', path = '/work3/s212373/ted_lium/TEDLIUM_release-3/legacy/test'):
        self.file_ids = ['DanBarber_2010',
                         'JaneMcGonigal_2010',
                         'BillGates_2010',
                         'TomWujec_2010U',
                         'GaryFlake_2010',
                         'EricMead_2009P',
                         'MichaelSpecter_2010',
                         'DanielKahneman_2010',
                         'AimeeMullins_2009P',
                         'JamesCameron_2010',
                         'RobertGupta_2010U']
        self.audio_files = {file_id:f'sph_cut/{file_id}.sph' for file_id in self.file_ids}
        self.text_files = {file_id:f'text/{file_id}.txt' for file_id in self.file_ids}
        self.device = device
        self.path = path

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, item):
        file_id = self.file_ids[item]
        audio_file = self.audio_files[file_id]
        text_file = self.text_files[file_id]
        audio_file_path = f"{self.path}/{audio_file}"
        # audio = whisper.load_audio(audio_file_path)
        # try:
        #     assert sample_rate == 16000
        # except:
        #      audio = torchaudio.functional.resample(audio, sample_rate, 16000)
        with open(f"{self.path}/{text_file}", "r") as f:
            text = f.read()
        return (audio_file_path, text)