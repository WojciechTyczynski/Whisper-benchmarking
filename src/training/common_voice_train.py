import warnings

import audioread
import librosa
import pandas as pd
import torch
import torchaudio
import whisper
from datasets import Audio, load_dataset

warnings.filterwarnings("ignore")


class Common_voice_11(torch.utils.data.Dataset):
    """
    A simple class to wrap fleurs and trim/pad the audio to 30 seconds.
    """
    def __init__(self, tokenizer, split="test", device='cpu', language='da', path='/work3/s212373/common_voice_11'):
        self.tokenizer = tokenizer
        self.device = device
        self.device = device
        self.path = path
        self.split = split
        self.language = language
        self.dataset = pd.read_csv(f'{self.path}/{self.language}/{self.split}.tsv', sep='\t')
        self.dataset.drop(['client_id', 'up_votes', 'down_votes', 'age', 'gender', 'locale', 'segment'], axis=1, inplace=True)
        self.dataset.dropna(inplace=True)
        self.dataset = self.dataset.reset_index(drop=True)
        self.dataset['full_path'] = self.dataset['path'].apply(lambda x: f'{self.path}/{self.language}/clips/{x}')
        self.text_list = self.dataset['sentence']
        self.file_names = self.dataset['full_path']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        audio_file = audioread.audio_open(self.file_names[item])
        audio, sample_rate = librosa.load(audio_file, sr=16000)
        audio = torch.from_numpy(audio)
        text = self.text_list[item]
        text = [
            *self.tokenizer.sot_sequence_including_notimestamps
        ] + self.tokenizer.encode(text)
        labels = text[1:] + [self.tokenizer.eot]
        try:
            assert sample_rate == 16000
        except:
            audio = torchaudio.functional.resample(audio, sample_rate, 16000)
        audio = whisper.pad_or_trim(audio.flatten()).to(self.device)
        mel = whisper.log_mel_spectrogram(audio)
        
        
        return {"input_ids": mel, "labels": labels, "dec_input_ids": text}