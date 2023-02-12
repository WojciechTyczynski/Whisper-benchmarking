import torch
import os 
import torchaudio
import whisper
import pandas as pd


class NST_dk(torch.utils.data.Dataset):
    """
    A simple class to wrap NST_dk dataset and trim/pad the audio to 30 seconds.
    It will drop the last few seconds of a very small portion of the utterances.
    """
    def __init__(
        self, device="cpu", path="/work3/s183954/NST_dk/", split="train"
    ):
        self.device = device
        self.path = path
        self.split = split
        if split == "train":
            df = pd.read_csv(path + "NST_dk_clean.csv", sep=",")
            self.path = path + "dk/"
            self.file_names = df["filename_both_channels"]
        else:
            df = pd.read_csv(path + "supplement_dk_clean.csv", sep=",")
            self.path = path + "supplement_dk/testdata/audio/"
            self.file_names = df["filename_channel_1"]
        self.text_list = df["text"]

    def __len__(self):
        return len(self.text_list)

    def __getitem__(self, item):
        text = self.text_list[item]
        file_name = self.file_names[item]
        folder = file_name.split("_")[0]
        if self.split != "train":
            file_name = file_name.split("_")[1]
        file_path = f"{self.path}{folder}/{file_name.lower()}"
        audio, sample_rate = torchaudio.load(file_path)
        assert sample_rate == 16000
        audio = whisper.pad_or_trim(audio.flatten()).to(self.device)
        mel = whisper.log_mel_spectrogram(audio)
        return (mel, text)


