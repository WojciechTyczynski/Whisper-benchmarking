import torch
import torchaudio
import pandas as pd
import whisper

class FTSpeech(torch.utils.data.Dataset):
    """
    A simple class to wrap ftspeech and trim/pad the audio to 30 seconds.
    It will drop the last few seconds of a very small portion of the utterances.
    """
    def __init__(self,path = "/dtu/blackhole/1f/137151/ftspeech/", split="ft-speech_dev-balanced", device='cpu', remove_long = True):
        self.path = path
        self.data = pd.read_csv(f'{path}text/{split}.tsv',sep='\t')
        if remove_long:
            self.data['duration'] = self.data['end_time'] - self.data['start_time']
            self.data = self.data[self.data['duration'] <= 30]
        self.data = self.data.reset_index(drop=True)        
        self.device = device
        self.last_filepath = ""
        self.last_audio = None

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        folder = self.data.iloc[item]['utterance_id'].split('_')[1][:-1]
        filename = f"{self.data.iloc[item]['utterance_id'][5:15]}.wav"
        filepath = f'{self.path}audio/{folder}/{filename}'
        # Since we are just using part of the audio file for each round. 
        # Then we check if it is the same as for the previous transcript. 
        if filepath == self.last_filepath:
            audio = self.last_audio
            sample_rate = 16000
        else:
            self.last_filepath = filepath
            audio, sample_rate = torchaudio.load(filepath)
            self.last_audio = audio
        start_sample = round(self.data.iloc[item]['start_time']*sample_rate)
        end_sample = round(self.data.iloc[item]['end_time']*sample_rate)         
        # Find chunk that we need
        audio = audio[:,start_sample:end_sample]
        text = self.data.iloc[item]['transcript']
        text = text.replace("<UNK>", "")
        assert sample_rate == 16000
        audio = whisper.pad_or_trim(audio.flatten()).to(self.device)
        mel = whisper.log_mel_spectrogram(audio)
        return (mel, text)