from datasets import load_dataset
import torch
import torchaudio
import whisper
from datasets import Audio

class Common_voice(torch.utils.data.Dataset):
    """
    A simple class to wrap fleurs and trim/pad the audio to 30 seconds.
    """
    def __init__(self, split="test", device='cpu', language='da'):
        self.dataset = load_dataset("mozilla-foundation/common_voice_11_0", language, split=split, cache_dir='/work3/s212373/common_voice')
        self.dataset =  self.dataset.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])
        self.data = self.dataset.cast_column("audio", Audio(sampling_rate=16000))
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        audio =  torch.from_numpy(self.dataset[item]['audio']['array'])
        text = self.dataset[item]['sentence']
        sample_rate = self.dataset[item]['audio']['sampling_rate']
        try:
            assert sample_rate == 16000
        except:
            audio = torchaudio.functional.resample(audio, sample_rate, 16000)
        audio = whisper.pad_or_trim(audio.float()).to(self.device)
        mel = whisper.log_mel_spectrogram(audio)
        
        return (mel, text)
