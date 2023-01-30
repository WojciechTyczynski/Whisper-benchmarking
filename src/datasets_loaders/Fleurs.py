from datasets import load_dataset
import torch
import whisper

class Fleurs(torch.utils.data.Dataset):
    """
    A simple class to wrap fleurs and trim/pad the audio to 30 seconds.
    """
    def __init__(self, split="test", device='cpu', language='da_dk'):
        self.dataset = load_dataset("google/fleurs", language, split=split)
        self.dataset =  self.dataset.remove_columns(['id', 'num_samples', 'path',
                                                     'transcription', 'gender',
                                                     'lang_id', 'language', 'lang_group_id'])
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        audio =  torch.from_numpy(self.dataset[item]['audio']['array'])
        text = self.dataset[item]['raw_transcription']
        sample_rate = self.dataset[item]['audio']['sampling_rate']
        assert sample_rate == 16000
        audio = whisper.pad_or_trim(audio.float()).to(self.device)
        mel = whisper.log_mel_spectrogram(audio)
        
        return (mel, text)
