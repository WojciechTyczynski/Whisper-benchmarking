import torch
import torchaudio



class FTSpeech(torch.utils.data.Dataset):
    """
    A simple class to wrap ftspeech and trim/pad the audio to 30 seconds.
    It will drop the last few seconds of a very small portion of the utterances.
    """
    def __init__(self,path = "/dtu/blackhole/1f/137151/ftspeech/", split="ft-speech_dev-balanced", device='cpu'):
        self.data = pd.read_csv(f'{path}text/{split}.tsv',sep='\t')        
        self.device = device

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, item):
        folder = data.iloc[item]['utterance_id'].split('_')[1][:-1]
        filename = f"{data.iloc[item]['utterance_id'][5:15]}.wav"
        audio, sample_rate = torchaudio.load(f'{path}audio/{folder}/{filename}')
        start_sample = round(data.iloc[item]['start_time']*samplerate)
        end_sample = round(data.iloc[item]['end_time']*samplerate)
#         Find chunk that we need
        audio = audio[:,start_sample:end_sample]
        text = data.iloc[item]['transcript']
        text = text.replace("<UNK>", "")
        assert sample_rate == 16000
        audio = whisper.pad_or_trim(audio.flatten()).to(self.device)
        mel = whisper.log_mel_spectrogram(audio)
        return (mel, text)