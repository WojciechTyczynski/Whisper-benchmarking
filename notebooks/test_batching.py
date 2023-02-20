import whisper
import torch
import glob 
import os


model = whisper.load_model("base", device=torch.device('cuda'))

options=whisper.DecodingOptions(fp16=True,
            language="da"
        )




test = ['/dtu/blackhole/1f/137151/ftspeech/audio/2010/20101_M001.wav']
model.transcribe(test, verbose=False, **{"language" : "da"})
model.transcribe(test*2, verbose=False, **{"language" : "da"})