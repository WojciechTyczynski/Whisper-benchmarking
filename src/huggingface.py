from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, pipeline

# rename datasets folder to make this work

# load dummy dataset and read audio files
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")


pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-large", device=0)

raw_audio = ds[0]['audio']['array']

output = pipeline(raw_audio, return_timestamps = True, generate_kwargs={"task" : "transcribe"}, chunk_length_s=30, batch_size=16)

print(output)
