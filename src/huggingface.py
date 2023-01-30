from transformers import WhisperForConditionalGeneration, pipeline
from datasets import load_dataset


# load dummy dataset and read audio files
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
raw_audio = ds[0]['audio']["array"]

pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-large")
output = pipeline(raw_audio, return_timestamps = True, generate_kwargs={"language" : "<|ja|>", "task" : "translate"}, batch_size=16)

print(output)
