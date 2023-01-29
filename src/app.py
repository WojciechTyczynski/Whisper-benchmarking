from fastapi import FastAPI, File, UploadFile
from typing import Union
from config import api_config
from transcribe import transcribe_api
import utilities as ut
from loguru import logger
import pandas as pd

app = FastAPI()

# Define health endpoint
@app.get('/health')
def health():
    return {'status': 'ok'}

# Define root endpoint
@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get('/greet', status_code=200)
def say_hello():
    return api_config.greet_message

# Define endpoint to transcribe a file
@app.post("/transcribe/")
def transcribe_file(audio_file: UploadFile):
    logger.info(f"File loaded: {audio_file.filename}")
    logger.info(f"Converting audio file...")
    audio = ut.load_audio_file(audio_file.file)
    logger.info(f"Audio file converted")
    logger.info(f"Transcribing audio file...")
    transcribtion = transcribe_api(api_config, audio)
    df = pd.DataFrame(transcribtion)
    results = df[['start', 'end', 'text']].to_dict('dict')
    # results = ut.load_audio_file(audio_file.file)
    return {"filename": audio_file.filename, "file_type": audio_file.content_type, "results": results}

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    return {"filename": file.filename}
