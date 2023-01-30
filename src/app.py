from fastapi import FastAPI, File, UploadFile
from typing import Union
from config import api_config
from transcribe import transcribe_api
import utilities as ut
from loguru import logger
import pandas as pd
import uvicorn
from fastapi.responses import HTMLResponse
from omegaconf import OmegaConf

app = FastAPI()

# Define health endpoint
@app.get('/health')
def health():
    return {'status': 'ok'}

# Define root endpoint
@app.get("/")
async def main():
    content = """
<body>
<form action="/transcribe/" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple>
<input type="submit">
</form>
<form action="/uploadfiles/" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple>
<input type="submit">
</form>
</body>
    """
    return HTMLResponse(content=content)

@app.get('/greet', status_code=200)
def say_hello():
    return api_config.greet_message

@app.get('/config', status_code=200)
def get_config():
    return {'config': OmegaConf.to_container(api_config)}

# Define endpoint to transcribe a file
@app.post("/transcribe/")
def transcribe_file(
    audio_files: list[UploadFile] = File()
):
    response = {}
    for audio_file in audio_files:
        logger.info(f"File loaded: {audio_file.filename}")
        logger.info(f"Converting audio file...")
        audio = ut.load_audio_file(audio_file.file)
        logger.info(f"Audio file converted")
        logger.info(f"Transcribing audio file...")
        transcribtion = transcribe_api(api_config, audio)
        df = pd.DataFrame(transcribtion)
        results = df[['start', 'end', 'text']].to_dict('dict')
        # results = ut.load_audio_file(audio_file.file)
        response[audio_file.filename] = results
    return response

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    return {"filename": file.filename}



if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)