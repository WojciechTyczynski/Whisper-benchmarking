from fastapi import FastAPI, File, UploadFile
from typing import Union

app = FastAPI()

# Define health endpoint
@app.get('/health')
def health():
    return {'status': 'ok'}

# Define root endpoint
@app.get("/")
async def root():
    return {"message": "Hello World"}

# Define endpoint to transcribe a file
@app.post("/transcribe")
def transcribe_file(file: Union[UploadFile, None] = None):
    if not file:
        return {"message": "No upload file sent"}
    else:
        return {"filename": file.filename}
