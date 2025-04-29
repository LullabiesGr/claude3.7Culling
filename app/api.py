from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import shutil
import uuid

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/analyze-photo/")
async def analyze_photo(file: UploadFile = File(...)):
    # Save photo to disk
    photo_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{photo_id}_{file.filename}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Return fake score for now
    return {
        "status": "success",
        "filename": file.filename,
        "score": 87,  # We'll calculate real score later
        "approved": True  # Or False based on score
    }

if __name__ == "__main__":
    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)

