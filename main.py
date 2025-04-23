from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import shutil
import os
import whisper
import json
from difflib import SequenceMatcher
import re

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

model = whisper.load_model("base")

# Load Quran ayat (seluruh Al-Qur'an)
with open("quran_full.json", "r", encoding="utf-8") as f:
    ayat_list = json.load(f)

class AnalysisResult(BaseModel):
    ayat: str
    tajwid: dict
    feedback: str
    score: int

def transcribe_audio(path: str) -> str:
    result = model.transcribe(path, language='ar')
    return result["text"]

def match_ayah(transcribed_text, ayah_list):
    best_match = None
    best_score = 0.0
    for ayah in ayah_list:
        score = SequenceMatcher(None, transcribed_text.strip(), ayah["text"].strip()).ratio()
        if score > best_score:
            best_match = ayah
            best_score = score
    return best_match, best_score

def analyze_tajwid(text: str) -> dict:
    result = {
        "mad": [],
        "ikhfa": [],
        "idgham": [],
        "iqlab": [],
        "ghunnah": [],
        "qalqalah": []
    }

    # Pola sederhana
    mad_patterns = [r"[اآووي]{2,}", r"ٰ"]
    ikhfa_patterns = [r"نْ ?[تثدذرزسشصضطظ]", r"ن ٱ[تثدذرزسشصضطظ]"]
    idgham_patterns = [r"نْ ?[يومنلو]", r"ن ٱ[يومنلو]"]
    iqlab_patterns = [r"نْ ?ب"]
    ghunnah_patterns = [r"(مّ|نّ)"]
    qalqalah_patterns = [r"(بْ|جْ|دْ|طْ|قْ)"]

    for pat in mad_patterns:
        result["mad"].extend(re.findall(pat, text))
    for pat in ikhfa_patterns:
        result["ikhfa"].extend(re.findall(pat, text))
    for pat in idgham_patterns:
        result["idgham"].extend(re.findall(pat, text))
    for pat in iqlab_patterns:
        result["iqlab"].extend(re.findall(pat, text))
    for pat in ghunnah_patterns:
        result["ghunnah"].extend(re.findall(pat, text))
    for pat in qalqalah_patterns:
        result["qalqalah"].extend(re.findall(pat, text))

    return result

@app.post("/api/analyze")
async def analyze_audio(audio: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_DIR, audio.filename)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)

    transcribed_text = transcribe_audio(file_location)
    matched_ayah, confidence = match_ayah(transcribed_text, ayat_list)
    ayat_text = matched_ayah["text"] if matched_ayah else transcribed_text
    tajwid_result = analyze_tajwid(ayat_text)

    result = AnalysisResult(
        ayat=ayat_text,
        tajwid=tajwid_result,
        feedback=f"Kecocokan ayat: {confidence:.2f}. Tajwid dianalisis otomatis.",
        score=int(confidence * 100)
    )

    return JSONResponse(content=result.dict())
