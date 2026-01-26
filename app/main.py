import io
import os
import cv2
import json
import base64
import requests
import numpy as np
from ultralytics import YOLO
from typing import List, Optional, Dict, Any
from app.config import URL, MODEL_NAME
from app.prompt import LLM_PROMPT

from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from pydantic import BaseModel
from PIL import Image


# =========================
# APP INITIALIZATION
# =========================
app = FastAPI(title="Constellation Recognition API")


# =========================
# RESPONSE MODELS
# =========================
class DetectorResponse(BaseModel):
    yolo_img_result: str
    yolo_class_result: List[str]

class ExplainerResponse(BaseModel):
    llm_result: str


# =========================
# CONFIG
# =========================
# MODEL_NAME = "qwen3:4b"
# URL = "http://10.88.0.1:11434/api/chat"
# URL = "http://localhost:11434/api/chat"
# URL = "http://ollama:11434/v1/chat/completions"
# LLM_PROMPT = """You are an astronomy knowledge assistant.

# Input:
# - Detected constellations: <<CONSTELLATION_LIST>>
# - Output language: <<LANGUAGE>>

# Instructions:
# 1. ONLY if the detected constellations list is truly empty, respond ONLY with:
#    "No constellation found, please upload another image."
#    translated into the specified output language.
# 2. If the list is NOT empty, always generate information for the provided items, even if the image contains only the Moon.
# 3. For EACH detected constellation, provide:
#    - History
#    - Cultural significance
#    - Notable features
#    Each section should be 2–3 short, clear, factual sentences.
# 4. Output must be plain text only (no markdown, no emojis, no bullet symbols).
# 5. Do NOT invent or infer constellations beyond those explicitly listed.
# 6. Language rules:
#    - If the output language is English, show ONLY the English constellation name.
#    - If the output language is NOT English, show the name in English first, followed by the specified language.
# 7. Use the following format exactly:

# If LANGUAGE is English:
# Constellation Name
# History: ...
# Cultural Significance: ...
# Notable Features: ...

# If LANGUAGE is not English:
# Constellation Name in English / Constellation Name in <<LANGUAGE>>
# History: ...
# Cultural Significance: ...
# Notable Features: ...
# """


# =========================
# MODEL INITIALIZATION
# =========================
constellation_detectoer_model = YOLO("/app/yolo11-best-model.pt")


# =========================
# HELPER FUNCTIONS
# =========================
def file_to_pil(uploaded_file: UploadFile):
    image_bytes = uploaded_file.file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return img

def encoded_img(img: np.array) -> str:
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")


# =========================
# API ENDPOINTS
# =========================
@app.post("/constellation_detector", response_model=DetectorResponse)
def constellation_detector(files: List[UploadFile] = File(...)):
    try:
        img = file_to_pil(files[0])
        results = constellation_detectoer_model.predict(img, imgsz=640, conf=0.4)

        class_name_list = []

        boxes = results[0].boxes
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = constellation_detectoer_model.names[class_id]
            class_name_list.append(class_name)
            
        img_result = encoded_img(results[0].plot())

        if img_result is None:
            raise ValueError("YOLO returned no plot image")

        return DetectorResponse(
            yolo_img_result=img_result,
            yolo_class_result=class_name_list
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 


@app.post("/constellation_explainer", response_model=ExplainerResponse)
async def constellation_explainer(const_list: List[str] = Query(...), lang: str = Query(...)):
    QWEN_PROMPT = (LLM_PROMPT.replace("<<CONSTELLATION_LIST>>", str(const_list)).replace("<<LANGUAGE>>", lang))
    
    payload = {
        "model": MODEL_NAME,  # must match the model you pulled
        "messages": [
            {
                "role": "user", 
                "content": QWEN_PROMPT
            }
        ],
        "stream": False,
        "temperature": 0.2,
        "top_p": 0.4,
        "top_k": 40,
        "seed": 42,
    }

    try:
        response = requests.post(URL, json=payload)
        response.raise_for_status()
        result = response.json()["message"]["content"]

        return ExplainerResponse(
            llm_result=result
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 


