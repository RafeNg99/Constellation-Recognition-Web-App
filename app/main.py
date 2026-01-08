import io
import os
import cv2
import json
import base64
import requests
import numpy as np
from ultralytics import YOLO
from typing import List, Optional, Dict, Any

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
MODEL_NAME = "qwen3:4b"
URL = "http://localhost:11434/api/chat"


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

def llm_prompt(constellation_list, language):
    LLM_PROMPT = f"""You are an astronomy knowledge assistant.

Input:
- Detected constellations: {constellation_list}
- Output language: {language}

Instructions:
1. If the detected constellations list is empty, respond ONLY with:
   "No constellation found, please upload another image."
   in the specified output language.
2. If constellations are present, for EACH constellation provide:
   - History
   - Cultural significance
   - Notable features
3. Output must be plain text (no markdown, no emojis, no bullet symbols).
4. Do NOT invent constellations that are not in the detected list.
5. Keep the explanation short, clear, concise, and factual.
6. Write everything strictly in the specified output language.
"""

    return LLM_PROMPT


# =========================
# API ENDPOINTS
# =========================
@app.post("/constellation_detector", response_model=DetectorResponse)
def constellation_detector(files: List[UploadFile] = File(...)):
    try:
        img = file_to_pil(files[0])
        results = constellation_detectoer_model(img)

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
    QWEN_PROMPT = llm_prompt(const_list, lang)
    payload = {
        "model": MODEL_NAME,
        "prompt": QWEN_PROMPT,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "top_p": 0.4,
            "top_k": 40,
            "seed": 42
        }
    }

    try:
        response = requests.post(URL, json=payload)
        response.raise_for_status()
        result = response.json()["response"]

        return ExplainerResponse(
            llm_result=result
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 





