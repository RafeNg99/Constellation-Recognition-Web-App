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
from app.logger import get_logger

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
    result_logger = get_logger()
    try:
        img = file_to_pil(files[0])
        results = constellation_detectoer_model.predict(img, imgsz=640, conf=0.4)

        class_name_list = []

        boxes = results[0].boxes
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = constellation_detectoer_model.names[class_id]
            class_name_list.append(class_name)

        result_logger.log_info(f"Filename: {files[0].filename}")
        result_logger.log_info(f"Detected Constellations: {class_name_list}")
            
        img_result = encoded_img(results[0].plot())

        if img_result is None:
            result_logger.log_error("ValueError: YOLO returned no plot image")
            raise ValueError("YOLO returned no plot image\n\n" + "-" * 100 + "\n")

        return DetectorResponse(
            yolo_img_result=img_result,
            yolo_class_result=class_name_list
        )

    except Exception as e:
        result_logger.log_error(f"HTTPException 500: {e}\n\n" + "-" * 100 + "\n")
        raise HTTPException(status_code=500, detail=str(e)) 


@app.post("/constellation_explainer", response_model=ExplainerResponse)
async def constellation_explainer(const_list: List[str] = Query(...), lang: str = Query(...)):
    result_logger = get_logger()
    QWEN_PROMPT = (LLM_PROMPT.replace("<<CONSTELLATION_LIST>>", str(const_list)).replace("<<LANGUAGE>>", lang))
    
    payload = {
        "model": MODEL_NAME,  
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

        result_logger.log_info(f"Langauge: {lang}")
        result_logger.log_info(f"LLM Result: {result}\n\n" + "-" * 100 + "\n")

        return ExplainerResponse(
            llm_result=result
        )

    except Exception as e:
        result_logger.log_error(f"HTTPException 500: {e}\n\n" + "-" * 100 + "\n")
        raise HTTPException(status_code=500, detail=str(e)) 


