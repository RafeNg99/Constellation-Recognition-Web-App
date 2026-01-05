import io
import os
import cv2
import base64
import numpy as np
from ultralytics import YOLO
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from pydantic import BaseModel
from PIL import Image

app = FastAPI()



class DetectorResponse(BaseModel):
    yolo_img_result: str
    yolo_class_result: List[str]

class ExplainerResponse(BaseModel):
    llm_result: str




LLM_PROMPT = """
"""

constellation_detectoer_model = YOLO("/app/yolo11-best-model.pt")

def file_to_pil(uploaded_file: UploadFile):
    image_bytes = uploaded_file.file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return img

def encoded_img(img: np.array) -> str:
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")


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
async def constellation_explainer(params: List[str] = Query(...)):
    try:


        return

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 





