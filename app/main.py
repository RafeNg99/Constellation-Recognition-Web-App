import io
import os
import base64
from ultralytics import YOLO
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from pydantic import BaseModel
from PIL import Image

app = FastAPI()



class DetectorResponse(BaseModel):
    yolo_result: List[Any]

class ExplainerResponse(BaseModel):
    llm_result: List[Any]




LLM_PROMPT = """
"""

constellation_detectoer_model = YOLO("yolo11-best-model.pt")

def file_to_pil(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    return img



@app.post("/constellation_detector", response_model=DetectorResponse)
async def constellation_detector(files: List[UploadFile] = File(...)):
    try:
        img = file_to_pil(files[0])
        results = constellation_detectoer_model(img)

        return

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 



@app.post("/constellation_explainer", response_model=ExplainerResponse)
async def constellation_explainer(params: str = Query(...)):
    try:


        return

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 





