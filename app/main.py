import io
import os
import base64
from typing import List, Optional, Dict, Any

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()



class DetectorResponse(BaseModel):
    yolo_result:List[Any]

class ExplainerResponse(BaseModel):
    llm_result:List[Any]




llm_prompt = """
"""





@app.post("/constellation_detector")
async def constellation_detector():





    return 



@app.post("/constellation_explainer")
async def constellation_explainer():







    return





