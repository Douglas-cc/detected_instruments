import numpy as np

from fastapi import (
  APIRouter,
  File,
  UploadFile
)

from io import BytesIO
from librosa import load
from librosa.feature import mfcc
from loguru import logger


router = APIRouter()


# @router.post("/files/")
# async def create_file(file: bytes = File()):
#   return {"file_size": len(file)}


# @router.post("/uploadfile/")
# async def create_upload_file(file: UploadFile):
#   return {"filename": file.filename}
  
  

@router.post("/files/")
async def create_file(file: bytes = File()):
  audio = BytesIO(file) 
    
  y, sr = load(audio)
  feature = mfcc(y=y, sr=sr)
  
  # for data in range(feature.shape[0]):
  #   value = np.mean(data)  
  
  logger.info(feature)
  
  return feature.tolist()

