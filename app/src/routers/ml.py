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
# from schemas.schema import LoadAudio

router = APIRouter()

# @router.post("/files/")
# def mfcc_enpoint(audio: LoadAudio):  
#   feature = mfcc(y=audio.y, sr=audio.sr)

#   for data in range(feature.shape[0]):
#     value = np.mean(data)  

#   return value


@router.post("/files/")
def mfcc_enpoint(audio: bytes = File(...)):  
  y, sr = load(BytesIO(audio))
  feature = mfcc(y=y, sr=sr)
  logger.info(feature)

  # for data in range(feature.shape[0]):
  #   value = np.mean(data)  

  return feature[0]
