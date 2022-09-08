from pydantic import BaseModel
from typing import Optional, List

class LoadAudio(BaseModel):
    id: Optional[int] = None
    y: List [float]
    sr: float 
    
    class Config:
        orm_mode = True  