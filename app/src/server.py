from fastapi import FastAPI
from routers import ml


app = FastAPI(title='API classificação de instrumentos')
app.include_router(ml.router)
