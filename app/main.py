import time
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.classifier import get_engine
from app import config

@asynccontextmanager
async def lifespan(app: FastAPI):
    print('[Startup] Pre-loading BioCLIP TreeOfLife model...')
    get_engine(device=config.MODEL_DEVICE)
    print('[Startup] BioCLIP engine ready for inference.')
    yield
    print('[Shutdown] Service shutting down.')

app = FastAPI(
    title='Nature AI - BioCLIP Species Identification API',
    description='Production-grade self-hosted biological identification API using BioCLIP / iNaturalist Tree of Life.',
    version='1.0.0',
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.get('/')
def root():
    return {
        'status': 'online',
        'service': 'Nature AI BioCLIP Species Identification Engine',
        'version': '1.0.0',
        'endpoints': {
            'identify': '/v1/identify (POST multipart/form-data)',
            'health': '/health (GET)'
        }
    }

@app.get('/health')
def health():
    return {'status': 'healthy', 'device': config.MODEL_DEVICE}

@app.post('/v1/identify')
async def identify(
    file: UploadFile = File(...),
    category: Optional[str] = Form(None)
):
    try:
        t0 = time.time()
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail='Empty image file provided')

        engine = get_engine(device=config.MODEL_DEVICE)
        predictions = engine.predict(contents, k=5)
        inference_time_ms = round((time.time() - t0) * 1000, 2)

        if not predictions:
            raise HTTPException(status_code=404, detail='No species match found')

        top_match = predictions[0]

        return {
            'status': 'success',
            'inference_time_ms': inference_time_ms,
            'top_match': top_match,
            'species_name': top_match['common_name'],
            'scientific_name': top_match['scientific_name'],
            'confidence': top_match['score'],
            'formatted_output': top_match['formatted_name'],
            'candidates': predictions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('app.main:app', host=config.HOST, port=config.PORT, reload=False)
