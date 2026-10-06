import os
import time
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app import config

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
int8_model = os.path.join(base_dir, "models", "bioclip_visual_int8.onnx")
fp32_model = os.path.join(base_dir, "models", "bioclip_visual_fp32.onnx")
USE_ONNX = os.path.exists(int8_model) or os.path.exists(fp32_model)

if USE_ONNX:
    from app.onnx_classifier import get_onnx_engine
    def get_active_engine():
        return get_onnx_engine()
else:
    from app.classifier import get_engine
    def get_active_engine():
        return get_engine(device=config.MODEL_DEVICE)

@asynccontextmanager
async def lifespan(app: FastAPI):
    mode = "ONNX INT8 (Ultra-Low RAM ~200MB)" if USE_ONNX else f"PyTorch ({config.MODEL_DEVICE})"
    print(f"[Startup] Pre-loading BioCLIP TreeOfLife engine in {mode} mode...")
    get_active_engine()
    print("[Startup] BioCLIP engine ready for inference.")
    yield
    print("[Shutdown] Service shutting down.")

app = FastAPI(
    title='Nature AI - BioCLIP Species Identification API',
    description='Production-grade self-hosted biological identification API using BioCLIP / iNaturalist Tree of Life.',
    version='2.0.0',
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
        'version': '2.0.0',
        'engine': 'ONNX INT8' if USE_ONNX else 'PyTorch',
        'endpoints': {
            'identify': '/v1/identify (POST multipart/form-data)',
            'health': '/health (GET)'
        }
    }

@app.get('/health')
def health():
    return {
        'status': 'healthy',
        'engine': 'ONNX INT8' if USE_ONNX else 'PyTorch',
        'device': 'cpu' if USE_ONNX else config.MODEL_DEVICE
    }

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

        engine = get_active_engine()
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
