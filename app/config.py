import os

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
MODEL_DEVICE = os.getenv("MODEL_DEVICE", "cpu")  # 'cpu' or 'cuda'
CACHE_DIR = os.getenv("CACHE_DIR", os.path.join(os.path.dirname(__file__), "..", "models_cache"))
