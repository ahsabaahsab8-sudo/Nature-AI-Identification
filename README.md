# Nature AI - BioCLIP / iNaturalist Species Identification API

Production-ready, self-hosted biological computer vision API powered by **BioCLIP** (CVPR 2024) and the **Tree of Life** taxonomy (covering over **454,000+ species/taxa** across plants, insects, fish, and birds).

Featuring an **Ultra-Lightweight ONNX INT8 Runtime Engine** that runs within **~200 MB of RAM**—perfect for budget 1 GB VPS servers, Raspberry Pi, or free cloud containers without Out-of-Memory (OOM) crashes.

---

## ⚡ Highlights

- **454,000+ Biological Taxa / 384,490 Species**: Covers plants, insects, birds, mammals, fish, fungi, and reptiles across the entire tree of life.
- **Ultra-Low Memory Footprint (~200 MB RAM)**: Utilizes INT8 quantized ViT-B/16 visual backbone and OS memory-mapped (`mmap`) text embeddings (`txt_emb_species.npy`).
- **Blazing Fast**: Inference takes ~150 - 250 ms on standard CPU.
- **No Heavy PyTorch Required for ONNX Mode**: Deploy without downloading heavy 3GB PyTorch wheels.

---

## 📂 Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── config.py             # Configuration (Host, Port, Device)
│   ├── classifier.py         # BioCLIP PyTorch Engine
│   ├── onnx_classifier.py    # Ultra-lightweight ONNX INT8 Engine (~200MB RAM)
│   └── main.py               # FastAPI REST Server (auto-selects ONNX if present)
├── models/
│   └── bioclip_visual_int8.onnx # Quantized 84MB visual backbone model
├── test_images/              # Verification images (Plants, Insects, Birds, Fish)
├── test_onnx_cli.py          # Fast CLI test for ONNX INT8 engine
├── test_cli.py               # CLI test for PyTorch engine
├── export_onnx.py            # PyTorch to ONNX INT8 exporter
├── requirements-onnx.txt     # Minimal dependencies for 1GB VPS (ONNX mode)
├── requirements.txt          # Full PyTorch dependencies
├── Dockerfile                # Production Docker container setup
├── deploy_vps.sh             # 1-Click setup script for VPS (with swap protection)
└── README.md
```

---

## 🚀 Quickstart (Ultra-Low RAM / ONNX Mode)

### 1. Install Dependencies
```bash
pip install -r requirements-onnx.txt
```

### 2. Verify with CLI Test
```bash
python test_onnx_cli.py
```

### 3. Start the FastAPI Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
API Documentation available at: `http://localhost:8000/docs`

---

## 🌐 API Schema

### **POST /v1/identify**
* **Request**: `multipart/form-data` with file key `file` (Image)
* **Response**:
```json
{
  "status": "success",
  "inference_time_ms": 152.0,
  "top_match": {
    "common_name": "Mallard",
    "scientific_name": "Anas platyrhynchos",
    "score": 0.978,
    "confidence_percent": 97.8,
    "formatted_name": "Mallard (Anas platyrhynchos) (Mallard)",
    "taxonomy": {
      "kingdom": "Animalia",
      "phylum": "Chordata",
      "class": "Aves",
      "order": "Anseriformes",
      "family": "Anatidae",
      "genus": "Anas",
      "species": "Anas platyrhynchos"
    }
  },
  "species_name": "Mallard",
  "scientific_name": "Anas platyrhynchos",
  "confidence": 0.978,
  "formatted_output": "Mallard (Anas platyrhynchos) (Mallard)",
  "candidates": [ ... ]
}
```

---

## 🖥️ VPS Deployment (1 GB RAM Ready)

### 1-Click Setup Script:
```bash
git clone https://github.com/ahsabaahsab8-sudo/Nature-AI-Identification.git
cd Nature-AI-Identification
chmod +x deploy_vps.sh
./deploy_vps.sh
```
The deploy script automatically provisions a 2 GB swap file to protect the VPS from any memory spikes, installs `requirements-onnx.txt`, and boots the service.
