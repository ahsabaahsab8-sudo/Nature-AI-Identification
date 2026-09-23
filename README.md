# Nature AI - BioCLIP / iNaturalist Species Identification API

Production-ready, self-hosted biological computer vision API powered by **BioCLIP** (CVPR 2024) and the **Tree of Life** taxonomy (covering over **454,000+ species** across plants, insects, fish, and birds).

---

## 📁 Project Structure

`
.
├── app/
│   ├── __init__.py
│   ├── config.py           # Configuration (Host, Port, Device)
│   ├── classifier.py       # BioCLIP TreeOfLifeClassifier Engine
│   └── main.py             # FastAPI REST Server
├── test_images/            # Verification images (Plants, Insects, Birds, Fish)
├── test_cli.py             # Local CLI verification test
├── requirements.txt        # Python dependencies
├── Dockerfile              # Production Docker container setup
├── deploy_vps.sh           # 1-Click setup script for Ubuntu/Debian VPS
└── README.md
`

---

## 🚀 Quickstart

### 1. Local CLI Test
`ash
python test_cli.py
`

### 2. Start the FastAPI Server
`ash
uvicorn app.main:app --host 0.0.0.0 --port 8000
`
Swagger UI available at: http://localhost:8000/docs

---

## 📡 API Schema

### **POST /v1/identify**
* **Request**: multipart/form-data with key ile (Image)
* **Response**:
`json
{
  status: success,
  inference_time_ms: 274.06,
  top_match: {
    common_name: bloodwort,
    scientific_name: Achillea millefolium,
    score: 0.3537,
    confidence_percent: 35.37,
    formatted_name: bloodwort (Achillea millefolium) (bloodwort),
    taxonomy: {
      kingdom: Plantae,
      phylum: Tracheophyta,
      class: Magnoliopsida,
      order: Asterales,
      family: Asteraceae,
      genus: Achillea,
      species: Achillea millefolium
    }
  },
  species_name: bloodwort,
  scientific_name: Achillea millefolium,
  confidence: 0.3537,
  formatted_output: bloodwort (Achillea millefolium) (bloodwort)
}
`

---

## ☁️ VPS Deployment (24/7 Uptime)

### Method 1: Using deploy script
`ash
git clone https://github.com/ahsabaahsab8-sudo/Nature-AI-Identification.git
cd Nature-AI-Identification
chmod +x deploy_vps.sh
./deploy_vps.sh
`

### Method 2: Docker
`ash
docker build -t bioclip-api .
docker run -d --name bioclip -p 8000:8000 --restart always bioclip-api
`