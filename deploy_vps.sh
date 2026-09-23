#!/bin/bash
# VPS Setup and Start Script for Nature AI BioCLIP Backend
set -e

echo "=== Updating system and installing dependencies ==="
sudo apt-get update && sudo apt-get install -y python3-pip python3-venv git curl

echo "=== Setting up Python Virtual Environment ==="
python3 -m venv venv
source venv/bin/activate

echo "=== Installing Python Requirements ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Starting BioCLIP FastAPI Service ==="
uvicorn app.main:app --host 0.0.0.0 --port 8000
