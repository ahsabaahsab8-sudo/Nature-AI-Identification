#!/bin/bash
# VPS Setup and Start Script for Nature AI BioCLIP Backend (ONNX INT8 / Low-RAM Mode)
set -e

echo "=== 1. Checking / Creating 2GB Swap Memory (Protects 1GB VPS against OOM) ==="
if [ $(free -m | awk '/^Swap:/ {print $2}') -eq 0 ]; then
    echo "Creating 2GB swapfile..."
    sudo fallocate -l 2G /swapfile || sudo dd if=/dev/zero of=/swapfile bs=1M count=2048
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo "/swapfile none swap sw 0 0" | sudo tee -a /etc/fstab
    echo "Swap enabled successfully."
else
    echo "Swap already configured."
fi

echo "=== 2. Updating system and installing dependencies ==="
sudo apt-get update && sudo apt-get install -y python3-pip python3-venv git curl

echo "=== 3. Setting up Python Virtual Environment ==="
python3 -m venv venv
source venv/bin/activate

echo "=== 4. Installing Ultra-Lightweight ONNX Requirements (No PyTorch needed) ==="
pip install --upgrade pip
if [ -f "requirements-onnx.txt" ]; then
    pip install -r requirements-onnx.txt
else
    pip install -r requirements.txt
fi

echo "=== 5. Starting Nature AI BioCLIP Service on port 8000 ==="
uvicorn app.main:app --host 0.0.0.0 --port 8000
