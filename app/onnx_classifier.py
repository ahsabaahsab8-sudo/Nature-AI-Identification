import os
import io
import json
import time
from typing import List, Dict, Any, Union
import numpy as np
from PIL import Image
import onnxruntime as ort
from huggingface_hub import hf_hub_download

class OnnxBioClipEngine:
    _instance = None

    def __init__(self, model_path: str = None):
        print("[OnnxBioClipEngine] Initializing ultra-lightweight ONNX runtime engine...")
        t0 = time.time()
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if model_path is None:
            # Prefer INT8 quantized model for minimal RAM
            int8_path = os.path.join(base_dir, "models", "bioclip_visual_int8.onnx")
            fp32_path = os.path.join(base_dir, "models", "bioclip_visual_fp32.onnx")
            model_path = int8_path if os.path.exists(int8_path) else fp32_path
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model file not found at {model_path}. Please run export_onnx.py first.")

        # 1. Initialize ONNX Runtime Session (uses only ~30 MB RAM)
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = max(1, os.cpu_count() or 1)
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(
            model_path, 
            sess_options=sess_options,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        print(f"[OnnxBioClipEngine] Loaded ONNX model from {os.path.basename(model_path)} ({os.path.getsize(model_path)/(1024*1024):.1f} MB)")

        # 2. Memory-map species embeddings (takes ~0 MB active RAM!)
        print("[OnnxBioClipEngine] Memory-mapping Tree-of-Life 454,000+ species embeddings...")
        txt_emb_path = hf_hub_download(
            repo_id="imageomics/TreeOfLife-10M", 
            filename="embeddings/txt_emb_species.npy", 
            repo_type="dataset"
        )
        self.txt_emb = np.load(txt_emb_path, mmap_mode="r")
        print(f"[OnnxBioClipEngine] Mmapped {self.txt_emb.shape[1]:,} species embeddings (Shape: {self.txt_emb.shape})")

        # 3. Load species taxonomy labels
        txt_names_path = hf_hub_download(
            repo_id="imageomics/TreeOfLife-10M", 
            filename="embeddings/txt_emb_species.json", 
            repo_type="dataset"
        )
        with open(txt_names_path, "r", encoding="utf-8") as f:
            self.txt_names = json.load(f)
            
        print(f"[OnnxBioClipEngine] Ready in {time.time() - t0:.2f}s! (Ultra-low RAM mode)")

    @classmethod
    def get_instance(cls, model_path: str = None) -> "OnnxBioClipEngine":
        if cls._instance is None:
            cls._instance = OnnxBioClipEngine(model_path=model_path)
        return cls._instance

    @staticmethod
    def preprocess_image(image: Image.Image) -> np.ndarray:
        # Resize to 224x224
        img = image.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
        # Convert to numpy float32 [0.0, 1.0]
        arr = np.array(img, dtype=np.float32) / 255.0
        # Normalize (CLIP mean & std)
        mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
        arr = (arr - mean) / std
        # Transpose HWC -> CHW and add batch dimension -> NCHW (1, 3, 224, 224)
        arr = np.transpose(arr, (2, 0, 1))
        return np.expand_dims(arr, axis=0)

    def predict(
        self, 
        image_data: Union[str, bytes, Image.Image], 
        k: int = 5
    ) -> List[Dict[str, Any]]:
        if isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data))
        elif isinstance(image_data, str):
            image = Image.open(image_data)
        elif isinstance(image_data, Image.Image):
            image = image_data
        else:
            raise ValueError(f"Unsupported image type: {type(image_data)}")

        # 1. Preprocess
        input_tensor = self.preprocess_image(image)

        # 2. Run ONNX Visual Inference
        img_features = self.session.run(
            [self.output_name], 
            {self.input_name: input_tensor}
        )[0][0]  # Shape: (512,)

        # 3. Compute Cosine Similarity against all 454,000+ species (Memory-Mapped)
        # Both img_features and txt_emb are L2-normalized, so dot product = cosine similarity
        scores = np.dot(img_features, self.txt_emb)

        # 4. Find Top-K indices using fast argpartition
        top_k_idx = np.argpartition(scores, -k)[-k:]
        top_k_idx = top_k_idx[np.argsort(-scores[top_k_idx])]

        # Softmax on top-k logits for clean confidence percentages
        exp_scores = np.exp(scores[top_k_idx] * 100.0)
        probs = exp_scores / np.sum(exp_scores)

        results = []
        for i, idx in enumerate(top_k_idx):
            meta = self.txt_names[idx]
            # In TreeOfLife-10M txt_emb_species.json:
            # meta is [taxa_list, common_name_str]
            # taxa_list: [kingdom, phylum, class, order, family, genus, species_epithet]
            taxa_list = meta[0] if isinstance(meta, (list, tuple)) and len(meta) > 0 else []
            common_name = meta[1] if isinstance(meta, (list, tuple)) and len(meta) > 1 and meta[1] else ""

            kingdom = taxa_list[0] if len(taxa_list) > 0 else ""
            phylum = taxa_list[1] if len(taxa_list) > 1 else ""
            cls_name = taxa_list[2] if len(taxa_list) > 2 else ""
            order = taxa_list[3] if len(taxa_list) > 3 else ""
            family = taxa_list[4] if len(taxa_list) > 4 else ""
            genus = taxa_list[5] if len(taxa_list) > 5 else ""
            epithet = taxa_list[6] if len(taxa_list) > 6 else ""
            
            sci_name = f"{genus} {epithet}".strip() or genus or family or "Unknown"
            display_common = common_name or sci_name
            score = round(float(scores[idx]), 4)
            conf_pct = round(float(probs[i]) * 100.0, 2)

            formatted = f"{display_common} ({sci_name}) ({display_common})"

            results.append({
                "common_name": display_common,
                "scientific_name": sci_name,
                "score": score,
                "confidence_percent": conf_pct,
                "formatted_name": formatted,
                "taxonomy": {
                    "kingdom": kingdom,
                    "phylum": phylum,
                    "class": cls_name,
                    "order": order,
                    "family": family,
                    "genus": genus,
                    "species": sci_name
                }
            })

        return results

def get_onnx_engine(model_path: str = None) -> OnnxBioClipEngine:
    return OnnxBioClipEngine.get_instance(model_path=model_path)

