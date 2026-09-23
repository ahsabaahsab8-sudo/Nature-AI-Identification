import io
import time
from typing import List, Dict, Any, Union
from PIL import Image
from bioclip import TreeOfLifeClassifier, Rank
from bioclip._constants import BIOCLIP_V1_MODEL_STR

class BioClipEngine:
    _instance = None

    def __init__(self, device: str = "cpu"):
        print(f"[BioClipEngine] Initializing TreeOfLifeClassifier on {device}...")
        t0 = time.time()
        self.classifier = TreeOfLifeClassifier(model_str=BIOCLIP_V1_MODEL_STR, device=device)
        print(f"[BioClipEngine] Initialized in {time.time() - t0:.2f}s")

    @classmethod
    def get_instance(cls, device: str = "cpu") -> "BioClipEngine":
        if cls._instance is None:
            cls._instance = BioClipEngine(device=device)
        return cls._instance

    def predict(
        self, 
        image_data: Union[str, bytes, Image.Image], 
        k: int = 5,
        target_rank: Rank = Rank.SPECIES
    ) -> List[Dict[str, Any]]:
        if isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
        elif isinstance(image_data, str):
            image = Image.open(image_data).convert("RGB")
        elif isinstance(image_data, Image.Image):
            image = image_data.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image_data)}")

        # pybioclip expects a list of images: [image]
        predictions = self.classifier.predict([image], target_rank, k=k)
        
        results = []
        for pred in predictions:
            sci_name = pred.get("species") or pred.get("genus") or pred.get("family") or "Unknown"
            common_name = pred.get("common_name") or sci_name
            score = round(float(pred.get("score", 0.0)), 4)
            
            # Format according to Nature AI standard: Common Name (Scientific Name) (Common Name)
            formatted = f"{common_name} ({sci_name}) ({common_name})"
            
            results.append({
                "common_name": common_name,
                "scientific_name": sci_name,
                "score": score,
                "confidence_percent": round(score * 100, 2),
                "formatted_name": formatted,
                "taxonomy": {
                    "kingdom": pred.get("kingdom"),
                    "phylum": pred.get("phylum"),
                    "class": pred.get("class"),
                    "order": pred.get("order"),
                    "family": pred.get("family"),
                    "genus": pred.get("genus"),
                    "species": pred.get("species"),
                }
            })
            
        return results

def get_engine(device: str = "cpu") -> BioClipEngine:
    return BioClipEngine.get_instance(device=device)
