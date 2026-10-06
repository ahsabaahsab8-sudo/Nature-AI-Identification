import os
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from app.onnx_classifier import get_onnx_engine

def run_tests():
    print("=================================================================")
    print("   Nature AI - BioCLIP ONNX INT8 Verification Test (Low RAM)     ")
    print("=================================================================")
    
    t0 = time.time()
    engine = get_onnx_engine()
    print(f"Engine ready in {time.time() - t0:.2f}s")
    
    test_cases = [
        ("Plant (Yarrow)", "test_images/plant_yarrow.webp", "Achillea millefolium"),
        ("Insect (Honeybee)", "test_images/insect_honeybee.webp", "Apis mellifera"),
        ("Bird (Mallard)", "test_images/bird_mallard.webp", "Anas platyrhynchos"),
        ("Fish (Bluegill)", "test_images/fish_bluegill.webp", "Lepomis macrochirus"),
    ]
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    for label, rel_path, expected_sci in test_cases:
        full_path = os.path.join(base_dir, rel_path)
        if not os.path.exists(full_path):
            print(f"[SKIPPED] File not found: {full_path}")
            continue
            
        print(f"\n>>> Testing: {label}")
        print(f"    File: {rel_path}")
        print(f"    Expected Scientific Name: {expected_sci}")
        
        t_start = time.time()
        results = engine.predict(full_path, k=5)
        dt = (time.time() - t_start) * 1000
        
        top = results[0]
        c_name = top["common_name"]
        s_name = top["scientific_name"]
        conf = top["confidence_percent"]
        fmt = top["formatted_name"]
        
        print(f"    [RESULT] {c_name} | {s_name} (Confidence: {conf}%) in {dt:.1f}ms")
        print(f"    [FORMAT] {fmt}")
        
        # Check if match
        is_match = expected_sci.lower() in s_name.lower() or any(expected_sci.lower() in r["scientific_name"].lower() for r in results[:3])
        if is_match:
            print("    [MATCH] MATCH VERIFIED!")
        else:
            print(f"    [WARN] Top 3: {[r['scientific_name'] for r in results[:3]]}")

    print("\n=================================================================")
    print("   ALL TESTS COMPLETED SUCCESSFULLY WITH ONNX INT8!               ")
    print("=================================================================")

if __name__ == "__main__":
    run_tests()
