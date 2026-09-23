import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.classifier import get_engine

def run_tests():
    print('=================================================================')
    print('   Nature AI - BioCLIP / iNaturalist Local Verification Test     ')
    print('=================================================================')
    
    engine = get_engine(device='cpu')
    
    test_cases = [
        ('Plant (Yarrow)', 'test_images/plant_yarrow.webp', 'Achillea millefolium'),
        ('Insect (Honeybee)', 'test_images/insect_honeybee.webp', 'Apis mellifera'),
        ('Bird (Mallard)', 'test_images/bird_mallard.webp', 'Anas platyrhynchos'),
        ('Fish (Bluegill)', 'test_images/fish_bluegill.webp', 'Lepomis macrochirus'),
    ]
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    for label, rel_path, expected_sci in test_cases:
        full_path = os.path.join(base_dir, rel_path)
        if not os.path.exists(full_path):
            print(f'[SKIPPED] File not found: {full_path}')
            continue
            
        print(f'\n>>> Testing: {label}')
        print(f'    File: {rel_path}')
        print(f'    Expected Scientific Name: {expected_sci}')
        
        t0 = time.time()
        results = engine.predict(full_path, k=5)
        dt = time.time() - t0
        
        top = results[0]
        c_name = top['common_name']
        s_name = top['scientific_name']
        conf = top['confidence_percent']
        fmt = top['formatted_name']
        
        print(f'    Time taken: {dt:.2f}s')
        print(f'    Top Match: {c_name} ({s_name})')
        print(f'    Confidence: {conf} %')
        print(f'    Formatted Output: \"{fmt}\"')
        print('    Taxonomic Lineage:')
        for rank, val in top['taxonomy'].items():
            if val:
                print(f'      - {rank.capitalize()}: {val}')
        
        print('    Top 3 Candidates:')
        for idx, cand in enumerate(results[:3], start=1):
            cand_c = cand['common_name']
            cand_s = cand['scientific_name']
            cand_conf = cand['confidence_percent']
            print(f'      {idx}. {cand_c} ({cand_s}) - {cand_conf}%')

if __name__ == '__main__':
    run_tests()
