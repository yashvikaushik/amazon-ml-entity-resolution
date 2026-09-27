"""
High-performance Normalization Evaluation Suite.
"""
import sys
import os
import time
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Robust path setup relative to script location
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR / "src"
for p in [str(CURRENT_DIR), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from normalization import (
    normalize_name_core,
    normalize_address,
    batch_generate_keys,
    clean_string
)
from benchmark_normalization import load_data, evaluate_recall

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate normalization recall across strategies")
    parser.add_argument("--sample-size", type=int, default=10000, help="Number of S1 samples to evaluate (default: 10,000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    # Load S1 sample and corresponding ground truth
    s1, s2, s3, gt = load_data(sample_size=args.sample_size, seed=args.seed)
    
    print("\n==================================================================")
    print("      NORMALIZATION RECALL BENCHMARK EXPERIMENTS")
    print("==================================================================")
    
    # -------------------------------------------------------------
    # 1. Baseline: Raw alphanumeric without legal cleaning
    # -------------------------------------------------------------
    t0 = time.time()
    s1["raw_norm"] = [clean_string(x).replace(" ", "") for x in s1["business_name"]]
    s2["raw_norm"] = [clean_string(x).replace(" ", "") for x in s2["business_name"]]
    s3["raw_norm"] = [clean_string(x).replace(" ", "") for x in s3["business_name"]]
    print(f"\n[Generated Raw Alphanumeric in {time.time()-t0:.2f}s]")
    r1 = evaluate_recall(s1, s2, s3, gt, "raw_norm", "raw_norm", "raw_norm", label="1. Baseline Raw Name")
    
    # -------------------------------------------------------------
    # 2. Normalized Core Name (Legal stripped + Multilingual Unidecode)
    # -------------------------------------------------------------
    t0 = time.time()
    s1["name_core"] = [normalize_name_core(x) for x in s1["business_name"]]
    s2["name_core"] = [normalize_name_core(x) for x in s2["business_name"]]
    s3["name_core"] = [normalize_name_core(x) for x in s3["business_name"]]
    print(f"\n[Generated Core Name in {time.time()-t0:.2f}s]")
    r2 = evaluate_recall(s1, s2, s3, gt, "name_core", "name_core", "name_core", label="2. Core Name (Legal + Diacritics Cleaned)")
    
    # -------------------------------------------------------------
    # 3. Normalized Address (Canonicalized abbreviations + Token sorted)
    # -------------------------------------------------------------
    t0 = time.time()
    s1["addr_norm"] = [normalize_address(x) for x in s1["business_address"]]
    s2["addr_norm"] = [normalize_address(x) for x in s2["business_address"]]
    s3["addr_norm"] = [normalize_address(x) for x in s3["business_address"]]
    print(f"\n[Generated Address in {time.time()-t0:.2f}s]")
    r3 = evaluate_recall(s1, s2, s3, gt, "addr_norm", "addr_norm", "addr_norm", label="3. Normalized Address (Order Invariant)")
    
    # -------------------------------------------------------------
    # 4. Multi-Key: Core Name + Address
    # -------------------------------------------------------------
    t0 = time.time()
    s1["keys_name_addr"] = batch_generate_keys(list(s1["business_name"]), list(s1["business_address"]), list(s1["country"]), include_tokens=False, include_address=True)
    s2["keys_name_addr"] = batch_generate_keys(list(s2["business_name"]), list(s2["business_address"]), list(s2["country"]), include_tokens=False, include_address=True)
    s3["keys_name_addr"] = batch_generate_keys(list(s3["business_name"]), list(s3["business_address"]), list(s3["country"]), include_tokens=False, include_address=True)
    print(f"\n[Generated Name+Address Keys in {time.time()-t0:.2f}s]")
    r4 = evaluate_recall(s1, s2, s3, gt, "keys_name_addr", "keys_name_addr", "keys_name_addr", label="4. Multi-Key: Core Name + Address")
    
    # -------------------------------------------------------------
    # 5. Multi-Key: Core Name + Address + Distinctive Name Tokens
    # -------------------------------------------------------------
    t0 = time.time()
    s1["keys_all"] = batch_generate_keys(list(s1["business_name"]), list(s1["business_address"]), list(s1["country"]), include_tokens=True, include_address=True)
    s2["keys_all"] = batch_generate_keys(list(s2["business_name"]), list(s2["business_address"]), list(s2["country"]), include_tokens=True, include_address=True)
    s3["keys_all"] = batch_generate_keys(list(s3["business_name"]), list(s3["business_address"]), list(s3["country"]), include_tokens=True, include_address=True)
    print(f"\n[Generated Full Multi-Keys in {time.time()-t0:.2f}s]")
    r5 = evaluate_recall(s1, s2, s3, gt, "keys_all", "keys_all", "keys_all", label="5. Multi-Key: Core Name + Address + Tokens")
    
    print("\n==================================================================")
    print("                     FINAL RECALL SUMMARY")
    print("==================================================================")
    print(f"1. Baseline Raw Name               : {r1*100:.2f}%")
    print(f"2. Core Name (Cleaned)             : {r2*100:.2f}%  (+{(r2-r1)*100:.2f}%)")
    print(f"3. Normalized Address              : {r3*100:.2f}%")
    print(f"4. Core Name + Address             : {r4*100:.2f}%  (+{(r4-r1)*100:.2f}%)")
    print(f"5. Core Name + Address + Tokens    : {r5*100:.2f}%  (+{(r5-r1)*100:.2f}%)")
    print("==================================================================")
