"""
Fast normalization & blocking recall benchmark harness.
"""
import time
import re
import pandas as pd
import numpy as np

def load_data(sample_size=None, seed=42):
    print("Loading datasets...")
    t0 = time.time()
    import os
    if os.path.exists("student_resource/dataset/train"):
        BASE = "student_resource/dataset/train"
    else:
        BASE = "dataset/train"
    
    s1 = pd.read_csv(f"{BASE}/train_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
    s2 = pd.read_csv(f"{BASE}/train_source2.tsv", sep="\t", dtype=str, keep_default_na=False)
    s3 = pd.read_csv(f"{BASE}/train_source3.tsv", sep="\t", dtype=str, keep_default_na=False)
    gt = pd.read_csv(f"{BASE}/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
    
    if sample_size and sample_size < len(gt):
        np.random.seed(seed)
        sampled_s1_ids = set(np.random.choice(gt[gt["matched_entity_ids"] != ""]["source1_entity_id"], size=sample_size, replace=False))
        gt = gt[gt["source1_entity_id"].isin(sampled_s1_ids)].copy()
        s1 = s1[s1["entity_id"].isin(sampled_s1_ids)].copy()
        print(f"Sampled {len(s1)} S1 records for fast evaluation.")
        
    print(f"Loaded in {time.time() - t0:.2f}s | S1: {len(s1):,}, S2: {len(s2):,}, S3: {len(s3):,}, GT: {len(gt):,}")
    return s1, s2, s3, gt


def evaluate_recall(s1, s2, s3, gt, s1_key_col, s2_key_col, s3_key_col, label=""):
    """
    Computes candidate recall by matching keys.
    Can handle single key (str) or multiple keys (list/set of str).
    """
    t0 = time.time()
    
    # Build lookups
    # If key is list/set, map id -> set(keys)
    # If key is str, map id -> str
    is_multi_s1 = isinstance(s1[s1_key_col].iloc[0], (list, set, tuple)) if len(s1) > 0 else False
    is_multi_s2 = isinstance(s2[s2_key_col].iloc[0], (list, set, tuple)) if len(s2) > 0 else False
    is_multi_s3 = isinstance(s3[s3_key_col].iloc[0], (list, set, tuple)) if len(s3) > 0 else False
    
    s1_lookup = dict(zip(s1["entity_id"], s1[s1_key_col]))
    s2_lookup = dict(zip(s2["entity_id"], s2[s2_key_col]))
    s3_lookup = dict(zip(s3["entity_id"], s3[s3_key_col]))
    
    total_true_matches = 0
    retrieved_matches = 0
    
    # Count matches
    for s1_id, matched_str in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        if not matched_str:
            continue
            
        s1_keys = s1_lookup.get(s1_id)
        if s1_keys is None:
            continue
            
        if not is_multi_s1:
            s1_key_set = {s1_keys} if s1_keys else set()
        else:
            s1_key_set = set(s1_keys) if s1_keys else set()
            
        # Ignore empty key string in set
        s1_key_set.discard("")
        if not s1_key_set:
            matched_list = matched_str.split(",")
            total_true_matches += len(matched_list)
            continue
            
        matched_list = matched_str.split(",")
        for match_id in matched_list:
            total_true_matches += 1
            
            if match_id.startswith("S2-"):
                cand_keys = s2_lookup.get(match_id)
            else:
                cand_keys = s3_lookup.get(match_id)
                
            if cand_keys is None:
                continue
                
            if isinstance(cand_keys, (list, set, tuple)):
                cand_set = set(cand_keys)
            else:
                cand_set = {cand_keys} if cand_keys else set()
            cand_set.discard("")
            
            # If intersection > 0, we retrieved this match!
            if s1_key_set.intersection(cand_set):
                retrieved_matches += 1
                
    recall = retrieved_matches / total_true_matches if total_true_matches > 0 else 0.0
    elapsed = time.time() - t0
    print(f"[{label}] True Matches: {total_true_matches:,} | Retrieved: {retrieved_matches:,} | Recall: {recall*100:.2f}% (Time: {elapsed:.2f}s)")
    return recall
