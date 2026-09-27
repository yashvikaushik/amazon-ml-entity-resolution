"""
Experiment 2: Multi-Representation Normalization (Name + Address + Token Inverted Index).
Tests how multi-representation normalization drives recall toward >90%.
"""
import re
import unicodedata
import string
import time
import pandas as pd
import numpy as np
from unidecode import unidecode
from benchmark_normalization import load_data, evaluate_recall

# -------------------------------------------------------------
# Comprehensive Name & Address Normalizers
# -------------------------------------------------------------

LEGAL_STOPWORDS = {
    # US / UK / Global
    'llc', 'llp', 'inc', 'incorporated', 'corp', 'corporation',
    'ltd', 'limited', 'pvt', 'private', 'pvtltd', 'privatelimited',
    'co', 'company', 'plc', 'pc', 'pa', 'gmbh',
    # French (for test set)
    'sarl', 'sa', 'sas', 'sasu', 'snc', 'sci', 'eurl',
    # Common generic business descriptors
    'services', 'service', 'solutions', 'solution', 'holdings', 'holding',
    'group', 'enterprises', 'enterprise', 'associates', 'associate',
    'consulting', 'consultants', 'consultant', 'international', 'mgmt', 'management',
    'center', 'centre', 'the', 'and', 'in', 'of', 'for', 'at', 'by'
}

ADDRESS_ABBREVIATIONS = {
    r'\bstreet\b': 'st',
    r'\bsaint\b': 'st',
    r'\broad\b': 'rd',
    r'\bavenue\b': 'ave',
    r'\bdrive\b': 'dr',
    r'\bboulevard\b': 'blvd',
    r'\blane\b': 'ln',
    r'\bcourt\b': 'ct',
    r'\bparkway\b': 'pkwy',
    r'\bhighway\b': 'hwy',
    r'\bterrace\b': 'ter',
    r'\bplace\b': 'pl',
    r'\bcircle\b': 'cir',
    r'\bapartment\b': 'apt',
    r'\bsuite\b': 'ste',
    r'\bbuilding\b': 'bldg',
    r'\bfloor\b': 'fl',
    r'\bnumber\b': 'no',
    r'\bplot\b': 'plot',
    r'\bdoor\b': 'door',
    r'\bnorth\b': 'n',
    r'\bsouth\b': 's',
    r'\beast\b': 'e',
    r'\bwest\b': 'w',
    r'\bnortheast\b': 'ne',
    r'\bnorthwest\b': 'nw',
    r'\bsoutheast\b': 'se',
    r'\bsouthwest\b': 'sw',
}

def clean_text(text):
    if not text:
        return ""
    # Fast transliteration + accent stripping
    text = unidecode(str(text))
    text = text.lower()
    # Strip URL domains
    text = re.sub(r'\.(com|in|co\.in|org|net|io|ai|biz|info|us|fr|gov)\b', ' ', text)
    text = re.sub(r'\b(www|http|https)\b', ' ', text)
    text = text.replace('&', ' and ').replace('+', ' plus ')
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return " ".join(text.split())

def normalize_name_core(name):
    clean = clean_text(name)
    tokens = [t for t in clean.split() if t not in LEGAL_STOPWORDS]
    return "".join(tokens)

def normalize_address(address):
    if not address or str(address).strip().lower() in {'', 'null', 'none'}:
        return ""
    clean = clean_text(address)
    for pattern, repl in ADDRESS_ABBREVIATIONS.items():
        clean = re.sub(pattern, repl, clean)
    # Token sort to be invariant to Street, City, State vs City, State, Street
    tokens = sorted(clean.split())
    return "".join(tokens)

def extract_distinctive_name_tokens(name, max_tokens=3):
    """Extract up to 3 longest / most distinctive non-legal tokens"""
    clean = clean_text(name)
    tokens = [t for t in clean.split() if t not in LEGAL_STOPWORDS and len(t) >= 4]
    # Sort by length descending (longer words are much more distinctive and less likely to collide)
    tokens = sorted(tokens, key=len, reverse=True)
    return tokens[:max_tokens]

def generate_multi_keys(row):
    """
    Generates a set of normalized blocking keys for a single record.
    If ANY key matches between S1 and S2/S3, the pair is captured!
    """
    keys = []
    
    # 1. Core Name (legal stripped)
    core_name = normalize_name_core(row["business_name"])
    if core_name:
        keys.append(f"name:{core_name}")
        
    # 2. Normalized Address (if present)
    norm_addr = normalize_address(row["business_address"])
    if norm_addr and len(norm_addr) >= 6:
        keys.append(f"addr:{norm_addr}")
        
    # 3. Distinctive Name Tokens (Inverted Index)
    dist_tokens = extract_distinctive_name_tokens(row["business_name"], max_tokens=2)
    for tok in dist_tokens:
        # Prefix with country to ensure no cross-country collisions
        country = str(row.get("country", "")).strip().lower()
        keys.append(f"tok:{country}:{tok}")
        
    return keys


if __name__ == "__main__":
    s1, s2, s3, gt = load_data(sample_size=100000, seed=42)
    
    print("\n--- Testing Multi-Representation Normalization Keys ---")
    
    # Test 1: Name Core only
    t0 = time.time()
    s1["name_core"] = s1["business_name"].apply(normalize_name_core)
    s2["name_core"] = s2["business_name"].apply(normalize_name_core)
    s3["name_core"] = s3["business_name"].apply(normalize_name_core)
    print(f"Computed Name Core in {time.time()-t0:.2f}s")
    evaluate_recall(s1, s2, s3, gt, "name_core", "name_core", "name_core", label="Key 1: Core Name")
    
    # Test 2: Address only
    t0 = time.time()
    s1["norm_addr"] = s1["business_address"].apply(normalize_address)
    s2["norm_addr"] = s2["business_address"].apply(normalize_address)
    s3["norm_addr"] = s3["business_address"].apply(normalize_address)
    print(f"Computed Address in {time.time()-t0:.2f}s")
    evaluate_recall(s1, s2, s3, gt, "norm_addr", "norm_addr", "norm_addr", label="Key 2: Normalized Address")
    
    # Test 3: Multi-Key Union (Core Name + Address + Distinctive Tokens)
    t0 = time.time()
    s1["multi_keys"] = s1.apply(generate_multi_keys, axis=1)
    s2["multi_keys"] = s2.apply(generate_multi_keys, axis=1)
    s3["multi_keys"] = s3.apply(generate_multi_keys, axis=1)
    print(f"Computed Multi-Keys in {time.time()-t0:.2f}s")
    evaluate_recall(s1, s2, s3, gt, "multi_keys", "multi_keys", "multi_keys", label="Key 3: Multi-Key Union (Name + Address + Tokens)")
