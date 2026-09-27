"""
Progressive normalization experiment runner.
Tests multiple stages of normalization to observe the recall progression.
"""
import re
import unicodedata
import string
import time
import pandas as pd
import numpy as np

# Load benchmark harness
from benchmark_normalization import load_data, evaluate_recall

# -------------------------------------------------------------
# Stage 1: Baseline (from notebook)
# -------------------------------------------------------------
def norm_v1_baseline(text):
    if not text:
        return ""
    text = str(text).lower()
    return re.sub(r'[^a-z0-9]', '', text)

# -------------------------------------------------------------
# Stage 2: Unicode NFKD + Accent Stripping + Basic URL/Punctuation
# -------------------------------------------------------------
def strip_accents(text):
    if not text:
        return ""
    # Unicode NFKD decomposition
    text = unicodedata.normalize('NFKD', str(text))
    # Strip non-spacing marks (accents)
    return "".join(c for c in text if not unicodedata.combining(c))

URL_PATTERN = re.compile(r'\b(https?://|www\.)\S+|\S+\.(com|org|net|in|co|io|ai|biz|info|us|fr)\b', re.IGNORECASE)

def norm_v2_accents_and_urls(text):
    if not text:
        return ""
    text = strip_accents(text).lower()
    # Remove web domain endings like .com, .in, etc.
    text = re.sub(r'\.(com|in|co\.in|org|net|io|ai|biz|info|us|fr|gov)\b', ' ', text)
    text = re.sub(r'\b(www|http|https)\b', ' ', text)
    # Replace symbols with words
    text = text.replace('&', ' and ').replace('+', ' plus ')
    # Clean non-alphanumeric
    text = re.sub(r'[^a-z0-9]', '', text)
    return text

# -------------------------------------------------------------
# Stage 3: Legal Entity Suffix & Prefix Stripping
# -------------------------------------------------------------
LEGAL_TERMS = {
    # US / UK / Global
    'llc', 'llp', 'inc', 'incorporated', 'corp', 'corporation',
    'ltd', 'limited', 'pvt', 'private', 'pvtltd', 'privatelimited',
    'co', 'company', 'plc', 'pc', 'pa', 'gmbh',
    # French (for test set)
    'sarl', 'sa', 'sas', 'sasu', 'snc', 'sci', 'eurl',
    # Common generic business descriptors often attached/detached
    'services', 'service', 'solutions', 'solution', 'holdings', 'holding',
    'group', 'enterprises', 'enterprise', 'associates', 'associate',
    'consulting', 'consultants', 'consultant', 'international', 'mgmt', 'management'
}

# Regex to strip legal suffixes and standalone tokens
LEGAL_REGEX = re.compile(r'\b(' + '|'.join(sorted(LEGAL_TERMS, key=len, reverse=True)) + r')\b', re.IGNORECASE)

def norm_v3_legal_stripped(text):
    if not text:
        return ""
    text = strip_accents(text).lower()
    text = re.sub(r'\.(com|in|co\.in|org|net|io|ai|biz|info|us|fr|gov)\b', ' ', text)
    text = re.sub(r'\b(www|http|https)\b', ' ', text)
    text = text.replace('&', ' and ').replace('+', ' plus ')
    # Replace punctuation with space to isolate words
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # Remove legal terms
    tokens = [t for t in text.split() if t not in LEGAL_TERMS]
    return "".join(tokens)

# -------------------------------------------------------------
# Stage 4: Word-Sorted Normalized Name (neutralizes word order variations)
# -------------------------------------------------------------
def norm_v4_sorted_tokens(text):
    if not text:
        return ""
    text = strip_accents(text).lower()
    text = re.sub(r'\.(com|in|co\.in|org|net|io|ai|biz|info|us|fr|gov)\b', ' ', text)
    text = re.sub(r'\b(www|http|https)\b', ' ', text)
    text = text.replace('&', ' and ').replace('+', ' plus ')
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = sorted(t for t in text.split() if t not in LEGAL_TERMS)
    return " ".join(tokens)

# -------------------------------------------------------------
# Stage 5: Indic Transliteration Mapping (Devanagari, Tamil, etc.)
# -------------------------------------------------------------
# Basic high-frequency Indic character map for Devanagari & Tamil transliteration
INDIC_VOWELS_CONSONANTS = {
    # Devanagari
    'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'ng',
    'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ञ': 'ny',
    'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n',
    'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
    'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm',
    'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v', 'श': 'sh', 'ष': 'sh', 'स': 's', 'ह': 'h',
    'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ee', 'उ': 'u', 'ऊ': 'oo', 'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au',
    'ा': 'a', 'ि': 'i', 'ी': 'ee', 'ु': 'u', 'ू': 'oo', 'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au', '्': '',
    # Tamil
    'க': 'k', 'ங': 'ng', 'ச': 's', 'ஞ': 'ny', 'ட': 't', 'ண': 'n', 'த': 't', 'ந': 'n',
    'ப': 'p', 'ம': 'm', 'ய': 'y', 'ர': 'r', 'ல': 'l', 'வ': 'v', 'ழ': 'zh', 'ள': 'l',
    'ற': 'r', 'ன': 'n', 'ஜ': 'j', 'ஷ': 'sh', 'ஸ': 's', 'ஹ': 'h',
    'அ': 'a', 'ஆ': 'aa', 'இ': 'i', 'ஈ': 'ee', 'உ': 'u', 'ஊ': 'oo', 'எ': 'e', 'ஏ': 'ee', 'ஐ': 'ai', 'ஒ': 'o', 'ஓ': 'oo', 'ஔ': 'au',
    'ா': 'a', 'ி': 'i', 'ீ': 'ee', 'ு': 'u', 'ூ': 'oo', 'ெ': 'e', 'ே': 'ee', 'ை': 'ai', 'ொ': 'o', 'ோ': 'oo', 'ௌ': 'au', '்': ''
}

def transliterate_indic(text):
    out = []
    for ch in text:
        if ch in INDIC_VOWELS_CONSONANTS:
            out.append(INDIC_VOWELS_CONSONANTS[ch])
        else:
            out.append(ch)
    return "".join(out)

def norm_v5_multilingual_sorted(text):
    if not text:
        return ""
    text = transliterate_indic(str(text))
    text = strip_accents(text).lower()
    text = re.sub(r'\.(com|in|co\.in|org|net|io|ai|biz|info|us|fr|gov)\b', ' ', text)
    text = re.sub(r'\b(www|http|https)\b', ' ', text)
    text = text.replace('&', ' and ').replace('+', ' plus ')
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = sorted(t for t in text.split() if t not in LEGAL_TERMS)
    return " ".join(tokens)


if __name__ == "__main__":
    # Test on a 100,000 S1 sample for rapid iteration
    s1, s2, s3, gt = load_data(sample_size=100000, seed=42)
    
    print("\n--- Benchmarking Normalization Strategies ---")
    
    # 1. Baseline
    s1["v1"] = s1["business_name"].apply(norm_v1_baseline)
    s2["v1"] = s2["business_name"].apply(norm_v1_baseline)
    s3["v1"] = s3["business_name"].apply(norm_v1_baseline)
    evaluate_recall(s1, s2, s3, gt, "v1", "v1", "v1", label="V1: Baseline Alphanumeric")
    
    # 2. Accents + URLs
    s1["v2"] = s1["business_name"].apply(norm_v2_accents_and_urls)
    s2["v2"] = s2["business_name"].apply(norm_v2_accents_and_urls)
    s3["v2"] = s3["business_name"].apply(norm_v2_accents_and_urls)
    evaluate_recall(s1, s2, s3, gt, "v2", "v2", "v2", label="V2: Accents & Domains Cleaned")
    
    # 3. Legal Stripped
    s1["v3"] = s1["business_name"].apply(norm_v3_legal_stripped)
    s2["v3"] = s2["business_name"].apply(norm_v3_legal_stripped)
    s3["v3"] = s3["business_name"].apply(norm_v3_legal_stripped)
    evaluate_recall(s1, s2, s3, gt, "v3", "v3", "v3", label="V3: Legal Suffixes Stripped")
    
    # 4. Token Sorted
    s1["v4"] = s1["business_name"].apply(norm_v4_sorted_tokens)
    s2["v4"] = s2["business_name"].apply(norm_v4_sorted_tokens)
    s3["v4"] = s3["business_name"].apply(norm_v4_sorted_tokens)
    evaluate_recall(s1, s2, s3, gt, "v4", "v4", "v4", label="V4: Legal Stripped + Token Sorted")
    
    # 5. Multilingual Transliteration + Sorted
    s1["v5"] = s1["business_name"].apply(norm_v5_multilingual_sorted)
    s2["v5"] = s2["business_name"].apply(norm_v5_multilingual_sorted)
    s3["v5"] = s3["business_name"].apply(norm_v5_multilingual_sorted)
    evaluate_recall(s1, s2, s3, gt, "v5", "v5", "v5", label="V5: Indic Translit + Legal Stripped + Sorted")
