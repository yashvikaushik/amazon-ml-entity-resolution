"""
Comprehensive Pattern Analysis for the 1,000 Missed Match Sample.
Generates student_resource/v2_miss_patterns.txt
"""

import os
import sys
import re
import difflib
from pathlib import Path
from collections import defaultdict
from unidecode import unidecode
import pandas as pd

# Load sample
SAMPLE_FILE = "student_resource/v2_missed_sample.tsv"
OUTPUT_FILE = "student_resource/v2_miss_patterns.txt"

df = pd.read_csv(SAMPLE_FILE, sep="\t", keep_default_na=False)

LEGAL_TERMS = {
    'llc', 'inc', 'corp', 'ltd', 'pvt', 'limited', 'private', 'co', 'company',
    'cie', 'plc', 'gmbh', 'sa', 'sas', 'sarl', 'services', 'service', 'solutions',
    'solution', 'holdings', 'group', 'enterprises', 'associates', 'consulting',
    'partners', 'partner', 'mr', 'ms', 'dr', 'm/s'
}

ADDR_ABBRS = {
    'st', 'street', 'rd', 'road', 'dr', 'drive', 'ave', 'avenue', 'ln', 'lane',
    'ct', 'court', 'blvd', 'boulevard', 'pkwy', 'parkway', 'hwy', 'highway',
    'apt', 'apartment', 'ste', 'suite', 'fl', 'floor', 'unit', 'po', 'box',
    'nc', 'north carolina', 'ok', 'oklahoma', 'az', 'arizona', 'md', 'maryland',
    'oh', 'ohio', 'tn', 'tennessee', 'in', 'indiana', 'tx', 'texas', 'ca', 'california',
    'wb', 'west bengal', 'dl', 'delhi', 'mp', 'madhya pradesh', 'mh', 'maharashtra'
}

COMMON_STOPWORDS = {
    'st', 'street', 'rd', 'road', 'dr', 'drive', 'ave', 'avenue', 'ln', 'lane', 'ct', 'court',
    'blvd', 'boulevard', 'hwy', 'highway', 'way', 'pl', 'place', 'us', 'india', 'unit', 'apt',
    'apartment', 'ste', 'suite', 'fl', 'floor', 'no', 'box', 'po', 'north', 'south', 'east', 'west',
    'n', 's', 'e', 'w', 'null', 'nan', 'the', 'in', 'of', 'for', 'at', 'by', 'and'
}

def clean_toks(text):
    return set(re.findall(r'[a-z0-9]+', unidecode(str(text)).lower()))

def extract_hn_list(addr):
    return re.findall(r'\b\d{1,6}\b', str(addr))

def extract_city_field(addr):
    parts = str(addr).split(',')
    if len(parts) >= 2:
        return parts[-2].strip().lower()
    return ''

def has_non_ascii(s):
    return any(ord(c) >= 128 for c in str(s))

def has_domain(s):
    return bool(re.search(r'(\.(com|in|co|org|net|io|ai)\b|@|www\.|http)', str(s).lower()))


# 12 Category Definitions
categories = {
    1: {"name": "Name typo / spelling variation", "matches": []},
    2: {"name": "Name token variation / word reorder", "matches": []},
    3: {"name": "Legal suffix / prefix variation", "matches": []},
    4: {"name": "Completely different business names", "matches": []},
    5: {"name": "Missing S2/S3 address", "matches": []},
    6: {"name": "Address typo / spelling variation", "matches": []},
    7: {"name": "Address abbreviation variation", "matches": []},
    8: {"name": "House number mismatch/missing", "matches": []},
    9: {"name": "City/locality variation", "matches": []},
    10: {"name": "Transliteration / script variation", "matches": []},
    11: {"name": "Domain / website-like business name", "matches": []},
    12: {"name": "Other", "matches": []},
}

for idx, row in df.iterrows():
    s1_name = row['s1_business_name']
    m_name = row['missed_business_name']
    s1_addr = row['s1_business_address']
    m_addr = row['missed_business_address']

    s1_toks = clean_toks(s1_name)
    m_toks = clean_toks(m_name)
    s1_addr_toks = clean_toks(s1_addr)
    m_addr_toks = clean_toks(m_addr)

    c_s1_name = ' '.join(re.findall(r'[a-z0-9]+', unidecode(s1_name).lower()))
    c_m_name = ' '.join(re.findall(r'[a-z0-9]+', unidecode(m_name).lower()))
    name_ratio = difflib.SequenceMatcher(None, c_s1_name, c_m_name).ratio()

    hn1 = extract_hn_list(s1_addr)
    hn2 = extract_hn_list(m_addr)
    hn1_norm = {x.lstrip('0') for x in hn1 if x.lstrip('0')}
    hn2_norm = {x.lstrip('0') for x in hn2 if x.lstrip('0')}

    matched = False

    # 1. Name typo / spelling variation
    name_typo = False
    for t1 in s1_toks - m_toks:
        if len(t1) >= 4:
            for t2 in m_toks - s1_toks:
                if len(t2) >= 4 and difflib.SequenceMatcher(None, t1, t2).ratio() >= 0.70:
                    name_typo = True
                    break
    if name_typo or (0.65 <= name_ratio < 0.98 and s1_toks != m_toks and (s1_toks & m_toks)):
        categories[1]["matches"].append(idx)
        matched = True

    # 2. Name token variation / word reorder
    diff_toks = s1_toks ^ m_toks
    is_pure_legal = diff_toks and all(t in LEGAL_TERMS for t in diff_toks)
    if s1_toks != m_toks and (s1_toks & m_toks) and not is_pure_legal:
        categories[2]["matches"].append(idx)
        matched = True

    # 3. Legal suffix / prefix variation
    if (diff_toks and any(t in LEGAL_TERMS for t in diff_toks)) or is_pure_legal:
        categories[3]["matches"].append(idx)
        matched = True

    # 4. Completely different business names
    if not (s1_toks & m_toks) and name_ratio < 0.40:
        categories[4]["matches"].append(idx)
        matched = True

    # 5. Missing S2/S3 address
    if not m_addr.strip() or len(m_addr_toks) <= 1 or m_addr.strip().lower() in {'null', 'none', 'nan', 'india', 'us'}:
        categories[5]["matches"].append(idx)
        matched = True

    # 6. Address typo / spelling variation
    addr_diff = False
    for t1 in s1_addr_toks - m_addr_toks:
        if len(t1) >= 4 and t1 not in ADDR_ABBRS:
            for t2 in m_addr_toks - s1_addr_toks:
                if len(t2) >= 4 and t2 not in ADDR_ABBRS and difflib.SequenceMatcher(None, t1, t2).ratio() >= 0.75:
                    addr_diff = True
                    break
    if addr_diff:
        categories[6]["matches"].append(idx)
        matched = True

    # 7. Address abbreviation variation
    if any(tok in ADDR_ABBRS for tok in s1_addr_toks ^ m_addr_toks):
        categories[7]["matches"].append(idx)
        matched = True

    # 8. House number mismatch/missing
    if (hn1 and not hn2) or (hn2 and not hn1) or (hn1 and hn2 and not (hn1_norm & hn2_norm)) or (hn1 != hn2 and (hn1_norm & hn2_norm)):
        categories[8]["matches"].append(idx)
        matched = True

    # 9. City/locality variation
    city1 = extract_city_field(s1_addr)
    city2 = extract_city_field(m_addr)
    if (city1 and not city2) or (city2 and not city1) or (city1 and city2 and city1 != city2):
        categories[9]["matches"].append(idx)
        matched = True

    # 10. Transliteration / script variation
    if has_non_ascii(s1_name) or has_non_ascii(m_name) or has_non_ascii(s1_addr) or has_non_ascii(m_addr):
        categories[10]["matches"].append(idx)
        matched = True

    # 11. Domain / website-like business name
    if has_domain(s1_name) or has_domain(m_name):
        categories[11]["matches"].append(idx)
        matched = True

    # 12. Other
    if not matched:
        categories[12]["matches"].append(idx)


# Calculate Overall Metrics
same_hn_count = 0
shared_addr_tok_count = 0
meaningful_name_tok_count = 0
strong_addr_weak_name = 0
strong_name_weak_addr = 0

for _, row in df.iterrows():
    s1_name, m_name = row['s1_business_name'], row['missed_business_name']
    s1_addr, m_addr = row['s1_business_address'], row['missed_business_address']

    hn1_norm = {x.lstrip('0') for x in extract_hn_list(s1_addr) if x.lstrip('0')}
    hn2_norm = {x.lstrip('0') for x in extract_hn_list(m_addr) if x.lstrip('0')}
    if hn1_norm and hn2_norm and (hn1_norm & hn2_norm):
        same_hn_count += 1

    tok1 = {t for t in clean_toks(s1_addr) if len(t) >= 4 and t not in COMMON_STOPWORDS}
    tok2 = {t for t in clean_toks(m_addr) if len(t) >= 4 and t not in COMMON_STOPWORDS}
    if tok1 & tok2:
        shared_addr_tok_count += 1

    ntok1 = {t for t in clean_toks(s1_name) if len(t) >= 3 and t not in LEGAL_TERMS and t not in COMMON_STOPWORDS}
    ntok2 = {t for t in clean_toks(m_name) if len(t) >= 3 and t not in LEGAL_TERMS and t not in COMMON_STOPWORDS}
    if ntok1 & ntok2:
        meaningful_name_tok_count += 1

    addr_sim = float(row['address_token_overlap'])
    name_jaccard = len(ntok1 & ntok2) / len(ntok1 | ntok2) if (ntok1 | ntok2) else 0.0

    if addr_sim >= 0.40 and name_jaccard < 0.30:
        strong_addr_weak_name += 1
    if name_jaccard >= 0.50 and addr_sim < 0.30:
        strong_name_weak_addr += 1


total_samples = len(df)

# Write output file
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("=" * 80 + "\n")
    f.write("V1 CANDIDATE-MISS FAILURE PATTERN ANALYSIS (1,000 SAMPLE)\n")
    f.write("=" * 80 + "\n\n")

    f.write("SUMMARY OF 12 PRACTICAL FAILURE CATEGORIES:\n")
    f.write("-" * 80 + "\n")

    for cat_id in range(1, 13):
        cat = categories[cat_id]
        name = cat["name"]
        matches = cat["matches"]
        count = len(matches)
        pct = (count / total_samples) * 100

        f.write(f"\nCategory {cat_id}: {name}\n")
        f.write(f"  Count: {count} / {total_samples} ({pct:.1f}%)\n")
        f.write("  Representative Examples (5):\n")

        examples = matches[:5]
        for i, idx in enumerate(examples, 1):
            r = df.iloc[idx]
            f.write(f"    Example {i} [{r['source1_entity_id']} <-> {r['missed_entity_id']} ({r['source']})]:\n")
            f.write(f"      S1 Name:     {r['s1_business_name']}\n")
            f.write(f"      Missed Name: {r['missed_business_name']}\n")
            f.write(f"      S1 Address:  {r['s1_business_address']}\n")
            f.write(f"      Missed Addr: {r['missed_business_address']}\n")

    f.write("\n\n" + "=" * 80 + "\n")
    f.write("CROSS-FEATURE RETRIEVAL SIGNAL STATISTICS (1,000 SAMPLE):\n")
    f.write("=" * 80 + "\n")
    f.write(f"1. Same house number (normalized leading digits):            {same_hn_count:4d} / {total_samples} ({same_hn_count/total_samples*100:5.1f}%)\n")
    f.write(f"2. Shared distinctive address token (length >= 4):           {shared_addr_tok_count:4d} / {total_samples} ({shared_addr_tok_count/total_samples*100:5.1f}%)\n")
    f.write(f"3. Meaningful name-token overlap (excluding legal/noise):    {meaningful_name_tok_count:4d} / {total_samples} ({meaningful_name_tok_count/total_samples*100:5.1f}%)\n")
    f.write(f"4. Strong address similarity despite weak/different names:   {strong_addr_weak_name:4d} / {total_samples} ({strong_addr_weak_name/total_samples*100:5.1f}%)\n")
    f.write(f"5. Strong name similarity despite weak/missing addresses:    {strong_name_weak_addr:4d} / {total_samples} ({strong_name_weak_addr/total_samples*100:5.1f}%)\n")

    f.write("\n\n" + "=" * 80 + "\n")
    f.write("TOP 3 V2 FAILURE PATTERNS\n")
    f.write("=" * 80 + "\n")
    f.write("""1. NAME TYPOS & TOKEN REORDER / VARIATION (73.6% - 61.6% of misses):
   - Cause: V1 relies primarily on exact normalized core name equality (`normalize_name_core`), which collapses if there are character edits, token insertions/deletions (e.g. 'Orelee\\'s Barbershop' vs 'Orelee\\'s Services'), or multi-word permutations.
   - Impact: Misses legitimate matches when business names have slight variations or extra descriptors not caught by standard legal stopword removal.

2. CITY & LOCALITY VARIATIONS / MISMATCHES (66.2% of misses):
   - Cause: V1 City+House blocking assumes the city string extracted by split-comma indexing exactly matches across sources. In reality, sources frequently format locality differently (e.g., 'Columus CDP' vs 'Columbus', 'Howrah' vs 'Kolkata', or missing city tokens).
   - Impact: Blocks on City+House fail because the city key never intersects even when the house number and street are identical.

3. HOUSE NUMBER FORMATTING & STREET PREFIX MISMATCHES (65.8% of misses):
   - Cause: V1 House+Street requires exact matching on the initial house number token and full street token. Sources exhibit zero-padding ('0017560' vs '17560'), prefix numbers ('HN 567 797' vs '797'), missing house numbers in secondary sources ('Westchester Dr' vs '1795 Westchester Drive'), and street abbreviation variations.
   - Impact: Exact composite key `house_number:street` fails when one source truncates, prepends, or formats the house number differently.
""")

    f.write("\n" + "=" * 80 + "\n")
    f.write("RECOMMENDED RETRIEVAL SIGNAL TO TEST FIRST\n")
    f.write("=" * 80 + "\n")
    f.write("""RECOMMENDATION: DISTINCTIVE ADDRESS TOKEN INDEXING & PREFIX-AWARE HOUSE BLOCKING

Primary Signal to Test First:
1. Distinctive Address Token Inverted Index (Street / Locality token overlap):
   - Finding: 90.9% of all missed pairs share at least one distinctive address token (length >= 4).
   - Strategy: Build an inverted index on normalized, distinctive address tokens (with appropriate frequency limits/caps <= 100 or 200 per source, or combined with country partitioning) to capture records where business names are noisy or house numbers are formatted inconsistently.

Secondary Signal (High Precision Complement):
2. Normalized House Number + Street Prefix (Street 4-prefix with zero-stripped house number):
   - Finding: 59.8% of missed pairs share the exact normalized house number, but failed V1 due to leading zeros ('0017560' vs '17560') or street token variations.
   - Strategy: Strip leading zeros and use a 3-4 character street prefix key (e.g. '17560:elli') with block size <= 200-500.
""")

print("v2_miss_patterns.txt successfully generated.")
