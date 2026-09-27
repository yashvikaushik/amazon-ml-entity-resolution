"""
Evaluation script for V2 Blocker #3: Distinctive Business Name Token Retrieval.

Evaluates thresholds <= 50 and <= 100 incrementally over current V2 (V1 + V2-A <= 100 + V2-B <= 100).
Evaluated on COMPLETE train ground truth (2,206,821 records, 7,638,365 true matches).
Does NOT modify candidate_pairs.tsv or any existing candidate files.
"""

import os
import sys
import time
import re
from collections import Counter, defaultdict
from pathlib import Path

# Ensure src/ is on python path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from normalization import clean_string, LEGAL_STOPWORDS

# Generic address tokens for V2-A
GENERIC_ADDR_TOKENS = {
    'st', 'street', 'rd', 'road', 'dr', 'drive', 'ave', 'avenue', 'ln', 'lane',
    'ct', 'court', 'blvd', 'boulevard', 'pkwy', 'parkway', 'hwy', 'highway',
    'way', 'pl', 'place', 'apt', 'apartment', 'ste', 'suite', 'fl', 'floor',
    'unit', 'po', 'box', 'no', 'north', 'south', 'east', 'west', 'n', 's', 'e', 'w',
    'null', 'none', 'nan', 'us', 'usa', 'india', 'in', 'of', 'for', 'at', 'by', 'and', 'the'
}

PREFIX_STOPWORDS = {
    'hn', 'no', 'plot', 'shop', 'flat', 'unit', 'bldg', 'building',
    'apt', 'ste', 'suite', 'floor', 'pmb', 'hno', 'house'
}


def get_address_tokens(address: str) -> set[str]:
    """Tokenize address for V2-A distinctive token blocking."""
    if not address:
        return set()
    raw_tokens = re.findall(r"[a-z0-9]+", str(address).lower())
    return {t for t in raw_tokens if len(t) >= 3 and t not in GENERIC_ADDR_TOKENS}


def extract_house_street_keys(address: str, prefix_len: int = 4) -> list[str]:
    """Extract normalized house number + street prefix keys for V2-B."""
    if not address:
        return []
    tokens = re.findall(r"[a-z0-9]+", str(address).lower())
    if not tokens:
        return []
    
    keys = []
    for i, tok in enumerate(tokens):
        if tok.isdigit():
            num = tok.lstrip("0")
            if not num or len(num) > 6:
                continue
            for j in range(i + 1, min(i + 4, len(tokens))):
                cand = tokens[j]
                if cand not in PREFIX_STOPWORDS and not cand.isdigit() and len(cand) >= 2:
                    keys.append(f"{num}:{cand[:prefix_len]}")
                    break
    return list(set(keys))


def get_name_tokens(name: str) -> set[str]:
    """
    Extract distinctive business name tokens for V2-C.
    - Strips legal and generic business stopwords
    - Excludes tokens < 4 chars
    - Excludes pure numbers
    """
    if not name:
        return set()
    cleaned = clean_string(name)
    tokens = cleaned.split()
    return {t for t in tokens if len(t) >= 4 and t not in LEGAL_STOPWORDS and not t.isdigit()}


def evaluate_blocker3(
    train_dir: str = "student_resource/dataset/train",
    v1_candidate_file: str = "student_resource/candidate_pairs.tsv",
):
    start_time = time.time()

    print("=" * 80)
    print("V2 BLOCKER #3: DISTINCTIVE BUSINESS NAME TOKEN RETRIEVAL EVALUATION")
    print("=" * 80)

    gt_file = os.path.join(train_dir, "train_ground_truth.tsv")
    s1_file = os.path.join(train_dir, "train_source1.tsv")
    s2_file = os.path.join(train_dir, "train_source2.tsv")
    s3_file = os.path.join(train_dir, "train_source3.tsv")

    # 1. Load Ground Truth
    t0 = time.time()
    print("\n[1/6] Loading ground truth...")
    gt = {}
    total_gt_matches = 0
    with open(gt_file, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            line = line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split("\t")
            s1_id = parts[0]
            if len(parts) > 1 and parts[1]:
                matches = set(parts[1].split(","))
                gt[s1_id] = matches
                total_gt_matches += len(matches)

    print(f"      Loaded {len(gt):,} S1 entries ({total_gt_matches:,} true matches) in {time.time()-t0:.2f}s")

    # 2. Build V2-A (Address Tokens <= 100) indexes
    t0 = time.time()
    print("\n[2/6] Building V2-A (Address Tokens <= 100) indexes...")
    s2_tok_freq = Counter()
    s3_tok_freq = Counter()

    with open(s2_file, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.split("\t")
            if len(parts) > 2:
                for tok in get_address_tokens(parts[2]):
                    s2_tok_freq[tok] += 1

    with open(s3_file, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.split("\t")
            if len(parts) > 2:
                for tok in get_address_tokens(parts[2]):
                    s3_tok_freq[tok] += 1

    s2_v2a_tokens = {tok for tok, c in s2_tok_freq.items() if c <= 100}
    s3_v2a_tokens = {tok for tok, c in s3_tok_freq.items() if c <= 100}
    del s2_tok_freq, s3_tok_freq

    s2_v2a_index = defaultdict(list)
    s3_v2a_index = defaultdict(list)

    with open(s2_file, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.split("\t")
            if len(parts) > 2:
                e_id = parts[0]
                for tok in get_address_tokens(parts[2]):
                    if tok in s2_v2a_tokens:
                        s2_v2a_index[tok].append(e_id)

    with open(s3_file, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.split("\t")
            if len(parts) > 2:
                e_id = parts[0]
                for tok in get_address_tokens(parts[2]):
                    if tok in s3_v2a_tokens:
                        s3_v2a_index[tok].append(e_id)

    s2_v2a_index = {k: tuple(v) for k, v in s2_v2a_index.items()}
    s3_v2a_index = {k: tuple(v) for k, v in s3_v2a_index.items()}
    del s2_v2a_tokens, s3_v2a_tokens
    print(f"      V2-A indexes built in {time.time()-t0:.2f}s")

    # 3. Build V2-B (House + Street Prefix <= 100) indexes
    t0 = time.time()
    print("\n[3/6] Building V2-B (House + Street Prefix <= 100) indexes...")
    s2_hs_freq = Counter()
    s3_hs_freq = Counter()

    with open(s2_file, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.split("\t")
            if len(parts) > 2:
                for k in extract_house_street_keys(parts[2]):
                    s2_hs_freq[k] += 1

    with open(s3_file, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.split("\t")
            if len(parts) > 2:
                for k in extract_house_street_keys(parts[2]):
                    s3_hs_freq[k] += 1

    s2_hs_valid = {k for k, c in s2_hs_freq.items() if c <= 100}
    s3_hs_valid = {k for k, c in s3_hs_freq.items() if c <= 100}
    del s2_hs_freq, s3_hs_freq

    s2_hs_index = defaultdict(list)
    s3_hs_index = defaultdict(list)

    with open(s2_file, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.split("\t")
            if len(parts) > 2:
                e_id = parts[0]
                for k in extract_house_street_keys(parts[2]):
                    if k in s2_hs_valid:
                        s2_hs_index[k].append(e_id)

    with open(s3_file, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.split("\t")
            if len(parts) > 2:
                e_id = parts[0]
                for k in extract_house_street_keys(parts[2]):
                    if k in s3_hs_valid:
                        s3_hs_index[k].append(e_id)

    s2_hs_index = {k: tuple(v) for k, v in s2_hs_index.items()}
    s3_hs_index = {k: tuple(v) for k, v in s3_hs_index.items()}
    del s2_hs_valid, s3_hs_valid
    print(f"      V2-B indexes built in {time.time()-t0:.2f}s")

    # 4. Build V2-C (Distinctive Name Tokens <= 100) indexes
    t0 = time.time()
    print("\n[4/6] Building V2-C (Distinctive Name Tokens <= 100) indexes...")
    s2_name_freq = Counter()
    s3_name_freq = Counter()

    with open(s2_file, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.split("\t")
            if len(parts) > 1:
                for t in get_name_tokens(parts[1]):
                    s2_name_freq[t] += 1

    with open(s3_file, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.split("\t")
            if len(parts) > 1:
                for t in get_name_tokens(parts[1]):
                    s3_name_freq[t] += 1

    s2_name_oversized_50 = sum(1 for c in s2_name_freq.values() if c > 50)
    s2_name_oversized_100 = sum(1 for c in s2_name_freq.values() if c > 100)

    s3_name_oversized_50 = sum(1 for c in s3_name_freq.values() if c > 50)
    s3_name_oversized_100 = sum(1 for c in s3_name_freq.values() if c > 100)

    total_oversized_50 = s2_name_oversized_50 + s3_name_oversized_50
    total_oversized_100 = s2_name_oversized_100 + s3_name_oversized_100

    s2_name_valid_100 = {t for t, c in s2_name_freq.items() if c <= 100}
    s3_name_valid_100 = {t for t, c in s3_name_freq.items() if c <= 100}
    del s2_name_freq, s3_name_freq

    s2_name_index = defaultdict(list)
    s3_name_index = defaultdict(list)

    with open(s2_file, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.split("\t")
            if len(parts) > 1:
                e_id = parts[0]
                for t in get_name_tokens(parts[1]):
                    if t in s2_name_valid_100:
                        s2_name_index[t].append(e_id)

    with open(s3_file, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.split("\t")
            if len(parts) > 1:
                e_id = parts[0]
                for t in get_name_tokens(parts[1]):
                    if t in s3_name_valid_100:
                        s3_name_index[t].append(e_id)

    s2_name_index = {k: tuple(v) for k, v in s2_name_index.items()}
    s3_name_index = {k: tuple(v) for k, v in s3_name_index.items()}
    del s2_name_valid_100, s3_name_valid_100
    print(f"      V2-C indexes built in {time.time()-t0:.2f}s")

    # 5. Stream S1 and evaluate V1 + V2-A + V2-B + V2-C
    print("\n[5/6] Streaming S1 and evaluating incremental recall over current V2 (V1 + V2-A + V2-B)...")
    t_eval0 = time.time()

    v1_total_candidates = 0
    v1_recovered = 0
    current_v2_candidates = 0
    current_v2_recovered = 0

    # V2-C <= 50
    standalone_50 = 0
    recovered_50 = 0
    new_recovered_50 = 0
    combined_cands_50 = 0

    # V2-C <= 100
    standalone_100 = 0
    recovered_100 = 0
    new_recovered_100 = 0
    combined_cands_100 = 0

    processed_s1 = 0

    with open(s1_file, "r", encoding="utf-8") as f_s1, open(v1_candidate_file, "r", encoding="utf-8") as f_v1:
        next(f_s1)
        next(f_v1)

        for line_s1, line_v1 in zip(f_s1, f_v1):
            processed_s1 += 1

            parts_s1 = line_s1.split("\t")
            s1_id = parts_s1[0]
            s1_name = parts_s1[1] if len(parts_s1) > 1 else ""
            s1_addr = parts_s1[2] if len(parts_s1) > 2 else ""

            # V1 candidates
            idx_v1 = line_v1.find("\t")
            v1_cands_str = line_v1[idx_v1+1:].rstrip("\r\n") if idx_v1 != -1 else ""
            v1_set = set(v1_cands_str.split(",")) if v1_cands_str else set()
            v1_total_candidates += len(v1_set)

            gt_set = gt.get(s1_id, set())
            if gt_set:
                v1_hits = gt_set.intersection(v1_set)
                v1_recovered += len(v1_hits)

            # V2-A candidates (Tokens <= 100)
            v2a_cands = set()
            s1_addr_toks = get_address_tokens(s1_addr)
            for tok in s1_addr_toks:
                if tok in s2_v2a_index:
                    v2a_cands.update(s2_v2a_index[tok])
                if tok in s3_v2a_index:
                    v2a_cands.update(s3_v2a_index[tok])

            # V2-B candidates (House + Street Prefix <= 100)
            v2b_cands = set()
            s1_hs_keys = extract_house_street_keys(s1_addr)
            for k in s1_hs_keys:
                if k in s2_hs_index:
                    v2b_cands.update(s2_hs_index[k])
                if k in s3_hs_index:
                    v2b_cands.update(s3_hs_index[k])

            # Current V2 set (V1 | V2-A | V2-B)
            current_v2_set = v1_set | v2a_cands | v2b_cands
            current_v2_candidates += len(current_v2_set)
            if gt_set:
                v2_hits = gt_set.intersection(current_v2_set)
                current_v2_recovered += len(v2_hits)

            # V2-C candidates
            s1_name_toks = get_name_tokens(s1_name)

            # --- Threshold <= 50 ---
            cands_50 = set()
            for tok in s1_name_toks:
                if tok in s2_name_index and len(s2_name_index[tok]) <= 50:
                    cands_50.update(s2_name_index[tok])
                if tok in s3_name_index and len(s3_name_index[tok]) <= 50:
                    cands_50.update(s3_name_index[tok])

            standalone_50 += len(cands_50)
            if gt_set:
                hits_50 = gt_set.intersection(cands_50)
                recovered_50 += len(hits_50)
                new_recovered_50 += len(hits_50 - current_v2_set)
            combined_cands_50 += len(current_v2_set | cands_50)

            # --- Threshold <= 100 ---
            cands_100 = set()
            for tok in s1_name_toks:
                if tok in s2_name_index:
                    cands_100.update(s2_name_index[tok])
                if tok in s3_name_index:
                    cands_100.update(s3_name_index[tok])

            standalone_100 += len(cands_100)
            if gt_set:
                hits_100 = gt_set.intersection(cands_100)
                recovered_100 += len(hits_100)
                new_recovered_100 += len(hits_100 - current_v2_set)
            combined_cands_100 += len(current_v2_set | cands_100)

            if processed_s1 % 500_000 == 0:
                print(f"      Processed {processed_s1:,} / 2,206,821 S1 records...")

    eval_time = time.time() - t_eval0
    total_runtime = time.time() - start_time
    print(f"      Evaluation stream completed in {eval_time:.2f}s (Total runtime: {total_runtime:.2f}s)")

    # 6. Compile and Print Results
    v1_recall = (v1_recovered / total_gt_matches) * 100
    current_v2_recall = (current_v2_recovered / total_gt_matches) * 100

    # Threshold 50
    inc_recall_50 = (new_recovered_50 / total_gt_matches) * 100
    comb_recall_50 = ((current_v2_recovered + new_recovered_50) / total_gt_matches) * 100
    pool_inc_50 = combined_cands_50 - current_v2_candidates
    pool_inc_pct_50 = (pool_inc_50 / current_v2_candidates) * 100

    # Threshold 100
    inc_recall_100 = (new_recovered_100 / total_gt_matches) * 100
    comb_recall_100 = ((current_v2_recovered + new_recovered_100) / total_gt_matches) * 100
    pool_inc_100 = combined_cands_100 - current_v2_candidates
    pool_inc_pct_100 = (pool_inc_100 / current_v2_candidates) * 100

    print("\n" + "=" * 80)
    print("DETAILED RESULTS: V2-C THRESHOLD <= 50")
    print("=" * 80)
    print(f"- Standalone blocker candidate pairs:          {standalone_50:,}")
    print(f"- True matches recovered by blocker:           {recovered_50:,} ({(recovered_50/total_gt_matches)*100:.4f}%)")
    print(f"- NEW true matches recovered beyond current V2:{new_recovered_50:,}")
    print(f"- Incremental recall over current V2:          +{inc_recall_50:.4f}%")
    print(f"- Combined recall (V1 + V2-A + V2-B + V2-C):   {comb_recall_50:.4f}%")
    print(f"- Combined candidate count:                    {combined_cands_50:,}")
    print(f"- Candidate-pool increase (over current V2):   +{pool_inc_50:,} (+{pool_inc_pct_50:.2f}%)")
    print(f"- Oversized/skipped blocks:                    {total_oversized_50:,}")
    print(f"- Runtime:                                     {total_runtime:.2f}s")

    print("\n" + "=" * 80)
    print("DETAILED RESULTS: V2-C THRESHOLD <= 100")
    print("=" * 80)
    print(f"- Standalone blocker candidate pairs:          {standalone_100:,}")
    print(f"- True matches recovered by blocker:           {recovered_100:,} ({(recovered_100/total_gt_matches)*100:.4f}%)")
    print(f"- NEW true matches recovered beyond current V2:{new_recovered_100:,}")
    print(f"- Incremental recall over current V2:          +{inc_recall_100:.4f}%")
    print(f"- Combined recall (V1 + V2-A + V2-B + V2-C):   {comb_recall_100:.4f}%")
    print(f"- Combined candidate count:                    {combined_cands_100:,}")
    print(f"- Candidate-pool increase (over current V2):   +{pool_inc_100:,} (+{pool_inc_pct_100:.2f}%)")
    print(f"- Oversized/skipped blocks:                    {total_oversized_100:,}")
    print(f"- Runtime:                                     {total_runtime:.2f}s")

    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    print(f"{'threshold':<14} | {'blocker_pairs':<14} | {'new_true_matches':<16} | {'incremental_recall':<18} | {'combined_recall':<15} | {'combined_candidates':<19} | {'runtime':<10}")
    print("-" * 117)
    print(f"{'V1 Baseline':<14} | {'246,533,496':<14} | {'-':<16} | {'-':<18} | {f'{v1_recall:.4f}%':<15} | {'246,533,496':<19} | {'-':<10}")
    print(f"{'Current V2':<14} | {'-':<14} | {'-':<16} | {'-':<18} | {f'{current_v2_recall:.4f}%':<15} | {'346,186,836':<19} | {'-':<10}")
    print(f"{'V2-C (<= 50)':<14} | {f'{standalone_50:,}':<14} | {f'{new_recovered_50:,}':<16} | {f'+{inc_recall_50:.4f}%':<18} | {f'{comb_recall_50:.4f}%':<15} | {f'{combined_cands_50:,}':<19} | {f'{total_runtime:.1f}s':<10}")
    print(f"{'V2-C (<= 100)':<14} | {f'{standalone_100:,}':<14} | {f'{new_recovered_100:,}':<16} | {f'+{inc_recall_100:.4f}%':<18} | {f'{comb_recall_100:.4f}%':<15} | {f'{combined_cands_100:,}':<19} | {f'{total_runtime:.1f}s':<10}")
    print("=" * 80)


if __name__ == "__main__":
    evaluate_blocker3()
