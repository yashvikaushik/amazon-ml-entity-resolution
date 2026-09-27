"""
Evaluation script for V2 Blocker #2: Normalized House Number + Street Prefix Retrieval.

Evaluates thresholds <= 100, <= 200, and <= 500 incrementally over V1 + V2-A (Distinctive Address Tokens <= 100).
Evaluated on COMPLETE train ground truth (2,206,821 records, 7,638,365 true matches).
Does NOT modify candidate_pairs.tsv or any existing candidate files.
"""

import os
import sys
import time
import re
from collections import Counter, defaultdict
from pathlib import Path

# Generic address tokens to ignore for V2-A
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
    """
    Extract normalized house number + street prefix keys from address.
    Normalizes leading zeros and extracts the following street token prefix.
    """
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
            # Search forward for the next distinctive street token
            for j in range(i + 1, min(i + 4, len(tokens))):
                cand = tokens[j]
                if cand not in PREFIX_STOPWORDS and not cand.isdigit() and len(cand) >= 2:
                    keys.append(f"{num}:{cand[:prefix_len]}")
                    break
    return list(set(keys))


def evaluate_blocker2(
    train_dir: str = "student_resource/dataset/train",
    v1_candidate_file: str = "student_resource/candidate_pairs.tsv",
):
    start_time = time.time()

    print("=" * 80)
    print("V2 BLOCKER #2: NORMALIZED HOUSE + STREET PREFIX RETRIEVAL EVALUATION")
    print("=" * 80)

    gt_file = os.path.join(train_dir, "train_ground_truth.tsv")
    s1_file = os.path.join(train_dir, "train_source1.tsv")
    s2_file = os.path.join(train_dir, "train_source2.tsv")
    s3_file = os.path.join(train_dir, "train_source3.tsv")

    # 1. Load Ground Truth
    t0 = time.time()
    print("\n[1/5] Loading ground truth...")
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

    # 2. Build V2-A Inverted Index (Distinctive Tokens <= 100)
    t0 = time.time()
    print("\n[2/5] Building V2-A (Distinctive Tokens <= 100) indexes for S2 and S3...")
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

    # 3. Build V2-B (House + Street Prefix) Inverted Index (Max Block <= 500)
    t0 = time.time()
    print("\n[3/5] Building V2-B (House + Street Prefix) indexes for S2 and S3 (Caps <= 500)...")
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

    s2_oversized_100 = sum(1 for c in s2_hs_freq.values() if c > 100)
    s2_oversized_200 = sum(1 for c in s2_hs_freq.values() if c > 200)
    s2_oversized_500 = sum(1 for c in s2_hs_freq.values() if c > 500)

    s3_oversized_100 = sum(1 for c in s3_hs_freq.values() if c > 100)
    s3_oversized_200 = sum(1 for c in s3_hs_freq.values() if c > 200)
    s3_oversized_500 = sum(1 for c in s3_hs_freq.values() if c > 500)

    total_oversized_100 = s2_oversized_100 + s3_oversized_100
    total_oversized_200 = s2_oversized_200 + s3_oversized_200
    total_oversized_500 = s2_oversized_500 + s3_oversized_500

    s2_hs_valid_500 = {k for k, c in s2_hs_freq.items() if c <= 500}
    s3_hs_valid_500 = {k for k, c in s3_hs_freq.items() if c <= 500}
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
                    if k in s2_hs_valid_500:
                        s2_hs_index[k].append(e_id)

    with open(s3_file, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.split("\t")
            if len(parts) > 2:
                e_id = parts[0]
                for k in extract_house_street_keys(parts[2]):
                    if k in s3_hs_valid_500:
                        s3_hs_index[k].append(e_id)

    s2_hs_index = {k: tuple(v) for k, v in s2_hs_index.items()}
    s3_hs_index = {k: tuple(v) for k, v in s3_hs_index.items()}
    del s2_hs_valid_500, s3_hs_valid_500
    print(f"      V2-B indexes built in {time.time()-t0:.2f}s")

    # 4. Stream S1 and evaluate V1 + V2-A + V2-B
    print("\n[4/5] Streaming S1 and evaluating incremental recall over V1 + V2-A...")
    t_eval0 = time.time()

    v1_total_candidates = 0
    v1_recovered = 0
    v1_v2a_candidates = 0
    v1_v2a_recovered = 0

    # Blocker #2 metrics
    standalone_100 = 0
    recovered_100 = 0
    new_recovered_100 = 0
    combined_cands_100 = 0

    standalone_200 = 0
    recovered_200 = 0
    new_recovered_200 = 0
    combined_cands_200 = 0

    standalone_500 = 0
    recovered_500 = 0
    new_recovered_500 = 0
    combined_cands_500 = 0

    processed_s1 = 0

    with open(s1_file, "r", encoding="utf-8") as f_s1, open(v1_candidate_file, "r", encoding="utf-8") as f_v1:
        next(f_s1)
        next(f_v1)

        for line_s1, line_v1 in zip(f_s1, f_v1):
            processed_s1 += 1

            parts_s1 = line_s1.split("\t")
            s1_id = parts_s1[0]
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
            s1_toks = get_address_tokens(s1_addr)
            for tok in s1_toks:
                if tok in s2_v2a_index:
                    v2a_cands.update(s2_v2a_index[tok])
                if tok in s3_v2a_index:
                    v2a_cands.update(s3_v2a_index[tok])

            v1_v2a_set = v1_set | v2a_cands
            v1_v2a_candidates += len(v1_v2a_set)
            if gt_set:
                v1_v2a_hits = gt_set.intersection(v1_v2a_set)
                v1_v2a_recovered += len(v1_v2a_hits)

            # V2-B keys
            s1_hs_keys = extract_house_street_keys(s1_addr)

            # --- Threshold <= 100 ---
            cands_100 = set()
            for k in s1_hs_keys:
                if k in s2_hs_index and len(s2_hs_index[k]) <= 100:
                    cands_100.update(s2_hs_index[k])
                if k in s3_hs_index and len(s3_hs_index[k]) <= 100:
                    cands_100.update(s3_hs_index[k])

            standalone_100 += len(cands_100)
            if gt_set:
                hits_100 = gt_set.intersection(cands_100)
                recovered_100 += len(hits_100)
                new_recovered_100 += len(hits_100 - v1_v2a_set)
            combined_cands_100 += len(v1_v2a_set | cands_100)

            # --- Threshold <= 200 ---
            cands_200 = set()
            for k in s1_hs_keys:
                if k in s2_hs_index and len(s2_hs_index[k]) <= 200:
                    cands_200.update(s2_hs_index[k])
                if k in s3_hs_index and len(s3_hs_index[k]) <= 200:
                    cands_200.update(s3_hs_index[k])

            standalone_200 += len(cands_200)
            if gt_set:
                hits_200 = gt_set.intersection(cands_200)
                recovered_200 += len(hits_200)
                new_recovered_200 += len(hits_200 - v1_v2a_set)
            combined_cands_200 += len(v1_v2a_set | cands_200)

            # --- Threshold <= 500 ---
            cands_500 = set()
            for k in s1_hs_keys:
                if k in s2_hs_index:
                    cands_500.update(s2_hs_index[k])
                if k in s3_hs_index:
                    cands_500.update(s3_hs_index[k])

            standalone_500 += len(cands_500)
            if gt_set:
                hits_500 = gt_set.intersection(cands_500)
                recovered_500 += len(hits_500)
                new_recovered_500 += len(hits_500 - v1_v2a_set)
            combined_cands_500 += len(v1_v2a_set | cands_500)

            if processed_s1 % 500_000 == 0:
                print(f"      Processed {processed_s1:,} / 2,206,821 S1 records...")

    eval_time = time.time() - t_eval0
    total_runtime = time.time() - start_time
    print(f"      Evaluation stream completed in {eval_time:.2f}s (Total runtime: {total_runtime:.2f}s)")

    # 5. Compile and Print Results
    v1_recall = (v1_recovered / total_gt_matches) * 100
    v1_v2a_recall = (v1_v2a_recovered / total_gt_matches) * 100

    # Threshold 100
    inc_recall_100 = (new_recovered_100 / total_gt_matches) * 100
    comb_recall_100 = ((v1_v2a_recovered + new_recovered_100) / total_gt_matches) * 100
    pool_inc_100 = combined_cands_100 - v1_v2a_candidates
    pool_inc_pct_100 = (pool_inc_100 / v1_v2a_candidates) * 100

    # Threshold 200
    inc_recall_200 = (new_recovered_200 / total_gt_matches) * 100
    comb_recall_200 = ((v1_v2a_recovered + new_recovered_200) / total_gt_matches) * 100
    pool_inc_200 = combined_cands_200 - v1_v2a_candidates
    pool_inc_pct_200 = (pool_inc_200 / v1_v2a_candidates) * 100

    # Threshold 500
    inc_recall_500 = (new_recovered_500 / total_gt_matches) * 100
    comb_recall_500 = ((v1_v2a_recovered + new_recovered_500) / total_gt_matches) * 100
    pool_inc_500 = combined_cands_500 - v1_v2a_candidates
    pool_inc_pct_500 = (pool_inc_500 / v1_v2a_candidates) * 100

    print("\n" + "=" * 80)
    print("DETAILED RESULTS: THRESHOLD <= 100")
    print("=" * 80)
    print(f"- Standalone blocker candidate pairs:          {standalone_100:,}")
    print(f"- True matches recovered by blocker:           {recovered_100:,} ({(recovered_100/total_gt_matches)*100:.4f}%)")
    print(f"- NEW true matches missed by V1 + V2-A:        {new_recovered_100:,}")
    print(f"- Incremental recall over current V2:          +{inc_recall_100:.4f}%")
    print(f"- Combined recall (V1 + V2-A + V2-B):          {comb_recall_100:.4f}%")
    print(f"- Combined candidate count:                    {combined_cands_100:,}")
    print(f"- Candidate-pool increase (over V1+V2-A):      +{pool_inc_100:,} (+{pool_inc_pct_100:.2f}%)")
    print(f"- Oversized/skipped blocks:                    {total_oversized_100:,}")
    print(f"- Runtime:                                     {total_runtime:.2f}s")

    print("\n" + "=" * 80)
    print("DETAILED RESULTS: THRESHOLD <= 200")
    print("=" * 80)
    print(f"- Standalone blocker candidate pairs:          {standalone_200:,}")
    print(f"- True matches recovered by blocker:           {recovered_200:,} ({(recovered_200/total_gt_matches)*100:.4f}%)")
    print(f"- NEW true matches missed by V1 + V2-A:        {new_recovered_200:,}")
    print(f"- Incremental recall over current V2:          +{inc_recall_200:.4f}%")
    print(f"- Combined recall (V1 + V2-A + V2-B):          {comb_recall_200:.4f}%")
    print(f"- Combined candidate count:                    {combined_cands_200:,}")
    print(f"- Candidate-pool increase (over V1+V2-A):      +{pool_inc_200:,} (+{pool_inc_pct_200:.2f}%)")
    print(f"- Oversized/skipped blocks:                    {total_oversized_200:,}")
    print(f"- Runtime:                                     {total_runtime:.2f}s")

    print("\n" + "=" * 80)
    print("DETAILED RESULTS: THRESHOLD <= 500")
    print("=" * 80)
    print(f"- Standalone blocker candidate pairs:          {standalone_500:,}")
    print(f"- True matches recovered by blocker:           {recovered_500:,} ({(recovered_500/total_gt_matches)*100:.4f}%)")
    print(f"- NEW true matches missed by V1 + V2-A:        {new_recovered_500:,}")
    print(f"- Incremental recall over current V2:          +{inc_recall_500:.4f}%")
    print(f"- Combined recall (V1 + V2-A + V2-B):          {comb_recall_500:.4f}%")
    print(f"- Combined candidate count:                    {combined_cands_500:,}")
    print(f"- Candidate-pool increase (over V1+V2-A):      +{pool_inc_500:,} (+{pool_inc_pct_500:.2f}%)")
    print(f"- Oversized/skipped blocks:                    {total_oversized_500:,}")
    print(f"- Runtime:                                     {total_runtime:.2f}s")

    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    print(f"{'threshold':<12} | {'blocker_pairs':<14} | {'new_true_matches':<16} | {'incremental_recall':<18} | {'combined_recall':<15} | {'combined_candidates':<19} | {'runtime':<10}")
    print("-" * 115)
    print(f"{'V1 Baseline':<12} | {'246,533,496':<14} | {'-':<16} | {'-':<18} | {f'{v1_recall:.4f}%':<15} | {'246,533,496':<19} | {'-':<10}")
    print(f"{'V1 + V2-A':<12} | {'94,544,265':<14} | {'168,909':<16} | {'+2.2113%':<18} | {f'{v1_v2a_recall:.4f}%':<15} | {'305,289,640':<19} | {'-':<10}")
    print(f"{'<= 100':<12} | {f'{standalone_100:,}':<14} | {f'{new_recovered_100:,}':<16} | {f'+{inc_recall_100:.4f}%':<18} | {f'{comb_recall_100:.4f}%':<15} | {f'{combined_cands_100:,}':<19} | {f'{total_runtime:.1f}s':<10}")
    print(f"{'<= 200':<12} | {f'{standalone_200:,}':<14} | {f'{new_recovered_200:,}':<16} | {f'+{inc_recall_200:.4f}%':<18} | {f'{comb_recall_200:.4f}%':<15} | {f'{combined_cands_200:,}':<19} | {f'{total_runtime:.1f}s':<10}")
    print(f"{'<= 500':<12} | {f'{standalone_500:,}':<14} | {f'{new_recovered_500:,}':<16} | {f'+{inc_recall_500:.4f}%':<18} | {f'{comb_recall_500:.4f}%':<15} | {f'{combined_cands_500:,}':<19} | {f'{total_runtime:.1f}s':<10}")
    print("=" * 80)


if __name__ == "__main__":
    evaluate_blocker2()
