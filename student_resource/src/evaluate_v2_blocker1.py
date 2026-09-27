"""
Evaluation script for V2 Blocker #1: Distinctive Address Token Expansion.

Evaluates thresholds <= 100 and <= 200 incrementally against V1 baseline on COMPLETE train ground truth.
Does NOT modify candidate_pairs.tsv or any V1 logic.
"""

import os
import sys
import time
import re
from collections import Counter, defaultdict
from pathlib import Path

# Common generic address tokens to ignore
GENERIC_ADDR_TOKENS = {
    'st', 'street', 'rd', 'road', 'dr', 'drive', 'ave', 'avenue', 'ln', 'lane',
    'ct', 'court', 'blvd', 'boulevard', 'pkwy', 'parkway', 'hwy', 'highway',
    'way', 'pl', 'place', 'apt', 'apartment', 'ste', 'suite', 'fl', 'floor',
    'unit', 'po', 'box', 'no', 'north', 'south', 'east', 'west', 'n', 's', 'e', 'w',
    'null', 'none', 'nan', 'us', 'usa', 'india', 'in', 'of', 'for', 'at', 'by', 'and', 'the'
}


def get_address_tokens(address: str) -> set[str]:
    """Tokenize address into lowercase alphanumeric tokens, filtering generic stopwords and short tokens."""
    if not address:
        return set()
    raw_tokens = re.findall(r"[a-z0-9]+", str(address).lower())
    return {t for t in raw_tokens if len(t) >= 3 and t not in GENERIC_ADDR_TOKENS}


def evaluate_blocker1(
    train_dir: str = "student_resource/dataset/train",
    v1_candidate_file: str = "student_resource/candidate_pairs.tsv",
):
    start_time = time.time()

    print("=" * 80)
    print("V2 BLOCKER #1: DISTINCTIVE ADDRESS TOKEN EXPANSION EVALUATION")
    print("=" * 80)

    gt_file = os.path.join(train_dir, "train_ground_truth.tsv")
    s1_file = os.path.join(train_dir, "train_source1.tsv")
    s2_file = os.path.join(train_dir, "train_source2.tsv")
    s3_file = os.path.join(train_dir, "train_source3.tsv")

    # 1. Load Ground Truth
    t0 = time.time()
    print("\n[1/4] Loading ground truth...")
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

    # 2. Build S2 and S3 token frequencies and indexes up to freq <= 200
    t0 = time.time()
    print("\n[2/4] Building distinctive address token indexes for S2 and S3 (Max Freq <= 200)...")

    # Pass 1: Count token frequencies
    s2_freq = Counter()
    s3_freq = Counter()

    print("      Counting token frequencies in S2...")
    with open(s2_file, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.split("\t")
            if len(parts) > 2:
                for tok in get_address_tokens(parts[2]):
                    s2_freq[tok] += 1

    print("      Counting token frequencies in S3...")
    with open(s3_file, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.split("\t")
            if len(parts) > 2:
                for tok in get_address_tokens(parts[2]):
                    s3_freq[tok] += 1

    s2_tokens_100 = {tok for tok, c in s2_freq.items() if c <= 100}
    s2_tokens_200 = {tok for tok, c in s2_freq.items() if c <= 200}
    s2_oversized_100 = len(s2_freq) - len(s2_tokens_100)
    s2_oversized_200 = len(s2_freq) - len(s2_tokens_200)

    s3_tokens_100 = {tok for tok, c in s3_freq.items() if c <= 100}
    s3_tokens_200 = {tok for tok, c in s3_freq.items() if c <= 200}
    s3_oversized_100 = len(s3_freq) - len(s3_tokens_100)
    s3_oversized_200 = len(s3_freq) - len(s3_tokens_200)

    total_oversized_100 = s2_oversized_100 + s3_oversized_100
    total_oversized_200 = s2_oversized_200 + s3_oversized_200

    print(f"      S2 tokens <= 100: {len(s2_tokens_100):,} (Oversized skipped: {s2_oversized_100:,})")
    print(f"      S2 tokens <= 200: {len(s2_tokens_200):,} (Oversized skipped: {s2_oversized_200:,})")
    print(f"      S3 tokens <= 100: {len(s3_tokens_100):,} (Oversized skipped: {s3_oversized_100:,})")
    print(f"      S3 tokens <= 200: {len(s3_tokens_200):,} (Oversized skipped: {s3_oversized_200:,})")

    # Pass 2: Inverted Index population up to <= 200
    s2_index = defaultdict(list)
    s3_index = defaultdict(list)

    print("      Populating S2 inverted index...")
    with open(s2_file, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.split("\t")
            if len(parts) > 2:
                e_id = parts[0]
                for tok in get_address_tokens(parts[2]):
                    if tok in s2_tokens_200:
                        s2_index[tok].append(e_id)

    print("      Populating S3 inverted index...")
    with open(s3_file, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.split("\t")
            if len(parts) > 2:
                e_id = parts[0]
                for tok in get_address_tokens(parts[2]):
                    if tok in s3_tokens_200:
                        s3_index[tok].append(e_id)

    # Convert to compact tuples
    s2_index = {k: tuple(v) for k, v in s2_index.items()}
    s3_index = {k: tuple(v) for k, v in s3_index.items()}
    del s2_freq, s3_freq
    print(f"      Indexes built in {time.time()-t0:.2f}s")

    # 3. Stream S1 and compare against V1 candidates
    print("\n[3/4] Streaming S1 and evaluating incremental recall over V1...")
    t_eval0 = time.time()

    v1_total_candidates = 0
    v1_recovered = 0

    standalone_cands_100 = 0
    recovered_100 = 0
    new_recovered_100 = 0
    combined_cands_100 = 0

    standalone_cands_200 = 0
    recovered_200 = 0
    new_recovered_200 = 0
    combined_cands_200 = 0

    processed_s1 = 0

    with open(s1_file, "r", encoding="utf-8") as f_s1, open(v1_candidate_file, "r", encoding="utf-8") as f_v1:
        next(f_s1)
        next(f_v1)

        for line_s1, line_v1 in zip(f_s1, f_v1):
            processed_s1 += 1

            # S1 fields
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

            s1_toks = get_address_tokens(s1_addr)

            # --- Threshold <= 100 ---
            cands_100 = set()
            for tok in s1_toks:
                if tok in s2_tokens_100 and tok in s2_index:
                    cands_100.update(s2_index[tok])
                if tok in s3_tokens_100 and tok in s3_index:
                    cands_100.update(s3_index[tok])

            standalone_cands_100 += len(cands_100)
            if gt_set:
                hits_100 = gt_set.intersection(cands_100)
                recovered_100 += len(hits_100)
                new_recovered_100 += len(hits_100 - v1_set)
            combined_cands_100 += len(v1_set | cands_100)

            # --- Threshold <= 200 ---
            cands_200 = set()
            for tok in s1_toks:
                if tok in s2_index:
                    cands_200.update(s2_index[tok])
                if tok in s3_index:
                    cands_200.update(s3_index[tok])

            standalone_cands_200 += len(cands_200)
            if gt_set:
                hits_200 = gt_set.intersection(cands_200)
                recovered_200 += len(hits_200)
                new_recovered_200 += len(hits_200 - v1_set)
            combined_cands_200 += len(v1_set | cands_200)

            if processed_s1 % 500_000 == 0:
                print(f"      Processed {processed_s1:,} / 2,206,821 S1 records...")

    eval_time = time.time() - t_eval0
    total_runtime = time.time() - start_time
    print(f"      Evaluation stream completed in {eval_time:.2f}s (Total runtime: {total_runtime:.2f}s)")

    # 4. Detailed Metrics and Reports
    v1_recall = (v1_recovered / total_gt_matches) * 100

    inc_recall_100 = (new_recovered_100 / total_gt_matches) * 100
    comb_recall_100 = ((v1_recovered + new_recovered_100) / total_gt_matches) * 100
    pool_inc_100 = combined_cands_100 - v1_total_candidates
    pool_inc_pct_100 = (pool_inc_100 / v1_total_candidates) * 100

    inc_recall_200 = (new_recovered_200 / total_gt_matches) * 100
    comb_recall_200 = ((v1_recovered + new_recovered_200) / total_gt_matches) * 100
    pool_inc_200 = combined_cands_200 - v1_total_candidates
    pool_inc_pct_200 = (pool_inc_200 / v1_total_candidates) * 100

    print("\n" + "=" * 80)
    print("DETAILED RESULTS: THRESHOLD <= 100")
    print("=" * 80)
    print(f"- Standalone blocker candidate pairs:          {standalone_cands_100:,}")
    print(f"- True matches recovered by blocker:           {recovered_100:,} ({(recovered_100/total_gt_matches)*100:.4f}%)")
    print(f"- True matches NEWLY recovered that V1 missed: {new_recovered_100:,}")
    print(f"- Incremental recall percentage points:        +{inc_recall_100:.4f}%")
    print(f"- Combined V1 + V2 recall:                     {comb_recall_100:.4f}%")
    print(f"- Final V1 + V2 candidate count:               {combined_cands_100:,}")
    print(f"- Candidate-pool increase:                     +{pool_inc_100:,} (+{pool_inc_pct_100:.2f}%)")
    print(f"- Maximum block size (per source):             100 (combined <= 200)")
    print(f"- Number of oversized/skipped blocks:          {total_oversized_100:,}")
    print(f"- Runtime:                                     {total_runtime:.2f}s")

    print("\n" + "=" * 80)
    print("DETAILED RESULTS: THRESHOLD <= 200")
    print("=" * 80)
    print(f"- Standalone blocker candidate pairs:          {standalone_cands_200:,}")
    print(f"- True matches recovered by blocker:           {recovered_200:,} ({(recovered_200/total_gt_matches)*100:.4f}%)")
    print(f"- True matches NEWLY recovered that V1 missed: {new_recovered_200:,}")
    print(f"- Incremental recall percentage points:        +{inc_recall_200:.4f}%")
    print(f"- Combined V1 + V2 recall:                     {comb_recall_200:.4f}%")
    print(f"- Final V1 + V2 candidate count:               {combined_cands_200:,}")
    print(f"- Candidate-pool increase:                     +{pool_inc_200:,} (+{pool_inc_pct_200:.2f}%)")
    print(f"- Maximum block size (per source):             200 (combined <= 400)")
    print(f"- Number of oversized/skipped blocks:          {total_oversized_200:,}")
    print(f"- Runtime:                                     {total_runtime:.2f}s")

    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    print(f"{'threshold':<12} | {'blocker_pairs':<14} | {'new_true_matches':<16} | {'incremental_recall':<18} | {'combined_recall':<15} | {'combined_candidates':<19} | {'runtime':<10}")
    print("-" * 115)
    print(f"{'V1 Baseline':<12} | {'246,533,496':<14} | {'-':<16} | {'-':<18} | {f'{v1_recall:.4f}%':<15} | {'246,533,496':<19} | {'-':<10}")
    print(f"{'<= 100':<12} | {f'{standalone_cands_100:,}':<14} | {f'{new_recovered_100:,}':<16} | {f'+{inc_recall_100:.4f}%':<18} | {f'{comb_recall_100:.4f}%':<15} | {f'{combined_cands_100:,}':<19} | {f'{total_runtime:.1f}s':<10}")
    print(f"{'<= 200':<12} | {f'{standalone_cands_200:,}':<14} | {f'{new_recovered_200:,}':<16} | {f'+{inc_recall_200:.4f}%':<18} | {f'{comb_recall_200:.4f}%':<15} | {f'{combined_cands_200:,}':<19} | {f'{total_runtime:.1f}s':<10}")
    print("=" * 80)


if __name__ == "__main__":
    evaluate_blocker1()
