"""
Analyze V1 Blocking Misses against Ground Truth.

Computes:
- V1 candidate recall and miss counts
- Full missed matches TSV (student_resource/v2_missed_matches.tsv)
- Summary report (student_resource/v2_miss_summary.txt)
- Diagnostic sample TSV (student_resource/v2_missed_sample.tsv)
"""

import os
import sys
import time
import re
from pathlib import Path

# Ensure src/ is on python path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from normalization import normalize_name_core, clean_string


def extract_house_number(address: str) -> str:
    """Extract first house/building number from an address."""
    if not address:
        return ""
    match = re.search(r"\b\d{1,6}\b", str(address))
    return match.group(0) if match else ""


def extract_city(address: str) -> str:
    """Extract city from comma-separated business address (second-to-last field)."""
    if not address:
        return ""
    parts = str(address).lower().split(",")
    if len(parts) >= 2:
        return parts[-2].strip()
    return ""


def get_token_set(text: str) -> set[str]:
    """Extract lowercase alphanumeric tokens from text."""
    if not text:
        return set()
    cleaned = clean_string(text)
    return set(cleaned.split())


def compute_jaccard(tokens1: set[str], tokens2: set[str]) -> float:
    """Compute Jaccard similarity between two token sets."""
    if not tokens1 and not tokens2:
        return 0.0
    union = tokens1 | tokens2
    if not union:
        return 0.0
    intersection = tokens1 & tokens2
    return len(intersection) / len(union)


def run_analysis(
    train_dir: str = "student_resource/dataset/train",
    candidate_file: str = "student_resource/candidate_pairs.tsv",
    missed_matches_file: str = "student_resource/v2_missed_matches.tsv",
    miss_summary_file: str = "student_resource/v2_miss_summary.txt",
    missed_sample_file: str = "student_resource/v2_missed_sample.tsv",
    sample_size: int = 1000,
):
    start_time = time.time()

    gt_file = os.path.join(train_dir, "train_ground_truth.tsv")
    s1_file = os.path.join(train_dir, "train_source1.tsv")
    s2_file = os.path.join(train_dir, "train_source2.tsv")
    s3_file = os.path.join(train_dir, "train_source3.tsv")

    print("=" * 65)
    print("V1 CANDIDATE-MISS ANALYSIS")
    print("=" * 65)

    # 1. Load Ground Truth
    t0 = time.time()
    print("[1/5] Loading ground truth...")
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

    print(f"Loaded {len(gt):,} S1 ground-truth entries ({total_gt_matches:,} matches) in {time.time()-t0:.2f}s")

    # 2. Stream candidate_pairs.tsv and compute misses
    t1 = time.time()
    print("[2/5] Streaming V1 candidate pairs and computing misses...")
    v1_total_candidates = 0
    v1_recovered = 0
    v1_missed = 0
    missed_s2 = 0
    missed_s3 = 0

    all_missed = []
    needed_s1 = set()
    needed_s2 = set()
    needed_s3 = set()

    with open(candidate_file, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            line = line.rstrip("\r\n")
            if not line:
                continue
            idx = line.find("\t")
            if idx == -1:
                s1_id = line
                cand_str = ""
            else:
                s1_id = line[:idx]
                cand_str = line[idx+1:]

            cands = set(cand_str.split(",")) if cand_str else set()
            v1_total_candidates += len(cands)

            gt_matches = gt.get(s1_id)
            if gt_matches:
                hits = gt_matches.intersection(cands)
                misses = gt_matches - cands
                v1_recovered += len(hits)
                v1_missed += len(misses)
                for m in misses:
                    if m.startswith("S2"):
                        missed_s2 += 1
                        needed_s2.add(m)
                    else:
                        missed_s3 += 1
                        needed_s3.add(m)
                    all_missed.append((s1_id, m))
                    needed_s1.add(s1_id)

    print(f"Compared candidates in {time.time()-t1:.2f}s")
    print(f"Found {len(all_missed):,} missed matches ({missed_s2:,} S2, {missed_s3:,} S3)")

    # 3. Stream source files for metadata
    t2 = time.time()
    print("[3/5] Extracting entity details for missed matches...")
    s1_data = {}
    with open(s1_file, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            line = line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split("\t")
            entity_id = parts[0]
            if entity_id in needed_s1:
                b_name = parts[1] if len(parts) > 1 else ""
                b_addr = parts[2] if len(parts) > 2 else ""
                country = parts[3] if len(parts) > 3 else ""
                s1_data[entity_id] = (b_name, b_addr, country)

    s2_data = {}
    with open(s2_file, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            line = line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split("\t")
            entity_id = parts[0]
            if entity_id in needed_s2:
                b_name = parts[1] if len(parts) > 1 else ""
                b_addr = parts[2] if len(parts) > 2 else ""
                country = parts[3] if len(parts) > 3 else ""
                s2_data[entity_id] = (b_name, b_addr, country)

    s3_data = {}
    with open(s3_file, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            line = line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split("\t")
            entity_id = parts[0]
            if entity_id in needed_s3:
                b_name = parts[1] if len(parts) > 1 else ""
                b_addr = parts[2] if len(parts) > 2 else ""
                country = parts[3] if len(parts) > 3 else ""
                s3_data[entity_id] = (b_name, b_addr, country)

    print(f"Extracted entity metadata in {time.time()-t2:.2f}s")

    # 4. Write full missed matches TSV and summary
    t3 = time.time()
    print(f"[4/5] Writing missed matches to {missed_matches_file}...")
    with open(missed_matches_file, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmissed_entity_id\tsource\ts1_business_name\tmissed_business_name\ts1_business_address\tmissed_business_address\tcountry\n")
        for s1_id, missed_id in all_missed:
            s1_info = s1_data.get(s1_id, ("", "", ""))
            if missed_id.startswith("S2"):
                source = "S2"
                m_info = s2_data.get(missed_id, ("", "", ""))
            else:
                source = "S3"
                m_info = s3_data.get(missed_id, ("", "", ""))

            s1_name, s1_addr, s1_country = s1_info
            m_name, m_addr, m_country = m_info
            country = s1_country if s1_country else m_country

            f.write(f"{s1_id}\t{missed_id}\t{source}\t{s1_name}\t{m_name}\t{s1_addr}\t{m_addr}\t{country}\n")

    v1_recall_pct = (v1_recovered / total_gt_matches) * 100 if total_gt_matches else 0.0
    missed_pct = (v1_missed / total_gt_matches) * 100 if total_gt_matches else 0.0

    print(f"Writing summary to {miss_summary_file}...")
    with open(miss_summary_file, "w", encoding="utf-8") as f:
        f.write("=" * 50 + "\n")
        f.write("V1 CANDIDATE-MISS SUMMARY REPORT\n")
        f.write("=" * 50 + "\n")
        f.write(f"total ground-truth matches: {total_gt_matches:,}\n")
        f.write(f"total V1 candidate pairs: {v1_total_candidates:,}\n")
        f.write(f"total V1 missed true matches: {v1_missed:,}\n")
        f.write(f"V1 candidate recall: {v1_recall_pct:.4f}%\n")
        f.write(f"missed S2 matches: {missed_s2:,}\n")
        f.write(f"missed S3 matches: {missed_s3:,}\n")
        f.write(f"missed-match percentage: {missed_pct:.4f}%\n")

    # 5. Diagnostic Sample TSV
    t4 = time.time()
    print(f"[5/5] Creating diagnostic sample in {missed_sample_file}...")
    sample_items = all_missed[:sample_size]

    with open(missed_sample_file, "w", encoding="utf-8") as f:
        f.write(
            "source1_entity_id\tmissed_entity_id\tsource\ts1_business_name\tmissed_business_name\t"
            "s1_business_address\tmissed_business_address\tcountry\t"
            "name_exact_normalized\thouse_number_same\taddress_token_overlap\tcity_similarity\n"
        )
        for s1_id, missed_id in sample_items:
            s1_info = s1_data.get(s1_id, ("", "", ""))
            if missed_id.startswith("S2"):
                source = "S2"
                m_info = s2_data.get(missed_id, ("", "", ""))
            else:
                source = "S3"
                m_info = s3_data.get(missed_id, ("", "", ""))

            s1_name, s1_addr, s1_country = s1_info
            m_name, m_addr, m_country = m_info
            country = s1_country if s1_country else m_country

            # Diagnostics
            norm_s1_name = normalize_name_core(s1_name)
            norm_m_name = normalize_name_core(m_name)
            name_exact_norm = bool(norm_s1_name and norm_m_name and (norm_s1_name == norm_m_name))

            hn_s1 = extract_house_number(s1_addr)
            hn_m = extract_house_number(m_addr)
            house_same = bool(hn_s1 and hn_m and (hn_s1 == hn_m))

            s1_addr_tokens = get_token_set(s1_addr)
            m_addr_tokens = get_token_set(m_addr)
            addr_overlap = round(compute_jaccard(s1_addr_tokens, m_addr_tokens), 4)

            city_s1 = extract_city(s1_addr)
            city_m = extract_city(m_addr)
            city_s1_tokens = get_token_set(city_s1)
            city_m_tokens = get_token_set(city_m)
            city_sim = round(compute_jaccard(city_s1_tokens, city_m_tokens), 4)

            f.write(
                f"{s1_id}\t{missed_id}\t{source}\t{s1_name}\t{m_name}\t"
                f"{s1_addr}\t{m_addr}\t{country}\t"
                f"{name_exact_norm}\t{house_same}\t{addr_overlap}\t{city_sim}\n"
            )

    runtime = time.time() - start_time
    runtime_str = f"{int(runtime // 60)}m {runtime % 60:.2f}s" if runtime >= 60 else f"{runtime:.2f}s"

    print("\n" + "=" * 65)
    print("FINAL SUMMARY")
    print("=" * 65)
    print(f"V1 total candidates: {v1_total_candidates:,}")
    print(f"Ground-truth matches: {total_gt_matches:,}")
    print(f"V1 recovered: {v1_recovered:,}")
    print(f"V1 missed: {v1_missed:,}")
    print(f"V1 recall: {v1_recall_pct:.4f}%")
    print(f"Missed S2: {missed_s2:,}")
    print(f"Missed S3: {missed_s3:,}")
    print(f"Runtime: {runtime_str}")


if __name__ == "__main__":
    run_analysis()
