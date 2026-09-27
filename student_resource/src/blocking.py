"""
Data blocking / candidate generation utilities for CODEHex.

Blocking V1 Strategies (Fixed Logic):
1. V3 normalized business-name blocking
2. House + Street blocking
3. City + House blocking (block size <= 500)
4. Rare address-token blocking (frequency <= 50)

The final candidate set is the deduplicated union of all 4 blocking strategies.

Optimized Architecture:
- Direct single-pass unified inverted index construction across S2 and S3.
- Upfront index pre-filtering (City+House <= 500, Rare Tokens <= 50) converted to compact tuples.
- Chunk-based streaming candidate generation (memory-safe, constant low RAM footprint).
- Atomic temporary file creation with post-generation lightweight validation.
"""

import os
import sys
import re
import time
import argparse
from pathlib import Path
from collections import defaultdict
import pandas as pd

# Ensure src/ is on python path for standalone execution
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from normalization import normalize_name_core


# ============================================================
# NORMALIZATION / KEY FUNCTIONS
# ============================================================

def normalize_name(name: str) -> str:
    """
    V3 business-name normalization.
    Uses the project's normalization module (legal suffix stripping,
    accent removal, URL/domain cleanup).
    """
    return normalize_name_core(name)


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


def house_street_key(address: str) -> str:
    """
    Build a house + street blocking key.
    Example: '1795 Westchester Drive' -> '1795:westchester'
    """
    if not address:
        return ""
    text = str(address).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    if len(tokens) < 2 or not tokens[0].isdigit():
        return ""
    return f"{tokens[0]}:{tokens[1]}"


def get_address_tokens(address: str) -> list[str]:
    """Tokenize an address for rare-token blocking."""
    if not address:
        return []
    return list(set(re.findall(r"[a-z0-9]+", str(address).lower())))


def city_house_key(city: str, house_number: str) -> str:
    """Build city + house blocking key."""
    if not city or not house_number:
        return ""
    return f"{city}:{house_number}"


# ============================================================
# HARD-CASE EXPERIMENT (Documented, Not Used in V1 Production)
# ============================================================

def street_prefix_key(address: str, prefix_len: int = 4) -> str:
    """
    Experimental hard-case blocking key:
    house number + first characters of street token.
    Example: '1795 Westchester Drive' -> '1795:west'
    """
    if not address:
        return ""
    text = str(address).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    if len(tokens) < 2 or not tokens[0].isdigit():
        return ""
    street = tokens[1]
    if not street:
        return ""
    return f"{tokens[0]}:{street[:prefix_len]}"


# ============================================================
# OPTIMIZED CANDIDATE GENERATION PIPELINE
# ============================================================

def run_blocking_pipeline(
    train_dir: str = "student_resource/dataset/train",
    output_file: str = "student_resource/candidate_pairs.tsv",
    flat_output_file: str = "student_resource/candidate_pairs_flat.tsv",
    city_house_max_block_size: int = 500,
    rare_token_frequency_limit: int = 50,
    chunk_size: int = 100_000,
    limit_s1: int = None,
    limit_s2_s3: int = None
):
    """
    Run the complete, high-performance Blocking V1 candidate-generation pipeline.

    Blocking V1 Strategies:
        1. V3 normalized business name
        2. House + Street
        3. City + House (max block size <= 500)
        4. Rare address token (frequency <= 50)
    """
    total_pipeline_t0 = time.time()

    print("=" * 75)
    print("CODEHex - OPTIMIZED BLOCKING V1 CANDIDATE GENERATION")
    print("=" * 75)

    # Resolve train_dir path dynamically
    if not os.path.exists(train_dir):
        alt_path = os.path.join("dataset", "train")
        if os.path.exists(alt_path):
            train_dir = alt_path

    # Ensure output parent directories exist
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(flat_output_file)), exist_ok=True)

    # --------------------------------------------------------
    # 1. Load data
    # --------------------------------------------------------
    print(f"\n[1/6] Loading source data from {train_dir}...")
    t0 = time.time()

    s1 = pd.read_csv(
        os.path.join(train_dir, "train_source1.tsv"),
        sep="\t",
        dtype=str,
        keep_default_na=False
    )
    s2 = pd.read_csv(
        os.path.join(train_dir, "train_source2.tsv"),
        sep="\t",
        dtype=str,
        keep_default_na=False
    )
    s3 = pd.read_csv(
        os.path.join(train_dir, "train_source3.tsv"),
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    if limit_s1 is not None and limit_s1 > 0:
        print(f"      [SMOKE-TEST MODE] Subsetting S1 to first {limit_s1:,} records.")
        s1 = s1.head(limit_s1).copy()

    if limit_s2_s3 is not None and limit_s2_s3 > 0:
        print(f"      [SMOKE-TEST MODE] Subsetting S2/S3 to first {limit_s2_s3:,} records.")
        s2 = s2.head(limit_s2_s3).copy()
        s3 = s3.head(limit_s2_s3).copy()

    total_s1 = len(s1)
    print(f"      S1: {total_s1:,} | S2: {len(s2):,} | S3: {len(s3):,} (Loaded in {time.time()-t0:.2f}s)")

    # --------------------------------------------------------
    # 2. Build Unified Inverted Indexes directly from S2 and S3
    # --------------------------------------------------------
    print("\n[2/6] Building unified inverted indexes across S2 and S3 (Pass 1)...")
    t0 = time.time()

    unified_name_index = defaultdict(list)
    unified_hs_index = defaultdict(list)
    unified_city_house_index = defaultdict(list)
    s2_rare_tokens = set()
    s3_rare_tokens = set()

    for label, df in [("S2", s2), ("S3", s3)]:
        sub_t0 = time.time()
        ids = list(df["entity_id"])
        names = list(df["business_name"])
        addresses = list(df["business_address"])

        raw_ch_index = defaultdict(list)
        token_freq = defaultdict(int)

        for entity_id, name, addr in zip(ids, names, addresses):
            # Strategy 1: V3 normalized business name
            n_key = normalize_name(name)
            if n_key:
                unified_name_index[n_key].append(entity_id)

            # Strategy 2: House + Street
            hs_k = house_street_key(addr)
            if hs_k:
                unified_hs_index[hs_k].append(entity_id)

            # Strategy 3: City + House
            if addr:
                c = extract_city(addr)
                h = extract_house_number(addr)
                if c and h:
                    raw_ch_index[f"{c}:{h}"].append(entity_id)

                # Strategy 4: Address token frequency counting (per-source)
                toks = set(re.findall(r"[a-z0-9]+", str(addr).lower()))
                for tok in toks:
                    token_freq[tok] += 1

        # Apply per-source max block size <= 500 for City+House
        for ch_k, entity_list in raw_ch_index.items():
            if len(entity_list) <= city_house_max_block_size:
                unified_city_house_index[ch_k].extend(entity_list)
        del raw_ch_index

        # Identify rare tokens (frequency <= 50 per source)
        rare_toks = {tok for tok, count in token_freq.items() if count <= rare_token_frequency_limit}
        if label == "S2":
            s2_rare_tokens = rare_toks
        else:
            s3_rare_tokens = rare_toks
        del token_freq

        print(f"      Processed {label} ({len(df):,} records) in {time.time()-sub_t0:.2f}s")

    # Convert Name, House+Street, and City+House lists to compact tuples
    unified_name_index = {k: tuple(v) for k, v in unified_name_index.items()}
    unified_hs_index = {k: tuple(v) for k, v in unified_hs_index.items()}
    unified_city_house_index = {k: tuple(v) for k, v in unified_city_house_index.items()}

    print(f"      Pass 1 completed in {time.time()-t0:.2f}s")
    print(f"      Identified S2 rare tokens: {len(s2_rare_tokens):,}, S3 rare tokens: {len(s3_rare_tokens):,}")

    # --------------------------------------------------------
    # 3. Build Rare Token Inverted Index (Pass 2)
    # --------------------------------------------------------
    print("\n[3/6] Populating rare token inverted index (Pass 2)...")
    t0 = time.time()

    unified_rare_token_index = defaultdict(list)
    for label, df, rare_tokens in [("S2", s2, s2_rare_tokens), ("S3", s3, s3_rare_tokens)]:
        ids = list(df["entity_id"])
        addresses = list(df["business_address"])
        for entity_id, addr in zip(ids, addresses):
            if addr:
                toks = set(re.findall(r"[a-z0-9]+", str(addr).lower()))
                for tok in toks:
                    if tok in rare_tokens:
                        unified_rare_token_index[tok].append(entity_id)

    unified_rare_token_index = {k: tuple(v) for k, v in unified_rare_token_index.items()}
    del s2_rare_tokens, s3_rare_tokens
    del s2, s3  # Free DataFrame memory immediately

    print(f"      Unified Name blocks       : {len(unified_name_index):,}")
    print(f"      Unified House+Street blocks: {len(unified_hs_index):,}")
    print(f"      Unified City+House (<=500) : {len(unified_city_house_index):,}")
    print(f"      Unified Rare Tokens (<=50) : {len(unified_rare_token_index):,}")
    print(f"      Indexes finalized in {time.time()-t0:.2f}s")

    # --------------------------------------------------------
    # 4. Stream Candidate Generation in Chunks to Temporary Files
    # --------------------------------------------------------
    tmp_output = output_file + ".tmp"
    tmp_flat_output = flat_output_file + ".tmp"

    print("\n[4/6] Streaming candidate generation chunk-by-chunk to disk...")
    print(f"      Chunk size: {chunk_size:,} records")
    print(f"      Temporary grouped file: {tmp_output}")
    print(f"      Temporary flat file   : {tmp_flat_output}")

    gen_t0 = time.time()
    total_candidate_pairs = 0
    zero_candidate_s1_count = 0
    unique_s1_with_candidates = 0

    # Extract S1 columns as lists for maximum iteration speed
    s1_ids = list(s1["entity_id"])
    s1_names = [normalize_name(x) for x in s1["business_name"]]
    s1_addresses = list(s1["business_address"])
    s1_hs = [house_street_key(x) for x in s1_addresses]
    s1_ch = [city_house_key(extract_city(x), extract_house_number(x)) for x in s1_addresses]
    s1_tokens = [get_address_tokens(x) for x in s1_addresses]
    del s1, s1_addresses

    with open(tmp_output, "w", encoding="utf-8", buffering=1024*1024*8) as f_grp, \
         open(tmp_flat_output, "w", encoding="utf-8", buffering=1024*1024*16) as f_flat:

        # Write exact required headers
        f_grp.write("source1_entity_id\tcandidate_entity_ids\n")
        f_flat.write("source1_entity_id\tcandidate_entity_id\n")

        for start in range(0, total_s1, chunk_size):
            end = min(start + chunk_size, total_s1)
            chunk_t0 = time.time()

            chunk_ids = s1_ids[start:end]
            chunk_names = s1_names[start:end]
            chunk_hs = s1_hs[start:end]
            chunk_ch = s1_ch[start:end]
            chunk_tokens = s1_tokens[start:end]

            chunk_cands_count = 0

            for s1_id, name_key, hs_key, ch_key, tokens in zip(
                chunk_ids, chunk_names, chunk_hs, chunk_ch, chunk_tokens
            ):
                cands = set()

                # Strategy 1: V3 normalized business name
                if name_key:
                    cands.update(unified_name_index.get(name_key, ()))

                # Strategy 2: House + Street
                if hs_key:
                    cands.update(unified_hs_index.get(hs_key, ()))

                # Strategy 3: City + House (pre-filtered to block size <= 500)
                if ch_key:
                    cands.update(unified_city_house_index.get(ch_key, ()))

                # Strategy 4: Rare address tokens (pre-filtered to frequency <= 50)
                for tok in tokens:
                    if tok in unified_rare_token_index:
                        cands.update(unified_rare_token_index[tok])

                if not cands:
                    zero_candidate_s1_count += 1
                    f_grp.write(f"{s1_id}\t\n")
                else:
                    unique_s1_with_candidates += 1
                    # Sort candidates for deterministic output
                    sorted_cands = sorted(cands)
                    chunk_cands_count += len(sorted_cands)

                    # Write grouped line
                    f_grp.write(f"{s1_id}\t{','.join(sorted_cands)}\n")

                    # Write flat lines
                    for cand_id in sorted_cands:
                        f_flat.write(f"{s1_id}\t{cand_id}\n")

            total_candidate_pairs += chunk_cands_count
            chunk_elapsed = time.time() - chunk_t0
            rate = len(chunk_ids) / chunk_elapsed if chunk_elapsed > 0 else 0
            pct = (end / total_s1) * 100

            print(
                f"      Processed {end:9,d} / {total_s1:,} S1 ({pct:5.1f}%) | "
                f"Chunk rate: {rate:7.0f} rec/s | "
                f"Total candidates: {total_candidate_pairs:11,d}"
            )

    print(f"\n      Candidate generation finished in {time.time()-gen_t0:.2f}s")

    # --------------------------------------------------------
    # 5. Lightweight Verification Before Atomic Move
    # --------------------------------------------------------
    print("\n[5/6] Performing lightweight safety validation...")
    val_t0 = time.time()

    assert os.path.exists(tmp_output), f"Missing {tmp_output}"
    assert os.path.exists(tmp_flat_output), f"Missing {tmp_flat_output}"

    # Verify grouped output
    grouped_row_count = 0
    seen_s1 = set()
    dup_s1 = 0
    with open(tmp_output, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        assert header == ["source1_entity_id", "candidate_entity_ids"], f"Invalid grouped header: {header}"
        for line in f:
            grouped_row_count += 1
            s1_id = line.split("\t", 1)[0]
            if s1_id in seen_s1:
                dup_s1 += 1
            seen_s1.add(s1_id)

    # Verify flat header
    with open(tmp_flat_output, "r", encoding="utf-8") as f:
        flat_header = f.readline().rstrip("\n").split("\t")
        assert flat_header == ["source1_entity_id", "candidate_entity_id"], f"Invalid flat header: {flat_header}"

    assert grouped_row_count == total_s1, f"Grouped row count mismatch: {grouped_row_count} != {total_s1}"
    assert dup_s1 == 0, f"Found {dup_s1} duplicate S1 rows in grouped output!"

    print(f"      Grouped rows verified: {grouped_row_count:,} (Expected: {total_s1:,})")
    print(f"      Duplicate S1 rows    : {dup_s1}")
    print(f"      Validation passed in {time.time()-val_t0:.2f}s")

    # --------------------------------------------------------
    # 6. Atomic Replace & Final Report
    # --------------------------------------------------------
    print("\n[6/6] Finalizing output files...")
    os.replace(tmp_output, output_file)
    os.replace(tmp_flat_output, flat_output_file)

    grouped_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    flat_size_mb = os.path.getsize(flat_output_file) / (1024 * 1024)
    total_elapsed = time.time() - total_pipeline_t0

    print("\n" + "=" * 75)
    print("BLOCKING V1 CANDIDATE GENERATION COMPLETED SUCCESSFULLY")
    print("=" * 75)
    print(f"1. Official Grouped File : {output_file} ({grouped_size_mb:.1f} MB)")
    print(f"2. Matcher Flat File     : {flat_output_file} ({flat_size_mb:.1f} MB)")
    print(f"3. Total S1 Records      : {total_s1:,}")
    print(f"4. Total Candidate Pairs : {total_candidate_pairs:,}")
    print(f"5. S1 with Candidates    : {unique_s1_with_candidates:,} ({(unique_s1_with_candidates/total_s1)*100:.2f}%)")
    print(f"6. S1 Zero Candidates    : {zero_candidate_s1_count:,} ({(zero_candidate_s1_count/total_s1)*100:.2f}%)")
    print(f"7. Total Pipeline Runtime: {total_elapsed:.2f}s ({total_elapsed/60:.2f} min)")
    print("=" * 75)

    return {
        "output_file": output_file,
        "flat_output_file": flat_output_file,
        "total_s1": total_s1,
        "total_candidate_pairs": total_candidate_pairs,
        "zero_candidate_s1_count": zero_candidate_s1_count,
        "runtime_seconds": total_elapsed
    }


# ============================================================
# COMMAND LINE ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CODEHex Blocking V1 candidate generation")
    parser.add_argument("--train-dir", default="student_resource/dataset/train", help="Path to train dataset directory")
    parser.add_argument("--output", default="student_resource/candidate_pairs.tsv", help="Path to official grouped TSV")
    parser.add_argument("--flat-output", default="student_resource/candidate_pairs_flat.tsv", help="Path to matcher-ready flat TSV")
    parser.add_argument("--chunk-size", type=int, default=100000, help="Chunk size for streaming generation")
    parser.add_argument("--limit-s1", type=int, default=None, help="Optional limit on S1 records for quick validation dry-run")
    parser.add_argument("--limit-s2-s3", type=int, default=None, help="Optional limit on S2/S3 records for micro smoke test")

    args = parser.parse_args()

    run_blocking_pipeline(
        train_dir=args.train_dir,
        output_file=args.output,
        flat_output_file=args.flat_output,
        chunk_size=args.chunk_size,
        limit_s1=args.limit_s1,
        limit_s2_s3=args.limit_s2_s3
    )