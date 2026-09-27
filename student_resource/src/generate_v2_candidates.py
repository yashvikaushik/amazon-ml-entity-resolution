"""
Production Generation Pipeline for Blocking V2 Candidate Pairs.

Generates:
1. student_resource/candidate_pairs_v2.tsv (Grouped format)
2. student_resource/candidate_pairs_v2_flat.tsv (Flat format)

Combines frozen V1 candidates with V2-A (<=100), V2-B (<=100), and V2-C (<=50).
Expected unique candidate pairs: 361,997,158
Expected candidate recall: 90.7717%
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
    """Extract distinctive business name tokens for V2-C."""
    if not name:
        return set()
    cleaned = clean_string(name)
    tokens = cleaned.split()
    return {t for t in tokens if len(t) >= 4 and t not in LEGAL_STOPWORDS and not t.isdigit()}


def generate_v2_candidates(
    train_dir: str = "student_resource/dataset/train",
    v1_candidate_file: str = "student_resource/candidate_pairs.tsv",
    output_file: str = "student_resource/candidate_pairs_v2.tsv",
    flat_output_file: str = "student_resource/candidate_pairs_v2_flat.tsv",
):
    start_time = time.time()

    print("=" * 80)
    print("CODEHex - PRODUCTION BLOCKING V2 CANDIDATE GENERATION")
    print("=" * 80)

    s1_file = os.path.join(train_dir, "train_source1.tsv")
    s2_file = os.path.join(train_dir, "train_source2.tsv")
    s3_file = os.path.join(train_dir, "train_source3.tsv")

    # 1. Check baseline V1 file
    if not os.path.exists(v1_candidate_file):
        raise FileNotFoundError(f"Missing V1 candidates file: {v1_candidate_file}")
    v1_mtime_before = os.path.getmtime(v1_candidate_file)
    v1_size_before = os.path.getsize(v1_candidate_file)

    # 2. Build V2-A (Distinctive Address Tokens <= 100) indexes
    t0 = time.time()
    print("\n[1/4] Building V2-A (Distinctive Address Tokens <= 100) indexes...")
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
    print("\n[2/4] Building V2-B (House + Street Prefix <= 100) indexes...")
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

    # 4. Build V2-C (Distinctive Name Tokens <= 50) indexes
    t0 = time.time()
    print("\n[3/4] Building V2-C (Distinctive Name Tokens <= 50) indexes...")
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

    s2_name_valid_50 = {t for t, c in s2_name_freq.items() if c <= 50}
    s3_name_valid_50 = {t for t, c in s3_name_freq.items() if c <= 50}
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
                    if t in s2_name_valid_50:
                        s2_name_index[t].append(e_id)

    with open(s3_file, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.split("\t")
            if len(parts) > 1:
                e_id = parts[0]
                for t in get_name_tokens(parts[1]):
                    if t in s3_name_valid_50:
                        s3_name_index[t].append(e_id)

    s2_name_index = {k: tuple(v) for k, v in s2_name_index.items()}
    s3_name_index = {k: tuple(v) for k, v in s3_name_index.items()}
    del s2_name_valid_50, s3_name_valid_50
    print(f"      V2-C indexes built in {time.time()-t0:.2f}s")

    # 5. Stream candidate generation chunk-by-chunk to temporary files
    tmp_output = output_file + ".tmp"
    tmp_flat_output = flat_output_file + ".tmp"

    print("\n[4/4] Streaming candidate generation chunk-by-chunk to temporary files...")
    print(f"      Temporary grouped file: {tmp_output}")
    print(f"      Temporary flat file   : {tmp_flat_output}")

    gen_t0 = time.time()
    total_candidate_pairs = 0
    zero_candidate_s1_count = 0
    processed_s1 = 0

    with open(tmp_output, "w", encoding="utf-8", buffering=1024*1024*16) as f_grp, \
         open(tmp_flat_output, "w", encoding="utf-8", buffering=1024*1024*32) as f_flat, \
         open(s1_file, "r", encoding="utf-8", buffering=1024*1024*8) as f_s1, \
         open(v1_candidate_file, "r", encoding="utf-8", buffering=1024*1024*16) as f_v1:

        # Write exact required headers
        f_grp.write("source1_entity_id\tcandidate_entity_ids\n")
        f_flat.write("source1_entity_id\tcandidate_entity_id\n")

        # Skip headers of input files
        next(f_s1)
        next(f_v1)

        for line_s1, line_v1 in zip(f_s1, f_v1):
            processed_s1 += 1

            parts_s1 = line_s1.split("\t")
            s1_id = parts_s1[0]
            s1_name = parts_s1[1] if len(parts_s1) > 1 else ""
            s1_addr = parts_s1[2] if len(parts_s1) > 2 else ""

            # 1. Base V1 candidates
            idx_v1 = line_v1.find("\t")
            v1_cands_str = line_v1[idx_v1+1:].rstrip("\r\n") if idx_v1 != -1 else ""
            cands = set(v1_cands_str.split(",")) if v1_cands_str else set()

            # 2. V2-A: Address tokens <= 100
            s1_addr_toks = get_address_tokens(s1_addr)
            for tok in s1_addr_toks:
                if tok in s2_v2a_index:
                    cands.update(s2_v2a_index[tok])
                if tok in s3_v2a_index:
                    cands.update(s3_v2a_index[tok])

            # 3. V2-B: House + Street Prefix <= 100
            s1_hs_keys = extract_house_street_keys(s1_addr)
            for k in s1_hs_keys:
                if k in s2_hs_index:
                    cands.update(s2_hs_index[k])
                if k in s3_hs_index:
                    cands.update(s3_hs_index[k])

            # 4. V2-C: Distinctive Name Tokens <= 50
            s1_name_toks = get_name_tokens(s1_name)
            for tok in s1_name_toks:
                if tok in s2_name_index:
                    cands.update(s2_name_index[tok])
                if tok in s3_name_index:
                    cands.update(s3_name_index[tok])

            cand_count = len(cands)
            total_candidate_pairs += cand_count

            if cand_count == 0:
                zero_candidate_s1_count += 1
                f_grp.write(f"{s1_id}\t\n")
            else:
                cands_list = sorted(cands)
                f_grp.write(f"{s1_id}\t{','.join(cands_list)}\n")
                for c_id in cands_list:
                    f_flat.write(f"{s1_id}\t{c_id}\n")

            if processed_s1 % 500_000 == 0:
                print(f"      Processed {processed_s1:,} / 2,206,821 S1 records ({total_candidate_pairs:,} candidate pairs)...")

    gen_time = time.time() - gen_t0
    print(f"      Candidate stream completed in {gen_time:.2f}s")
    print(f"      Total S1 processed: {processed_s1:,}")
    print(f"      Total candidate pairs: {total_candidate_pairs:,}")
    print(f"      Zero-candidate S1: {zero_candidate_s1_count:,}")

    # 6. Streaming validation before finalizing
    print("\n[VALIDATION] Running lightweight streaming validation on temporary files...")
    val_t0 = time.time()

    # Validate Grouped File
    grouped_rows = 0
    grouped_zero_count = 0
    grouped_total_cands = 0
    grouped_first_5 = []
    grouped_last_5 = []

    with open(tmp_output, "r", encoding="utf-8") as f:
        h = next(f).rstrip("\r\n")
        assert h == "source1_entity_id\tcandidate_entity_ids", f"Invalid grouped header: {h}"
        for line in f:
            grouped_rows += 1
            parts = line.rstrip("\r\n").split("\t")
            assert len(parts) == 2, f"Malformed grouped line {grouped_rows}: {line[:100]}"
            c_str = parts[1]
            if not c_str:
                grouped_zero_count += 1
            else:
                grouped_total_cands += c_str.count(",") + 1
            if grouped_rows <= 5:
                grouped_first_5.append(line.rstrip("\r\n"))
            if len(grouped_last_5) >= 5:
                grouped_last_5.pop(0)
            grouped_last_5.append(line.rstrip("\r\n"))

    assert grouped_rows == 2206821, f"Expected 2,206,821 grouped rows, got {grouped_rows}"
    assert grouped_total_cands == total_candidate_pairs, f"Grouped candidate mismatch: {grouped_total_cands} vs {total_candidate_pairs}"

    # Validate Flat File
    flat_rows = 0
    flat_unique_s1 = set()
    flat_first_5 = []
    flat_last_5 = []

    with open(tmp_flat_output, "r", encoding="utf-8") as f:
        h = next(f).rstrip("\r\n")
        assert h == "source1_entity_id\tcandidate_entity_id", f"Invalid flat header: {h}"
        for line in f:
            flat_rows += 1
            parts = line.rstrip("\r\n").split("\t")
            assert len(parts) == 2, f"Malformed flat line {flat_rows}: {line[:100]}"
            flat_unique_s1.add(parts[0])
            if flat_rows <= 5:
                flat_first_5.append(line.rstrip("\r\n"))
            if len(flat_last_5) >= 5:
                flat_last_5.pop(0)
            flat_last_5.append(line.rstrip("\r\n"))

    assert flat_rows == total_candidate_pairs, f"Expected {total_candidate_pairs} flat rows, got {flat_rows}"
    assert len(flat_unique_s1) == (grouped_rows - grouped_zero_count), "Flat unique S1 mismatch"

    # Confirm V1 candidate file was untouched
    v1_mtime_after = os.path.getmtime(v1_candidate_file)
    v1_size_after = os.path.getsize(v1_candidate_file)
    assert v1_mtime_before == v1_mtime_after and v1_size_before == v1_size_after, "CRITICAL ERROR: V1 candidate_pairs.tsv was modified!"

    print(f"      Validation passed in {time.time()-val_t0:.2f}s!")

    # 7. Atomic Rename
    print("\n[ATOMIC RENAME] Moving temporary files to final target paths...")
    os.replace(tmp_output, output_file)
    os.replace(tmp_flat_output, flat_output_file)

    grouped_size_bytes = os.path.getsize(output_file)
    flat_size_bytes = os.path.getsize(flat_output_file)

    grouped_size_str = f"{grouped_size_bytes / (1024**3):.2f} GB ({grouped_size_bytes:,} bytes)"
    flat_size_str = f"{flat_size_bytes / (1024**3):.2f} GB ({flat_size_bytes:,} bytes)"

    total_runtime = time.time() - start_time
    runtime_str = f"{int(total_runtime // 60)}m {total_runtime % 60:.2f}s" if total_runtime >= 60 else f"{total_runtime:.2f}s"

    print("\n" + "=" * 80)
    print("FINAL V2 VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Grouped rows:           {grouped_rows:,}")
    print(f"Flat candidate pairs:   {flat_rows:,}")
    print(f"Unique S1 with cands:   {len(flat_unique_s1):,}")
    print(f"Zero-candidate S1:      {grouped_zero_count:,}")
    print(f"Grouped file size:      {grouped_size_str}")
    print(f"Flat file size:         {flat_size_str}")
    print(f"Total pipeline runtime: {runtime_str}")
    print("V1 candidate file intact: CONFIRMED")

    print("\nFirst 5 grouped rows:")
    for l in grouped_first_5:
        print("  ", l[:120] + ("..." if len(l) > 120 else ""))

    print("\nLast 5 grouped rows:")
    for l in grouped_last_5:
        print("  ", l[:120] + ("..." if len(l) > 120 else ""))

    print("\nFirst 5 flat rows:")
    for l in flat_first_5:
        print("  ", l)

    print("\nLast 5 flat rows:")
    for l in flat_last_5:
        print("  ", l)

    print("\n" + "=" * 80)
    print("FINAL V2 GENERATION COMPLETE")
    print("=" * 80)
    print(f"Grouped file:\n{os.path.abspath(output_file)}\n")
    print(f"Flat file:\n{os.path.abspath(flat_output_file)}\n")
    print(f"S1 rows:\n{grouped_rows:,}\n")
    print(f"Unique candidate pairs:\n{flat_rows:,}\n")
    print("Candidate recall:\n90.7717% expected\n")
    print(f"Zero-candidate S1:\n{grouped_zero_count:,}\n")
    print(f"Grouped file size:\n{grouped_size_str}\n")
    print(f"Flat file size:\n{flat_size_str}\n")
    print(f"Runtime:\n{runtime_str}")


if __name__ == "__main__":
    generate_v2_candidates()
