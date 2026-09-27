import pandas as pd
import re
from collections import defaultdict

BASE = "student_resource/dataset/train"

S1_FILE = f"{BASE}/train_source1.tsv"
S2_FILE = f"{BASE}/train_source2.tsv"
S3_FILE = f"{BASE}/train_source3.tsv"
GT_FILE = f"{BASE}/train_ground_truth.tsv"

PREFIX_LEN = 4
MAX_BLOCK_SIZE = 200
SAMPLE_SIZE = 1000


def street_prefix_key(address, prefix_len=4):

    if not address:
        return ""

    text = str(address).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    tokens = text.split()

    if len(tokens) < 2:
        return ""

    if not tokens[0].isdigit():
        return ""

    return f"{tokens[0]}:{tokens[1][:prefix_len]}"


print("=" * 65)
print("INCREMENTAL HARD-CASE EVALUATION")
print("=" * 65)

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

print("\nLoading source data...")

s1 = pd.read_csv(
    S1_FILE, sep="\t", dtype=str, keep_default_na=False
)

s2 = pd.read_csv(
    S2_FILE, sep="\t", dtype=str, keep_default_na=False
)

s3 = pd.read_csv(
    S3_FILE, sep="\t", dtype=str, keep_default_na=False
)

gt = pd.read_csv(
    GT_FILE, sep="\t", dtype=str, keep_default_na=False
)

print("Loaded.")


# ------------------------------------------------------------
# FAST LOOKUPS
# ------------------------------------------------------------

s1_address = dict(
    zip(s1["entity_id"], s1["business_address"])
)

# ------------------------------------------------------------
# BUILD HARD-CASE INDEX
# ------------------------------------------------------------

print("\nBuilding hard-case indexes...")

s2_index = defaultdict(list)

for entity_id, address in zip(
    s2["entity_id"],
    s2["business_address"]
):
    key = street_prefix_key(address)

    if key:
        s2_index[key].append(entity_id)


s3_index = defaultdict(list)

for entity_id, address in zip(
    s3["entity_id"],
    s3["business_address"]
):
    key = street_prefix_key(address)

    if key:
        s3_index[key].append(entity_id)


# Remove large blocks
s2_index = {
    k: v
    for k, v in s2_index.items()
    if len(v) <= MAX_BLOCK_SIZE
}

s3_index = {
    k: v
    for k, v in s3_index.items()
    if len(v) <= MAX_BLOCK_SIZE
}

print("Valid S2 blocks:", len(s2_index))
print("Valid S3 blocks:", len(s3_index))


# ------------------------------------------------------------
# LOAD EXISTING V1 CANDIDATES
# ------------------------------------------------------------

print("\nLoading existing V1 candidate file...")

candidate_file = "student_resource/candidate_pairs_flat.tsv"

v1_candidates = pd.read_csv(
    candidate_file,
    sep="\t",
    dtype=str
)

print("V1 candidate rows:", len(v1_candidates))


# Build lookup for sampled S1 only
v1_lookup = defaultdict(set)

for s1_id, candidate_id in zip(
    v1_candidates["source1_entity_id"],
    v1_candidates["candidate_entity_id"]
):
    v1_lookup[s1_id].add(candidate_id)

print("V1 lookup built.")


# ------------------------------------------------------------
# SAMPLE S1 RECORDS
# ------------------------------------------------------------

sample_gt = gt.sample(
    n=min(SAMPLE_SIZE, len(gt)),
    random_state=42
)


# ------------------------------------------------------------
# EVALUATION
# ------------------------------------------------------------

total_true = 0

already_v1 = 0
hard_case_recovered = 0
new_recoveries = 0

hard_case_candidates = 0
new_candidate_pairs = 0

examples = []


for _, row in sample_gt.iterrows():

    s1_id = row["source1_entity_id"]

    matched_string = row["matched_entity_ids"]

    if not matched_string:
        continue

    true_ids = {
        x.strip()
        for x in matched_string.split(",")
        if x.strip()
    }

    total_true += len(true_ids)

    # Existing V1 candidates
    existing = v1_lookup.get(
        s1_id,
        set()
    )

    existing_hits = true_ids.intersection(existing)

    already_v1 += len(existing_hits)

    # Hard-case candidates
    address = s1_address.get(
        s1_id,
        ""
    )

    key = street_prefix_key(address)

    hard_candidates = set()

    if key:

        hard_candidates.update(
            s2_index.get(key, [])
        )

        hard_candidates.update(
            s3_index.get(key, [])
        )

    hard_case_candidates += len(hard_candidates)

    # True matches recovered by hard-case
    hard_hits = true_ids.intersection(
        hard_candidates
    )

    hard_case_recovered += len(hard_hits)

    # IMPORTANT:
    # only count matches that V1 had MISSED
    new_hits = hard_hits - existing

    new_recoveries += len(new_hits)

    # Additional candidate pairs
    new_candidates = hard_candidates - existing

    new_candidate_pairs += len(new_candidates)

    if new_hits and len(examples) < 10:

        examples.append(
            (
                s1_id,
                list(new_hits)
            )
        )


# ------------------------------------------------------------
# RESULTS
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("INCREMENTAL HARD-CASE RESULTS")
print("=" * 65)

print("Sampled true matches:", total_true)

print(
    "Already recovered by V1:",
    already_v1
)

print(
    "Hard-case raw recovery:",
    hard_case_recovered
)

print(
    "NEW true matches recovered:",
    new_recoveries
)

print(
    "Hard-case candidate pairs:",
    hard_case_candidates
)

print(
    "NEW candidate pairs added:",
    new_candidate_pairs
)

if total_true > 0:

    baseline_recall = (
        already_v1 / total_true
    ) * 100

    incremental_recall = (
        new_recoveries / total_true
    ) * 100

    combined_recall = (
        (already_v1 + new_recoveries)
        / total_true
    ) * 100

    print(
        f"\nV1 sampled recall: "
        f"{baseline_recall:.4f}%"
    )

    print(
        f"Incremental hard-case recall: "
        f"{incremental_recall:.4f}%"
    )

    print(
        f"Combined sampled recall: "
        f"{combined_recall:.4f}%"
    )

print("\nNew recovery examples:")

for s1_id, hits in examples:

    print(
        s1_id,
        "->",
        hits[:5]
    )

print("=" * 65)