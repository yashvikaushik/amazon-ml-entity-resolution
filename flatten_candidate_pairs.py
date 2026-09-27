import os
import sys
import time

def main():
    input_path = "student_resource/candidate_pairs.tsv"
    output_path = "student_resource/candidate_pairs_flat_FULL.tsv"

    print(f"Starting candidate pairs flattening...")
    print(f"Input file:  {input_path}")
    print(f"Output file: {output_path}")

    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} does not exist!")
        sys.exit(1)

    start_time = time.time()
    s1_count = 0
    pairs_count = 0
    zero_cand_count = 0

    BUFFER_SIZE = 100000
    buffer = []

    with open(input_path, "r", encoding="utf-8", buffering=32 * 1024 * 1024) as infile, \
         open(output_path, "w", encoding="utf-8", buffering=32 * 1024 * 1024) as outfile:
        
        # Read header
        header = infile.readline()
        outfile.write("source1_entity_id\tcandidate_entity_id\n")

        for line in infile:
            s1_count += 1
            line = line.rstrip("\r\n")
            if not line:
                zero_cand_count += 1
                continue

            parts = line.split("\t", 1)
            s1_id = parts[0]

            if len(parts) < 2 or not parts[1].strip():
                zero_cand_count += 1
                continue

            candidates = parts[1].strip().split(",")
            valid_cands = [c.strip() for c in candidates if c.strip()]

            if not valid_cands:
                zero_cand_count += 1
                continue

            for cand in valid_cands:
                buffer.append(f"{s1_id}\t{cand}\n")
                pairs_count += 1

            if len(buffer) >= BUFFER_SIZE:
                outfile.writelines(buffer)
                buffer.clear()

            if s1_count % 100000 == 0:
                elapsed = time.time() - start_time
                print(f"[Progress] S1 rows: {s1_count:,} | Pairs written: {pairs_count:,} | Zero-cand S1: {zero_cand_count:,} | Elapsed: {elapsed:.1f}s")
                sys.stdout.flush()

        if buffer:
            outfile.writelines(buffer)
            buffer.clear()

    elapsed = time.time() - start_time
    file_size_bytes = os.path.getsize(output_path)
    file_size_gb = file_size_bytes / (1024 ** 3)
    file_size_mb = file_size_bytes / (1024 ** 2)

    print("\n" + "=" * 60)
    print("FLATTENING COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Total S1 rows processed:      {s1_count:,}")
    print(f"Total flat candidate pairs:   {pairs_count:,}")
    print(f"Zero-candidate S1 count:      {zero_cand_count:,}")
    print(f"Output file size:             {file_size_bytes:,} bytes ({file_size_gb:.2f} GB / {file_size_mb:.2f} MB)")
    print(f"Output path:                  {os.path.abspath(output_path)}")
    print(f"Total time elapsed:           {elapsed:.1f}s")
    print("=" * 60)

if __name__ == "__main__":
    main()
