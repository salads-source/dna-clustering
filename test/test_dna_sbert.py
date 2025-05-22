import time
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util

# ───────── test_params ─────────
MODEL_DIR   = "dna_sbert_trainer"
INPUT_FILE  = "NoisyStrands.txt"
OUTPUT_FILE = "PredictedClusters.txt"

BATCH_SIZE  = 1024
THRESHOLD   = 0.88
MIN_SIZE    = 5      
# ──────────────────────────

model = SentenceTransformer(MODEL_DIR)
model.max_seq_length = 202

with open(INPUT_FILE) as fh:
    strands = [l.strip().upper() for l in fh if l.strip()]
    strands = strands[:10000]
N = len(strands)
print(f"   {N:,} sequences")

emb = model.encode(
    strands,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True,   
    convert_to_tensor=True,
)

print("Clustering started")

t0 = time.time()
raw_clusters = util.community_detection(
    emb,
    min_community_size=MIN_SIZE,
    threshold=THRESHOLD,
)

print("raw clusters & sizes (first 10):",
      [len(c) for c in raw_clusters[:10]])
print("largest cluster size:", max(len(c) for c in raw_clusters))
print(f"raw clusters: {len(raw_clusters):,}")

# 5-member clusters
clusters = [c for c in raw_clusters if len(c) == 5]

print(f"writing results to {OUTPUT_FILE} …")
with open(OUTPUT_FILE, "w") as fh:
    for cid, member_ids in enumerate(clusters):
        fh.write(f"CLUSTER {cid}\n")
        for idx in member_ids:
            fh.write(f"{strands[idx]}\n")
        fh.write("\n")

