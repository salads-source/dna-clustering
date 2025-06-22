import numpy as np, faiss, time
from collections import defaultdict
from sentence_transformers import SentenceTransformer

MODEL_DIR   = "dna_sbert_trainer"
INPUT_FILE  = "NoisyStrands.txt"
OUTPUT_FILE = "PredictedClusters.txt"

BATCH_SIZE  = 1024        
BLOCK       = 50_000        # process 50 k embeddings per chunk
THRESH_PAIR = 0.92         
MIN_SIZE    = 100

print("load model")
model = SentenceTransformer(MODEL_DIR)
model.max_seq_length = 230

print("load sequences")
with open(INPUT_FILE) as fh:
    seqs = [l.strip().upper() for l in fh if l.strip()]
N = len(seqs)

print("encode all (fp16) …")
emb = model.encode(seqs, batch_size=BATCH_SIZE,
                   normalize_embeddings=True,
                   convert_to_numpy=True).astype("float16")

parent = np.arange(N, dtype=np.int32)     # union find

def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i
def union(i, j):
    ri, rj = find(i), find(j)
    if ri != rj:
        parent[rj] = ri

print("block-wise range search …")
index = faiss.IndexFlatIP(256)            # cosine on L2 norm
start = 0
while start < N:
    end = min(start + BLOCK, N)
    index.reset()
    index.add(emb[start:end])
    lim, D, I = index.range_search(emb[start:end], THRESH_PAIR)
    for idx in range(end - start):
        for j_ptr in range(lim[idx], lim[idx+1]):
            j = I[j_ptr]
            if idx != j:                 
                union(start + idx, start + j)
    start = end
    print(f"block {end//BLOCK} done")

print("collect clusters")
clusters = defaultdict(list)
for i in range(N):
    clusters[find(i)].append(i)
final = [m for m in clusters.values() if len(m) == MIN_SIZE]
print(f"clusters size {MIN_SIZE}: {len(final)}")

print("write file")
with open(OUTPUT_FILE, "w") as fh:
    for cid, members in enumerate(final):
        fh.write(f"CLUSTER {cid}\n")
        for idx in members:
            fh.write(f"{seqs[idx]}\n")
        fh.write("\n")
print("done")
