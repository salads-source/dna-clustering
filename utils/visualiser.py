import re, random, pathlib, numpy as np, matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer

MODEL_DIR   = "dna_sbert_trainer"
STRANDS_TXT = "NoisyStrands.txt"
SAMPLE      = 10_000
PERPLEXITY  = 30
TSNE_ITERS  = 750
SEED        = 42

random.seed(SEED)
np.random.seed(SEED)

with open(STRANDS_TXT) as f:
    strands = [s.strip().upper() for s in f if s.strip()]
if len(strands) > SAMPLE:
    strands = random.sample(strands, SAMPLE)

ckpts = sorted(
    [p for p in pathlib.Path(MODEL_DIR).glob("checkpoint-*") if p.is_dir()],
    key=lambda p: int(re.search(r"\d+", p.name).group()))
if not ckpts:
    ckpts = [pathlib.Path(MODEL_DIR)]

for cp in ckpts:
    tag = cp.name if cp.name.startswith("checkpoint") else "final"
    model = SentenceTransformer(str(cp), device="cpu")
    emb = model.encode(strands, batch_size=512, normalize_embeddings=True,
                       convert_to_numpy=True, dtype=np.float16)
    emb = PCA(n_components=50, random_state=SEED).fit_transform(emb)
    tsne = TSNE(n_components=2, perplexity=PERPLEXITY, n_iter=TSNE_ITERS,
                init="random", learning_rate="auto", random_state=SEED)
    xy = tsne.fit_transform(emb)
    plt.figure(figsize=(6,6))
    plt.scatter(xy[:,0], xy[:,1], s=4, alpha=0.6)
    plt.title(f"t-SNE – {tag}")
    plt.savefig(f"tsne_{tag}.png", dpi=200, bbox_inches="tight")
    plt.close()
