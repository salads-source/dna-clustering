import argparse, random, re, pathlib
import numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer, util

def args_():
    p = argparse.ArgumentParser()
    p.add_argument("--model_root", default="dna_sbert_trainer",
                   help="folder that contains checkpoint-* subdirs")
    p.add_argument("--strands_file", default="NoisyStrands.txt")
    p.add_argument("--subset", type=int, default=512,
                   help="#strands to sample for the distance matrix")
    p.add_argument("--outfile", default="pairwise_distances.xlsx")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def load_subset(path, n, seed):
    random.seed(seed)
    with open(path) as fh:
        strands = [l.strip().upper() for l in fh if l.strip()]
    if len(strands) <= n:
        return strands
    return random.sample(strands, n)

def checkpoints(root):
    p = pathlib.Path(root)
    ck = sorted((d for d in p.glob("checkpoint-*") if d.is_dir()),
                key=lambda d: int(re.search(r"\d+", d.name).group()))
    return ck or [p]     # fall back to final model only

def main():
    cfg = args_()
    subset = load_subset(cfg.strands_file, cfg.subset, cfg.seed)
    n = len(subset)
    print(f"subset size = {n}")

    writer = pd.ExcelWriter(cfg.outfile, engine="openpyxl")

    for ck in checkpoints(cfg.model_root):
        tag = ck.name if ck.name.startswith("checkpoint") else "final"
        print(f"embedding with {tag} …")
        model = SentenceTransformer(str(ck), device="cpu")
        emb = model.encode(subset, batch_size=512, convert_to_numpy=True,
                           normalize_embeddings=True, dtype=np.float16)

        # cosine distance = 1 - cosine similarity
        dist = 1.0 - (emb @ emb.T).astype(np.float32)
        df = pd.DataFrame(dist, columns=[f"s{i}" for i in range(n)],
                                 index  =[f"s{i}" for i in range(n)])
        sheet = f"{tag[:28]}"          
        df.to_excel(writer, sheet_name=sheet)
        df.to_csv(f"pairwise_{tag}.csv")

    writer.close()
    print("wrote", cfg.outfile)

if __name__ == "__main__":
    main()
