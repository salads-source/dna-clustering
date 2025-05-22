import argparse, os, random, json, time
from collections import defaultdict
from datasets import Dataset

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from transformers import BertConfig, BertModel

from sentence_transformers import util
import numpy as np

# params
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--clusters_file", required=True)
    p.add_argument("--output_dir",   default="dna_sbert_trainer")
    p.add_argument("--epochs",       type=int, default=5)
    p.add_argument("--batch_size",   type=int, default=256)
    p.add_argument("--lr",           type=float, default=5e-5)
    p.add_argument("--max_len",      type=int,  default=230)
    return p.parse_args()


def load_clusters(path):
    clusters = defaultdict(list)
    cid = -1
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line: continue
            if line.startswith("CLUSTER"):
                cid = int(line.split()[1])
            else:
                clusters[cid].append(line.upper())
    return clusters

def make_dataset(clusters):
    anchors, positives = [], []
    for seqs in clusters.values():      # each cluster len = 100
        for i, s1 in enumerate(seqs):
            anchors.append(s1)
            positives.append(seqs[(i+1) % len(seqs)])
    shuffled = list(zip(anchors, positives))
    random.shuffle(shuffled)
    a, p = zip(*shuffled)
    return Dataset.from_dict({"anchor": a, "positive": p})

def build_tokenizer(save_dir):
    vocab = {"[PAD]":0,"[CLS]":1,"[SEP]":2,"[MASK]":3,"[UNK]":4,
             "A":5,"C":6,"G":7,"T":8}
    tok = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]",
        mask_token="[MASK]", unk_token="[UNK]"
    )
    fast.save_pretrained(save_dir)

def build_bert(vocab_size, max_len, save_dir):
    cfg = BertConfig(
        vocab_size=vocab_size,
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=4,
        intermediate_size=1024,
        max_position_embeddings=max_len)
    model = BertModel(cfg); model.save_pretrained(save_dir)
    with open(os.path.join(save_dir,"sentence_bert_config.json"),"w") as fh:
        json.dump({"max_seq_length": max_len}, fh)

def sanity_checks(model, strands):
    import numpy as np

    # sim spread on 2000 random pairs
    pairs = random.sample(list(zip(strands, strands[1:])), k=2000)
    sims  = [float(util.cos_sim(
              model.encode(a, normalize_embeddings=True),
              model.encode(b, normalize_embeddings=True))) for a, b in pairs]

    sample = random.sample(strands, k=10_000)
    emb = model.encode(sample, batch_size=512,
                       normalize_embeddings=True, convert_to_tensor=True)
    clusters = util.community_detection(
        emb, min_community_size=100, threshold=0.92)
    sizes = [len(c) for c in clusters]
    print(f"🧩  Probe clustering (10 k strands, thr=0.92): "
          f"{len(clusters)} groups; median size {np.median(sizes):.0f}, "
          f"max {max(sizes)}\n")

# main training routine
def train(cfg):
    from sentence_transformers import SentenceTransformer, models, util
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    from sentence_transformers.training_args import (
        SentenceTransformerTrainingArguments, BatchSamplers)
    from sentence_transformers import SentenceTransformerTrainer

    clusters = load_clusters(cfg.clusters_file)
    dataset  = make_dataset(clusters)
    split    = dataset.train_test_split(test_size=0.01, shuffle=True)
    train_ds, eval_ds = split["train"], split["test"]

    os.makedirs(cfg.output_dir, exist_ok=True)
    model_dir = tok_dir = os.path.join(cfg.output_dir, "bert")
    build_tokenizer(tok_dir)
    build_bert(9, cfg.max_len, model_dir)

    transformer = models.Transformer(model_dir,
                                     tokenizer_name_or_path=tok_dir,
                                     max_seq_length=cfg.max_len)
    pooling = models.Pooling(
        transformer.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True)
    model = SentenceTransformer(modules=[transformer, pooling])

    loss_fn = MultipleNegativesRankingLoss(model)
    args = SentenceTransformerTrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        learning_rate=cfg.lr,
        warmup_ratio=0.1,
        fp16=False,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="steps", eval_steps=1000,
        save_strategy="steps", save_steps=1000,
        logging_steps=200,
        run_name="dna-sbert")
    trainer = SentenceTransformerTrainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=eval_ds, loss=loss_fn)

    trainer.train()
    model.save_pretrained(cfg.output_dir)
    print(f"\nModel saved to {cfg.output_dir}")

    # sanity check (cosine similarity)
    # print("\nRunning post-training sanity checks …")
    # all_strands = [ex["anchor"] for ex in dataset]
    # sanity_checks(model, all_strands)

if __name__ == "__main__":
    cfg = get_args()
    start = time.time()
    train(cfg)
    print(f"Total train + sanity time: {time.time() - start:.1f}s")
