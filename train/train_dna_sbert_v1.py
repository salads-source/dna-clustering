import argparse, os, random, json
from collections import defaultdict

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from sentence_transformers import SentenceTransformerTrainer
from sentence_transformers import SentenceTransformer, models

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--clusters_file", required=True)
    p.add_argument("--output_dir",   default="dna_sbert_trainer")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_len", type=int, default=202)   # 200 bp 
    return p.parse_args()

def load_clusters(path):
    clusters = defaultdict(list)
    cid = -1
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith("CLUSTER"):
                cid = int(line.split()[1])
            else:
                clusters[cid].append(line.upper())
    return clusters

def build_tokenizer(save_dir):

    vocab = {
        "[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[MASK]": 3, "[UNK]": 4,
        "A": 5, "C": 6, "G": 7, "T": 8,
    }

    tok = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    tokenizer.save_pretrained(save_dir)
    return tokenizer

def build_bert(vocab_size, max_len, save_dir):
    from transformers import BertConfig, BertModel

    cfg = BertConfig(
        vocab_size=vocab_size,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=1024,
        max_position_embeddings=max_len,
    )
    model = BertModel(cfg)
    model.save_pretrained(save_dir)

    #  for sentence-transformers to know the max sequence length
    with open(os.path.join(save_dir, "sentence_bert_config.json"), "w") as fh:
        json.dump({"max_seq_length": max_len}, fh)

    return model
  
def make_dataset(clusters):
    from datasets import Dataset

    anchors, positives = [], []
    for seqs in clusters.values():
        k = len(seqs)
        for i, s1 in enumerate(seqs):
            anchors.append(s1)
            positives.append(seqs[(i + 1) % k])   
    #  identical shuffling
    shuffled = list(zip(anchors, positives))
    random.shuffle(shuffled)
    anchors, positives = zip(*shuffled)

    return Dataset.from_dict({"anchor": anchors, "positive": positives})


#  main training routine
def train(cfg):
    #  data split
    clusters = load_clusters(cfg.clusters_file)
    dataset = make_dataset(clusters)
    split = dataset.train_test_split(test_size=0.01, shuffle=True)
    train_ds = split["train"]
    eval_ds = split["test"]

    #  tokenizer + encoder
    os.makedirs(cfg.output_dir, exist_ok=True)
    tok_dir = os.path.join(cfg.output_dir, "bert") 
    model_dir = tok_dir
    tokenizer = build_tokenizer(tok_dir)
    _bert = build_bert(len(tokenizer), cfg.max_len, model_dir)

    transformer = models.Transformer(
        model_dir,
        tokenizer_name_or_path=tok_dir,
        max_seq_length=cfg.max_len,
    )
    
    pooling = models.Pooling(
        transformer.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
    )
    
    model = SentenceTransformer(modules=[transformer, pooling])

    # loss func
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    loss_fn = MultipleNegativesRankingLoss(model)

    # load training args
    from sentence_transformers.training_args import (
        SentenceTransformerTrainingArguments, BatchSamplers)

    args = SentenceTransformerTrainingArguments(
        output_dir            = cfg.output_dir,
        num_train_epochs      = cfg.epochs,
        per_device_train_batch_size = cfg.batch_size,
        per_device_eval_batch_size  = cfg.batch_size,
        learning_rate         = cfg.lr,
        warmup_ratio          = 0.1,
        fp16                  = True,          
        batch_sampler         = BatchSamplers.NO_DUPLICATES,
        eval_strategy         = "steps",
        eval_steps            = 100,
        save_strategy         = "steps",
        save_steps            = 100,
        logging_steps         = 50,
        run_name              = "dna-sbert",  
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        loss=loss_fn,
    )

    trainer.train()
    
    model.save_pretrained(cfg.output_dir)

if __name__ == "__main__":
    cfg = get_args()
    train(cfg)


# sample cmd line arg
'''
py train_dna_sbert.py --clusters_file UnderlyingClusters.txt --output_dir dna_sbert_trainer --epochs 3 --batch_size 256 --lr 2e-5
'''