#!/usr/bin/env python
import argparse, os, json, random, time
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, BertConfig, BertModel
from sentence_transformers import SentenceTransformer, models, util
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import SentenceTransformerTrainingArguments, BatchSamplers
from sentence_transformers import SentenceTransformerTrainer
from datasets import IterableDataset, Features, Value

def args_():
    p = argparse.ArgumentParser()
    p.add_argument("--clusters_file", required=True)
    p.add_argument("--output_dir", default="dna_sbert_trainer")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max_len", type=int, default=230)
    p.add_argument("--fp16", type=int, default=1)
    return p.parse_args()

def pair_gen(path):
    buf=[]
    with open(path) as fh:
        for ln in fh:
            t=ln.strip()
            if not t: continue
            if t.startswith("CLUSTER"):
                if buf:
                    for i,s in enumerate(buf):
                        yield {"anchor":s,"positive":buf[(i+1)%len(buf)]}
                buf=[]
            else:
                buf.append(t.upper())
    if buf:
        for i,s in enumerate(buf):
            yield {"anchor":s,"positive":buf[(i+1)%len(buf)]}

def count_pairs(path):
    total,cur=0,0
    with open(path) as fh:
        for ln in fh:
            if ln.startswith("CLUSTER"):
                if cur: total+=cur
                cur=0
            elif ln.strip():
                cur+=1
    return total+cur

def tok(save):
    vocab={"[PAD]":0,"[CLS]":1,"[SEP]":2,"[MASK]":3,"[UNK]":4,"A":5,"C":6,"G":7,"T":8}
    t=Tokenizer(WordLevel(vocab,unk_token="[UNK]")); t.pre_tokenizer=Whitespace()
    fast=PreTrainedTokenizerFast(tokenizer_object=t,
        pad_token="[PAD]",cls_token="[CLS]",sep_token="[SEP]",
        mask_token="[MASK]",unk_token="[UNK]")
    fast.save_pretrained(save)

def bert(vsz,ml,save):
    cfg=BertConfig(vocab_size=vsz,hidden_size=256,num_hidden_layers=6,
                   num_attention_heads=4,intermediate_size=1024,
                   max_position_embeddings=ml)
    BertModel(cfg).save_pretrained(save)
    with open(os.path.join(save,"sentence_bert_config.json"),"w") as f:
        json.dump({"max_seq_length":ml},f)

def main():
    cfg=args_()
    os.makedirs(cfg.output_dir,exist_ok=True)
    tok_dir=os.path.join(cfg.output_dir,"bert")
    tok(tok_dir); bert(9,cfg.max_len,tok_dir)
    transformer=models.Transformer(tok_dir,tokenizer_name_or_path=tok_dir,
                                   max_seq_length=cfg.max_len)
    pool=models.Pooling(transformer.get_word_embedding_dimension(),
                        pooling_mode_mean_tokens=True)
    model=SentenceTransformer(modules=[transformer,pool])
    features=Features({"anchor":Value("string"),"positive":Value("string")})
    train_ds=IterableDataset.from_generator(lambda: pair_gen(cfg.clusters_file),
                                            features=features)
    eval_ds=train_ds.take(100_000)
    steps=count_pairs(cfg.clusters_file)//cfg.batch_size*cfg.epochs
    args=SentenceTransformerTrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        learning_rate=cfg.lr,
        warmup_ratio=0.1,
        fp16=bool(cfg.fp16),
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        max_steps=steps,
        eval_strategy="steps",eval_steps=2000,
        save_strategy="steps",save_steps=2000,
        logging_steps=500,
        gradient_checkpointing=True,
        run_name="dna-sbert-100-stream")
    trainer=SentenceTransformerTrainer(model=model,args=args,
                                       train_dataset=train_ds,
                                       eval_dataset=eval_ds,
                                       loss=MultipleNegativesRankingLoss(model))
    trainer.train(); model.save_pretrained(cfg.output_dir)
    sample=[next(pair_gen(cfg.clusters_file))["anchor"] for _ in range(10000)]
    sims=[float(util.cos_sim(model.encode(a,normalize_embeddings=True),
                             model.encode(b,normalize_embeddings=True)))
          for a,b in random.sample(list(zip(sample,sample[1:])),2000)]
    print(min(sims),max(sims))

if __name__=="__main__":
    t=time.time(); main(); print("elapsed",time.time()-t)
