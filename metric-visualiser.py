#!/usr/bin/env python
"""
plot_loss_lr.py
---------------
Read `trainer_state.json` from a Sentence-Transformers run and plot:

• training loss  
• evaluation loss (if present)  
• learning-rate schedule

Usage
-----
python plot_loss_lr.py dna_sbert_trainer  # folder that contains trainer_state.json
"""

import argparse, json, pathlib, matplotlib.pyplot as plt

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("model_dir", help="directory with trainer_state.json")
    p.add_argument("--out", default="loss_lr.png", help="output PNG file")
    return p.parse_args()

def main():
    cfg = get_args()
    state_file = pathlib.Path(cfg.model_dir, "trainer_state.json")
    if not state_file.exists():
        raise SystemExit(f"trainer_state.json not found in {cfg.model_dir}")

    with open(state_file) as f:
        state = json.load(f)["log_history"]

    steps, train_loss, eval_loss, lrs = [], [], [], []
    for entry in state:
        if "step" not in entry:
            continue
        steps.append(entry["step"])
        if "loss" in entry:          # training loss
            train_loss.append(entry["loss"])
        if "eval_loss" in entry:     # eval loss (may be sparse)
            eval_loss.append((entry["step"], entry["eval_loss"]))
        if "learning_rate" in entry:
            lrs.append(entry["learning_rate"])

    # ------- plot ---------
    fig, ax1 = plt.subplots(figsize=(6,4))
    ax1.plot(steps[:len(train_loss)], train_loss, label="train loss", color="tab:blue")
    if eval_loss:
        ev_steps, ev_vals = zip(*eval_loss)
        ax1.plot(ev_steps, ev_vals, label="eval loss", color="tab:green")
    ax1.set_xlabel("global step")
    ax1.set_ylabel("loss")
    ax1.legend(loc="upper right")

    ax2 = ax1.twinx()
    ax2.plot(steps[:len(lrs)], lrs, label="learning rate", color="tab:orange")
    ax2.set_ylabel("LR")
    ax2.legend(loc="upper left")

    plt.title("Training Loss / Eval Loss / Learning-Rate")
    plt.tight_layout()
    plt.savefig(cfg.out, dpi=200)
    print("saved", cfg.out)

if __name__ == "__main__":
    main()
