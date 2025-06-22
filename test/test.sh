#!/bin/bash
#SBATCH --job-name=dna_sbert_train
#SBATCH --gres=gpu:h100-96:1          # Request 1 NVIDIA H100 GPU (change if needed)
#SBATCH --time=3:00:00                 # Max run time 3 hours
#SBATCH -C h100                       # Feature name for GPU type
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --mail-user=youremail@nus.edu.sg
#SBATCH --mail-type=BEGIN,END,FAIL

# Load conda and activate env
. ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv

# Run your training script with args
# python train_dna_sbert_100.py --clusters_file UnderlyingClusters_100.txt --output_dir dna_sbert_trainer --epochs 3 --batch_size 128 --lr 5e-5
python test_dna_sbert_v1.py