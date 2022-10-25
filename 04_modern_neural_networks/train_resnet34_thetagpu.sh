#!/bin/bash -l
#COBALT -t 60
#COBALT -q single-gpu
#COBALT -A ALCFAITP
#COBALT -n 1
<<<<<<< HEAD

#COBALT --attrs filesystems=home:grand
=======
#COBALT --attrs filesystems=home,grand
>>>>>>> upstream/main

# Set up software deps:
module load conda/2022-07-01
conda activate

# You have to point this to YOUR local copy of ai-science-training-series
cd /home/kprabhu/ai-science-training-series/04_modern_neural_networks

export TF_XLA_FLAGS="--tf_xla_auto_jit=2"
python train_resnet34.py
