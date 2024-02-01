#!/bin/bash
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=ksun
#SBATCH --job-name=classinfo_mutation
#SBATCH --output=/projects/ksun3/%u/sbatch_log/%x-%N-%j.out
#SBATCH --error=/projects/ksun3/%u/sbatch_log/%x-%N-%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100.80gb:4
#SBATCH --ntasks-per-node=20
#SBATCH --mem=480G
#SBATCH --export=ALL
#SBATCH --time=5-00:00:00

set echo
umask 0022

# to see ID and state of GPUs assigned
nvidia-smi

module load gnu10

source 	~/Anaconda/etc/profile.d/conda.sh
conda activate dschat

accelerate launch llama_sft.py \
    --model_name_or_path codellama/CodeLlama-7b-hf \
    --data_path /projects/ksun3/dwu25/apr_datasets_processing/java_mutation/data/classinfo_mutation \
    --is_lora True \
    --model_max_length 1124 \
    --do_train \
    --do_eval True \
    --fp16 True \
    --output_dir /projects/ksun3/dwu25/trained_models/classinfo_mutation_output \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --eval_steps 1000 \
    --save_steps 150 \
    --learning_rate 5e-4 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --ddp_find_unused_parameters False \
