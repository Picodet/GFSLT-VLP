# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**GFSLT-VLP** (Gloss-Free Sign Language Translation via Visual-Language Pretraining) is a PyTorch-based deep learning project for sign language translation. It implements visual-language pretraining approaches to improve sign language translation without glosses.

Key features:
- Three training pipelines: VLP (Visual-Language Pretraining), VLP-V2 (joint visual-text pretraining), and SLT (downstream Sign Language Translation)
- Uses MBart as the text decoder and custom vision encoders (ResNet-based)
- Distributed training support via `torch.distributed.launch`
- Comprehensive evaluation metrics (BLEU, METEOR, ROUGE_L, CIDEr, WER)

## Setup & Dependencies

```bash
conda create -n gfslt python==3.8
conda activate gfslt
pip install -r requirements.txt
```

Key dependencies: PyTorch 1.11.0 (CUDA 11.3), transformers 4.32.0, timm 0.6.11, sacrebleu, wandb, loguru

## Common Commands

### Training

**VLP (Visual-Language Pretraining - basic visual encoder alignment):**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=1236 --use_env train_vlp.py --batch-size 4 --epochs 80 --opt sgd --lr 0.01 --output_dir out/vlp
```

**VLP-V2 (Joint visual encoder + text decoder pretraining with CLIP + masked self-supervision):**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=1236 --use_env train_vlp_v2.py --batch-size 4 --epochs 80 --opt sgd --lr 0.01 --output_dir out/vlp_v2 --training-refurbish True --noise-rate 0.15 --noise-type omit_last --random-shuffle False
```

**GFSLT-VLP (downstream task - sign language translation with finetuning):**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=1236 --use_env train_slt.py --batch-size 2 --epochs 200 --opt sgd --lr 0.01 --output_dir out/Gloss-Free --finetune ./out/vlp/checkpoint.pth
```

### Evaluation

Requires `nlgeval` package for full metrics. Run on single GPU for stability:
```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=1236 --use_env train_slt.py --batch-size 2 --epochs 200 --opt sgd --lr 0.01 --output_dir out/Gloss-Free --resume out/Gloss-Free/best_checkpoint.pth --eval
```

See [metrics/README.md](metrics/README.md) for nlgeval installation instructions.

## Architecture

### Core Training Scripts

- **train_vlp.py**: Visual-language pretraining with basic visual-text alignment (SLRCLIP model)
- **train_vlp_v2.py**: Advanced pretraining combining CLIP-style contrastive learning + masked language modeling
- **train_slt.py**: Downstream sign language translation task (gloss_free_model)

Both use hyperparameter management via `hpargparse` and `hpman` (hyperparameter manager).

### Key Modules

- **models.py**: Vision encoders (ResNet-based) and model architectures (SLRCLIP, gloss_free_model). Uses einops for tensor operations.
- **datasets.py**: S2T_Dataset (sign-to-text) with LMDB caching, video augmentation (vidaug), and normalization
- **augmentation.py**: Video augmentation strategies (SomeOf mixer for selecting augmentations)
- **utils.py**: Training utilities (optimizer creation, checkpointing, distributed helpers)
- **metrics.py**: Evaluation metrics computation (WER, BLEU, METEOR, ROUGE_L, CIDEr)
- **definition.py**: Global constants/definitions

### Data & Config

- **data/**: Dataset storage (Phonexi-2014T included)
- **configs/config_gloss_free.yaml**: Main configuration file for GFSLT-VLP
- **pretrain_models/**: Instructions for preparing MBart weights and GFSLT pretrained models

### Tools

- **tools/mbart_download.py**: Download MBart model weights
- **tools/save_vlp_features.py**: Extract and save precomputed VLP features
- **tools/trim_model.py**: Model weight pruning utility

## Design Patterns

### Distributed Training
All training scripts use `torch.distributed.launch` with DDP. Models use `nn.SyncBatchNorm` for distributed sync.

### Checkpointing
Checkpoints saved at `output_dir/checkpoint.pth` and `best_checkpoint.pth`. Resume with `--resume` flag.

### Hyperparameter Management
Uses `hpargparse` for command-line argument parsing and `hpman` for tracking hyperparameter definitions. All experiments logged to wandb.

### Video Input Pipeline
Videos loaded from LMDB or disk, normalized to [-1, 1], augmented with vidaug, and padded to batch via `pad_sequence`.

## Development Notes

- Text decoding supports both custom decoder and official 12-layer MBart decoder (set `--decoder-type LLMD` after VLP pretraining)
- Evaluation requires single GPU for accurate metric computation (multi-GPU evaluation not reliable)
- nlgeval package is optional but recommended for comprehensive evaluation
- All logging handled by loguru; wandb integration for experiment tracking
