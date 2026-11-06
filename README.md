<div align="center">
 
# RNAtomy: A Structure-Aware RNA Foundation Model for Transferable Structural and Functional Representations


</div>

**RNAtomy** is a self-supervised RNA foundation model that learns the structural *anatomy* of RNA directly from
**inter-nucleotide geometry** and **residue-level conformational descriptors**, instead of from sequence or
secondary structure alone. It is pretrained on 391,845 non-redundant RNAs from RNAcentral spanning 16 RNA categories,
a much smaller corpus than those used by the RNA language models it is compared against.

Three design choices set it apart:

1. **Joint masked reconstruction** of a pairwise geometry tensor (one distance and three orientation channels
   from trRosettaRNA) and a residue-wise descriptor matrix (23 DSSR features per nucleotide), which ties
   spatial organization to local conformation.
2. **A hybrid Mamba–Transformer backbone.** Bidirectional Mamba blocks carry context across the whole
   pairwise map, and shifted-window attention blocks add content-dependent interactions inside local regions
   of the map, which Mamba alone does not model.
3. **Axis-aware FiLM.** Nucleotide-level conformations modulate the row and column axes of the pairwise tokens
   separately, so position (*i*, *j*) is distinguished from (*j*, *i*).

RNAtomy is evaluated on seven structural, interaction and functional tasks (see [Results](#results)). The same
pretrained encoder can produce per-pair, per-nucleotide, per-patch or whole-RNA embeddings.

---

## Table of Contents

1. [Features](#features)
2. [Repository layout](#repository-layout)
3. [Installation](#installation)
4. [Input features](#input-features)
5. [Quick start: use the encoder](#quick-start-use-the-encoder)
6. [Pretraining](#pretraining)
7. [Downstream tasks](#downstream-tasks)
8. [Results](#results)
9. [Interpreting the model](#interpreting-the-model)
10. [Checkpoints and data](#checkpoints-and-data)
11. [Citation](#citation)
12. [License](#license)
13. [Contact](#contact)

---

## Features

- **Two structural modalities.** A 4-channel pairwise geometry map `X ∈ ℝ^{4×L×L}` and a 23-dimensional
  residue-wise descriptor matrix `F ∈ ℝ^{L×23}`.
- **Hybrid backbone.** 16 blocks, arranged as four groups of three bidirectional Mamba blocks followed by one
  shifted-window attention block (12 Mamba + 4 attention), about 75M parameters.
- **Axis-specific positional embeddings** for rows and columns, so directional orientation channels are
  represented correctly.
- **Modality dropout.** The 1D track can be withheld at pretraining time, so the encoder also runs with the
  pairwise map alone.
- **Four output modes** from one wrapper (`RNAEncoder`): `encode_pairs`, `encode_nucleotides`, `encode_tokens`, `encode_pooled`.
- **Ready-made task heads** for all seven downstream tasks, with frozen, partially unfrozen or full fine-tuning.
- **Length curriculum and shape-aware batching** for pairwise maps of very different sizes (50–1024 nt).

## Repository layout

```text
RNAtomy/
├── main_mine.py                  # self-supervised pretraining entry point
├── engine.py  datasets.py  augment.py  utils.py
├── models/
│   ├── hybridnet.py              # Mamba + shifted-window attention backbone, FiLM, 1D encoder
│   ├── rna_encoder.py            # RNAEncoder: loads a checkpoint and exposes encode_* outputs
│   └── lm_encoders.py            # wrappers for baseline RNA language models
├── finetune/                     # dataset classes and shared utilities for the downstream tasks
├── rna_task_heads/               # stand-alone task heads (contact, RSA, LRBP, RRI, RBP, MRL, IRES)
├── Final_extacting_dssr.py       # DSSR -> 23-d residue descriptors (feat1d_norm.npy)
├── finetune_rsa.py               # solvent accessibility
├── finetune_contact_map.py       # 3D contact maps
├── finetune_lrbp.py              # long-range base pairs
├── finetune_rri.py               # RNA-RNA interaction
├── finetune_rna_protein.py       # RNA-protein binding (17 RBPs)
├── finetune_utr_mrl.py           # 5'UTR mean ribosome load
├── finetune_ires.py              # IRES identification
├── visualization/                # embedding, attention and prediction figures
└── assets/                       # figures used in this README
```

## Installation

RNAtomy needs a CUDA GPU because the Mamba blocks use the `mamba_ssm` kernels.

```bash
conda create -n rnatomy python=3.10 -y
conda activate rnatomy

# PyTorch matching your CUDA version: https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Mamba kernels
pip install causal-conv1d mamba-ssm

# remaining dependencies
pip install timm numpy scipy pandas scikit-learn matplotlib mlflow
```

DSSR is required only to compute the residue-wise descriptors for your own RNAs. It is distributed separately
under its own licence: <http://x3dna.org/>.

## Input features

RNAtomy takes two files per RNA. Both are produced from a sequence (no experimental structure is needed).

| File | Shape | Content | Produced by |
|---|---|---|---|
| `<id>.npy` | `L × L × 4` | trRosettaRNA distance map and three orientation maps | [trRosettaRNA](https://yanglab.qd.sdu.edu.cn/trRosettaRNA/) |
| `<id>/feat1d_norm.npy` | `L × 23` | sin/cos of 6 backbone torsions and χ, `ssZp`, `Dp`, 7 binary structural indicators | `Final_extacting_dssr.py` on a predicted 3D model |

```bash
# DSSR descriptors for every PDB in ./training_set (writes ./out/<RNA>/feat1d_norm.npy)
python Final_extacting_dssr.py
```

> **Provenance matters.** Both inputs should be computed from *predicted* structures. If a benchmark's
> labels come from experimental structures (for example contact maps or solvent accessibility), do not derive the
> 1D descriptors from those same experimental structures, or the label leaks into the input.

## Quick start: use the encoder

```python
import numpy as np, torch
from models.rna_encoder import build_encoder_from_checkpoint

enc = build_encoder_from_checkpoint(
    "checkpoints/rnatomy_pretrained.pth",   # see "Checkpoints and data"
    patch_size=8, in_chans=4, embed_dim=768, depth=16, window_size=16, window_num_heads=12,
    device="cuda",
).eval()

MEAN = torch.tensor([28.472, 93.731, 89.640, -8.994])   # pretraining statistics (see --norm-mean / --norm-std)
STD  = torch.tensor([6.018, 15.843, 22.327, 21.095])

x = torch.from_numpy(np.load("assets/example/URS0000A0B8F6.npy")).float()   # [L, L, 4]
x = (x - MEAN) / STD                       # per-channel z-score
x = x.permute(2, 0, 1)[None].cuda()        # [1, 4, L, L]
feats = torch.from_numpy(np.load("<id>/feat1d_norm.npy")).float()[None].cuda()   # [1, L, 23] from Final_extacting_dssr.py

with torch.no_grad():
    pair = enc.encode_pairs(x, nt_features=feats, seq_len=x.shape[-1])   # [1, L, L, D]  contact / distance maps
    nt   = enc.encode_nucleotides(x, nt_features=feats)                 # [1, L, D]     per-nucleotide tasks
    rna  = enc.encode_pooled(x, nt_features=feats)                      # [1, D]        whole-RNA tasks
```

The architecture arguments must match the checkpoint. Check the shapes printed by `build_encoder_from_checkpoint`,
and read the module docstring of `models/rna_encoder.py` for the freezing helpers (`freeze`,
`unfreeze_last_n_blocks`, `unfreeze_all`). To run without the residue-wise track, pass `feat1d_mode="none"`
(this requires a checkpoint pretrained with modality dropout).

## Pretraining

```bash
python main_mine.py \
    --data-path <dir with <id>.npy maps> --feat1d-root <dir with <id>/feat1d_norm.npy> \
    --prot-patch-size 8 --depth 16 --mamba-ratio 3 \
    --swin-embed-dim 768 --swin-window-size 16 --swin-num-heads 12 \
    --mask-ratio-min 0.45 --mask-ratio-max 0.65 \
    --recon-1d-weight 0.5 --recon-1d-mask-ratio 0.15 --recon-1d-span 4 --moddrop-prob 0.3 \
    --length-curriculum --output_dir runs/pretrain
```

Pretraining objective:

```
L = L_2D + λ_1D · L_1D        (λ_1D = 0.5)
```

- `L_2D` reconstructs standardized distance and orientation values of masked patches, with signal-aware weights so
  background patches do not dominate.
- `L_1D` reconstructs the 23 descriptors of masked spans (about 15% of nucleotides): circular losses for the
  sin/cos torsions, a robust loss for continuous descriptors, binary cross-entropy for indicators.
- Full optimizer settings, masking algorithms and ablations are in the paper's appendix; run
  `python main_mine.py -h` for every option.

## Downstream tasks

All task scripts share one pattern: load the pretrained encoder (`--pretrain-ckpt`), attach a task head, and
train the head alone (`--freeze-backbone`), the last *n* blocks, or the full network. Validation picks the best
epoch, and the test set is evaluated once.

| # | Task | Script | Level | Metric |
|---|---|---|---|---|
| 1 | Solvent accessibility (RSA) | `finetune_rsa.py` | nucleotide | PCC, SCC, MAE (Å²) |
| 2 | 3D contact maps | `finetune_contact_map.py` | nucleotide pair | long-range Top-L/k precision |
| 3 | Long-range base pairs | `finetune_lrbp.py` | nucleotide pair | F1 (overall, pseudoknot classes, long-distance) |
| 4 | RNA–RNA interaction | `finetune_rri.py` | RNA pair | accuracy, F1, AUC |
| 5 | RNA–protein binding | `finetune_rna_protein.py` | nucleotide / RNA | AUROC, AUPRC |
| 6 | 5′UTR ribosome load | `finetune_utr_mrl.py` | RNA | R² |
| 7 | IRES identification | `finetune_ires.py` | RNA | accuracy, F1, MCC, AUROC |

Example: contact-map prediction with a frozen backbone.

```bash
python finetune_contact_map.py \
    --pretrain-ckpt checkpoints/rnatomy_pretrained.pth \
    --train-split data/train.fasta --val-split data/val.fasta --test-split data/test.fasta \
    --train-npy-root data/train/npy --val-npy-root data/val/npy --test-npy-root data/test/npy \
    --train-feat1d-root data/train/dssr --val-feat1d-root data/val/dssr --test-feat1d-root data/test/dssr \
    --contact-root data/contact_maps \
    --norm-mean "28.472,93.731,89.640,-8.994,0.0" --norm-std "6.018,15.843,22.327,21.095,1.0" \
    --head-input pair --focal-alpha 0.5 --focal-gamma 2.0 \
    --freeze-backbone --unfreeze-last-n-blocks 0 \
    --lr 1e-5 --epochs 200 --early-stop-patience 15 \
    --output-dir runs/contact
```

Re-evaluate a trained head without training: add `--resume runs/contact/checkpoint_best.pth --eval-only`.
Every script documents its own arguments and data format at the top of the file (`python finetune_<task>.py -h`).

## Results

Selected numbers from the paper (see the paper for all baselines and for the frozen-backbone variant RNAtomy⁻).

| Task | Metric | RNAtomy | Best compared method |
|---|---|---|---|
| Solvent accessibility (TS44) | PCC / SCC / MAE | **0.51** / **0.43** / **33.8** | RNAsol 0.43 / 0.42 / 34.0 |
| Contact map | Top-L/5 precision | **0.91** | RNAcontact 0.88, ERNIE-RNA attention-map 0.78 |
| Long-range base pairs (TR0) | overall F1 / long-distance | **0.533** / **0.498** | RNA-km 0.527 / 0.228 |
| RNA–RNA interaction (DeepMirTar) | F1 / AUC | 0.887 / 0.957 | RNAErnie 0.957 / 0.988 |
| RNA–protein binding (17 RBPs) | mean AUPRC | 0.855 | ERNIE-RNA 0.865 |
| 5′UTR ribosome load | R² (Random / Human 7600) | **0.95** / **0.88** | ERNIE-RNA-conv 0.92 / 0.86 |
| IRES identification | AUROC / F1 | **0.75** / **0.73** | structRFM 0.71 / 0.70 |

<details open>
<summary><b>Structural tasks</b></summary>
<img src="assets/fig_results_structural.png" width="100%">
</details>

<details>
<summary><b>Interaction tasks</b></summary>
<img src="assets/fig_results_interaction.png" width="100%">
</details>

<details>
<summary><b>Functional tasks</b></summary>
<img src="assets/fig_results_functional.png" width="100%">
</details>

### Predicted contact maps

Ten test RNAs, one drawn at random from each of ten equal-size length groups (fixed seed): ground truth on the left of
each pair, RNAtomy prediction on the right. Only pairs with |i − j| ≥ 24 are labelled and scored.

<img src="assets/contact_gallery.png" width="100%">

## Interpreting the model

`visualization/` contains scripts that run a fine-tuned checkpoint on a test set and draw what it learned, without
modifying any model code: predictions, PCA of patch embeddings across blocks, window attention of the four
attention blocks, and per-head attention for a chosen patch.

<p align="center">
<img src="assets/embedding_1gax_D.png" width="49%">
<img src="assets/attention_1gax_D.png" width="49%">
</p>

```bash
python visualization/rnatomy_extract.py --run <run dir> --npy-root <maps> --feat1d-root <dssr> --out internals --verify
python visualization/rnatomy_internals_plots.py dossier --internals internals --run <run dir> --ids 1gax_D --out figures
python visualization/rnatomy_internals_plots.py testset --internals internals --run <run dir> --out figures
```

Attention is local to a window of 16 × 16 patches (128 nt) and shifts by half a window in every second attention
block, so for RNAs longer than 128 nt a patch attends only within its window.

## Checkpoints and data

| Resource | Link |
|---|---|
| Pretrained RNAtomy encoder |  |
| Fine-tuned task heads |  |
| Example trRosettaRNA map (`URS0000A0B8F6`, `L × L × 4`) | `assets/example/` |

Downstream benchmarks are public: RNAsolvent/RNAsol (TR95/VL24/TS44), RNAcontact (ERNIE-RNA), bpRNA-1m and
RNA-km (TR0), DeepMirTar, PrismNet (17 RBPs, HeLa), Optimus 5-Prime (Random/Human 7600) and the structRFM IRES set.
Please cite the original benchmark papers when you use them.


## Contact

**Azam Shirali**, Bioinformatics Research Group (BioRG), Knight Foundation School of Computing and Information
Sciences, Florida International University
📧 [ashir018@fiu.edu](mailto:ashir018@fiu.edu)

