# Pretrained RNAtomy weights

The pretrained checkpoint (about 1.3 GB) is too large for GitHub and is hosted on Zenodo.

**Download:** [https://doi.org/10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX)

Save the file in this folder as `rnatomy_pretrained.pth`, or from the command line:

```bash
wget -O pretrained_model/rnatomy_pretrained.pth "https://zenodo.org/records/XXXXXXX/files/rnatomy_pretrained.pth?download=1"
```

## Model

| | |
|---|---|
| Parameters | 75.6M |
| Patch size | 8 |
| Blocks | 16 (12 bidirectional Mamba + 4 shifted-window attention) |
| Embedding dimension | 768 |
| Window size | 16 patches (128 nt) |
| Attention heads | 12 |
| Input channels | 4 (trRosettaRNA pairwise map) + 23-d DSSR residue descriptors |

## Usage

Load the encoder and extract embeddings:

```python
from models.rna_encoder import build_encoder_from_checkpoint

enc = build_encoder_from_checkpoint("pretraining_model/pretrained_model/rnatomy_pretrained.pth")
```

Fine-tune on a downstream task:

```bash
python finetune_contact.py \
    --pretrain-ckpt pretraining_model/pretrained_model/rnatomy_pretrained.pth \
    ...
```

See the main [README](../../README.md) for the full list of downstream tasks and options.

## Citation

If you use these weights, please cite the RNAtomy paper (see the main README) and the Zenodo record.
