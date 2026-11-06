# Pretrained RNAtomy weights and pretraining data

The pretrained model and the data used for pretraining are too large for GitHub and are hosted on Zenodo:

**[https://doi.org/10.5281/zenodo.23968854](https://doi.org/10.5281/zenodo.23968854)** (v1, CC BY 4.0, 2.3 GB in total)

| File | Size | Contents | MD5 |
|---|---|---|---|
| `rnatomy.pth` | 1.2 GB | Pretrained RNAtomy checkpoint | `2180156f2cf824fb1165c6ae639f93d8` |
| `fasta.zip` | 90.8 MB | FASTA sequences of the pretraining (training and validation) RNAs, one file per RNAcentral ID | `4884100b7783d5694d2c6fd6f6df70ea` |
| `1d_feats.zip` | 943.7 MB | Residue-wise 23-d DSSR descriptors (`feat1d_norm.npy`) for 114,936 RNAs | `e29be7897e83a1a2676481dc50912d72` |
| `2d_maps_examples.zip` | 101.1 MB | 70 example trRosettaRNA pairwise maps (`L × L × 4`) | `7c87ad06fa078c96d3ed3c88279bb847` |

## Download

```bash
BASE=https://zenodo.org/records/23968854/files
wget -O pretraining_model/pretrained_model/rnatomy.pth "$BASE/rnatomy.pth?download=1"
wget -O fasta.zip                "$BASE/fasta.zip?download=1"
wget -O 1d_feats.zip             "$BASE/1d_feats.zip?download=1"
wget -O 2d_maps_examples.zip     "$BASE/2d_maps_examples.zip?download=1"
```

Only `rnatomy.pth` is needed to use the model or to fine-tune it. Check a download with `md5sum rnatomy.pth`.

## Data layout

After unzipping:

```
fasta_files/<RNAcentral ID>.fasta          # one sequence per file, 246,458 RNAs
1d_feats/<RNAcentral ID>/feat1d_norm.npy   # L x 23 residue descriptors, 114,936 RNAs
2d_maps_examples/<name>.npy                # L x L x 4 trRosettaRNA maps, 70 examples
```

- The residue descriptors are what `--feat1d-root` expects in `pretrain.py`: one folder per RNA containing
  `feat1d_norm.npy`.
- The pairwise maps of the full pretraining set are not included because of their size. They are generated from the
  FASTA files with trRosettaRNA; the 70 maps in `2d_maps_examples.zip` show the expected format and are enough to
  try the encoder.

## Using the checkpoint

```python
from models.rna_encoder import build_encoder_from_checkpoint

enc = build_encoder_from_checkpoint("pretraining_model/pretrained_model/rnatomy.pth")
```

To fine-tune on a downstream task, pass the same path to `--pretrain-ckpt`. See the main [README](../../README.md)
for the available tasks and options.

## Model

| | |
|---|---|
| Parameters | 75.6M |
| Patch size | 8 |
| Blocks | 16 (12 bidirectional Mamba + 4 shifted-window attention) |
| Embedding dimension | 768 |
| Window size | 16 patches (128 nt) |
| Attention heads | 12 |
| Inputs | trRosettaRNA pairwise map (4 channels) + 23-d DSSR residue descriptors |

