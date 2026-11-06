# RNAtomy model card

## Model details

| | |
|---|---|
| Name | RNAtomy |
| Type | self-supervised RNA encoder (foundation model) |
| Version | v1, November 2026 |
| Architecture | hybrid bidirectional Mamba and shifted-window Transformer, 16 blocks (12 Mamba, 4 attention), width 768, 12 heads, patch size 8, 75.6M parameters |
| Weights | [Zenodo, 10.5281/zenodo.23968854](https://doi.org/10.5281/zenodo.23968854), `rnatomy.pth`, CC BY 4.0 |
| Code | this repository |
| Paper | see the citation in the [README](README.md#citation) |

## Intended use

RNAtomy turns an RNA into vectors that carry structural information: one vector per RNA, one per nucleotide, or one
per nucleotide pair. They are meant as features for research on RNA structure and function. The paper evaluates them
on seven tasks:

- structural: solvent accessibility, 3D contact maps, long-range base pairs;
- interaction: miRNA–target interaction, RNA–protein binding;
- functional: 5′UTR ribosome load, IRES identification.

The encoder can be used as it is, with a light task head on frozen embeddings, or fine-tuned together with a head.

**Out of scope.** The model has not been validated for clinical or diagnostic decisions, for predicting the effect
of individual mutations, for RNA design, or for tasks outside the seven above.

## Inputs

| Input | Shape | Source | Required |
|---|---|---|---|
| Pairwise geometry | `L × L × 4` | trRosettaRNA: one distance-derived and three orientation-derived channels | yes |
| Residue descriptors | `L × 23` | DSSR on a predicted 3D model: sine/cosine of the backbone torsions and χ, two phosphate descriptors, seven binary structural indicators | no |

The pairwise map is normalized with the pretraining channel statistics (mean 28.472, 93.731, 89.640, −8.994; standard
deviation 6.018, 15.843, 22.327, 21.095). During pretraining the descriptors were withheld for 30% of the RNAs, so the
model also runs without them.

## Training data

- Source: the RNAcentral active-sequence release (40,712,942 sequences).
- Selection: sequences were assigned to 16 categories by keyword matching on their FASTA descriptions, sampled so that
  rarer classes are better represented, and clustered with CD-HIT-EST at 95% identity with at least 80% coverage.
  This gave 391,845 non-redundant candidates.
- Filters: length from 50 to 1,024 nt, and successful pairwise geometry prediction, 3D model generation and DSSR
  processing.
- Final set: 200,158 RNAs, split at random into 180,142 for training and 20,016 for validation.
- Composition of the candidate collection: rRNA 38.2%, tRNA 21.4%, lncRNA 15.2%, other 10.1%, and the remaining
  classes together about 15%. The category labels were used only to guide sampling and were not given to the model.

All structural inputs are predictions, not experimental structures.

## Training procedure

Joint masked reconstruction, `L = L_2D + 0.5 · L_1D`.

- Pairwise branch: 45–65% of valid patches are replaced by a learned mask token and reconstructed, with weights that
  favour patches that carry geometric signal.
- Residue-wise branch: about 15% of nucleotides, in spans of four, are replaced by a learned token and their 23
  descriptors are reconstructed from the pairwise representation.
- Modality dropout: the descriptor track is removed for 30% of the samples, which teaches the model to recover
  descriptors from the pairwise map alone.
- Optimization: AdamW, peak learning rate 2e-4, effective batch size 128, length curriculum, EMA weights.
- Compute: four NVIDIA A100 80 GB GPUs, about 85 hours.

The full hyperparameter list is in the [README](README.md#pretraining) and in the paper's appendix.

## Evaluation

Results on the seven tasks, with the compared methods and protocols, are in the paper and summarized in the
[README](README.md#results). The ablation in the paper's appendix shows that adding the residue-wise track lowers
the pairwise-map reconstruction error on 1,323 held-out RNAs, from 0.18 to 0.11 at width 768.

## Limitations

- **Static structure.** One predicted pairwise map and one descriptor set represent each RNA. Conformational ensembles,
  ligand-dependent states and cellular conditions are not modelled.
- **Dependence on predicted structures.** Errors from trRosettaRNA and DSSR propagate to the embeddings. Generating
  the inputs for a new RNA is the costly step, and DSSR needs a separate licence.
- **Length.** The pretraining range is 50–1,024 nt. Longer RNAs are processed in overlapping windows and their
  window outputs are averaged, which was not separately validated against full-length processing.
- **Training distribution.** rRNA and tRNA are about 60% of the candidate sequences, so performance on rare RNA
  classes may differ.
- **Benchmark scope.** Results are reported on the benchmarks listed in the paper. Performance on other datasets,
  species or conditions is unknown.
- **Hardware.** The Mamba kernels need a CUDA GPU.

## Licence and citation

The weights and data are released under CC BY 4.0, and the code under the MIT licence. Please cite the paper and the
Zenodo record, as given in the [README](README.md#citation).
