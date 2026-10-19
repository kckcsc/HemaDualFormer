# HemaDualFormer

HemaDualFormer is a Transformer-based framework for bone marrow single-cell
transcriptome analysis. This repository provides scripts for masked language
model pretraining, cell-type classification, and in silico gene deletion analysis.

The released workflow starts from processed, tokenized datasets. Raw sequencing
data are not included in this repository.

## Repository contents

```text
HemaDualFormer/
├── README.md
├── pretrain/
│   ├── pretrain.py
│   ├── token_dictionary.test.pkl
│   └── BoneMarrow_cell_markers.tsv
├── finetune/
│   ├── finetune_GSE116256.py
│   ├── finetune_GSE253355.py
│   ├── gse116256_splits.json
│   └── gse253355_splits.json
└── perturbation/
    └── perturbation.py
```

- `pretrain.py`: masked language model pretraining with marker-prioritized,
  rank-preserving gene selection.
- `finetune_GSE116256.py`: multi-task fine-tuning for cell type and health status
  with Class-Balanced cell-type loss.
- `finetune_GSE253355.py`: cell-type fine-tuning with Class-Balanced loss and
  five-fold sample-level cross-validation.
- `perturbation.py`: reference-state embedding extraction, single-gene deletion,
  and goal-state-shift statistics through Geneformer.

## Data and model resources

Large files are hosted separately:

[HemaDualFormer resources on Hugging Face](https://huggingface.co/dfgdfgcvqe/HemaDualFormer)

The resource repository contains:

| Resource | Purpose |
|---|---|
| `maskv3_all_datasets_test_tokenized.dataset/` | Pretraining dataset |
| `GSE116256_test_tokenized.dataset/` | GSE116256 fine-tuning and perturbation input |
| `GSE253355_test_filter_tokenized.dataset/` | GSE253355 fine-tuning input |
| `cell_high_marker_gene_dict3.pkl` | Per-cell marker gene lists |
| `cell_level_gene_list_dict.pkl` | Per-cell expression-ranked gene lists |
| `GSE116256_classifier_CB_id_class_dict.pkl` | Class mapping for the corresponding GSE116256 classifier |
| `checkpoint/` | Model checkpoint files |

Download these resources to a local directory, referred to below as
`RESOURCE_DIR`. Preserve the complete dataset directories, including their
metadata and data shards.

The token vocabulary, processed inputs, model configuration, and class mapping
must correspond to the same experiment. Only load pickle files from trusted
sources.

### Model checkpoints

Two model roles must be distinguished:

- **Pretrained encoder:** used to initialize fine-tuning.
- **Fine-tuned GSE116256 classifier:** used for perturbation analysis.

The original perturbation script used `checkpoint-17280` from fold 3.

Before running the commands below, replace:

- `<PRETRAINED_MODEL_DIR>` with the actual pretrained model directory.
- `<PERTURBATION_CHECKPOINT_DIR>` with the corresponding fine-tuned checkpoint
  directory.

Each model directory must contain its configuration and complete model weights.

## Environment

The scripts depend on Python, PyTorch, Transformers, datasets, Geneformer,
NumPy, pandas, and scikit-learn, together with the dependencies required by
the installed Geneformer version.

Exact experiment versions and the Geneformer revision must be documented
before the environment can be considered reproducible. The scripts use
version-dependent Geneformer and Transformers interfaces; installing the
latest releases is not a substitute for the original environment.

GPU memory requirements depend on sequence length and batch size. The
fine-tuning scripts retain FP16 settings from the original experiments.

## Usage

The commands below assume that the working directory is the repository root.
Replace example paths with the locations on your machine.

### 1. Pretraining

```bash
python pretrain/pretrain.py \
  --dataset RESOURCE_DIR/maskv3_all_datasets_test_tokenized.dataset \
  --token-dictionary pretrain/token_dictionary.test.pkl \
  --cell-markers RESOURCE_DIR/cell_high_marker_gene_dict3.pkl \
  --cell-genes RESOURCE_DIR/cell_level_gene_list_dict.pkl \
  --output-dir runs/pretrain
```

The dataset must contain `cell_id` values matching the keys in both per-cell
dictionaries.

Use a new or empty output directory for a new run. To resume an existing run,
use the same inputs and output directory and add:

```bash
--resume-checkpoint auto
```

### 2. GSE116256 fine-tuning

```bash
python finetune/finetune_GSE116256.py \
  --dataset RESOURCE_DIR/GSE116256_test_tokenized.dataset \
  --pretrained-model <PRETRAINED_MODEL_DIR> \
  --splits finetune/gse116256_splits.json \
  --output-dir runs/gse116256
```

The default metadata columns are `orig.ident`, `cell_type`, and a detected
health-status column. Column names can be specified through the command-line
options.

This script implements multi-task training. Its outputs should not be
presented as results from a separate single-task experiment.

### 3. GSE253355 fine-tuning

Edit the configuration block at the beginning of
`finetune/finetune_GSE253355.py`:

```python
MODEL_DIRECTORY = Path("<PRETRAINED_MODEL_DIR>")
INPUT_DATA_FILE = Path("RESOURCE_DIR/GSE253355_test_filter_tokenized.dataset")
OUTPUT_DIRECTORY = Path("runs/gse253355")
SPLITS_FILE = Path("finetune/gse253355_splits.json")
```

Then run:

```bash
python finetune/finetune_GSE253355.py
```

Relative paths are resolved from the working directory. Use a new or empty
output directory.

### 4. In silico gene deletion

Example for the GMP/GMP-like subset:

```bash
python perturbation/perturbation.py \
  --model-dir <PERTURBATION_CHECKPOINT_DIR> \
  --dataset RESOURCE_DIR/GSE116256_test_tokenized.dataset \
  --class-dictionary RESOURCE_DIR/GSE116256_classifier_CB_id_class_dict.pkl \
  --cell-types GMP GMP-like \
  --state-column health_status \
  --start-state aml \
  --goal-state health \
  --prefix GSE116256_AML_Healthy_GMP \
  --output-dir runs/perturbation_GMP
```

Cell-type and state labels are case-sensitive and must match the dataset.
Other lineage analyses require their actual experiment-specific filters and
output prefixes.

Outputs are organized as:

```text
runs/perturbation_GMP/
├── state_embeddings/
├── isp_runs/
└── isp_stats/
```

Virtual gene deletion measures changes in model representations. It is not
an experimental gene knockout or evidence of therapeutic efficacy.

## Reproducibility scope

- The released workflow begins with processed datasets.
- Sample partitions are provided in the two split JSON files.
- Dataset filenames containing `test` do not by themselves define a held-out
  test set; the relevant split and filtering logic determine the analysis set.
- Geneformer controls the internal perturbation and statistical procedures.
  Its version and any local modifications must be matched to the experiment.
- Reference-state and perturbed-cell embedding settings must be checked for
  compatibility in the Geneformer version used.
- The provided GSE116256 script implements multi-task fine-tuning.
- GO enrichment and cross-lineage downstream analyses require separate code
  and are not performed by `perturbation.py`.

## Data provenance

The manuscript describes a pretraining corpus assembled from 17 public
datasets and downstream analyses using GSE116256 and GSE253355.

The accession-level source list, selected samples, QC criteria, gene-ID
mapping, and tokenization procedure should accompany this release.
