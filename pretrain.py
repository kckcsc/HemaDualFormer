#!/usr/bin/env python
# coding: utf-8
# Usage: python pretrain.py --help
# Training settings retain the original experiment defaults.

import argparse
import os
import random
import pickle
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from transformers import BertConfig, BertForMaskedLM, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint   # Automatically locate the latest checkpoint.
from geneformer import GeneformerPretrainer

# -------------------------
# Set random seeds.
# -------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

# All machine-specific locations are supplied by the caller.
parser = argparse.ArgumentParser(description="DC-RPS masked language model pretraining")
parser.add_argument("--dataset", required=True, help="Dataset directory saved with datasets.save_to_disk")
parser.add_argument("--token-dictionary", required=True, help="Trusted pickle: gene/special token -> integer ID")
parser.add_argument("--cell-markers", required=True, help="Trusted pickle: cell_id -> marker gene list")
parser.add_argument("--cell-genes", required=True, help="Trusted pickle: cell_id -> expression-ranked gene list")
parser.add_argument("--output-dir", required=True, help="New run directory; reuse only with explicit resume")
parser.add_argument("--resume-checkpoint", help="Checkpoint directory, or 'auto' to find the last checkpoint in output-dir")
args = parser.parse_args()

DATASET_PATH = str(Path(args.dataset).expanduser().resolve())
TOKEN_DICT_PKL = str(Path(args.token_dictionary).expanduser().resolve())
CELL_MARKER_PKL = str(Path(args.cell_markers).expanduser().resolve())
CELL_GENE_LIST_PKL = str(Path(args.cell_genes).expanduser().resolve())
for label, location in [("dataset", DATASET_PATH), ("token dictionary", TOKEN_DICT_PKL),
                        ("cell markers", CELL_MARKER_PKL), ("cell genes", CELL_GENE_LIST_PKL)]:
    if not Path(location).exists():
        parser.error(f"Missing {label}: {location}")

training_output_dir = str(Path(args.output_dir).expanduser().resolve())
output_path = Path(training_output_dir)
if output_path.exists() and any(output_path.iterdir()) and not args.resume_checkpoint:
    parser.error("Output directory is not empty. Choose a new directory or explicitly resume.")
resume_ckpt = None
if args.resume_checkpoint == "auto":
    resume_ckpt = get_last_checkpoint(training_output_dir) if output_path.is_dir() else None
    if resume_ckpt is None:
        parser.error("No checkpoint found in output-dir; cannot resume automatically.")
elif args.resume_checkpoint:
    resume_ckpt = str(Path(args.resume_checkpoint).expanduser().resolve())
    if not Path(resume_ckpt).is_dir():
        parser.error(f"Checkpoint directory does not exist: {resume_ckpt}")
logging_dir = str(output_path / "logs")
model_output_dir = str(output_path / "final_model")
os.makedirs(training_output_dir, exist_ok=True)
os.makedirs(model_output_dir, exist_ok=True)

# -------------------------
# Load the token vocabulary and per-cell input dictionaries.
# -------------------------
with open(TOKEN_DICT_PKL, "rb") as f:
    token_dictionary = pickle.load(f)

REQUIRED_SPECIALS = ["<pad>", "<mask>", "<cls>", "<eos>"]
for sp in REQUIRED_SPECIALS:
    if sp not in token_dictionary:
        raise ValueError(f"token_dictionary is missing a required special token: {sp}")

PAD_ID  = token_dictionary["<pad>"]
MASK_ID = token_dictionary["<mask>"]
CLS_ID  = token_dictionary["<cls>"]
EOS_ID  = token_dictionary["<eos>"]
VOCAB_SIZE = len(token_dictionary)

with open(CELL_MARKER_PKL, "rb") as f:
    cell_high_marker_dict = pickle.load(f)  # cell_id -> [marker_gene,...]

with open(CELL_GENE_LIST_PKL, "rb") as f:
    gene_list_dict = pickle.load(f)  # cell_id -> genes ranked by decreasing expression

# -------------------------
# Model architecture.
# -------------------------
config = BertConfig(
    hidden_size=256,
    num_hidden_layers=6,
    initializer_range=0.02,
    layer_norm_eps=1e-12,
    attention_probs_dropout_prob=0.02,
    hidden_dropout_prob=0.02,
    intermediate_size=512,
    hidden_act="relu",
    max_position_embeddings=2048,
    model_type="bert",
    num_attention_heads=4,
    pad_token_id=PAD_ID,
    vocab_size=VOCAB_SIZE,
)
model = BertForMaskedLM(config).train()

# -------------------------
# Dual-channel Top-K selection: prioritize markers, then fill with non-markers.
# full_gene_list must already be sorted by decreasing expression.
# -------------------------
ADD_SPECIAL_TOKENS = True
MAX_INPUT = 2048

def select_topk_two_channel(full_gene_list, marker_set, cap):
    markers = [g for g in full_gene_list if g in marker_set]
    non_markers = [g for g in full_gene_list if g not in marker_set]
    take_m = markers[:cap]
    rem = max(cap - len(take_m), 0)
    take_u = non_markers[:rem]
    chosen = set(take_m + take_u)
    # Restore the original expression rank order.
    selected = [g for g in full_gene_list if g in chosen]
    return selected[:cap]

# -------------------------
# Calculate the selected sequence length, including <cls> and <eos>.
# -------------------------
def effective_len_for_cell(cell_id, topk_len=2048):
    reserve = 2 if ADD_SPECIAL_TOKENS else 0
    eff_topk = max(1, topk_len - reserve)
    full_list = gene_list_dict.get(cell_id, [])
    marker_set = set(cell_high_marker_dict.get(cell_id, []))
    selected = select_topk_two_channel(full_list, marker_set, eff_topk)
    usable = [g for g in selected if g in token_dictionary]
    return len(usable) + reserve

# -------------------------
# Self-contained uniform MLM collator.
# Select targets with probability 0.15; use 80% mask, 10% random, and 10% unchanged.
# -------------------------
class DCUniformMLMCollator:
    def __init__(self, mlm_probability=0.15, topk_len=2048):
        self.mlm_probability = mlm_probability
        self.topk_len = topk_len

    def _ids_for_cell(self, cell_id):
        reserve = 2 if ADD_SPECIAL_TOKENS else 0
        eff_topk = max(1, self.topk_len - reserve)
        full_list = gene_list_dict.get(cell_id, [])
        marker_set = set(cell_high_marker_dict.get(cell_id, []))
        selected = select_topk_two_channel(full_list, marker_set, eff_topk)
        ids = [token_dictionary[g] for g in selected if g in token_dictionary]
        if ADD_SPECIAL_TOKENS:
            if not ids or ids[0] != CLS_ID: ids = [CLS_ID] + ids
            if ids[-1] != EOS_ID: ids = ids + [EOS_ID]
        return ids

    def __call__(self, examples):
        batch_ids, batch_attn = [], []
        for ex in examples:
            cid = ex.get("cell_id", None)
            if cid is None:
                raise ValueError("Example is missing cell_id; cannot reconstruct dual-channel input_ids.")
            ids = self._ids_for_cell(cid)
            attn = [1]*len(ids)
            batch_ids.append(ids)
            batch_attn.append(attn)

        # Pad sequences to the longest sequence in the batch.
        max_len = max(len(x) for x in batch_ids)
        for i in range(len(batch_ids)):
            need = max_len - len(batch_ids[i])
            if need > 0:
                batch_ids[i]  = batch_ids[i]  + [PAD_ID]*need
                batch_attn[i] = batch_attn[i] + [0]*need

        input_ids = torch.tensor(batch_ids, dtype=torch.long)
        attention_mask = torch.tensor(batch_attn, dtype=torch.long)
        labels = torch.full_like(input_ids, -100)

        # Select MLM prediction targets.
        can_mask = (attention_mask == 1)
        for sp in [CLS_ID, EOS_ID, PAD_ID]:
            can_mask = can_mask & (input_ids != sp)

        mask_prob = torch.full_like(input_ids, fill_value=self.mlm_probability, dtype=torch.float)
        mask_prob = torch.where(can_mask, mask_prob, torch.zeros_like(mask_prob))
        mask = torch.bernoulli(mask_prob).bool()

        labels[mask] = input_ids[mask]

        # Replace 80% of selected targets with MASK_ID.
        mask80 = mask & (torch.rand_like(input_ids, dtype=torch.float) < 0.8)
        input_ids[mask80] = MASK_ID

        # Replace 10% of selected targets with random vocabulary tokens.
        remain = mask & ~mask80
        mask10 = remain & (torch.rand_like(input_ids, dtype=torch.float) < 0.5)
        rand_tokens = torch.randint(low=0, high=VOCAB_SIZE, size=input_ids.shape, dtype=torch.long)
        input_ids[mask10] = rand_tokens[mask10]
        # Keep the remaining 10% of selected targets unchanged.

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

# -------------------------
# Load the dataset and compute one sequence length per training example.
# -------------------------
train_dataset = load_from_disk(DATASET_PATH)
if "cell_id" not in train_dataset.column_names:
    raise ValueError("Dataset is missing the cell_id column required for dual-channel selection.")

# Reuse cached lengths on resume; the dataset and its order must remain unchanged.
lengths_pkl = os.path.join(training_output_dir, "train_lengths_dc.pkl")
if os.path.exists(lengths_pkl):
    print(f"[Resume] Found existing example_lengths_file: {lengths_pkl}; reusing cached lengths.")
else:
    lengths = [effective_len_for_cell(ex["cell_id"], topk_len=MAX_INPUT) for ex in train_dataset]
    with open(lengths_pkl, "wb") as f:
        pickle.dump([int(x) for x in lengths], f)
    print(f"[Saved] example_lengths_file -> {lengths_pkl} (n={len(train_dataset)})")

# -------------------------
# Training settings retained from the original experiment script.
# -------------------------
training_args = TrainingArguments(
    learning_rate=1e-3,
    do_train=True,
    do_eval=False,
    group_by_length=True,              # Enable length grouping with the supplied sequence lengths.
    length_column_name="length",       # Retain the length column setting for compatibility.
    disable_tqdm=False,
    lr_scheduler_type="linear",
    warmup_steps=10_000,
    weight_decay=0.001,
    per_device_train_batch_size=12,
    num_train_epochs=3,
    save_strategy="steps",
    save_steps=10000,                  # Save a checkpoint every 10000 steps.
    save_total_limit=5,                # Keep the five most recent checkpoints.
    overwrite_output_dir=False,        # Avoid overwriting an existing training run.
    logging_steps=1000,
    output_dir=training_output_dir,
    logging_dir=logging_dir,
    remove_unused_columns=False,
)

print("Starting training with Dual-Channel selection (marker first).")

# -------------------------
# Initialize the trainer with the custom collator and sequence lengths.
# -------------------------
dc_collator = DCUniformMLMCollator(mlm_probability=0.15, topk_len=MAX_INPUT)

trainer = GeneformerPretrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    example_lengths_file=lengths_pkl,      # Lengths correspond to the selected model inputs.
    token_dictionary=token_dictionary,
    data_collator=dc_collator,             # Dual-channel selection with uniform MLM.
)

# Resume is explicit; no machine-specific environment variable is required.
print(f"Resuming from: {resume_ckpt}" if resume_ckpt else "Starting a new training run.")

# Start or resume training.
trainer.train(resume_from_checkpoint=resume_ckpt)

# Save final model weights separately from intermediate checkpoints.
trainer.save_model(model_output_dir)
print(f"[Done] model saved to: {model_output_dir}")
