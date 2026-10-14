# -*- coding: utf-8 -*-
"""
Multi-task Fine-tuning + Class-Balanced Loss for Long-tail Cells
- Shared encoder with cell-type and health-status heads.
- Class-Balanced cell-type loss (Cui 2019) for class imbalance.
- Standard health-status cross-entropy weighted by lambda_health.
- Five-fold cross-validation, per-epoch CSV metrics, and CV summaries.
"""

import os, re, json, math
import argparse
from pathlib import Path
import pickle
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import (
    BertConfig, BertModel, TrainingArguments, Trainer
)
from transformers.trainer_utils import IntervalStrategy
from transformers import set_seed

# Global settings
IGNORE_INDEX = -100
set_seed(42)

# Metrics: accuracy and macro-F1
def _acc_macro_f1_from_logits(logits, labels, ignore_index=IGNORE_INDEX):
    """
    logits: [N, C]
    labels: [N]
    Include only examples whose label differs from ignore_index.
    """
    if hasattr(logits, "detach"):
        logits = logits.detach().cpu().numpy()
    if hasattr(labels, "detach"):
        labels = labels.detach().cpu().numpy()

    labels = labels.astype(int)
    mask = labels != ignore_index
    if mask.sum() == 0:
        return {"acc": float("nan"), "macro_f1": float("nan")}

    logits = logits[mask]
    y = labels[mask]
    y_pred = logits.argmax(axis=-1)

    acc = (y_pred == y).mean().item()

    classes = np.unique(y)
    f1s = []
    for c in classes:
        tp = ((y_pred == c) & (y == c)).sum()
        fp = ((y_pred == c) & (y != c)).sum()
        fn = ((y_pred != c) & (y == c)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    macro_f1 = float(np.mean(f1s)) if len(f1s) > 0 else float("nan")
    return {"acc": float(acc), "macro_f1": macro_f1}


def compute_metrics(eval_pred):
    """
    eval_pred.predictions: (cell_logits, health_logits)
    eval_pred.label_ids : (cell_labels, health_labels)
    """
    cell_logits, health_logits = eval_pred.predictions
    cell_labels, health_labels = eval_pred.label_ids
    m_cell   = _acc_macro_f1_from_logits(cell_logits,   cell_labels,   ignore_index=IGNORE_INDEX)
    m_health = _acc_macro_f1_from_logits(health_logits, health_labels, ignore_index=IGNORE_INDEX)
    return {
        "cell_acc":         m_cell["acc"],
        "cell_macro_f1":    m_cell["macro_f1"],
        "health_acc":       m_health["acc"],
        "health_macro_f1":  m_health["macro_f1"],
    }

# 0) Input paths and sample partitions
parser = argparse.ArgumentParser(description="Multi-task fine-tuning with Class-Balanced cell-type loss")
parser.add_argument("--dataset", required=True, help="Tokenized Dataset directory saved with save_to_disk")
parser.add_argument("--pretrained-model", required=True, help="Local pretrained BERT model directory")
parser.add_argument("--output-dir", required=True, help="New or empty experiment output directory")
parser.add_argument("--splits", default=str(Path(__file__).with_name("gse116256_splits.json")), help="JSON with train_ids, eval_ids_fixed, and test_ids")
parser.add_argument("--sample-column", default="orig.ident")
parser.add_argument("--cell-column", default="cell_type")
parser.add_argument("--health-column", default=None, help="Health label column; detect automatically if omitted")
parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16", help="Original experiment used fp16; choose fp32 for devices without FP16 support")
cli_args = parser.parse_args()
tokenized_dataset_path = str(Path(cli_args.dataset).expanduser().resolve())
pretrain_dir = str(Path(cli_args.pretrained_model).expanduser().resolve())
for location in [tokenized_dataset_path, pretrain_dir]:
    if not Path(location).is_dir():
        parser.error(f"Input directory does not exist: {location}")
with open(Path(cli_args.splits).expanduser(), encoding="utf-8") as f:
    split_config = json.load(f)
for key in ["train_ids", "eval_ids_fixed", "test_ids"]:
    values = split_config.get(key)
    if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
        parser.error(f"Split field {key} must be a list of sample identifiers")
    if len(values) != len(set(values)):
        parser.error(f"Duplicate sample identifiers in {key}")
train_ids = split_config["train_ids"]
eval_ids_fixed = split_config["eval_ids_fixed"]
test_ids = split_config["test_ids"]
if set(train_ids) & set(eval_ids_fixed) or (set(train_ids) | set(eval_ids_fixed)) & set(test_ids):
    parser.error("Sample identifiers must not overlap across the supplied partitions")
output_prefix = "GSE116256_multitask_CB"
output_path = Path(cli_args.output_dir).expanduser().resolve()
if output_path.exists() and (not output_path.is_dir() or any(output_path.iterdir())):
    parser.error("Output directory must be new or empty to avoid mixing experiment results")
output_dir = str(output_path)
os.makedirs(output_dir, exist_ok=True)
with open(output_path / "sample_splits.json", "w", encoding="utf-8") as f:
    json.dump(split_config, f, indent=2)

# 1) Build labeled datasets from tokenized inputs
print("[Build] loading original tokenized dataset ...")
full = load_from_disk(tokenized_dataset_path)

ID_COL = cli_args.sample_column
CELL_COL = cli_args.cell_column
HEALTH_CANDIDATES = [cli_args.health_column] if cli_args.health_column else ["health_status", "disease_group", "health", "disease_status"]
for column in [ID_COL, CELL_COL, "input_ids"] + ([cli_args.health_column] if cli_args.health_column else []):
    if column not in full.column_names:
        raise ValueError(f"Dataset is missing required column: {column}")

def in_ids(ids_set):
    return lambda ex: ex.get(ID_COL, None) in ids_set

# Combine train_ids and eval_ids_fixed into the cross-validation pool.
train_eval_raw = full.filter(in_ids(set(train_ids) | set(eval_ids_fixed)))
test_ds_raw    = full.filter(in_ids(set(test_ids)))

# 1.2 Map cell types to integer labels.
if CELL_COL not in train_eval_raw.column_names:
    raise ValueError(f"Dataset is missing {CELL_COL}; specify --cell-column.")
u = sorted({x for x in train_eval_raw[CELL_COL] if x is not None})
cell2id = {v: i for i, v in enumerate(u)}
id2cell = {i: v for v, i in cell2id.items()}

with open(f"{output_dir}/{output_prefix}_id_class_dict.pkl", "wb") as f:
    pickle.dump(id2cell, f)

def add_cell_labels(ex):
    ex["cell_labels"] = cell2id.get(ex.get(CELL_COL, None), IGNORE_INDEX)
    return ex

train_eval_raw = train_eval_raw.map(add_cell_labels)
test_ds_raw    = test_ds_raw.map(add_cell_labels)

# 1.3 Detect the health-status column and build health labels.
def detect_health_key(ds):
    for k in HEALTH_CANDIDATES:
        if k in ds.column_names:
            vals = [v for v in ds[k] if v is not None]
            if len(vals) > 0:
                return k
    return None

health_key = detect_health_key(train_eval_raw)
if health_key is None:
    print("[Warn] No health-status column found; all health labels will be -100. Health-task training requires valid labels.")
    health_l2i, health_i2l = {}, {}
else:
    uniq = sorted({v for v in (list(train_eval_raw[health_key]) + list(test_ds_raw[health_key])) if v is not None})
    health_l2i = {v: i for i, v in enumerate(uniq)}
    health_i2l = {i: v for v, i in health_l2i.items()}
    print("[Info] health label map:", health_l2i)

def add_health_labels(ex):
    if health_key is None:
        ex["health_labels"] = IGNORE_INDEX
    else:
        v = ex.get(health_key, None)
        ex["health_labels"] = health_l2i.get(v, IGNORE_INDEX)
    return ex

train_eval_raw = train_eval_raw.map(add_health_labels)
test_ds_raw    = test_ds_raw.map(add_health_labels)

print("[Info] sizes -> train_eval_raw:", len(train_eval_raw), " test:", len(test_ds_raw))

# 2) Load the pretrained encoder configuration and determine PAD_ID
print("[Info] using pretrained encoder from:", pretrain_dir)

PAD_ID = BertConfig.from_pretrained(pretrain_dir).pad_token_id
if PAD_ID is None:
    PAD_ID = 0
print("[Info] PAD_ID =", PAD_ID)

def ensure_attn(example):
    if "attention_mask" not in example or example["attention_mask"] is None:
        ids = example["input_ids"]
        example["attention_mask"] = [0 if t == PAD_ID else 1 for t in ids]
    return example

train_eval_raw = train_eval_raw.map(ensure_attn)
test_ds_raw    = test_ds_raw.map(ensure_attn)

# 3) Compute Class-Balanced cell-type weights (Cui 2019)
def compute_cb_weights_from_dataset(ds, label_key="cell_labels", ignore_index=IGNORE_INDEX):
    """
    Class-Balanced weights based on effective sample counts (Cui 2019).
    Count only examples whose label differs from ignore_index.
    """
    labels = [int(v) for v in ds[label_key] if int(v) != ignore_index]
    if len(labels) == 0:
        raise ValueError(f"[CB] No valid labels in {label_key}; cannot compute Class-Balanced weights.")

    class_counts = Counter(labels)
    num_classes = max(class_counts.keys()) + 1
    total_samples = sum(class_counts.values())

    if total_samples <= 1:
        beta = 0.999
    else:
        beta = (total_samples - 1) / float(total_samples)

    cb_weights = []
    for c in range(num_classes):
        n_c = class_counts.get(c, 0)
        if n_c <= 0:
            cb_weights.append(0.0)
        else:
            weight_c = (1.0 - beta) / (1.0 - math.pow(beta, n_c))
            cb_weights.append(weight_c)

    w_tensor = torch.tensor(cb_weights, dtype=torch.float32)
    return w_tensor, class_counts, num_classes, total_samples, beta

cb_cell_weights, class_counts, num_classes_cb, total_samples_cb, beta_cb = compute_cb_weights_from_dataset(
    train_eval_raw, label_key="cell_labels", ignore_index=IGNORE_INDEX
)
print("[CB-MT] num_classes =", num_classes_cb)
print("[CB-MT] total_samples =", total_samples_cb)
print("[CB-MT] beta =", beta_cb)
print("[CB-MT] example weights (first 10):", cb_cell_weights.tolist()[:10])

# 4) Pad input IDs, attention masks, and both label arrays
def simple_pad_collator(pad_id: int):
    def to_list(x):
        if isinstance(x, torch.Tensor):
            return x.tolist()
        if isinstance(x, np.ndarray):
            return x.tolist()
        return x

    def to_int(x):
        if isinstance(x, torch.Tensor):
            return int(x.item())
        if isinstance(x, np.generic):
            return int(x)
        return int(x)

    def collate(batch):
        seq_lens = [len(to_list(ex["input_ids"])) for ex in batch]
        max_len = max(seq_lens)
        ids, am = [], []
        y_cell, y_health = [], []

        for ex in batch:
            x = to_list(ex["input_ids"])
            m = to_list(ex["attention_mask"])
            pad = max_len - len(x)
            if pad > 0:
                x = x + [pad_id] * pad
                m = m + [0] * pad
            ids.append(torch.tensor(x, dtype=torch.long))
            am.append(torch.tensor(m, dtype=torch.long))
            y_cell.append(to_int(ex["cell_labels"]))
            y_health.append(to_int(ex["health_labels"]))

        return {
            "input_ids": torch.stack(ids),
            "attention_mask": torch.stack(am),
            "cell_labels": torch.tensor(y_cell, dtype=torch.long),
            "health_labels": torch.tensor(y_health, dtype=torch.long),
        }

    return collate

data_collator = simple_pad_collator(PAD_ID)

# 5) Shared encoder with two heads and Class-Balanced cell loss
class MultiTaskHead(nn.Module):
    def __init__(self, hidden_size, n_classes):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.cls  = nn.Linear(hidden_size, n_classes)

    def forward(self, pooled):
        return self.cls(self.norm(pooled))


class MultiTaskModel(nn.Module):
    def __init__(
        self,
        backbone_dir: str,
        num_cell_classes: int,
        num_health_classes: int,
        cb_cell_weights: torch.Tensor | None = None,
        pooling: str = "mean",
        lambda_health: float = 0.1,   # Weight of the health-status task
        freeze_layers: int = 2
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained(
            backbone_dir,
            add_pooling_layer=False,
            ignore_mismatched_sizes=True
        )
        self.config = BertConfig.from_pretrained(backbone_dir)
        self.pooling = pooling
        self.lambda_health = float(lambda_health)
        hidden = self.config.hidden_size

        self.cell_head   = MultiTaskHead(hidden, num_cell_classes)
        self.health_head = MultiTaskHead(hidden, num_health_classes)

        # Retain settings for experiment metadata.
        self._freeze_layers = freeze_layers
        self._lambda_health = float(lambda_health)

        # Register weights as a buffer so they follow the model device.
        if cb_cell_weights is not None:
            if cb_cell_weights.dim() != 1:
                raise ValueError(f"[CB-MT] cb_cell_weights must be one-dimensional; got {cb_cell_weights.shape}")
            self.register_buffer("cb_cell_weights", cb_cell_weights)
        else:
            self.cb_cell_weights = None

        # Freeze the specified number of initial encoder layers.
        if freeze_layers > 0:
            for i, layer in enumerate(self.bert.encoder.layer):
                if i < freeze_layers:
                    for p in layer.parameters():
                        p.requires_grad = False

        # Use unweighted cross-entropy for the health-status head.
        self.ce_health = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    def _pool(self, last_hidden_state, attention_mask, input_ids=None):
        if self.pooling == "cls":
            return last_hidden_state[:, 0]
        else:
            mask = attention_mask.unsqueeze(-1).float()
            s = (last_hidden_state * mask).sum(dim=1)
            d = mask.sum(dim=1).clamp(min=1e-6)
            return s / d

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        cell_labels=None,
        health_labels=None,
    ):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled = self._pool(out.last_hidden_state, attention_mask, input_ids)

        cell_logits   = self.cell_head(pooled)    # [B, C_cell]
        health_logits = self.health_head(pooled)  # [B, C_health]

        loss = None
        logs = {}

        # ====== Class-Balanced Loss for cell_labels ======
        if cell_labels is not None:
            logits_flat = cell_logits.view(-1, cell_logits.size(-1))
            labels_flat = cell_labels.view(-1)
            cell_loss = F.cross_entropy(
                logits_flat,
                labels_flat,
                weight=getattr(self, "cb_cell_weights", None),
                ignore_index=IGNORE_INDEX,
                reduction="mean",
            )
            loss = cell_loss
            logs["cell_loss"] = cell_loss

        # Standard cross-entropy for health labels
        if health_labels is not None:
            health_loss = self.ce_health(
                health_logits.view(-1, health_logits.size(-1)),
                health_labels.view(-1)
            )
            if loss is None:
                loss = health_loss
            else:
                loss = loss + self.lambda_health * health_loss
            logs["health_loss"] = self.lambda_health * health_loss

        return {
            "loss": loss,
            "cell_logits": cell_logits,
            "health_logits": health_logits,
            **logs
        }

# 6) Retain the supplied experiment hyperparameters
base_training_cfg = {
    "num_train_epochs": 30,
    "learning_rate": 0.0003461103096079109,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 993,
    "weight_decay": 0.29327198625766,
    "per_device_train_batch_size": 18,
    "per_device_eval_batch_size": 32,
    "save_strategy": IntervalStrategy.EPOCH,
    "evaluation_strategy": IntervalStrategy.EPOCH,
    "save_total_limit": 20,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_cell_macro_f1",
    "greater_is_better": True,
    "remove_unused_columns": False,
    "fp16": cli_args.precision == "fp16",
    "logging_steps": 50,
    "report_to": [],
    "seed": 42,
    "data_seed": 42,
}

# 7) Custom Trainer returning (cell, health) prediction and label tuples
class MTTrainer(Trainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # Keep only the model input fields.
        used = {k: v for k, v in inputs.items()
                if k in ["input_ids", "attention_mask", "cell_labels", "health_labels"]}

        with torch.no_grad():
            outputs = model(**used)
            loss = outputs["loss"].detach()

        if prediction_loss_only:
            return (loss, None, None)

        cell_logits = outputs["cell_logits"].detach().cpu()
        health_logits = outputs["health_logits"].detach().cpu()

        labels_cell = used.get("cell_labels", None)
        labels_health = used.get("health_labels", None)
        if labels_cell is not None:
            labels_cell = labels_cell.detach().cpu()
        if labels_health is not None:
            labels_health = labels_health.detach().cpu()

        logits = (cell_logits, health_logits)
        labels = (labels_cell, labels_health)
        return (loss, logits, labels)

# 8) Parse trainer_state.json
def _ensure_json_dump_safe():
    import json as _json
    if getattr(_json, "_gf_patched_dump", False):
        return

    def _to_jsonable(obj):
        try:
            import numpy as np
        except Exception:
            np = None
        try:
            import pandas as pd
        except Exception:
            pd = None

        if isinstance(obj, dict):
            return {k: _to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_jsonable(v) for v in obj]
        if np is not None and isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        if np is not None and hasattr(obj, "tolist"):
            try:
                return obj.tolist()
            except Exception:
                pass
        if pd is not None:
            try:
                import pandas as pd  # noqa
                if isinstance(obj, pd.DataFrame):
                    return obj.to_dict(orient="records")
                if isinstance(obj, pd.Series):
                    return obj.to_dict()
            except Exception:
                pass
        return obj

    _json._orig_dump = _json.dump

    def _dump_safe(obj, fp, *args, **kwargs):
        return _json._orig_dump(_to_jsonable(obj), fp, *args, **kwargs)

    _json.dump = _dump_safe
    _json._gf_patched_dump = True


def _find_latest_trainer_state(search_root: str) -> str | None:
    candidates = []
    for p, _, files in os.walk(search_root):
        if "trainer_state.json" in files:
            full = os.path.join(p, "trainer_state.json")
            try:
                mtime = os.path.getmtime(full)
            except FileNotFoundError:
                continue
            candidates.append((mtime, full))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _extract_epoch_rows_from_trainer_state(state_path: str):
    """
    Extract per-epoch metrics from trainer_state.json:
      - train loss（loss）
      - eval loss（eval_loss）
      - eval cell_acc
      - eval cell_macro_f1
    """
    _ensure_json_dump_safe()

    import json
    with open(state_path, "r") as f:
        st = json.load(f)
    logs = st.get("log_history", [])

    by_epoch = {}
    ACC_KEYS = ["eval_cell_acc", "eval_accuracy", "eval_acc"]
    F1_KEYS  = ["eval_cell_macro_f1", "eval_macro_f1", "eval_f1_macro", "eval_f1", "eval_f1_score", "eval_f1_weighted"]

    for rec in logs:
        ep = rec.get("epoch", None)
        if ep is None:
            continue
        ep = float(ep)
        row = by_epoch.get(ep, {
            "epoch": ep,
            "train_loss": None,
            "val_loss": None,
            "val_accuracy": None,
            "val_macro_f1": None
        })

        if "loss" in rec and row["train_loss"] is None:
            try:
                row["train_loss"] = float(rec["loss"])
            except Exception:
                pass

        if "eval_loss" in rec:
            try:
                row["val_loss"] = float(rec["eval_loss"])
            except Exception:
                pass

        for k in ACC_KEYS:
            if k in rec:
                try:
                    row["val_accuracy"] = float(rec[k])
                except Exception:
                    pass
                break

        for k in F1_KEYS:
            if k in rec:
                try:
                    row["val_macro_f1"] = float(rec[k])
                except Exception:
                    pass
                break

        by_epoch[ep] = row

    rows = [by_epoch[e] for e in sorted(by_epoch.keys())]

    best = None
    for r in rows:
        mf1 = r["val_macro_f1"]
        if mf1 is None:
            continue
        if (best is None) or (mf1 > best["val_macro_f1"]):
            best = r

    return rows, best


def _write_fold_epoch_csv(csv_path: str, rows):
    import csv as _csv

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["Epoch", "Training Loss", "Validation Loss", "Cell Accuracy", "Cell Macro F1"])
        for r in rows:
            w.writerow([
                r["epoch"],
                "" if r["train_loss"]   is None else f"{r['train_loss']:.6f}",
                "" if r["val_loss"]     is None else f"{r['val_loss']:.6f}",
                "" if r["val_accuracy"] is None else f"{r['val_accuracy']:.6f}",
                "" if r["val_macro_f1"] is None else f"{r['val_macro_f1']:.6f}",
            ])

    # Append metrics to all_folds_epoch_metrics.csv.
    m = re.search(r"_fold(\d+)_", os.path.basename(csv_path))
    fold_str = m.group(1) if m else ""

    root_dir  = Path(csv_path).parent.parent
    master_csv = root_dir / "all_folds_epoch_metrics.csv"

    need_header = not master_csv.exists()
    with open(master_csv, "a", newline="") as mf:
        w = _csv.writer(mf)
        if need_header:
            w.writerow(["Fold", "Epoch", "Training Loss", "Validation Loss", "Cell Accuracy", "Cell Macro F1"])
        for r in rows:
            w.writerow([
                fold_str,
                r["epoch"],
                "" if r["train_loss"]   is None else f"{r['train_loss']:.6f}",
                "" if r["val_loss"]     is None else f"{r['val_loss']:.6f}",
                "" if r["val_accuracy"] is None else f"{r['val_accuracy']:.6f}",
                "" if r["val_macro_f1"] is None else f"{r['val_macro_f1']:.6f}",
            ])

# 9) Count output classes
def count_classes_from_labels(ds, key):
    vals = [int(v) for v in ds[key] if int(v) != IGNORE_INDEX]
    return (max(vals) + 1) if len(vals) > 0 else 0

n_cell_classes = count_classes_from_labels(train_eval_raw, "cell_labels")
tmp_health_classes = count_classes_from_labels(train_eval_raw, "health_labels")
n_health_classes = tmp_health_classes if tmp_health_classes > 0 else 2
print(f"[Info] n_cell_classes={n_cell_classes}, n_health_classes={n_health_classes}")

# 10) Five-fold cross-validation grouped by the sample column
K = 5
seed_for_cv = 42

group_counts = defaultdict(int)
for ex in train_eval_raw:
    group_counts[ex[ID_COL]] += 1
groups = list(group_counts.keys())

rng = np.random.RandomState(seed_for_cv)
rng.shuffle(groups)

fold_groups = [[] for _ in range(K)]
for i, g in enumerate(groups):
    fold_groups[i % K].append(g)

all_metrics_cv = {}
summary_rows = []

for k in range(K):
    eval_groups = set(fold_groups[k])
    train_groups_fold = set(groups) - eval_groups

    def is_train_fold(ex):
        return ex.get(ID_COL, None) in train_groups_fold

    def is_eval_fold(ex):
        return ex.get(ID_COL, None) in eval_groups

    train_ds = train_eval_raw.filter(is_train_fold)
    eval_ds  = train_eval_raw.filter(is_eval_fold)

    fold_out_dir = os.path.join(output_dir, f"fold{k+1}")
    os.makedirs(fold_out_dir, exist_ok=True)

    print(f"\n[CV] Fold {k+1}/{K} | #train groups={len(train_groups_fold)} | #eval groups={len(eval_groups)}")
    print(f"[CV] train groups (head): {list(train_groups_fold)[:5]} ...")
    print(f"[CV] eval  groups (head): {list(eval_groups)[:5]} ...")

    args = TrainingArguments(
        output_dir=fold_out_dir,
        num_train_epochs=base_training_cfg["num_train_epochs"],
        learning_rate=base_training_cfg["learning_rate"],
        lr_scheduler_type=base_training_cfg["lr_scheduler_type"],
        warmup_steps=base_training_cfg["warmup_steps"],
        weight_decay=base_training_cfg["weight_decay"],
        per_device_train_batch_size=base_training_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=base_training_cfg["per_device_eval_batch_size"],
        evaluation_strategy=base_training_cfg["evaluation_strategy"],
        save_strategy=base_training_cfg["save_strategy"],
        save_total_limit=base_training_cfg["save_total_limit"],
        load_best_model_at_end=base_training_cfg["load_best_model_at_end"],
        metric_for_best_model=base_training_cfg["metric_for_best_model"],
        greater_is_better=base_training_cfg["greater_is_better"],
        remove_unused_columns=base_training_cfg["remove_unused_columns"],
        fp16=base_training_cfg["fp16"],
        logging_steps=base_training_cfg["logging_steps"],
        report_to=base_training_cfg["report_to"],
        seed=base_training_cfg["seed"],
        data_seed=base_training_cfg["data_seed"],
    )

    model = MultiTaskModel(
        backbone_dir=pretrain_dir,
        num_cell_classes=n_cell_classes,
        num_health_classes=n_health_classes,
        cb_cell_weights=cb_cell_weights,  # Pass the Class-Balanced weights.
        pooling="mean",
        lambda_health=0.1,  # Weight of the health-status loss
        freeze_layers=2
    )

    trainer = MTTrainer(
        model=model,
        args=args,
        data_collator=data_collator,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Evaluate the selected model on this fold validation set.
    metrics_k = trainer.evaluate(eval_dataset=eval_ds)
    all_metrics_cv[f"fold{k+1}"] = metrics_k

    # Parse trainer_state.json and write per-epoch CSV metrics.
    trainer_state = _find_latest_trainer_state(fold_out_dir)
    if trainer_state is None:
        print(f"[Warn] No trainer_state.json found in {fold_out_dir}; skipping CSV export for this fold.")
        continue

    rows, best = _extract_epoch_rows_from_trainer_state(trainer_state)
    fold_csv = os.path.join(fold_out_dir, f"{output_prefix}_fold{k+1}_epoch_metrics.csv")
    _write_fold_epoch_csv(fold_csv, rows)
    print(f"[CV] Per-epoch fold metrics saved to: {fold_csv}")

    if best is not None:
        summary_rows.append([
            f"fold{k+1}",
            f"{best['epoch']}",
            "" if best["val_accuracy"] is None else f"{best['val_accuracy']:.6f}",
            f"{best['val_macro_f1']:.6f}",
            os.path.dirname(trainer_state)
        ])
    else:
        summary_rows.append([f"fold{k+1}", "", "", "", os.path.dirname(trainer_state)])

# 11) Save the five-fold summary
cv_metrics_json = os.path.join(output_dir, f"{output_prefix}_cv_metrics.json")
with open(cv_metrics_json, "w") as f:
    json.dump(all_metrics_cv, f, indent=2, ensure_ascii=False)

cv_summary_csv = os.path.join(output_dir, f"{output_prefix}_cv_summary.csv")
import csv as _csv
with open(cv_summary_csv, "w", newline="") as f:
    w = _csv.writer(f)
    w.writerow(["Fold", "Best Epoch", "Best Val Cell Accuracy", "Best Val Cell Macro F1", "Run Dir"])
    for row in summary_rows:
        w.writerow(row)

print(f"\n[CV] Five-fold multi-task training with Class-Balanced loss completed.")
print(f"[CV] Fold evaluation metrics saved to: {cv_metrics_json}")
print(f"[CV] Best cell macro-F1 summary saved to: {cv_summary_csv}")
print(f"[Hint] Per-epoch metrics are saved in each fold directory (*_epoch_metrics.csv).")
