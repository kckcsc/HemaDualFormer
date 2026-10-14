# -*- coding: utf-8 -*-
"""Manual five-fold cross-validation with Class-Balanced loss (Cui 2019).
Patch the Geneformer Trainer to use effective-sample-count weighted cross-entropy."""

import os
import csv
import json
import glob
import pickle
import random
import datetime
import math
from collections import defaultdict, Counter
from pathlib import Path

import torch
import torch.nn as nn
from transformers import Trainer as HFTrainer
from transformers.trainer_callback import TrainerCallback
from datasets import load_from_disk
from geneformer import Classifier
import geneformer.classifier as gf_cls   # Patch Geneformer training components.
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import geneformer.classifier_utils as cu
# Runtime configuration.
# Edit these paths for your own machine. Relative paths use the kernel working directory.
MODEL_DIRECTORY = Path("models/pretrained")
INPUT_DATA_FILE = Path("data/gse253355_tokenized.dataset")
OUTPUT_DIRECTORY = Path("runs/gse253355_cb")
SPLITS_FILE = Path("gse253355_splits.json")
SAMPLE_COLUMN = "orig.ident"
CELL_COLUMN = "cell_type"
NPROC = 16
USE_FP16 = True
TEST_FORWARD_BATCH_SIZE = 200

for location in (MODEL_DIRECTORY, INPUT_DATA_FILE):
    if not location.is_dir():
        raise FileNotFoundError(f"Configure an existing input directory: {location.resolve()}")
with SPLITS_FILE.open(encoding="utf-8") as f:
    sample_splits = json.load(f)
for key in ("train_ids", "eval_ids", "test_ids"):
    values = sample_splits.get(key)
    if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
        raise ValueError(f"Split field {key} must be a list of sample identifiers")
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate identifiers in {key}")
if (set(sample_splits["train_ids"]) & set(sample_splits["eval_ids"]) or
    (set(sample_splits["train_ids"]) | set(sample_splits["eval_ids"])) & set(sample_splits["test_ids"])):
    raise ValueError("Sample partitions must not overlap")
if OUTPUT_DIRECTORY.exists() and (not OUTPUT_DIRECTORY.is_dir() or any(OUTPUT_DIRECTORY.iterdir())):
    raise ValueError("Choose a new or empty output directory; restarting does not resume training")
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
output_dir = str(OUTPUT_DIRECTORY.resolve())
output_prefix = "GSE253355_no_early_stopping_classifier"
model_directory = str(MODEL_DIRECTORY.resolve())
input_data_file = str(INPUT_DATA_FILE.resolve())
datestamp_min = datetime.datetime.now().strftime("%y%m%d")
with (OUTPUT_DIRECTORY / "sample_splits.json").open("w", encoding="utf-8") as f:
    json.dump(sample_splits, f, indent=2)
# Training hyperparameters retained from the original experiment.
training_args = {
    "num_train_epochs": 30,
    "learning_rate": 5.071449959818047e-05,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 1264,
    "weight_decay": 0.0938029929169129,
    "per_device_train_batch_size": 18,
    "seed": 42,
    "save_strategy": "epoch",
    "save_total_limit": 100,
    "evaluation_strategy": "epoch",
    "load_best_model_at_end": False,
    "fp16": USE_FP16,
}
# Initialize the cell classifier.
cc = Classifier(
    classifier="cell",
    cell_state_dict={"state_key": CELL_COLUMN, "states": "all"},
    filter_data=None,
    training_args=training_args,
    max_ncells=None,
    freeze_layers=2,
    num_crossval_splits=1,       # Use one internal split; supply each manual fold explicitly.
    forward_batch_size=32,
    nproc=NPROC
)
# Preserve the original sample partitions in an external configuration file.
train_ids = sample_splits["train_ids"]
eval_ids = sample_splits["eval_ids"]
test_ids = sample_splits["test_ids"]
train_test_id_split_dict = {
    "attr_key": SAMPLE_COLUMN,
    "train": train_ids + eval_ids,
    "test": test_ids,
}
# Prepare the training pool and held-out test dataset.
cc.prepare_data(
    input_data_file=input_data_file,
    output_directory=output_dir,
    output_prefix=output_prefix,
    split_id_dict=train_test_id_split_dict
)
prepared_train_file = f"{output_dir}/{output_prefix}_labeled_train.dataset"
prepared_test_file  = f"{output_dir}/{output_prefix}_labeled_test.dataset"
id_class_dict_file  = f"{output_dir}/{output_prefix}_id_class_dict.pkl"

# Load the training pool for class weights and fold grouping.
train_ds = load_from_disk(prepared_train_file)
print("[Info] Loaded labeled train dataset:", train_ds)

# Compute Class-Balanced weights from effective sample counts.
# The Geneformer cell classifier stores targets in label.
label_col = "label"
if label_col not in train_ds.column_names:
    raise ValueError(f"[CB] Missing {label_col}; available columns: {train_ds.column_names}")

label_list = list(train_ds[label_col])
class_counts = Counter(label_list)
num_classes = max(class_counts.keys()) + 1
total_samples = sum(class_counts.values())

# Set beta to (N - 1) / N, where N is the training-pool size.
beta = (total_samples - 1) / float(total_samples)
cb_weights = []
for c in range(num_classes):
    n_c = class_counts.get(c, 0)
    if n_c <= 0:
        cb_weights.append(0.0)
    else:
        # Class-Balanced weight: (1 - beta) / (1 - beta^n_c)
        weight_c = (1.0 - beta) / (1.0 - math.pow(beta, n_c))
        cb_weights.append(weight_c)

print("[CB] num_classes =", num_classes)
print("[CB] total_samples =", total_samples)
print("[CB] beta =", beta)
print("[CB] example weights (first 10):", cb_weights[:10])

# Expose weights to the custom Trainer.
gf_cls.CLASS_WEIGHTS = cb_weights
print("[CB] Registered geneformer.classifier.CLASS_WEIGHTS")
# Add rare-class macro-F1 to the metrics.

# Load label IDs and class names.
with open(id_class_dict_file, "rb") as f:
    id_class_dict = pickle.load(f)

# Specify the rare classes for this experiment.
# Retain the original list; validate its biological interpretation before publication.
RARE_LABEL_NAMES = ["earlyEry", "GMP", "cDC", "B", "ProB", "pDC"]

# Map rare-class names to label IDs.
rare_label_ids = []
for k, v in id_class_dict.items():
    try:
        cid = int(k)
    except Exception:
        cid = k
    if v in RARE_LABEL_NAMES:
        rare_label_ids.append(cid)

rare_label_ids = sorted(set(rare_label_ids))
print("[Rare-F1] Selected class names:", RARE_LABEL_NAMES)
print("[Rare-F1] Selected label IDs:", rare_label_ids)

# Keep the original metric function.
orig_compute_metrics = cu.compute_metrics


def compute_metrics_with_rare(pred):
    """Extend the original metrics with rare-class macro-F1, logged as eval_rare_macro_f1."""
    import numpy as np
    from sklearn.metrics import precision_recall_fscore_support

    # Compute the original accuracy and macro-F1.
    base_metrics = orig_compute_metrics(pred)

    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    # Handle both one-dimensional and flattened labels.
    if len(labels.shape) == 1:
        y_true = labels
        y_pred = preds
    else:
        flat_labels = labels.flatten().tolist()
        flat_preds = preds.flatten().tolist()
        logit_label_paired = [
            item for item in zip(flat_preds, flat_labels) if item[1] != -100
        ]
        y_pred = np.array([item[0] for item in logit_label_paired])
        y_true = np.array([item[1] for item in logit_label_paired])

    # Average F1 over the selected rare classes.
    if rare_label_ids:
        _, _, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=rare_label_ids,
            average=None,
            zero_division=0,
        )
        rare_macro_f1 = float(np.mean(f1))
    else:
        rare_macro_f1 = float("nan")

    base_metrics["rare_macro_f1"] = rare_macro_f1
    return base_metrics


# Install the extended metric function.
cu.compute_metrics = compute_metrics_with_rare
print("[Rare-F1] Patched compute_metrics to include eval_rare_macro_f1")
# Utilities for training-state parsing and CSV export.
def _ensure_json_dump_safe():
    """Patch json.dump once to serialize NumPy and pandas values."""
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
        if np is not None and hasattr(obj, "tolist"):  # ndarray
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
        return obj  # Leave other values to the standard JSON encoder.

    _json._orig_dump = _json.dump

    def _dump_safe(obj, fp, *args, **kwargs):
        return _json._orig_dump(_to_jsonable(obj), fp, *args, **kwargs)

    _json.dump = _dump_safe
    _json._gf_patched_dump = True


def _find_latest_trainer_state(search_root: str) -> str | None:
    """Recursively find the most recently modified trainer_state.json."""
    import os
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
    """Extract per-epoch training loss, validation loss, accuracy, macro-F1 and rare macro-F1. Return metric rows and the row with the highest validation macro-F1."""
    _ensure_json_dump_safe()

    import json
    with open(state_path, "r") as f:
        st = json.load(f)
    logs = st.get("log_history", [])

    by_epoch = {}
    ACC_KEYS  = ["eval_accuracy", "eval_acc"]
    F1_KEYS   = ["eval_macro_f1", "eval_f1_macro", "eval_f1", "eval_f1_score", "eval_f1_weighted"]
    RARE_KEYS = ["eval_rare_macro_f1", "eval_rare_f1", "eval_rare_macroF1"]

    for rec in logs:
        ep = rec.get("epoch", None)
        if ep is None:
            continue
        ep = float(ep)

        # Include rare-class macro-F1 in the epoch record.
        row = by_epoch.get(ep, {
            "epoch": ep,
            "train_loss": None,
            "val_loss": None,
            "val_accuracy": None,
            "val_macro_f1": None,
            "val_rare_macro_f1": None,
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

        # Read rare-class F1 from the log history.
        for k in RARE_KEYS:
            if k in rec:
                try:
                    row["val_rare_macro_f1"] = float(rec[k])
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
    """Write per-fold epoch metrics and append them to all_folds_epoch_metrics.csv."""
    import os, re, csv
    from pathlib import Path

    # Write this fold CSV.
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Epoch", "Training Loss", "Validation Loss", "Accuracy", "Macro F1", "Rare Macro F1"])
        for r in rows:
            w.writerow([
                r["epoch"],
                "" if r["train_loss"]        is None else f"{r['train_loss']:.6f}",
                "" if r["val_loss"]          is None else f"{r['val_loss']:.6f}",
                "" if r["val_accuracy"]      is None else f"{r['val_accuracy']:.6f}",
                "" if r["val_macro_f1"]      is None else f"{r['val_macro_f1']:.6f}",
                "" if r["val_rare_macro_f1"] is None else f"{r['val_rare_macro_f1']:.6f}",
            ])

    # Append the same rows to the cross-fold CSV.
    fold_match = re.search(r"_fold(\d+)_", os.path.basename(csv_path))
    fold_str = fold_match.group(1) if fold_match else ""

    root_dir = Path(csv_path).parent.parent
    master_csv = root_dir / "all_folds_epoch_metrics.csv"

    need_header = not master_csv.exists()
    with open(master_csv, "a", newline="") as mf:
        w = csv.writer(mf)
        if need_header:
            w.writerow(["Fold", "Epoch", "Training Loss", "Validation Loss", "Accuracy", "Macro F1", "Rare Macro F1"])
        for r in rows:
            w.writerow([
                fold_str,
                r["epoch"],
                "" if r["train_loss"]        is None else f"{r['train_loss']:.6f}",
                "" if r["val_loss"]          is None else f"{r['val_loss']:.6f}",
                "" if r["val_accuracy"]      is None else f"{r['val_accuracy']:.6f}",
                "" if r["val_macro_f1"]      is None else f"{r['val_macro_f1']:.6f}",
                "" if r["val_rare_macro_f1"] is None else f"{r['val_rare_macro_f1']:.6f}",
            ])

# Disable early stopping.
class _NoEarlyStopping(TrainerCallback):
    def __init__(self, *args, **kwargs):
        pass
    def on_evaluate(self, args, state, control, **kwargs):
        return control
    def on_train_end(self, args, state, control, **kwargs):
        return control

gf_cls.EarlyStoppingCallback = _NoEarlyStopping
cc.training_args["load_best_model_at_end"] = False
print("[Info] Early stopping disabled for five-fold training.")

class CBTrainer(HFTrainer):
    """Class-Balanced Trainer using weights from geneformer.classifier.CLASS_WEIGHTS."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Read class weights from the Geneformer module.
        try:
            cw = getattr(gf_cls, "CLASS_WEIGHTS", None)
        except Exception:
            cw = None
        if cw is not None:
            if isinstance(cw, torch.Tensor):
                self.class_weights = cw
            else:
                self.class_weights = torch.tensor(cw, dtype=torch.float32)
        else:
            self.class_weights = None

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
        **kwargs,
    ):
        """Accept num_items_in_batch for Trainer API compatibility; the custom loss does not use it."""
        # Accept either labels or label as the target key.
        label_key = None
        for lk in ("labels", "label"):
            if lk in inputs:
                label_key = lk
                break

        if label_key is None:
            # Fall back to the parent loss when no label key is present.
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
                **kwargs,
            )

        labels = inputs.pop(label_key)
        outputs = model(**inputs)
        logits = outputs.logits  # [batch, num_labels]

        # Move class weights to the logits device.
        if self.class_weights is not None:
            if self.class_weights.device != logits.device:
                self.class_weights = self.class_weights.to(logits.device)
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            loss_fct = nn.CrossEntropyLoss()

        loss = loss_fct(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )

        if return_outputs:
            return loss, outputs
        return loss
# Install the Class-Balanced Trainer.
gf_cls.Trainer = CBTrainer
print("[CB] Replaced geneformer.classifier.Trainer with CBTrainer")
# Run manual five-fold cross-validation.
K = 5
seed_for_cv = 42

# Count cells per sample in the training pool.
group_counts = defaultdict(int)
for ex in train_ds:
    group_counts[ex[SAMPLE_COLUMN]] += 1
groups = list(group_counts.keys())

# Shuffle with a fixed seed and assign samples to folds in round-robin order.
random.Random(seed_for_cv).shuffle(groups)
folds = [[] for _ in range(K)]
for i, g in enumerate(groups):
    folds[i % K].append(g)

# Train each fold and export validation metrics.
all_metrics_cv = {}
summary_rows = []  # [Fold, BestEpoch, BestValAcc, BestValMacroF1, RunDir]

for k in range(K):
    eval_ids_fold = folds[k]
    train_ids_fold = [g for i, gs in enumerate(folds) if i != k for g in gs]
    split_id_dict_fold = {"attr_key": SAMPLE_COLUMN, "train": train_ids_fold, "eval": eval_ids_fold}

    fold_out_dir = os.path.join(output_dir, f"fold{k+1}")
    os.makedirs(fold_out_dir, exist_ok=True)

    print(f"\n[CV] Fold {k+1}/{K} | #train groups={len(train_ids_fold)} | #eval groups={len(eval_ids_fold)}")
    print(f"[CV] train groups (head): {train_ids_fold[:5]} ...")
    print(f"[CV] eval  groups (head): {eval_ids_fold[:5]} ...")

    # Run training and validation with the patched Trainer.
    metrics_k = cc.validate(
        model_directory=model_directory,
        prepared_input_data_file=prepared_train_file,
        id_class_dict_file=id_class_dict_file,
        output_directory=fold_out_dir,
        output_prefix=f"{output_prefix}_fold{k+1}",
        split_id_dict=split_id_dict_fold,   # Pass this fold training and validation sample IDs.
        predict_trainer=True,               # Use Trainer predictions.
    )
    all_metrics_cv[f"fold{k+1}"] = metrics_k

    # Read the latest trainer state and export epoch metrics.
    trainer_state = _find_latest_trainer_state(fold_out_dir)
    if trainer_state is None:
        print(f"[Warn] No trainer_state.json in {fold_out_dir}; skipping CSV export.")
        continue

    rows, best = _extract_epoch_rows_from_trainer_state(trainer_state)

    fold_csv = os.path.join(fold_out_dir, f"{output_prefix}_fold{k+1}_epoch_metrics.csv")
    _write_fold_epoch_csv(fold_csv, rows)
    print(f"[CV] Per-epoch metrics saved to: {fold_csv}")

    # Record the best validation epoch metrics.
    if best is not None:
        summary_rows.append([
            f"fold{k+1}",
            f"{best['epoch']}",
            "" if best["val_accuracy"] is None else f"{best['val_accuracy']:.6f}",
            f"{best['val_macro_f1']:.6f}",
            os.path.dirname(trainer_state)  # Keep the run directory for traceability.
        ])
    else:
        summary_rows.append([f"fold{k+1}", "", "", "", os.path.dirname(trainer_state)])

# Save fold validation metrics and best-epoch summaries.
cv_metrics_json = os.path.join(output_dir, f"{output_prefix}_cv_metrics.json")
with open(cv_metrics_json, "w") as f:
    json.dump(all_metrics_cv, f, indent=2, ensure_ascii=False)

cv_summary_csv = os.path.join(output_dir, f"{output_prefix}_cv_summary.csv")
with open(cv_summary_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Fold", "Best Epoch", "Best Val Accuracy", "Best Val Macro F1", "Run Dir"])
    for row in summary_rows:
        w.writerow(row)

print("[CV] Five-fold training completed.")
print(f"[CV] Fold validation metrics: {cv_metrics_json}")
print(f"[CV] Best macro-F1 summary: {cv_summary_csv}")
print("[Hint] Per-epoch CSV files are in each fold directory.")

# Locate saved models and determine the plotting class order.
def _find_model_ksplit1_dir(fold_out_dir: str, output_prefix_fold: str) -> str | None:
    """Find the fold ksplit1 directory, first using the expected date-based path and then a recursive search for model weights or checkpoints."""
    import os
    expected = os.path.join(
        fold_out_dir,
        f"{datestamp_min}_geneformer_cellClassifier_{output_prefix_fold}",
        "ksplit1",
    )
    if os.path.isdir(expected):
        return expected

    # Search recursively if the expected directory is absent.
    candidates = []
    for p, dirs, files in os.walk(fold_out_dir):
        if os.path.basename(p) == "ksplit1":
            # Require checkpoint directories or model weight files.
            entries = set(os.listdir(p))
            has_ckpt = any(name.startswith("checkpoint-") for name in entries)
            has_weights = any(name.endswith((".bin", ".safetensors")) for name in entries)
            if has_ckpt or has_weights:
                candidates.append(p)
    if not candidates:
        return None
    candidates.sort(key=lambda d: os.path.getmtime(d), reverse=True)
    return candidates[0]


def _load_sorted_classes_from_id_dict(id_class_pkl: str):
    """Load a label-ID-to-class-name pickle and return alphabetically sorted class names for plots."""
    import pickle
    try:
        with open(id_class_pkl, "rb") as f:
            id2cls = pickle.load(f)
        return sorted(set(id2cls.values()))
    except Exception:
        return None


# Select a fold and evaluate on the fixed held-out test set.
# Evaluation workflow:
# Select the fold with the highest recorded validation macro-F1.
# Locate its ksplit1 model directory.
# Evaluate the saved model on the test set.
# Export plots and test metrics.
if summary_rows:
    def _safe_float(x):
        try:
            return float(x)
        except Exception:
            return float("-inf")

    best_row = max(summary_rows, key=lambda r: _safe_float(r[3]))  # r[3] = Best Val Macro F1
    best_fold_name = best_row[0]  # For example, fold3.
    best_fold_idx = int(best_fold_name.replace("fold", "")) - 1
    best_fold_dir = os.path.join(output_dir, best_fold_name)
    best_prefix_fold = f"{output_prefix}_fold{best_fold_idx+1}"

    print(f"[Test] Selected fold: {best_fold_name}; directory: {best_fold_dir}")

    model_dir_for_test = _find_model_ksplit1_dir(best_fold_dir, best_prefix_fold)
    if model_dir_for_test is None:
        print(f"[Warn] No ksplit1 directory in {best_fold_dir}; skipping test evaluation.")
    else:
        # Create a classifier for evaluation and plotting.
        cc_test = Classifier(
            classifier="cell",
            cell_state_dict={"state_key": CELL_COLUMN, "states": "all"},
            forward_batch_size=TEST_FORWARD_BATCH_SIZE,
            nproc=NPROC,
        )

        print(f"[Test] Evaluating model directory: {model_dir_for_test}")
        all_metrics_test = cc_test.evaluate_saved_model(
            model_directory=model_dir_for_test,
            id_class_dict_file=id_class_dict_file,
            test_data_file=prepared_test_file,
            output_directory=best_fold_dir,
            output_prefix=f"{best_prefix_fold}_test",
        )

        # Set an optional class order.
        custom_class_order = _load_sorted_classes_from_id_dict(id_class_dict_file)

        # Plot the confusion matrix.
        try:
            cm_dict = {"Geneformer": all_metrics_test["conf_matrix"]}
            kwargs = dict(
                conf_mat_dict=cm_dict,
                output_directory=best_fold_dir,
                output_prefix=f"{best_prefix_fold}_test",
            )
            if custom_class_order is not None:
                kwargs["custom_class_order"] = custom_class_order
            cc_test.plot_conf_mat(**kwargs)
            print(f"[Test] Confusion matrix saved to: {best_fold_dir}")
        except Exception as e:
            print(f"[Warn] Confusion matrix plotting failed: {e}")

        # Plot predictions saved by evaluate_saved_model.
        try:
            pred_pkl = os.path.join(best_fold_dir, f"{best_prefix_fold}_test_pred_dict.pkl")
            kwargs = dict(
                predictions_file=pred_pkl,
                id_class_dict_file=id_class_dict_file,
                title="cell_type",
                output_directory=best_fold_dir,
                output_prefix=f"{best_prefix_fold}_test",
            )
            if custom_class_order is not None:
                kwargs["custom_class_order"] = custom_class_order
            cc_test.plot_predictions(**kwargs)
            print(f"[Test] Prediction plot saved to: {best_fold_dir}")
        except Exception as e:
            print(f"[Warn] Prediction plotting failed: {e}")

        # Export test metrics to CSV.
        try:
            test_csv = os.path.join(best_fold_dir, f"{best_prefix_fold}_test_metrics.csv")
            with open(test_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["metric", "value"])
                for k, v in all_metrics_test.items():
                    if k == "conf_matrix":  # Exclude the confusion matrix from the scalar metric export.
                        continue
                    w.writerow([k, v])
            print(f"[Test] Test metrics saved to: {test_csv}")
        except Exception as e:
            print(f"[Warn] Test metric CSV export failed: {e}")
else:
    print("[Test] No fold summary found; skipping test evaluation.")
globals().get("all_metrics_test", "No test evaluation results available.")
# Export test metrics for each cell type.
try:
    # Skip per-class export when test evaluation was not completed.
    if "all_metrics_test" not in locals():
        print("[PerClass] No test results available; skipping per-class metrics.")
    else:
        import os
        import pickle
        import numpy as np
        import pandas as pd

        # Read the confusion matrix: rows are true labels, columns are predictions.
        cm_obj = all_metrics_test.get("conf_matrix", None)
        if cm_obj is None:
            # Try the alternative result key.
            cm_obj = all_metrics_test.get("conf_mat", None)

        if cm_obj is None:
            raise ValueError("Test results lack conf_matrix/conf_mat; cannot compute per-class metrics.")

        # Convert the confusion matrix to a labeled DataFrame.
        if isinstance(cm_obj, pd.DataFrame):
            cm_df = cm_obj
        else:
            with open(id_class_dict_file, "rb") as f:
                id_class_dict = pickle.load(f)
            classes = list(id_class_dict.values())
            cm_df = pd.DataFrame(cm_obj, index=classes, columns=classes)

        # Compute per-class counts.
        total_true = cm_df.sum(axis=1)                      # Number of cells with this true label.
        correct = np.diag(cm_df.values)                    # Correct predictions on the diagonal.
        wrong = total_true - correct                       # Incorrect predictions.
        acc_per_class = correct / total_true.replace(0, np.nan)  # Per-class recall, retained under the original accuracy column name.

        per_class_df = pd.DataFrame({
            "cell_type": cm_df.index,
            "n_true": total_true.astype(int).values,
            "n_correct": correct.astype(int),
            "n_wrong": wrong.astype(int).values,
            "accuracy": acc_per_class.values,
        })

        # Choose the output directory and filename.
        save_dir = best_fold_dir if "best_fold_dir" in locals() else output_dir
        prefix = best_prefix_fold if "best_prefix_fold" in locals() else output_prefix
        per_class_csv = os.path.join(
            save_dir,
            f"{prefix}_test_per_class_metrics.csv"
        )

        per_class_df.to_csv(per_class_csv, index=False)
        print(f"[PerClass] Metrics saved to: {per_class_csv}")

except Exception as e:
    print(f"[PerClass] Metric CSV export failed: {e}")
