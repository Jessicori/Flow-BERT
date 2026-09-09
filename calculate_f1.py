#!/usr/bin/env python3
"""Compute headline classification metrics for Flow-BERT evaluation runs.

Reads a gold TSV (a ``label`` column) and a prediction TSV whose first column
holds the predicted class id, then reports accuracy and macro/weighted/micro
precision, recall and F1.  Output format is kept stable so existing evaluation
logs remain comparable.

Usage:
    python3 calculate_f1.py --gold <gold.tsv> --pred <pred.tsv> \
        [--out_tsv <metrics.tsv>] [--out_json <metrics.json>]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List, Sequence

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)


def _gold_labels(path: str) -> List[int]:
    """Load true labels from the ``label`` column of a tab-separated file."""
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError(
                f"gold file must contain a 'label' column, got {reader.fieldnames!r}"
            )
        return [int(row["label"]) for row in reader]


def _predicted_labels(path: str) -> List[int]:
    """Load predicted class ids from the first column of a prediction file."""
    labels: List[int] = []
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        next(reader, None)  # skip header row
        for row in reader:
            if row and row[0].strip():
                labels.append(int(row[0].strip()))
    return labels


def evaluate(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    """Return the standard metric bundle for one evaluation slice."""
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"gold/pred length mismatch: gold={len(y_true)} pred={len(y_pred)}"
        )

    macro = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "n_samples": len(y_true),
        "AC": float(accuracy_score(y_true, y_pred)),
        "PR_macro": float(macro[0]),
        "RC_macro": float(macro[1]),
        "F1_macro": float(macro[2]),
        "PR_weighted": float(weighted[0]),
        "RC_weighted": float(weighted[1]),
        "F1_weighted": float(weighted[2]),
        "F1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
    }


_FIELD_ORDER = [
    "n_samples",
    "AC",
    "PR_macro",
    "RC_macro",
    "F1_macro",
    "PR_weighted",
    "RC_weighted",
    "F1_weighted",
    "F1_micro",
]


def _write_table(path: str, metrics: Dict[str, float]) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(_FIELD_ORDER)
        writer.writerow(
            [metrics["n_samples"]]
            + [f"{metrics[k]:.6f}" for k in _FIELD_ORDER[1:]]
        )


def _write_json(path: str, metrics: Dict[str, float]) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)


def _report(metrics: Dict[str, float]) -> None:
    for key in _FIELD_ORDER:
        if key == "n_samples":
            print(f"{key}={metrics[key]}")
        else:
            print(f"{key}={metrics[key]:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score Flow-BERT predictions against gold labels."
    )
    parser.add_argument("--gold", required=True, help="gold TSV with 'label' column")
    parser.add_argument("--pred", required=True, help="prediction TSV (first col = label)")
    parser.add_argument("--out_tsv", default=None, help="optional TSV dump of metrics")
    parser.add_argument("--out_json", default=None, help="optional JSON dump of metrics")
    args = parser.parse_args()

    metrics = evaluate(_gold_labels(args.gold), _predicted_labels(args.pred))
    _report(metrics)

    if args.out_tsv:
        _write_table(args.out_tsv, metrics)
        print(f"Saved TSV: {os.path.abspath(args.out_tsv)}")
    if args.out_json:
        _write_json(args.out_json, metrics)
        print(f"Saved JSON: {os.path.abspath(args.out_json)}")


if __name__ == "__main__":
    main()
