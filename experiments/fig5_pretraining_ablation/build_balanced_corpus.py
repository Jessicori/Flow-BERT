#!/usr/bin/env python3
"""Build per-class-balanced labeled pretrain TSVs (6000 rows/class, seed 7).

Source: <NAME>_finetune.tsv with header ``label \\t flow_tokens \\t text_a``.
Classes with fewer than 6000 rows are oversampled with replacement; classes
with more are randomly subsampled. Outputs
``corpora/cl_corpora/<NAME>_finetune_balanced.tsv``.
"""

import argparse
import os
import random
import sys
import time

HEADER = "label\tflow_tokens\ttext_a"
TARGET = 6000

DATASETS = {
    "Tor": 35,
    "NonTor": 39,
    "VPN": 29,
    "CIC": 10,
    "USTC": 20,
    "CSTNET": 120,
}


def build_one(name, nclasses, src_dir, out_dir, seed=7, idx=0, target=TARGET):
    src = os.path.join(src_dir, f"{name}_finetune.tsv")
    out = os.path.join(out_dir, f"{name}_finetune_balanced.tsv")
    if not os.path.isfile(src):
        raise SystemExit(f"missing source: {src}")
    rng = random.Random(seed * 1000 + idx)

    counts = {}
    with open(src, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n")
        if header != HEADER:
            raise SystemExit(f"{name}: unexpected header {header!r}")
        for line in f:
            if not line.strip():
                continue
            label = int(line.split("\t", 1)[0])
            counts[label] = counts.get(label, 0) + 1

    labels = sorted(counts)
    if len(labels) != nclasses:
        raise SystemExit(f"{name}: expected {nclasses} classes, found {len(labels)}")

    plan = {}
    for label in labels:
        cnt = counts[label]
        if cnt >= target:
            plan[label] = sorted(rng.sample(range(cnt), target))
        else:
            plan[label] = sorted(rng.randrange(cnt) for _ in range(target))

    seen = {label: 0 for label in labels}
    pos = {label: 0 for label in labels}
    written = 0
    tmp = out + ".tmp"
    with open(src, "r", encoding="utf-8") as fin, open(tmp, "w", encoding="utf-8") as fout:
        fout.write(HEADER + "\n")
        fin.readline()
        for line in fin:
            if not line.strip():
                continue
            label = int(line.split("\t", 1)[0])
            ord_ = seen[label]
            seen[label] += 1
            chosen = plan[label]
            p = pos[label]
            while p < len(chosen) and chosen[p] == ord_:
                fout.write(line)
                p += 1
                written += 1
            pos[label] = p

    for label in labels:
        if pos[label] != len(plan[label]):
            raise SystemExit(f"{name}: class {label} incomplete")
    if written != nclasses * target:
        raise SystemExit(f"{name}: wrote {written}, expected {nclasses * target}")
    os.replace(tmp, out)
    print(
        f"{name}: {written} rows ({nclasses} x {target}), "
        f"source min={min(counts.values())} max={max(counts.values())} -> {out}",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="*", default=list(DATASETS))
    ap.add_argument("--src-dir", default="/home/wzw/ET-BERT-main/corpora/ISCX/TSV")
    ap.add_argument("--out-dir", default="/home/wzw/Flow-BERT/corpora/cl_corpora")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--target", type=int, default=TARGET)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for i, name in enumerate(args.names):
        if name not in DATASETS:
            raise SystemExit(f"unknown dataset: {name}")
        t0 = time.time()
        build_one(name, DATASETS[name], args.src_dir, args.out_dir, args.seed, i, args.target)
        print(f"  ({time.time() - t0:.1f}s)", flush=True)
    print("BALANCED BUILD DONE")


if __name__ == "__main__":
    main()
