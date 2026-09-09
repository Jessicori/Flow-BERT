#!/usr/bin/env python3
"""Inference driver for Flow-BERT classification models.

Loads a fine-tuned dual-stream (flow + payload) or single-stream checkpoint and
writes predictions for one TSV test file.  Three prediction modes are supported
for dual-stream models:

* ``fused``          -- combine the flow and payload [CLS] logits (default);
* ``flow_only``      -- use only the flow [CLS] logits;
* ``payload_only``   -- use only the payload [CLS] logits;
* ``auto_pad_flow``  -- choose per row: payload branch when the flow stream is
  empty (all flow ids after [CLS] are PAD), otherwise fused.

The prediction file always contains a ``label`` column and may additionally
contain ``logits`` / ``prob`` columns when the corresponding flags are set.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "fine-tuning"))

from uer.model_loader import load_model
from uer.opts import infer_opts
from uer.utils import str2tokenizer
from uer.utils.config import load_hyperparam
from uer.utils.constants import CLS_TOKEN, PAD_ID

from run_classifier import Classifier, _pad_stream
from train_payload_aux_head import PayloadAuxHead


Tensor = torch.Tensor
Sample = Tuple[List[int], ...]


def _tokenize_dual(args, flow_text: str, payload_text: str) -> Tuple[List[int], ...]:
    """Encode flow/payload text into padded id + segment sequences."""
    flow_ids = args.tokenizer.convert_tokens_to_ids(
        [CLS_TOKEN] + args.tokenizer.tokenize(flow_text)
    )
    payload_ids = args.tokenizer.convert_tokens_to_ids(
        [CLS_TOKEN] + args.tokenizer.tokenize(payload_text)
    )
    flow_seg = [1] * len(flow_ids)
    payload_seg = [1] * len(payload_ids)

    flow_ids, flow_seg = _pad_stream(flow_ids, flow_seg, args.flow_seq_length, PAD_ID)
    payload_ids, payload_seg = _pad_stream(
        payload_ids, payload_seg, args.payload_seq_length, PAD_ID
    )
    return flow_ids, flow_seg, payload_ids, payload_seg


def _tokenize_single(args, text: str, seg_id: int = 1) -> Tuple[List[int], List[int]]:
    ids = args.tokenizer.convert_tokens_to_ids(
        [CLS_TOKEN] + args.tokenizer.tokenize(text)
    )
    seg = [seg_id] * len(ids)
    if len(ids) > args.seq_length:
        ids = ids[: args.seq_length]
        seg = seg[: args.seq_length]
    while len(ids) < args.seq_length:
        ids.append(0)
        seg.append(0)
    return ids, seg


def _tokenize_pair(args, text_a: str, text_b: str) -> Tuple[List[int], List[int]]:
    """Encode a sentence pair with [SEP] boundaries and 1/2 segment ids."""
    from uer.utils.constants import SEP_TOKEN

    ids = args.tokenizer.convert_tokens_to_ids(
        [CLS_TOKEN] + args.tokenizer.tokenize(text_a) + [SEP_TOKEN]
    )
    seg = [1] * len(ids)
    ids += args.tokenizer.convert_tokens_to_ids(
        args.tokenizer.tokenize(text_b) + [SEP_TOKEN]
    )
    seg += [2] * (len(ids) - len(seg))
    if len(ids) > args.seq_length:
        ids = ids[: args.seq_length]
        seg = seg[: args.seq_length]
    while len(ids) < args.seq_length:
        ids.append(0)
        seg.append(0)
    return ids, seg


def read_samples(args, path: str) -> List[Sample]:
    """Parse a test TSV into model-ready tensors (dual or single stream)."""
    samples: List[Sample] = []
    with open(path, "r", encoding="utf-8") as handle:
        header = handle.readline().strip().split("\t")
        columns = {name: i for i, name in enumerate(header)}

        for raw in handle:
            row = raw.strip().split("\t")
            text_a = row[columns["text_a"]]

            if getattr(args, "encoder", "transformer") == "dual_stream_cross":
                if "flow_tokens" not in columns:
                    raise ValueError(
                        "dual_stream_cross inference requires `flow_tokens` in TSV."
                    )
                flow_text = row[columns["flow_tokens"]].strip()
                if not flow_text:
                    raise ValueError(
                        "dual_stream_cross requires non-empty `flow_tokens` per row."
                    )
                samples.append(_tokenize_dual(args, flow_text, text_a))
                continue

            if "flow_tokens" in columns:
                flow_text = row[columns["flow_tokens"]].strip()
                if flow_text:
                    text_a = f"{flow_text} {text_a}".strip()

            if "text_b" in columns:
                text_b = row[columns["text_b"]]
                samples.append(_tokenize_pair(args, text_a, text_b))
            else:
                samples.append(_tokenize_single(args, text_a))

    return samples


def _batches(batch_size: int, *columns: Tensor) -> Iterable[Tuple[Tensor, ...]]:
    """Yield fixed-size slices (plus a final partial slice) of the columns."""
    count = columns[0].size(0)
    for start in range(0, count, batch_size):
        yield tuple(col[start : start + batch_size] for col in columns)


def _decode_mode(args) -> str:
    if args.flow_only_logits and args.payload_only_logits:
        raise ValueError("Only one of --flow_only_logits and --payload_only_logits can be set.")
    if args.flow_only_logits:
        return "flow_only"
    if args.payload_only_logits:
        return "payload_only"
    return args.dual_infer_logits_mode


def _build_aux_head(args, device: torch.device):
    """Optionally load a separately trained payload head checkpoint."""
    if not args.payload_aux_head_path:
        return None
    if args.dual_infer_logits_mode not in ("payload_only", "auto_pad_flow"):
        print(
            "Warning: --payload_aux_head_path is ignored unless mode is payload_only or auto_pad_flow."
        )
        return None

    checkpoint = torch.load(args.payload_aux_head_path, map_location=device)
    if checkpoint.get("labels_num") != args.labels_num:
        raise ValueError(
            f"payload_aux_head labels_num={checkpoint.get('labels_num')} "
            f"!= args.labels_num={args.labels_num}"
        )
    if checkpoint.get("hidden_size") != args.hidden_size:
        raise ValueError(
            f"payload_aux_head hidden_size={checkpoint.get('hidden_size')} "
            f"!= args.hidden_size={args.hidden_size}"
        )
    head = PayloadAuxHead(args.hidden_size, args.labels_num).to(device)
    head.load_state_dict(checkpoint["payload_aux_head"])
    head.eval()
    if torch.cuda.device_count() > 1:
        print("Note: DataParallel disabled when using --payload_aux_head_path (single-GPU inference).")
    return head


def _forward(model, args, payload_aux, batch: Sequence[Tensor]) -> Tensor:
    if args.dual_stream:
        flow_src, flow_seg, payload_src, payload_seg = batch
        _, logits = model(
            flow_src.to(args.device),
            None,
            flow_seg.to(args.device),
            None,
            payload_src.to(args.device),
            payload_seg.to(args.device),
            dual_logits_mode=args.dual_infer_logits_mode,
            payload_aux_head=payload_aux,
        )
    else:
        src, seg = batch
        _, logits = model(src.to(args.device), None, seg.to(args.device))
    return logits


def _write_predictions(path: str, rows: Iterable[Tuple[int, List[float], List[float]]],
                       include_logits: bool, include_prob: bool) -> None:
    header = "label"
    if include_logits:
        header += "\tlogits"
    if include_prob:
        header += "\tprob"

    with open(path, "w", encoding="utf-8") as handle:
        handle.write(header + "\n")
        for label, logits, probs in rows:
            fields = [str(label)]
            if include_logits:
                fields.append(" ".join(str(v) for v in logits))
            if include_prob:
                fields.append(" ".join(str(v) for v in probs))
            handle.write("\t".join(fields) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run Flow-BERT inference over a TSV test file.",
    )
    infer_opts(parser)
    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type.")
    parser.add_argument("--labels_num", type=int, required=True,
                        help="Number of prediction labels.")
    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="bert",
                        help="Tokenizer: bert / char / space.")
    parser.add_argument("--output_logits", action="store_true", help="Write logits column.")
    parser.add_argument("--output_prob", action="store_true", help="Write probabilities column.")
    parser.add_argument("--dual_infer_logits_mode",
                        choices=["fused", "flow_only", "payload_only", "auto_pad_flow"],
                        default="fused",
                        help="Dual-stream only: fused / flow_only / payload_only / auto_pad_flow.")
    parser.add_argument("--flow_only_logits", action="store_true",
                        help="Shorthand for --dual_infer_logits_mode flow_only.")
    parser.add_argument("--payload_only_logits", action="store_true",
                        help="Shorthand for --dual_infer_logits_mode payload_only.")
    parser.add_argument("--payload_aux_head_path", type=str, default=None,
                        help="Optional separately trained payload head checkpoint.")
    args = parser.parse_args()

    args.dual_infer_logits_mode = _decode_mode(args)
    args = load_hyperparam(args)
    args.dual_stream = args.encoder == "dual_stream_cross"

    if args.dual_stream and (
        args.flow_seq_length > args.max_seq_length
        or args.payload_seq_length > args.max_seq_length
    ):
        raise ValueError(
            "flow_seq_length and payload_seq_length must be <= max_seq_length from config."
        )

    args.tokenizer = str2tokenizer[args.tokenizer](args)
    args.soft_targets, args.soft_alpha = False, False

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Classifier(args)
    model = load_model(model, args.load_model_path).to(args.device)

    payload_aux = _build_aux_head(args, args.device)
    if torch.cuda.device_count() > 1 and payload_aux is None:
        print(f"{torch.cuda.device_count()} GPUs are available. Let's use them.")
        model = torch.nn.DataParallel(model)

    samples = read_samples(args, args.test_path)
    if args.dual_stream:
        columns = [torch.LongTensor([s[i] for s in samples]) for i in range(4)]
    else:
        columns = [torch.LongTensor([s[0] for s in samples]),
                   torch.LongTensor([s[1] for s in samples])]

    print("The number of prediction instances: ", columns[0].size(0))
    model.eval()

    rows = []
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        for batch in _batches(args.batch_size, *columns):
            logits = _forward(model, args, payload_aux, batch)
            pred = torch.argmax(logits, dim=1).cpu().tolist()
            logits_list = logits.cpu().tolist()
            prob_list = softmax(logits).cpu().tolist()
            rows.extend(zip(pred, logits_list, prob_list))

    _write_predictions(args.prediction_path, rows, args.output_logits, args.output_prob)


if __name__ == "__main__":
    main()
