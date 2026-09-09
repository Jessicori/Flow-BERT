"""
Train a new classification head on payload-stream [CLS] only.

The dual-stream encoder, embeddings, flow/payload heads, and fusion alpha stay
frozen. Optional distillation uses the frozen model's fused logits
(alpha * flow_head + (1-alpha) * payload_head) as teacher targets.

Example (repo root):
python fine-tuning/train_payload_aux_head.py \\
  --load_model_path models/dual_finetuned.bin \\
  --config_path assets/bert/base_config.json \\
  --vocab_path models/encryptd_vocab_flow_tor.txt \\
  --tokenizer space \\
  --train_path corpora/ISCX/TSV/Tor_finetune_train.tsv \\
  --dev_path corpora/ISCX/TSV/Tor_finetune_valid.tsv \\
  --output_aux_head_path models/payload_aux_head.bin
"""
import argparse
import math
import os
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, uer_dir)

from uer.model_loader import load_model
from uer.opts import model_opts
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.utils import str2tokenizer

from run_classifier import Classifier, batch_loader_dual, count_labels_num, read_dataset


class PayloadAuxHead(nn.Module):
    """Same shape as payload_head: [CLS] -> hidden -> tanh -> logits."""

    def __init__(self, hidden_size: int, labels_num: int):
        super().__init__()
        self.lin1 = nn.Linear(hidden_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, labels_num)

    def forward(self, vec_p: torch.Tensor) -> torch.Tensor:
        return self.lin2(torch.tanh(self.lin1(vec_p)))


@torch.no_grad()
def frozen_dual_forward(model: Classifier, src_f, seg_f, src_p, seg_p):
    """Encoder + existing heads; returns payload [CLS] and fused teacher logits."""
    emb_f = model.embedding_flow(src_f, seg_f)
    emb_p = model.embedding_payload(src_p, seg_p)
    out_f, out_p = model.encoder(emb_f, emb_p, seg_f, seg_p)
    vec_f = out_f[:, 0, :]
    vec_p = out_p[:, 0, :]
    logits_f = model.flow_head_2(torch.tanh(model.flow_head_1(vec_f)))
    logits_p = model.payload_head_2(torch.tanh(model.payload_head_1(vec_p)))
    alpha = torch.sigmoid(model.dual_alpha_logit)
    teacher = alpha * logits_f + (1.0 - alpha) * logits_p
    return vec_p, teacher


def evaluate_acc(model: Classifier, aux: PayloadAuxHead, args, path: str, device) -> float:
    ds = read_dataset(args, path)
    if len(ds) == 0:
        return 0.0
    src = torch.LongTensor([x[0] for x in ds])
    src_p = torch.LongTensor([x[1] for x in ds])
    tgt = torch.LongTensor([x[2] for x in ds])
    seg_f = torch.LongTensor([x[3] for x in ds])
    seg_p = torch.LongTensor([x[4] for x in ds])
    correct, total = 0, 0
    model.eval()
    aux.eval()
    bs = args.batch_size
    for batch in batch_loader_dual(bs, src, src_p, tgt, seg_f, seg_p, None):
        sf, sp, tb, segf, segp, _ = batch
        sf, segf = sf.to(device), segf.to(device)
        sp, segp = sp.to(device), segp.to(device)
        tb = tb.to(device)
        with torch.no_grad():
            vec_p, _ = frozen_dual_forward(model, sf, segf, sp, segp)
            pred = aux(vec_p).argmax(dim=-1)
        correct += (pred == tb).sum().item()
        total += tb.size(0)
    return correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train a new payload-[CLS] head; backbone and old heads frozen.",
    )
    # Same model switches as run_classifier / infer (embeddings + encoder need these attributes).
    model_opts(parser)
    parser.add_argument(
        "--pooling",
        choices=["mean", "max", "first", "last"],
        default="first",
        help="Classifier pooling flag (match finetuning).",
    )
    parser.add_argument(
        "--dual_aux_loss_weight",
        type=float,
        default=0.2,
        help="On Classifier for parity with run_classifier (not used when tgt is None).",
    )
    parser.add_argument("--load_model_path", type=str, required=True, help="Finetuned dual-stream checkpoint.")
    parser.add_argument("--config_path", type=str, default="assets/bert/base_config.json", help="Model JSON config.")
    parser.add_argument("--vocab_path", type=str, required=True, help="Vocab path used in finetuning.")
    parser.add_argument("--spm_model_path", type=str, default=None, help="Optional SPM model path.")
    parser.add_argument("--train_path", type=str, required=True, help="Dual TSV with flow_tokens + text_a + label.")
    parser.add_argument(
        "--dev_path",
        type=str,
        default=None,
        help="Optional validation TSV (same columns as train; e.g. Tor_finetune_valid.tsv). Omit to skip dev selection.",
    )
    parser.add_argument("--output_aux_head_path", type=str, default="models/payload_aux_head.bin", help="Where to save the new head only.")
    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="space", help="Tokenizer (Tor: space).")
    parser.add_argument("--labels_num", type=int, default=None, help="If omitted, inferred from train_path.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs_num", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--flow_seq_length", type=int, default=32)
    parser.add_argument("--payload_seq_length", type=int, default=128)
    parser.add_argument("--distill_weight", type=float, default=0.5, help="Weight of KL to fused teacher (0 = CE only).")
    parser.add_argument("--distill_temperature", type=float, default=2.0, help="Temperature T for distillation.")
    parser.add_argument("--scheduler", type=str, default="constant", choices=["constant", "linear", "cosine"])
    parser.add_argument("--warmup", type=float, default=0.0, help="Warmup fraction for linear/cosine schedulers.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay for aux head only.")
    parser.set_defaults(encoder="dual_stream_cross", embedding="word_pos_seg")
    args = parser.parse_args()

    if args.encoder != "dual_stream_cross":
        raise ValueError("This script only supports dual_stream_cross.")

    args = load_hyperparam(args)
    args.dual_stream = args.encoder == "dual_stream_cross"
    if args.flow_seq_length > args.max_seq_length or args.payload_seq_length > args.max_seq_length:
        raise ValueError("flow_seq_length and payload_seq_length must be <= max_seq_length from config.")

    set_seed(args.seed)
    args.soft_targets = False
    # Classifier.__init__ always reads soft_alpha (unused when soft_targets is False).
    args.soft_alpha = getattr(args, "soft_alpha", 0.5)

    if args.labels_num is None:
        args.labels_num = count_labels_num(args.train_path)

    args.tokenizer = str2tokenizer[args.tokenizer](args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    backbone = Classifier(args)
    backbone = load_model(backbone, args.load_model_path)
    backbone = backbone.to(device)
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()

    aux = PayloadAuxHead(args.hidden_size, args.labels_num).to(device)

    trainset = read_dataset(args, args.train_path)
    random.shuffle(trainset)
    src = torch.LongTensor([e[0] for e in trainset])
    src_p = torch.LongTensor([e[1] for e in trainset])
    tgt = torch.LongTensor([e[2] for e in trainset])
    seg_f = torch.LongTensor([e[3] for e in trainset])
    seg_p = torch.LongTensor([e[4] for e in trainset])

    opt = torch.optim.AdamW(aux.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    steps_per_epoch = max(len(trainset) // args.batch_size + (1 if len(trainset) % args.batch_size else 0), 1)
    total_steps = max(args.epochs_num * steps_per_epoch, 1)

    if args.scheduler == "constant":
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    elif args.scheduler == "linear":
        warm = int(total_steps * args.warmup)

        def lr_fn(s):
            if s < warm:
                return float(s) / max(warm, 1)
            return max(0.0, 1.0 - (s - warm) / max(total_steps - warm, 1))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    else:

        def lr_fn(s):
            warm = int(total_steps * args.warmup)
            if s < warm:
                return float(s) / max(warm, 1)
            t = (s - warm) / max(total_steps - warm, 1)
            return 0.5 * (1.0 + math.cos(math.pi * t))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)

    ce_loss = nn.CrossEntropyLoss()
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    T = args.distill_temperature
    w_dist = args.distill_weight

    best_dev = -1.0
    best_state = None

    n_train = len(trainset)
    n_batches = (n_train + args.batch_size - 1) // args.batch_size
    print(
        f"Training: {n_train} samples, batch_size={args.batch_size}, ~{n_batches} batches/epoch, "
        f"{args.epochs_num} epochs. (Outer epoch bar only advances after each full epoch; "
        f"watch the inner batch bar.)"
    )

    global_step = 0
    for epoch in range(1, args.epochs_num + 1):
        aux.train()
        perm = torch.randperm(n_train)
        src, src_p, tgt, seg_f, seg_p = src[perm], src_p[perm], tgt[perm], seg_f[perm], seg_p[perm]

        it = batch_loader_dual(args.batch_size, src, src_p, tgt, seg_f, seg_p, None)
        batch_iter = tqdm.tqdm(
            it,
            desc=f"epoch {epoch}/{args.epochs_num}",
            total=n_batches,
            leave=True,
        )
        for batch in batch_iter:
            sf, sp, tb, segf, segp, _ = batch
            sf, segf = sf.to(device), segf.to(device)
            sp, segp = sp.to(device), segp.to(device)
            tb = tb.to(device)

            with torch.no_grad():
                vec_p, teacher = frozen_dual_forward(backbone, sf, segf, sp, segp)

            logits_s = aux(vec_p)
            loss = ce_loss(logits_s, tb)
            if w_dist > 0.0:
                log_p_s = F.log_softmax(logits_s / T, dim=-1)
                p_t = F.softmax(teacher / T, dim=-1)
                loss = (1.0 - w_dist) * loss + w_dist * (T * T) * kl_loss(log_p_s, p_t)

            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            global_step += 1
            batch_iter.set_postfix(loss=f"{loss.item():.4f}")

        if args.dev_path:
            dev_acc = evaluate_acc(backbone, aux, args, args.dev_path, device)
            print(f"epoch {epoch} dev_acc={dev_acc:.4f}")
            if dev_acc > best_dev:
                best_dev = dev_acc
                best_state = {k: v.cpu().clone() for k, v in aux.state_dict().items()}
        else:
            best_state = {k: v.cpu().clone() for k, v in aux.state_dict().items()}

    if best_state is not None:
        aux.load_state_dict(best_state)

    out = {
        "payload_aux_head": aux.state_dict(),
        "hidden_size": args.hidden_size,
        "labels_num": args.labels_num,
        "encoder": args.encoder,
        "embedding": args.embedding,
    }
    out_dir = os.path.dirname(os.path.abspath(args.output_aux_head_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(out, args.output_aux_head_path)
    print(f"Saved payload-only auxiliary head to {args.output_aux_head_path}")

    if args.dev_path:
        print(f"Best dev accuracy: {best_dev:.4f}")


if __name__ == "__main__":
    main()
