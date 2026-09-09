"""
This script provides an exmaple to wrap UER-py for classification.
"""
import random
import argparse
import torch
import torch.nn as nn
from uer.layers import *
from uer.encoders import *
from uer.utils.vocab import Vocab
from uer.utils.constants import *
from uer.utils import *
from uer.utils.optimizers import *
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_saver import save_model
from uer.opts import finetune_opts
import tqdm
import numpy as np

class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.dual_stream = args.encoder == "dual_stream_cross"
        if self.dual_stream:
            self.embedding_flow = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
            self.embedding_payload = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        else:
            self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)
        self.labels_num = args.labels_num
        self.pooling = args.pooling
        self.soft_targets = args.soft_targets
        self.soft_alpha = args.soft_alpha
        self.dual_aux_loss_weight = getattr(args, "dual_aux_loss_weight", 0.2)
        if self.dual_stream:
            self.output_layer_1 = nn.Linear(2 * args.hidden_size, args.hidden_size)
            self.flow_head_1 = nn.Linear(args.hidden_size, args.hidden_size)
            self.flow_head_2 = nn.Linear(args.hidden_size, self.labels_num)
            self.payload_head_1 = nn.Linear(args.hidden_size, args.hidden_size)
            self.payload_head_2 = nn.Linear(args.hidden_size, self.labels_num)
            # Learnable fusion weight alpha in [0, 1] via sigmoid.
            self.dual_alpha_logit = nn.Parameter(torch.zeros(1))
        else:
            self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, self.labels_num)

    def _pool_hidden(self, hidden, seg):
        if self.pooling == "mean":
            return torch.mean(hidden, dim=1)
        if self.pooling == "max":
            return torch.max(hidden, dim=1)[0]
        if self.pooling == "last":
            return hidden[:, -1, :]
        return hidden[:, 0, :]

    def forward(
        self,
        src,
        tgt,
        seg,
        soft_tgt=None,
        src_payload=None,
        seg_payload=None,
        dual_logits_mode="fused",
        payload_aux_head=None,
    ):
        """
        Args:
            src: [batch_size x seq_length] (flow ids when dual_stream)
            tgt: [batch_size]
            seg: [batch_size x seq_length]
            src_payload, seg_payload: optional second stream for dual_stream_cross
            dual_logits_mode: dual_stream only — "fused" (default), "flow_only",
                "payload_only", "auto_pad_flow" (per row: if all flow token ids after [CLS]
                are PAD_ID, use payload head only; else fused).
            payload_aux_head: optional nn.Module(vec_p) -> logits; when dual_logits_mode is
                "payload_only" and this is set, used instead of the frozen payload_head (inference).
        """
        if self.dual_stream:
            emb_f = self.embedding_flow(src, seg)
            emb_p = self.embedding_payload(src_payload, seg_payload)
            out_f, out_p = self.encoder(emb_f, emb_p, seg, seg_payload)
            # Plan-A: use [CLS] from each stream for stream-specific heads.
            vec_f = out_f[:, 0, :]
            vec_p = out_p[:, 0, :]
            logits_f = self.flow_head_2(torch.tanh(self.flow_head_1(vec_f)))
            logits_p = self.payload_head_2(torch.tanh(self.payload_head_1(vec_p)))
            alpha = torch.sigmoid(self.dual_alpha_logit)
            logits_fused = alpha * logits_f + (1.0 - alpha) * logits_p
            if dual_logits_mode == "flow_only":
                logits = logits_f
            elif dual_logits_mode == "payload_only":
                if payload_aux_head is not None:
                    logits = payload_aux_head(vec_p)
                else:
                    logits = logits_p
            elif dual_logits_mode == "auto_pad_flow":
                only_p = (src[:, 1:] == PAD_ID).all(dim=1, keepdim=True)
                logits_p_branch = (
                    payload_aux_head(vec_p) if payload_aux_head is not None else logits_p
                )
                logits = torch.where(only_p.expand_as(logits_p), logits_p_branch, logits_fused)
            else:
                logits = logits_fused
        else:
            emb = self.embedding(src, seg)
            hidden = self.encoder(emb, seg)
            vec = self._pool_hidden(hidden, seg)
            output = torch.tanh(self.output_layer_1(vec))
            logits = self.output_layer_2(output)
        if tgt is not None:
            if self.dual_stream:
                loss_final = nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
                loss_flow = nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits_f), tgt.view(-1))
                loss_payload = nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits_p), tgt.view(-1))
                loss = loss_final + self.dual_aux_loss_weight * (loss_flow + loss_payload)
            elif self.soft_targets and soft_tgt is not None:
                loss = self.soft_alpha * nn.MSELoss()(logits, soft_tgt) + \
                       (1 - self.soft_alpha) * nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
            else:
                loss = nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
            return loss, logits
        else:
            return None, logits
            #return temp_output, logits


def count_labels_num(path):
    labels_set, columns = set(), {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.strip().split("\t")):
                    columns[column_name] = i
                continue
            line = line.strip().split("\t")
            label = int(line[columns["label"]])
            labels_set.add(label)
    return len(labels_set)


def load_or_initialize_parameters(args, model):
    if args.pretrained_model_path is not None:
        # Initialize with pretrained model. Use CPU when CUDA is unavailable (avoids deserialize error).
        map_location = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        state = torch.load(args.pretrained_model_path, map_location=map_location)
        model.load_state_dict(state, strict=False)
    else:
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if "gamma" not in n and "beta" not in n:
                p.data.normal_(0, 0.02)


def build_optimizer(args, model):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    if args.optimizer in ["adamw"]:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    else:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate,
                                                  scale_parameter=False, relative_step=False)
    if args.scheduler in ["constant"]:
        scheduler = str2scheduler[args.scheduler](optimizer)
    elif args.scheduler in ["constant_with_warmup"]:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps*args.warmup)
    else:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps*args.warmup, args.train_steps)
    return optimizer, scheduler


def batch_loader(batch_size, src, tgt, seg, soft_tgt=None):
    instances_num = src.size()[0]
    for i in range(instances_num // batch_size):
        src_batch = src[i * batch_size : (i + 1) * batch_size, :]
        tgt_batch = tgt[i * batch_size : (i + 1) * batch_size]
        seg_batch = seg[i * batch_size : (i + 1) * batch_size, :]
        if soft_tgt is not None:
            soft_tgt_batch = soft_tgt[i * batch_size : (i + 1) * batch_size, :]
            yield src_batch, tgt_batch, seg_batch, soft_tgt_batch
        else:
            yield src_batch, tgt_batch, seg_batch, None
    if instances_num > instances_num // batch_size * batch_size:
        src_batch = src[instances_num // batch_size * batch_size :, :]
        tgt_batch = tgt[instances_num // batch_size * batch_size :]
        seg_batch = seg[instances_num // batch_size * batch_size :, :]
        if soft_tgt is not None:
            soft_tgt_batch = soft_tgt[instances_num // batch_size * batch_size :, :]
            yield src_batch, tgt_batch, seg_batch, soft_tgt_batch
        else:
            yield src_batch, tgt_batch, seg_batch, None


def batch_loader_dual(batch_size, src_flow, src_payload, tgt, seg_flow, seg_payload, soft_tgt=None):
    instances_num = src_flow.size()[0]
    for i in range(instances_num // batch_size):
        sf = src_flow[i * batch_size : (i + 1) * batch_size, :]
        sp = src_payload[i * batch_size : (i + 1) * batch_size, :]
        tb = tgt[i * batch_size : (i + 1) * batch_size]
        segf = seg_flow[i * batch_size : (i + 1) * batch_size, :]
        segp = seg_payload[i * batch_size : (i + 1) * batch_size, :]
        st = soft_tgt[i * batch_size : (i + 1) * batch_size, :] if soft_tgt is not None else None
        yield sf, sp, tb, segf, segp, st
    rem = instances_num % batch_size
    if rem != 0:
        base = instances_num // batch_size * batch_size
        sf = src_flow[base:, :]
        sp = src_payload[base:, :]
        tb = tgt[base:]
        segf = seg_flow[base:, :]
        segp = seg_payload[base:, :]
        st = soft_tgt[base:, :] if soft_tgt is not None else None
        yield sf, sp, tb, segf, segp, st


def _pad_stream(ids, seg, max_len, pad_id):
    ids = ids[:max_len]
    seg = seg[:max_len]
    while len(ids) < max_len:
        ids.append(pad_id)
        seg.append(0)
    return ids, seg


def read_dataset(args, path):
    dataset, columns = [], {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.strip().split("\t")):
                    columns[column_name] = i
                continue
            line = line[:-1].split("\t")
            tgt = int(line[columns["label"]])
            if args.soft_targets and "logits" in columns.keys():
                soft_tgt = [float(value) for value in line[columns["logits"]].split(" ")]
            if "text_b" not in columns:  # Sentence classification.
                text_a = line[columns["text_a"]]
                if getattr(args, "encoder", "transformer") == "dual_stream_cross":
                    if "flow_tokens" not in columns:
                        raise ValueError("dual_stream_cross requires a `flow_tokens` column in the TSV.")
                    ft = line[columns["flow_tokens"]].strip()
                    if not ft:
                        raise ValueError("dual_stream_cross requires non-empty `flow_tokens` on each row.")
                    src_flow = args.tokenizer.convert_tokens_to_ids(
                        [CLS_TOKEN] + args.tokenizer.tokenize(ft)
                    )
                    src_payload = args.tokenizer.convert_tokens_to_ids(
                        [CLS_TOKEN] + args.tokenizer.tokenize(text_a)
                    )
                    seg_flow = [1] * len(src_flow)
                    seg_payload = [1] * len(src_payload)
                    src_flow, seg_flow = _pad_stream(
                        src_flow, seg_flow, args.flow_seq_length, PAD_ID
                    )
                    src_payload, seg_payload = _pad_stream(
                        src_payload, seg_payload, args.payload_seq_length, PAD_ID
                    )
                    if args.soft_targets and "logits" in columns.keys():
                        dataset.append((src_flow, src_payload, tgt, seg_flow, seg_payload, soft_tgt))
                    else:
                        dataset.append((src_flow, src_payload, tgt, seg_flow, seg_payload))
                    continue
                if "flow_tokens" in columns:
                    ft = line[columns["flow_tokens"]].strip()
                    if ft:
                        text_a = (ft + " " + text_a).strip()
                src = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text_a))
                seg = [1] * len(src)
            else:  # Sentence-pair classification.
                if getattr(args, "encoder", "transformer") == "dual_stream_cross":
                    raise ValueError("dual_stream_cross does not support sentence-pair TSV (text_b) in run_classifier.py.")
                text_a, text_b = line[columns["text_a"]], line[columns["text_b"]]
                src_a = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text_a) + [SEP_TOKEN])
                src_b = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(text_b) + [SEP_TOKEN])
                src = src_a + src_b
                seg = [1] * len(src_a) + [2] * len(src_b)

            if len(src) > args.seq_length:
                src = src[: args.seq_length]
                seg = seg[: args.seq_length]
            while len(src) < args.seq_length:
                src.append(0)
                seg.append(0)
            if args.soft_targets and "logits" in columns.keys():
                dataset.append((src, tgt, seg, soft_tgt))
            else:
                dataset.append((src, tgt, seg))

    return dataset


def train_model(
    args,
    model,
    optimizer,
    scheduler,
    src_batch,
    tgt_batch,
    seg_batch,
    soft_tgt_batch=None,
    src_payload_batch=None,
    seg_payload_batch=None,
):
    model.zero_grad()

    src_batch = src_batch.to(args.device)
    tgt_batch = tgt_batch.to(args.device)
    seg_batch = seg_batch.to(args.device)
    if soft_tgt_batch is not None:
        soft_tgt_batch = soft_tgt_batch.to(args.device)
    if getattr(args, "dual_stream", False):
        src_payload_batch = src_payload_batch.to(args.device)
        seg_payload_batch = seg_payload_batch.to(args.device)
        loss, _ = model(src_batch, tgt_batch, seg_batch, soft_tgt_batch, src_payload_batch, seg_payload_batch)
    else:
        loss, _ = model(src_batch, tgt_batch, seg_batch, soft_tgt_batch)
    if torch.cuda.device_count() > 1:
        loss = torch.mean(loss)

    if args.fp16:
        with args.amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()

    optimizer.step()
    scheduler.step()

    return loss


def evaluate(args, dataset, print_confusion_matrix=False):
    batch_size = args.batch_size
    if getattr(args, "dual_stream", False):
        src_flow = torch.LongTensor([sample[0] for sample in dataset])
        src_payload = torch.LongTensor([sample[1] for sample in dataset])
        tgt = torch.LongTensor([sample[2] for sample in dataset])
        seg_flow = torch.LongTensor([sample[3] for sample in dataset])
        seg_payload = torch.LongTensor([sample[4] for sample in dataset])
        loader = batch_loader_dual(batch_size, src_flow, src_payload, tgt, seg_flow, seg_payload, None)
    else:
        src = torch.LongTensor([sample[0] for sample in dataset])
        tgt = torch.LongTensor([sample[1] for sample in dataset])
        seg = torch.LongTensor([sample[2] for sample in dataset])
        loader = batch_loader(batch_size, src, tgt, seg)

    correct = 0
    # Confusion matrix.
    confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)

    args.model.eval()

    for batch in loader:
        if getattr(args, "dual_stream", False):
            src_batch, src_p_batch, tgt_batch, seg_batch, seg_p_batch, _ = batch
            src_batch = src_batch.to(args.device)
            src_p_batch = src_p_batch.to(args.device)
            tgt_batch = tgt_batch.to(args.device)
            seg_batch = seg_batch.to(args.device)
            seg_p_batch = seg_p_batch.to(args.device)
            with torch.no_grad():
                _, logits = args.model(src_batch, tgt_batch, seg_batch, None, src_p_batch, seg_p_batch)
        else:
            src_batch, tgt_batch, seg_batch, _ = batch
            src_batch = src_batch.to(args.device)
            tgt_batch = tgt_batch.to(args.device)
            seg_batch = seg_batch.to(args.device)
            with torch.no_grad():
                _, logits = args.model(src_batch, tgt_batch, seg_batch)
        pred = torch.argmax(nn.Softmax(dim=1)(logits), dim=1)
        gold = tgt_batch
        for j in range(pred.size()[0]):
            confusion[pred[j], gold[j]] += 1
        correct += torch.sum(pred == gold).item()

    if print_confusion_matrix:
        print("Confusion matrix:")
        print(confusion)
        cf_array = confusion.numpy()
        with open("/data2/lxj/pre-train/results/confusion_matrix",'w') as f:
            for cf_a in cf_array:
                f.write(str(cf_a)+'\n')
        print("Report precision, recall, and f1:")
        eps = 1e-9
        for i in range(confusion.size()[0]):
            p = confusion[i, i].item() / (confusion[i, :].sum().item() + eps)
            r = confusion[i, i].item() / (confusion[:, i].sum().item() + eps)
            if (p + r) == 0:
                f1 = 0
            else:
                f1 = 2 * p * r / (p + r)
            print("Label {}: {:.3f}, {:.3f}, {:.3f}".format(i, p, r, f1))

    print("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(correct / len(dataset), correct, len(dataset)))
    return correct / len(dataset), confusion


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    finetune_opts(parser)

    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type.")

    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="bert",
                        help="Specify the tokenizer."
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Space tokenizer segments sentences into words according to space."
                             )

    parser.add_argument("--soft_targets", action='store_true',
                        help="Train model with logits.")
    parser.add_argument("--soft_alpha", type=float, default=0.5,
                        help="Weight of the soft targets loss.")
    parser.add_argument("--dual_aux_loss_weight", type=float, default=0.2,
                        help="Weight of stream-specific auxiliary losses in dual-stream fusion.")
    
    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    args.dual_stream = args.encoder == "dual_stream_cross"
    if args.dual_stream:
        if args.flow_seq_length > args.max_seq_length or args.payload_seq_length > args.max_seq_length:
            raise ValueError(
                "flow_seq_length and payload_seq_length must be <= max_seq_length from config."
            )

    set_seed(args.seed)

    # Count the number of labels.
    args.labels_num = count_labels_num(args.train_path)

    # Build tokenizer.
    args.tokenizer = str2tokenizer[args.tokenizer](args)

    # Build classification model.
    model = Classifier(args)

    # Load or initialize parameters.
    load_or_initialize_parameters(args, model)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(args.device)

    # Training phase.
    trainset = read_dataset(args, args.train_path)
    random.shuffle(trainset)
    instances_num = len(trainset)
    batch_size = args.batch_size

    if args.dual_stream:
        src = torch.LongTensor([example[0] for example in trainset])
        src_payload = torch.LongTensor([example[1] for example in trainset])
        tgt = torch.LongTensor([example[2] for example in trainset])
        seg = torch.LongTensor([example[3] for example in trainset])
        seg_payload = torch.LongTensor([example[4] for example in trainset])
        if args.soft_targets:
            soft_tgt = torch.FloatTensor([example[5] for example in trainset])
        else:
            soft_tgt = None
    else:
        src = torch.LongTensor([example[0] for example in trainset])
        tgt = torch.LongTensor([example[1] for example in trainset])
        seg = torch.LongTensor([example[2] for example in trainset])
        if args.soft_targets:
            soft_tgt = torch.FloatTensor([example[3] for example in trainset])
        else:
            soft_tgt = None

    args.train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    print("Batch size: ", batch_size)
    print("The number of training instances:", instances_num)

    optimizer, scheduler = build_optimizer(args, model)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        args.amp = amp

    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    args.model = model

    total_loss, result, best_result = 0.0, 0.0, 0.0

    print("Start training.")

    for epoch in tqdm.tqdm(range(1, args.epochs_num + 1)):
        model.train()
        if args.dual_stream:
            batch_iter = batch_loader_dual(batch_size, src, src_payload, tgt, seg, seg_payload, soft_tgt)
        else:
            batch_iter = batch_loader(batch_size, src, tgt, seg, soft_tgt)
        for i, batch in enumerate(batch_iter):
            if args.dual_stream:
                src_b, src_pb, tgt_b, seg_b, seg_pb, st_b = batch
                loss = train_model(
                    args, model, optimizer, scheduler, src_b, tgt_b, seg_b, st_b, src_pb, seg_pb
                )
            else:
                src_batch, tgt_batch, seg_batch, soft_tgt_batch = batch
                loss = train_model(
                    args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch
                )
            total_loss += loss.item()
            if (i + 1) % args.report_steps == 0:
                print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i + 1, total_loss / args.report_steps))
                total_loss = 0.0

        result = evaluate(args, read_dataset(args, args.dev_path))
        if result[0] > best_result:
            best_result = result[0]
            save_model(model, args.output_model_path)

    # Evaluation phase.
    if args.test_path is not None:
        print("Test set evaluation.")
        _map = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        _state = torch.load(args.output_model_path, map_location=_map)
        if torch.cuda.device_count() > 1:
            model.module.load_state_dict(_state)
        else:
            model.load_state_dict(_state)
        evaluate(args, read_dataset(args, args.test_path), True)


if __name__ == "__main__":
    main()
