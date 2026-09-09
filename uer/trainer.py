import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from uer.model_loader import load_model
from uer.model_saver import save_model
from uer.model_builder import build_model
from uer.utils.optimizers import *
from uer.utils import *
from uer.utils.vocab import Vocab
from uer.utils.seed import set_seed


def train_and_validate(args):
    set_seed(args.seed)

    # Load vocabulary.
    if args.spm_model_path:
        try:
            import sentencepiece as spm
        except ImportError:
            raise ImportError("You need to install SentencePiece to use XLNetTokenizer: https://github.com/google/sentencepiece"
                              "pip install sentencepiece")
        sp_model = spm.SentencePieceProcessor()
        sp_model.Load(args.spm_model_path)
        args.vocab = {sp_model.IdToPiece(i): i for i
                      in range(sp_model.GetPieceSize())}
        args.tokenizer = str2tokenizer[args.tokenizer](args)
        if args.target == "seq2seq":
            tgt_sp_model = spm.SentencePieceProcessor()
            tgt_sp_model.Load(args.tgt_spm_model_path)
            args.tgt_vocab = {tgt_sp_model.IdToPiece(i): i for i
                              in range(tgt_sp_model.GetPieceSize())}
    else:
        args.tokenizer = str2tokenizer[args.tokenizer](args)
        args.vocab = args.tokenizer.vocab
        if args.target == "seq2seq":
            tgt_vocab = Vocab()
            tgt_vocab.load(args.tgt_vocab_path)
            args.tgt_vocab = tgt_vocab.w2i

    # Build model.
    model = build_model(args)

    # Load or initialize parameters.
    if args.pretrained_model_path is not None:
        # Initialize with pretrained model.
        model = load_model(model, args.pretrained_model_path) 
    else:
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if "gamma" not in n and "beta" not in n:
                p.data.normal_(0, 0.02)

    if args.dist_train:
        # Multiprocessing distributed mode.
        mp.spawn(worker, nprocs=args.ranks_num, args=(args.gpu_ranks, args, model), daemon=False)
    elif args.single_gpu:
        # Single GPU mode.
        worker(args.gpu_id, None, args, model)
    else:
        # CPU mode.
        worker(None, None, args, model)


class Trainer(object):
    def __init__(self, args):
        self.current_step = 1
        self.total_steps = args.total_steps
        self.accumulation_steps = args.accumulation_steps
        self.report_steps = args.report_steps
        self.save_checkpoint_steps = args.save_checkpoint_steps

        self.output_model_path = args.output_model_path

        self.start_time = time.time()
        self.total_loss = 0.0
        
        self.dist_train = args.dist_train
        self.batch_size = args.batch_size
        self.world_size = args.world_size

    def forward_propagation(self, batch, model):

        raise NotImplementedError

    def report_and_reset_stats(self):

        raise NotImplementedError

    def train(self, args, gpu_id, rank, loader, model, optimizer, scheduler):
        model.train()
        loader_iter = iter(loader)
        while True:
            if self.current_step == self.total_steps + 1:
                break
            batch = list(next(loader_iter))
            if len(batch) == 6 and getattr(args, "dual_stream_pretrain", False):
                self.seq_length = batch[0].size(1) + batch[1].size(1)
            else:
                self.seq_length = batch[0].size(1)
            if gpu_id is not None:
                for i in range(len(batch)):
                    batch[i] = batch[i].cuda(gpu_id)

            loss = self.forward_propagation(batch, model)

            if args.fp16:
                with args.amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if self.current_step % self.accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            if self.current_step % self.report_steps == 0 and \
                    (not self.dist_train or (self.dist_train and rank == 0)):
                self.report_and_reset_stats()
                self.start_time = time.time()

            if self.current_step % self.save_checkpoint_steps == 0 and \
                    (not self.dist_train or (self.dist_train and rank == 0)):
                save_model(model, self.output_model_path + "-" + str(self.current_step))

            self.current_step += 1


class MlmTrainer(Trainer):
    def __init__(self, args):
        super(MlmTrainer, self).__init__(args)
        self.total_correct = 0.0
        self.total_denominator = 0.0

    def forward_propagation(self, batch, model):
        src, tgt, seg = batch
        loss_info = model(src, tgt, seg)
        loss, correct, denominator = loss_info
        self.total_loss += loss.item()
        self.total_correct += correct.item()
        self.total_denominator += denominator.item()
        loss = loss / self.accumulation_steps
        return loss

    def report_and_reset_stats(self):
        done_tokens = self.batch_size * self.seq_length * self.report_steps
        if self.dist_train:
            done_tokens *= self.world_size
        print("| {:8d}/{:8d} steps"
              "| {:8.2f} tokens/s"
              "| loss {:7.2f}"
              "| acc: {:3.3f}".format(
            self.current_step,
            self.total_steps,
            done_tokens / (time.time() - self.start_time),
            self.total_loss / self.report_steps,
            self.total_correct / self.total_denominator))

        self.total_loss = 0.0
        self.total_correct = 0.0
        self.total_denominator = 0.0
        

class BertTrainer(Trainer):
    def __init__(self, args):
        super(BertTrainer, self).__init__(args)
        self.args = args
        self.total_loss_sp = 0.0
        self.total_correct_sp = 0.0
        self.total_instances = 0.0

        self.total_loss_mlm = 0.0
        self.total_correct_mlm = 0.0
        self.total_denominator = 0.0
        self.dual_stream_pretrain = getattr(args, "dual_stream_pretrain", False)
        self.cl_mode = getattr(args, "cl_loss", "none")
        self.lambda_mlm = getattr(args, "lambda_mlm", 1.0)
        self.lambda_cl = getattr(args, "lambda_cl", 1.0)

    def forward_propagation(self, batch, model):
        if self.dual_stream_pretrain:
            if len(batch) == 7:
                src_f, src_p, tgt_mlm_f, tgt_mlm_p, seg_f, seg_p, labels = batch
                tgt = (tgt_mlm_f, tgt_mlm_p, labels)
            else:
                src_f, src_p, tgt_mlm_f, tgt_mlm_p, seg_f, seg_p = batch
                tgt = (tgt_mlm_f, tgt_mlm_p)
            loss_info = model((src_f, src_p), tgt, (seg_f, seg_p))
            bs = src_f.size(0)
        else:
            src, tgt_mlm, tgt_sp, seg = batch
            loss_info = model(src, (tgt_mlm, tgt_sp), seg)
            bs = src.size(0)
        loss_mlm, loss_sp, correct_mlm, correct_sp, denominator = loss_info
        if self.cl_mode == "cl_only":
            loss = self.lambda_cl * loss_sp
        elif self.cl_mode == "mlm_cl":
            loss = self.lambda_mlm * loss_mlm + self.lambda_cl * loss_sp
        elif getattr(self.args, "dual_mlm_only", False):
            loss = loss_mlm
        else:
            loss = loss_mlm / 10 + loss_sp
        self.total_loss += loss.item()
        self.total_loss_mlm += loss_mlm.item()
        self.total_loss_sp += loss_sp.item()
        self.total_correct_mlm += correct_mlm.item()
        self.total_correct_sp += correct_sp.item()
        self.total_denominator += denominator.item()
        self.total_instances += bs
        loss = loss / self.accumulation_steps

        return loss

    def report_and_reset_stats(self):
        done_tokens = self.batch_size * self.seq_length * self.report_steps
        if self.dist_train:
            done_tokens *= self.world_size

        if self.cl_mode == "mlm_cl":
            print("| {:8d}/{:8d} steps"
                  "| {:8.2f} tokens/s"
                  "| loss {:7.2f}"
                  "| loss_mlm: {:3.3f}"
                  "| loss_cl: {:3.3f}"
                  "| acc_mlm: {:3.3f}".format(
                self.current_step,
                self.total_steps,
                done_tokens / (time.time() - self.start_time),
                self.total_loss / self.report_steps,
                self.total_loss_mlm / self.report_steps,
                self.total_loss_sp / self.report_steps,
                self.total_correct_mlm / self.total_denominator))
        elif self.cl_mode == "cl_only":
            print("| {:8d}/{:8d} steps"
                  "| {:8.2f} tokens/s"
                  "| loss {:7.2f}"
                  "| loss_cl: {:3.3f}".format(
                self.current_step,
                self.total_steps,
                done_tokens / (time.time() - self.start_time),
                self.total_loss / self.report_steps,
                self.total_loss_sp / self.report_steps))
        elif getattr(self.args, "dual_mlm_only", False):
            print("| {:8d}/{:8d} steps"
                  "| {:8.2f} tokens/s"
                  "| loss {:7.2f}"
                  "| loss_mlm: {:3.3f}"
                  "| acc_mlm: {:3.3f}".format(
                self.current_step,
                self.total_steps,
                done_tokens / (time.time() - self.start_time),
                self.total_loss / self.report_steps,
                self.total_loss_mlm / self.report_steps,
                self.total_correct_mlm / self.total_denominator))
        else:
            print("| {:8d}/{:8d} steps"
                  "| {:8.2f} tokens/s"
                  "| loss {:7.2f}"
                  "| loss_mlm: {:3.3f}"
                  "| loss_sp: {:3.3f}"
                  "| acc_mlm: {:3.3f}"
                  "| acc_sp: {:3.3f}".format(
                self.current_step,
                self.total_steps,
                done_tokens / (time.time() - self.start_time),
                self.total_loss / self.report_steps,
                self.total_loss_mlm / self.report_steps,
                self.total_loss_sp / self.report_steps,
                self.total_correct_mlm / self.total_denominator,
                self.total_correct_sp / self.total_instances))

        self.total_loss, self.total_loss_mlm, self.total_loss_sp = 0.0, 0.0, 0.0
        self.total_correct_mlm, self.total_denominator = 0.0, 0.0
        self.total_correct_sp, self.total_instances = 0.0, 0.0


class AlbertTrainer(BertTrainer):
    pass


class LmTrainer(MlmTrainer):
    pass


class BilmTrainer(Trainer):
    def __init__(self, args):
        super(BilmTrainer, self).__init__(args)
        self.total_loss_forward, self.total_loss_backward = 0.0, 0.0
        self.total_correct_forward, self.total_correct_backward = 0.0, 0.0
        self.total_denominator = 0.0

    def forward_propagation(self, batch, model):
        src, tgt_forward, tgt_backward, seg = batch
        loss_info = model(src, (tgt_forward, tgt_backward), seg)
        loss_forward, loss_backward, correct_forward, correct_backward, denominator = loss_info
        loss = loss_forward + loss_backward
        self.total_loss += loss.item()
        self.total_loss_forward += loss_forward.item()
        self.total_loss_backward += loss_backward.item()
        self.total_correct_forward += correct_forward.item()
        self.total_correct_backward += correct_backward.item()
        self.total_denominator += denominator.item()
        loss = loss / self.accumulation_steps
        return loss

    def report_and_reset_stats(self):
        done_tokens = self.batch_size * self.seq_length * self.report_steps
        if self.dist_train:
            done_tokens *= self.world_size
        print("| {:8d}/{:8d} steps"
                  "| {:8.2f} tokens/s"
                  "| loss {:7.2f}"
                  "| loss_forward {:3.3f}"
                  "| loss_backward {:3.3f}"
                  "| acc_forward: {:3.3f}"
                  "| acc_backward: {:3.3f}".format(
                    self.current_step,
                    self.total_steps, 
                    done_tokens / (time.time() - self.start_time), 
                    self.total_loss / self.report_steps,
                    self.total_loss_forward / self.report_steps,
                    self.total_loss_backward / self.report_steps,
                    self.total_correct_forward / self.total_denominator,
                    self.total_correct_backward / self.total_denominator))

        self.total_loss, self.total_loss_forward, self.total_loss_backward = 0.0, 0.0, 0.0
        self.total_correct_forward, self.total_correct_backward, self.total_denominator = 0.0, 0.0, 0.0


class ClsTrainer(Trainer):
    def __init__(self, args):
        super(ClsTrainer, self).__init__(args)
        self.total_correct = 0.0
        self.total_instances = 0.0

    def forward_propagation(self, batch, model):
        src, tgt, seg = batch
        loss_info = model(src, tgt, seg)
        loss, correct = loss_info
        self.total_loss += loss.item()
        self.total_correct += correct.item()
        self.total_instances += src.size(0)
        loss = loss / self.accumulation_steps
        return loss

    def report_and_reset_stats(self):
        done_tokens = self.batch_size * self.seq_length * self.report_steps
        if self.dist_train:
            done_tokens *= self.world_size
        print("| {:8d}/{:8d} steps"
              "| {:8.2f} tokens/s"
              "| loss {:7.2f}"
              "| acc: {:3.3f}".format(
            self.current_step,
            self.total_steps,
            done_tokens / (time.time() - self.start_time),
            self.total_loss / self.report_steps,
            self.total_correct / self.total_instances))

        self.total_loss = 0.0
        self.total_correct = 0.0
        self.total_instances = 0.0


class Seq2seqTrainer(Trainer):
    def __init__(self, args):
        super(Seq2seqTrainer, self).__init__(args)
        self.total_correct = 0.0
        self.total_denominator = 0.0

    def forward_propagation(self, batch, model):
        src, tgt_in, tgt_out, seg = batch
        loss_info = model(src, (tgt_in, tgt_out, src), seg)
        loss, correct, denominator = loss_info
        self.total_loss += loss.item()
        self.total_correct += correct.item()
        self.total_denominator += denominator.item()

        loss = loss / self.accumulation_steps

        return loss

    def report_and_reset_stats(self):
        done_tokens = self.batch_size * self.seq_length * self.report_steps
        if self.dist_train:
            done_tokens *= self.world_size

        print("| {:8d}/{:8d} steps"
              "| {:8.2f} tokens/s"
              "| loss {:7.2f}"
              "| acc: {:3.3f}".format(
            self.current_step,
            self.total_steps,
            done_tokens / (time.time() - self.start_time),
            self.total_loss / self.report_steps,
            self.total_correct / self.total_denominator))

        self.total_loss = 0.0
        self.total_correct = 0.0
        self.total_denominator = 0.0


class T5Trainer(Seq2seqTrainer):
    pass


class PrefixlmTrainer(MlmTrainer):
    pass


str2trainer = {"bert": BertTrainer, "mlm": MlmTrainer, "lm": LmTrainer,
               "albert": AlbertTrainer, "bilm": BilmTrainer, "cls": ClsTrainer,
               "seq2seq": Seq2seqTrainer, "t5": T5Trainer}

def worker(proc_id, gpu_ranks, args, model):
    """
    Args:
        proc_id: The id of GPU for single GPU mode;
                 The id of process (and GPU) for multiprocessing distributed mode.
        gpu_ranks: List of ranks of each process.
    """
    set_seed(args.seed)

    if args.dist_train:
        rank = gpu_ranks[proc_id]
        gpu_id = proc_id
    elif args.single_gpu:
        rank = None
        gpu_id = proc_id
    else:
        rank = None
        gpu_id = None

    if getattr(args, "dual_stream_pretrain", False):
        loader_cls = DualStreamBertDataLoader
        proc_id = rank if args.dist_train else 0
        proc_num = args.world_size if args.dist_train else 1
        train_loader = loader_cls(args, args.dataset_path, args.batch_size, proc_id, proc_num, True)
    elif args.dist_train:
        train_loader = str2dataloader[args.target](args, args.dataset_path, args.batch_size, rank, args.world_size, True)
    else:
        train_loader = str2dataloader[args.target](args, args.dataset_path, args.batch_size, 0, 1, True)

    if gpu_id is not None:
        torch.cuda.set_device(gpu_id)
        model.cuda(gpu_id)

    # Build optimizer.
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "gamma", "beta"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay_rate": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay_rate": 0.0}
    ]
    if args.optimizer in ["adamw"]:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    else:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate,
                                                  scale_parameter=False, relative_step=False)
    if args.scheduler in ["constant"]:
        scheduler = str2scheduler[args.scheduler](optimizer)
    elif args.scheduler in ["constant_with_warmup"]:
        scheduler = str2scheduler[args.scheduler](optimizer, args.total_steps*args.warmup)
    else:
        scheduler = str2scheduler[args.scheduler](optimizer, args.total_steps*args.warmup, args.total_steps)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        args.amp = amp

    if args.dist_train:
        # Initialize multiprocessing distributed training environment.
        dist.init_process_group(backend=args.backend,
                                init_method=args.master_ip,
                                world_size=args.world_size,
                                rank=rank)
        model = DistributedDataParallel(model, device_ids=[gpu_id], find_unused_parameters=True)
        print("Worker %d is training ... " % rank)
    else:
        print("Worker is training ...")

    trainer = str2trainer[args.target](args)
    trainer.train(args, gpu_id, rank, train_loader, model, optimizer, scheduler)
