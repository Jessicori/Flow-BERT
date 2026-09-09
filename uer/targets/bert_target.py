import torch
import torch.nn as nn
from uer.layers.layer_norm import LayerNorm
from uer.targets.mlm_target import MlmTarget
from uer.utils import str2act


class BertTarget(MlmTarget):
    """
    BERT exploits masked language modeling (MLM)
    and next sentence prediction (NSP) for pretraining.
    """

    def __init__(self, args, vocab_size):
        super(BertTarget, self).__init__(args, vocab_size)
        # NSP.
        self.nsp_linear_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.nsp_linear_2 = nn.Linear(args.hidden_size, 2)

    def forward(self, memory_bank, tgt):
        """
        Args:
            memory_bank: [batch_size x seq_length x hidden_size]
            tgt: tuple with tgt_mlm [batch_size x seq_length] and tgt_nsp [batch_size]

        Returns:
            loss_mlm: Masked language model loss.
            loss_nsp: Next sentence prediction loss.
            correct_mlm: Number of words that are predicted correctly.
            correct_nsp: Number of sentences that are predicted correctly.
            denominator: Number of masked words.
        """

        # Masked language model (MLM).
        assert type(tgt) == tuple
        tgt_mlm, tgt_nsp = tgt[0], tgt[1]
        loss_mlm, correct_mlm, denominator = self.mlm(memory_bank, tgt_mlm)
        #loss_mlm, correct_mlm, denominator = 0.0, 0.0, 0.0

        # Next sentence prediction (NSP).
        output_nsp = torch.tanh(self.nsp_linear_1(memory_bank[:, 0, :]))
        output_nsp = self.nsp_linear_2(output_nsp)
        loss_nsp = self.criterion(self.softmax(output_nsp), tgt_nsp)
        correct_nsp = self.softmax(output_nsp).argmax(dim=-1).eq(tgt_nsp).sum()

        #return loss_nsp, correct_nsp
        #return loss_mlm, correct_mlm, denominator
        return loss_mlm, loss_nsp, correct_mlm, correct_nsp, denominator


class BertDualStreamSplitMlmTarget(nn.Module):
    """
    Dual-stream pretraining **without NSP**:
    - **Flow-MLM**: separate head on ``h_flow``; labels only on flow positions (payload input unmasked).
    - **Payload-MLM**: separate head on ``h_payload``; labels only on payload positions (flow input unmasked).

    Encoder cross-attention lets each stream use the other stream as context when predicting masked tokens.
    Returns the same 5-tuple shape as ``BertTarget`` for ``BertTrainer`` compatibility; NSP slots are zeros.
    """

    def __init__(self, args, vocab_size):
        super(BertDualStreamSplitMlmTarget, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = args.hidden_size
        self.emb_size = args.emb_size
        self.factorized_embedding_parameterization = args.factorized_embedding_parameterization
        self.act = str2act[args.hidden_act]
        self.softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.NLLLoss()
        self.cl_alpha = getattr(args, "cl_alpha", 0.5)
        self.cl_beta = getattr(args, "cl_beta", 0.5)
        self.cl_margin = getattr(args, "cl_margin", 1.0)

        def _make_mlm_tower():
            if self.factorized_embedding_parameterization:
                return (
                    nn.Linear(args.hidden_size, args.emb_size),
                    LayerNorm(args.emb_size),
                    nn.Linear(args.emb_size, self.vocab_size),
                )
            return (
                nn.Linear(args.hidden_size, args.hidden_size),
                LayerNorm(args.hidden_size),
                nn.Linear(args.hidden_size, self.vocab_size),
            )

        self.flow_mlm_linear_1, self.flow_layer_norm, self.flow_mlm_linear_2 = _make_mlm_tower()
        self.payload_mlm_linear_1, self.payload_layer_norm, self.payload_mlm_linear_2 = _make_mlm_tower()

    def _mlm_one_stream(self, memory_bank, tgt_mlm, linear1, layer_norm, linear2):
        """Masked positions: tgt_mlm > 0. Returns (loss_scalar, correct, denom)."""
        output_mlm = self.act(linear1(memory_bank))
        output_mlm = layer_norm(output_mlm)
        if self.factorized_embedding_parameterization:
            output_mlm = output_mlm.contiguous().view(-1, self.emb_size)
        else:
            output_mlm = output_mlm.contiguous().view(-1, self.hidden_size)
        tgt_flat = tgt_mlm.contiguous().view(-1)
        output_mlm = output_mlm[tgt_flat > 0, :]
        tgt_flat = tgt_flat[tgt_flat > 0]
        n = output_mlm.size(0)
        if n == 0:
            z = memory_bank.sum() * 0.0
            return z, torch.tensor(0.0, device=memory_bank.device), torch.tensor(1e-6, device=memory_bank.device)
        output_mlm = linear2(output_mlm)
        output_mlm = self.softmax(output_mlm)
        denominator = torch.tensor(float(n) + 1e-6, device=memory_bank.device)
        correct_mlm = torch.sum((output_mlm.argmax(dim=-1).eq(tgt_flat)).float())
        loss_mlm = self.criterion(output_mlm, tgt_flat)
        return loss_mlm, correct_mlm, denominator

    def forward(self, memory_bank, tgt):
        assert type(tgt) == tuple and len(tgt) in (2, 3)
        tgt_flow, tgt_payload = tgt[0], tgt[1]
        h_f, h_p = memory_bank
        loss_f, cor_f, den_f = self._mlm_one_stream(
            h_f, tgt_flow, self.flow_mlm_linear_1, self.flow_layer_norm, self.flow_mlm_linear_2
        )
        loss_p, cor_p, den_p = self._mlm_one_stream(
            h_p, tgt_payload, self.payload_mlm_linear_1, self.payload_layer_norm, self.payload_mlm_linear_2
        )
        den_tot = den_f + den_p
        if den_tot.item() < 1e-5:
            loss_mlm = loss_f + loss_p
        else:
            loss_mlm = (loss_f * den_f + loss_p * den_p) / den_tot
        correct_mlm = cor_f + cor_p
        denominator = den_tot
        if len(tgt) == 3:
            labels = tgt[2]
            loss_cl = self._contrastive_loss(h_f, h_p, labels)
        else:
            loss_cl = loss_mlm * 0.0
        zero = loss_mlm * 0.0
        return loss_mlm, loss_cl, correct_mlm, zero, denominator

    def _contrastive_loss(self, h_f, h_p, labels):
        """Distance-based cross-modal contrastive loss (Eq. 9 of Flow-BERT).

        h_f[:, 0] / h_p[:, 0] are the [CLS] outputs of the behavior and payload
        streams.  Within the batch:
          * i == j             -> strong positive (same raw flow, y++)
          * same class, i != j -> positive (y+)
          * different class    -> negative (y-)
        """
        eps = 1e-12
        hb = h_f[:, 0, :]
        hp = h_p[:, 0, :]
        hb = hb / hb.norm(dim=-1, keepdim=True).clamp_min(eps)
        hp = hp / hp.norm(dim=-1, keepdim=True).clamp_min(eps)

        diff = hb.unsqueeze(1) - hp.unsqueeze(0)      # [B, B, H]
        d = diff.norm(dim=-1)                          # [B, B]
        d2 = d * d
        b = hb.size(0)
        device = hb.device
        eye = torch.eye(b, dtype=torch.bool, device=device)
        same_class = labels.unsqueeze(1) == labels.unsqueeze(0)
        pos = same_class & ~eye
        neg = ~same_class

        loss = d2[eye].sum() / max(float(b), 1.0)
        n_pos = pos.sum().clamp_min(1)
        loss = loss + self.cl_alpha * d2[pos].sum() / n_pos.float()
        n_neg = neg.sum().clamp_min(1)
        hinge = torch.clamp(self.cl_margin - d, min=0.0) ** 2
        loss = loss + self.cl_beta * hinge[neg].sum() / n_neg.float()
        return loss
