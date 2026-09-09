import torch
import torch.nn as nn
from uer.layers.transformer import TransformerLayer
from uer.layers.layer_norm import LayerNorm, T5LayerNorm
from uer.layers.multi_headed_attn import MultiHeadedAttention


def _fully_visible_self_mask(seg):
    """seg: [B, L] -> mask [B, 1, L, L] for additive attention scores."""
    batch_size, seq_length = seg.size()
    m = (seg > 0).unsqueeze(1).repeat(1, seq_length, 1).unsqueeze(1).float()
    return (1.0 - m) * -10000.0


def _cross_attention_mask(seg_q, seg_k):
    """
    seg_q: [B, Lq], seg_k: [B, Lk]
    Mask keys where seg_k==0; broadcast to [B, 1, Lq, Lk].
    """
    batch_size, lk = seg_k.size()
    lq = seg_q.size(1)
    k_valid = (seg_k > 0).float().unsqueeze(1).unsqueeze(2)
    m = (1.0 - k_valid) * -10000.0
    return m.expand(batch_size, 1, lq, lk)


class DualStreamSelfPairLayer(nn.Module):
    """One step: flow stream self-attn+FFN and payload stream self-attn+FFN (independent weights)."""

    def __init__(self, args):
        super(DualStreamSelfPairLayer, self).__init__()
        self.flow_self = TransformerLayer(args)
        self.payload_self = TransformerLayer(args)

    def forward(self, flow, payload, mask_flow, mask_payload, position_bias=None):
        flow = self.flow_self(flow, mask_flow, position_bias=position_bias)
        payload = self.payload_self(payload, mask_payload, position_bias=position_bias)
        return flow, payload


class DualStreamCrossInteraction(nn.Module):
    """
    Bidirectional cross-attention only (no self-attn).
    Flow attends to payload, then payload attends to updated flow.
    """

    def __init__(self, args):
        super(DualStreamCrossInteraction, self).__init__()
        self.layernorm_positioning = args.layernorm_positioning
        if args.layernorm == "t5":
            ln_cls = T5LayerNorm
        else:
            ln_cls = LayerNorm

        if hasattr(args, "attention_head_size"):
            attention_head_size = args.attention_head_size
        else:
            attention_head_size = args.hidden_size // args.heads_num
        has_bias = bool(1 - args.remove_transformer_bias)
        with_scale = bool(1 - args.remove_attention_scale)

        self.cross_flow = MultiHeadedAttention(
            args.hidden_size, args.heads_num, attention_head_size, args.dropout,
            has_bias=has_bias, with_scale=with_scale,
        )
        self.cross_payload = MultiHeadedAttention(
            args.hidden_size, args.heads_num, attention_head_size, args.dropout,
            has_bias=has_bias, with_scale=with_scale,
        )
        self.dropout_cf = nn.Dropout(args.dropout)
        self.dropout_cp = nn.Dropout(args.dropout)
        self.layer_norm_cf = ln_cls(args.hidden_size)
        self.layer_norm_cp = ln_cls(args.hidden_size)

    def forward(self, flow, payload, mask_cross_fp, mask_cross_pf):
        if self.layernorm_positioning == "post":
            qf = self.layer_norm_cf(flow)
            delta_f = self.dropout_cf(self.cross_flow(payload, payload, qf, mask_cross_fp, None))
            flow = flow + delta_f
            qp = self.layer_norm_cp(payload)
            delta_p = self.dropout_cp(self.cross_payload(flow, flow, qp, mask_cross_pf, None))
            payload = payload + delta_p
        else:
            qf = self.layer_norm_cf(flow)
            delta_f = self.dropout_cf(self.cross_flow(payload, payload, qf, mask_cross_fp, None))
            flow = flow + delta_f
            qp = self.layer_norm_cp(payload)
            delta_p = self.dropout_cp(self.cross_payload(flow, flow, qp, mask_cross_pf, None))
            payload = payload + delta_p
        return flow, payload


class DualStreamCrossEncoder(nn.Module):
    """
    Dual-stream encoder with sparse cross-attention:
    repeat `dual_stream_blocks` times:
        run `dual_self_layers_per_block` DualStreamSelfPairLayer (4 self per stream by default),
        then one DualStreamCrossInteraction.

    Default: blocks=3, self_per_block=4 => 12 self layers per stream and 3 cross rounds
    (vs. previous 12 cross rounds).
    `args.layers_num` from BERT config is not used for depth here; use dual_* args instead.
    """

    def __init__(self, args):
        super(DualStreamCrossEncoder, self).__init__()
        if getattr(args, "relative_position_embedding", False):
            raise ValueError("dual_stream_cross encoder does not support relative_position_embedding.")

        self.mask = args.mask
        if self.mask != "fully_visible":
            raise ValueError("dual_stream_cross encoder currently requires --mask fully_visible.")

        if getattr(args, "parameter_sharing", False):
            raise ValueError("dual_stream_cross does not support --parameter_sharing; use dual_stream_blocks / dual_self_layers_per_block.")

        self.blocks_num = getattr(args, "dual_stream_blocks", 3)
        self.self_per_block = getattr(args, "dual_self_layers_per_block", 4)
        self.factorized_embedding_parameterization = args.factorized_embedding_parameterization
        self.layernorm_positioning = args.layernorm_positioning

        total_self = self.blocks_num * self.self_per_block
        self.self_pairs = nn.ModuleList([DualStreamSelfPairLayer(args) for _ in range(total_self)])
        self.cross_blocks = nn.ModuleList(
            [DualStreamCrossInteraction(args) for _ in range(self.blocks_num)]
        )

        if self.factorized_embedding_parameterization:
            self.linear_flow = nn.Linear(args.emb_size, args.hidden_size)
            self.linear_payload = nn.Linear(args.emb_size, args.hidden_size)

        if self.layernorm_positioning == "pre":
            if args.layernorm == "t5":
                self.layer_norm_flow = T5LayerNorm(args.hidden_size)
                self.layer_norm_payload = T5LayerNorm(args.hidden_size)
            else:
                self.layer_norm_flow = LayerNorm(args.hidden_size)
                self.layer_norm_payload = LayerNorm(args.hidden_size)

    def forward(self, emb_flow, emb_payload, seg_flow, seg_payload):
        """
        Args:
            emb_flow: [B, Lf, emb_size]
            emb_payload: [B, Lp, emb_size]
            seg_flow: [B, Lf]
            seg_payload: [B, Lp]
        Returns:
            hidden_flow: [B, Lf, hidden_size]
            hidden_payload: [B, Lp, hidden_size]
        """
        if self.factorized_embedding_parameterization:
            emb_flow = self.linear_flow(emb_flow)
            emb_payload = self.linear_payload(emb_payload)

        mask_flow = _fully_visible_self_mask(seg_flow)
        mask_payload = _fully_visible_self_mask(seg_payload)
        mask_cross_fp = _cross_attention_mask(seg_flow, seg_payload)
        mask_cross_pf = _cross_attention_mask(seg_payload, seg_flow)

        flow = emb_flow
        payload = emb_payload
        position_bias = None

        self_idx = 0
        for b in range(self.blocks_num):
            for _ in range(self.self_per_block):
                flow, payload = self.self_pairs[self_idx](
                    flow, payload, mask_flow, mask_payload, position_bias=position_bias
                )
                self_idx += 1
            flow, payload = self.cross_blocks[b](flow, payload, mask_cross_fp, mask_cross_pf)

        if self.layernorm_positioning == "pre":
            flow = self.layer_norm_flow(flow)
            payload = self.layer_norm_payload(payload)

        return flow, payload
