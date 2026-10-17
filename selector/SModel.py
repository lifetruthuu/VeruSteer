import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_units, num_heads, dropout_rate):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0

        self.linear_q = nn.Linear(hidden_size, num_units)
        self.linear_k = nn.Linear(hidden_size, num_units)
        self.linear_v = nn.Linear(hidden_size, num_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, queries, keys):
        """
        :param queries: A 3d tensor with shape of [N, T_q, C_q]
        :param keys: A 3d tensor with shape of [N, T_k, C_k]

        :return: A 3d tensor with shape of (N, T_q, C)

        """
        Q = self.linear_q(queries)  # (N, T_q, C)
        K = self.linear_k(keys)  # (N, T_k, C)
        V = self.linear_v(keys)  # (N, T_k, C)

        # Split and Concat
        split_size = self.hidden_size // self.num_heads
        Q_ = torch.cat(torch.split(Q, split_size, dim=2), dim=0)  # (h*N, T_q, C/h)
        K_ = torch.cat(torch.split(K, split_size, dim=2), dim=0)  # (h*N, T_k, C/h)
        V_ = torch.cat(torch.split(V, split_size, dim=2), dim=0)  # (h*N, T_k, C/h)

        # Multiplication
        matmul_output = torch.bmm(Q_, K_.transpose(1, 2)) / split_size ** 0.5  # (h*N, T_q, T_k)

        # Key Masking
        key_mask = torch.sign(torch.abs(keys.sum(dim=-1))).repeat(self.num_heads, 1)  # (h*N, T_k)
        key_mask_reshaped = key_mask.unsqueeze(1).repeat(1, queries.shape[1], 1)  # (h*N, T_q, T_k)
        key_paddings = torch.ones_like(matmul_output) * (-2 ** 32 + 1)
        matmul_output_m1 = torch.where(torch.eq(key_mask_reshaped, 0), key_paddings, matmul_output)  # (h*N, T_q, T_k)

        # Causality - Future Blinding
        diag_vals = torch.ones_like(matmul_output[0, :, :])  # (T_q, T_k)
        tril = torch.tril(diag_vals)  # (T_q, T_k)
        causality_mask = tril.unsqueeze(0).repeat(matmul_output.shape[0], 1, 1)  # (h*N, T_q, T_k)
        causality_paddings = torch.ones_like(causality_mask) * (-2 ** 32 + 1)
        matmul_output_m2 = torch.where(torch.eq(causality_mask, 0), causality_paddings,
                                       matmul_output_m1)  # (h*N, T_q, T_k)

        # Activation
        matmul_output_sm = self.softmax(matmul_output_m2)  # (h*N, T_q, T_k)

        # Query Masking
        query_mask = torch.sign(torch.abs(queries.sum(dim=-1))).repeat(self.num_heads, 1)  # (h*N, T_q)
        query_mask = query_mask.unsqueeze(-1).repeat(1, 1, keys.shape[1])  # (h*N, T_q, T_k)
        matmul_output_qm = matmul_output_sm * query_mask

        # Dropout
        matmul_output_dropout = self.dropout(matmul_output_qm)

        # Weighted Sum
        output_ws = torch.bmm(matmul_output_dropout, V_)  # ( h*N, T_q, C/h)

        # Restore Shape
        output = torch.cat(torch.split(output_ws, output_ws.shape[0] // self.num_heads, dim=0), dim=2)  # (N, T_q, C)

        # Residual Connection
        output_res = output + queries

        return output_res

def extract_axis_1(data, indices):
    res = []
    for i in range(data.shape[0]):
        res.append(data[i, indices[i], :])
    res = torch.stack(res, dim=0).unsqueeze(1)
    return res

class SModel(nn.Module):
    def __init__(self, hidden_size, fail_type_num, state_size, dropout, device, num_heads=1, padding_val=0, emb_path=None):
        super(SModel, self).__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.fail_type_num = int(fail_type_num)
        self.padding_val = padding_val
        self.dropout = nn.Dropout(dropout)
        self.device = device
        emb_path = str(Path(emb_path) if emb_path is not None else Path(__file__).resolve().parent / "embedding_table") + "/"
        code_embeddings = torch.from_numpy(np.load(emb_path + "code_embeddings.npy")).float()
        verus_outputs_embeddings = torch.from_numpy(np.load(emb_path + "verus_outputs_embeddings.npy")).float()
        fail_types_embeddings = torch.from_numpy(np.load(emb_path + "failure_types_embeddings.npy")).float()
        self.code_embeddings = nn.Embedding.from_pretrained(
            code_embeddings,
            freeze=True
        )
        self.verus_embeddings = nn.Embedding.from_pretrained(
            verus_outputs_embeddings,
            freeze=True
        )
        self.fail_types_embeddings = nn.Embedding.from_pretrained(
            fail_types_embeddings,
            freeze=False,
            padding_idx=0
        )
        self.positional_embeddings = nn.Embedding(
            num_embeddings=state_size,
            embedding_dim=hidden_size
        )
        self.emb_map = nn.Linear(2 * code_embeddings.shape[1], hidden_size)
        self.fail_map = nn.Linear(code_embeddings.shape[1], hidden_size)
        self.emb_dropout = nn.Dropout(dropout)
        self.ln_01 = nn.LayerNorm(hidden_size)
        self.ln_02 = nn.LayerNorm(hidden_size)
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.ln_3 = nn.LayerNorm(hidden_size)
        self.mh_attn = MultiHeadAttention(hidden_size, hidden_size, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size, hidden_size, dropout)


    def forward(self, pos_seq, pos_target, neg_seq, neg_target):
        pos_code_emb = self.code_embeddings(pos_seq['code_ids'])
        pos_verus_emb = self.verus_embeddings(pos_seq['verus_ids'])
        pos_fail_type_emb = self.fail_types_embeddings(pos_seq['fail_type_ids'])
        pos_len_seq = pos_seq['len_seq']

        neg_code_emb = self.code_embeddings(neg_seq['code_ids'])
        neg_verus_emb = self.verus_embeddings(neg_seq['verus_ids'])
        neg_fail_type_emb = self.fail_types_embeddings(neg_seq['fail_type_ids'])
        neg_len_seq = neg_seq['len_seq']

        pos_state_emb = torch.cat(([pos_code_emb, pos_verus_emb]), dim=-1)
        pos_state_emb = self.emb_map(pos_state_emb)
        pos_state_emb = self.ln_01(pos_state_emb)
        pos_fail_type_emb = self.fail_map(pos_fail_type_emb)
        pos_fail_type_emb = self.ln_02(pos_fail_type_emb)

        neg_state_emb = torch.cat(([neg_code_emb, neg_verus_emb]), dim=-1)
        neg_state_emb = self.emb_map(neg_state_emb)
        neg_state_emb = self.ln_01(neg_state_emb)
        neg_fail_type_emb = self.fail_map(neg_fail_type_emb)
        neg_fail_type_emb = self.ln_02(neg_fail_type_emb)

        pos_emb = self.positional_embeddings(torch.arange(self.state_size).to(self.device))
        pos_inputs_emb = pos_state_emb + pos_emb
        pos_inputs_emb += pos_fail_type_emb
        neg_inputs_emb = neg_state_emb + pos_emb
        neg_inputs_emb += neg_fail_type_emb

        seq_pos = self.emb_dropout(pos_inputs_emb)
        mask = torch.ne(pos_seq['code_ids'], self.padding_val).float().unsqueeze(-1).to(self.device)
        seq_pos *= mask
        seq_pos_normalized = self.ln_1(seq_pos)
        mh_attn_out_pos = self.mh_attn(seq_pos_normalized, seq_pos)
        ff_out_pos = self.feed_forward(self.ln_2(mh_attn_out_pos))
        ff_out_pos *= mask
        ff_out_pos = self.ln_3(ff_out_pos)
        last_state_hidden_pos = extract_axis_1(ff_out_pos, pos_len_seq - 1)
        state_hidden_pos = ff_out_pos * mask
        all_fail_type_emb = self.ln_02(self.fail_map(self.fail_types_embeddings.weight)).T.unsqueeze(0)
        logits = torch.matmul(state_hidden_pos, all_fail_type_emb)


        seq_neg = self.emb_dropout(neg_inputs_emb)
        mask = torch.ne(neg_seq['code_ids'], self.padding_val).float().unsqueeze(-1).to(self.device)
        seq_neg *= mask
        seq_neg_normalized = self.ln_1(seq_neg)
        mh_attn_out_neg = self.mh_attn(seq_neg_normalized, seq_neg)
        ff_out_neg = self.feed_forward(self.ln_2(mh_attn_out_neg))
        ff_out_neg *= mask
        ff_out_neg = self.ln_3(ff_out_neg)
        last_state_hidden_neg = extract_axis_1(ff_out_neg, neg_len_seq - 1)


        pos_target_emb = self.fail_types_embeddings(pos_target).squeeze(1)
        neg_target_emb = self.fail_types_embeddings(neg_target).squeeze(1)
        pos_target_emb = self.fail_map(pos_target_emb)
        neg_target_emb = self.fail_map(neg_target_emb)
        pos_target_emb = self.ln_02(pos_target_emb)
        neg_target_emb = self.ln_02(neg_target_emb)
        #print((last_state_hidden_pos.squeeze(1) * pos_target_emb).shape)
        pos_scores = torch.sum(last_state_hidden_pos.squeeze(1) * pos_target_emb, dim=-1)
        neg_scores = torch.sum(last_state_hidden_neg.squeeze(1) * neg_target_emb, dim=-1)

        return logits, pos_scores, neg_scores

    def forward_eval(self, code_ids, verus_ids, fail_type_ids, len_seq):
        code_emb = self.code_embeddings(code_ids)
        verus_emb = self.verus_embeddings(verus_ids)
        fail_type_emb = self.fail_types_embeddings(fail_type_ids)

        state_emb = torch.cat(([code_emb, verus_emb]), dim=-1)
        state_emb = self.emb_map(state_emb)
        state_emb = self.ln_01(state_emb)
        fail_type_emb = self.fail_map(fail_type_emb)
        fail_type_emb = self.ln_02(fail_type_emb)
        pos_emb = self.positional_embeddings(torch.arange(self.state_size).to(self.device))
        inputs_emb = state_emb + pos_emb
        inputs_emb += fail_type_emb

        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(code_ids, self.padding_val).float().unsqueeze(-1).to(self.device)
        seq *= mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        last_state_hidden = extract_axis_1(ff_out, len_seq - 1)
        last_state_hidden = last_state_hidden.squeeze(1)
        last_logits = torch.matmul(last_state_hidden, self.ln_02(self.fail_map(self.fail_types_embeddings.weight)).T)
        return last_logits, last_state_hidden

    def infer(self, code_emb, verus_emb, fail_type_emb, len_seq):

        state_emb = torch.cat(([code_emb, verus_emb]), dim=-1)
        state_emb = self.emb_map(state_emb)
        state_emb = self.ln_01(state_emb)
        pos_emb = self.positional_embeddings(torch.arange(self.state_size).to(self.device))
        fail_type_emb = self.ln_02(fail_type_emb)
        inputs_emb = state_emb + pos_emb
        inputs_emb += fail_type_emb

        seq = inputs_emb
        pad_len = self.state_size - len_seq
        padding = torch.full((pad_len,), 0, dtype=seq.dtype).to(self.device)
        mask = torch.cat([torch.ones(len_seq).to(self.device), padding], dim=0).unsqueeze(-1).to(self.device)
        seq *= mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        state_hidden = ff_out[:, len_seq - 1, :]
        logits = torch.matmul(state_hidden, self.ln_02(self.fail_map(self.fail_types_embeddings.weight)).T).squeeze()
        return logits
