import pandas as pd
import ast
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import sys
import random
from tqdm import tqdm
from torch.utils.data import DataLoader
from SModel import SModel
from functools import partial
import argparse
from pathlib import Path
import copy

ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description="VeruSteer trainer")

parser.add_argument("--data_dir", type=Path, default=ROOT / "data/processed")
parser.add_argument("--emb_dir", type=Path, default=ROOT / "selector/embedding_table")
parser.add_argument("--output_dir", type=Path, default=ROOT / "outputs/checkpoints")
parser.add_argument("--pipeline", type=str, choices=["autoverus", "verusage"], default="verusage")

parser.add_argument("--seed", type=int, default=2022)
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--l2", type=float, default=1e-5)
parser.add_argument("--bsize", type=int, default=256)
parser.add_argument("--max_epoch", type=int, default=50)
parser.add_argument("--patience", type=int, default=10)
parser.add_argument("--bpr_weight", type=float, default=0.2)
parser.add_argument("--mask_prob", type=float, default=0.1)

parser.add_argument("--hidden_size", type=int, default=64)
parser.add_argument("--max_seq_len", type=int, choices=range(1, 6), default=5)
parser.add_argument("--dropout", type=float, default=0.2)

parser.add_argument("--cuda", type=int, default=0)

args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)
checkpoint_path = args.output_dir / f"{args.pipeline}.pth"

if args.bsize < 1 or args.max_epoch < 1 or args.patience < 1:
    parser.error("bsize, max_epoch and patience must be positive")
if not 0 <= args.mask_prob < 1:
    parser.error("mask_prob must be in [0, 1)")
if not 0 <= args.dropout < 1:
    parser.error("dropout must be in [0, 1)")
if args.bpr_weight < 0:
    parser.error("bpr_weight must be nonnegative")

my_seed = args.seed
torch.manual_seed(my_seed)
torch.cuda.manual_seed_all(my_seed)
random.seed(my_seed)
np.random.seed(my_seed)

MAX_SEQ_LEN = args.max_seq_len

train_file = args.data_dir / "train.csv"
val_file = args.data_dir / f"val_{args.pipeline}.csv"
test_file = args.data_dir / f"test_{args.pipeline}.csv"


class SeqDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        class_id = row['class_id']
        pos_code_ids = row['pos_code_ids']
        pos_verus_ids = row['pos_verus_ids']
        pos_fail_type_ids = row['pos_fail_type_ids']
        pos_candidate_fail_type_ids = row['pos_candidate_fail_type_ids']
        pos_target_fail_type_ids = pos_fail_type_ids
        pos_fail_type_ids, pos_target = [0] + pos_fail_type_ids[:-1], pos_fail_type_ids[-1:]

        neg_code_ids = row['neg_code_ids']
        neg_verus_ids = row['neg_verus_ids']
        neg_fail_type_ids = row['neg_fail_type_ids']
        neg_fail_type_ids, neg_target = [0] + neg_fail_type_ids[:-1], neg_fail_type_ids[-1:]
        return class_id, pos_code_ids, pos_verus_ids, pos_fail_type_ids, pos_target_fail_type_ids, pos_target, pos_candidate_fail_type_ids, neg_code_ids, neg_verus_ids, neg_fail_type_ids, neg_target

class SeqDataset_te(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        class_id = row['class_id']
        code_ids = row['code_ids']
        verus_ids = row['verus_ids']
        fail_type_ids = row['fail_type_ids']
        candidate_fail_type_ids = row['candidate_fail_type_ids']
        fail_type_ids, target = [0] + fail_type_ids[:-1], fail_type_ids[-1:]
        return class_id, code_ids, verus_ids, fail_type_ids, target, candidate_fail_type_ids

def right_pad_sequence(sequences, batch_first=True, padding_value=0, max_len=5):

    if max_len is None:
        max_len = MAX_SEQ_LEN
        
    # max_len = 5
    padded_seqs = []
    seqs_len = []
    for seq in sequences:
        seq = seq[-max_len:]
        
        pad_len = max_len - seq.size(0)
        seqs_len.append(seq.size(0))
        padding = torch.full((pad_len,), padding_value, dtype=seq.dtype)
        padded_seq = torch.cat([seq, padding], dim=0) 
        padded_seqs.append(padded_seq)
    return torch.stack(padded_seqs) if batch_first else torch.stack(padded_seqs).T, seqs_len

def collate_fn(batch):
    class_id, pos_code_ids, pos_verus_ids, pos_fail_type_ids, pos_target_fail_type_ids, pos_target, pos_candidate_fail_type_ids, neg_code_ids, neg_verus_ids, neg_fail_type_ids, neg_target = zip(*batch)
    class_id = torch.tensor(class_id, dtype=torch.long)

    pos_code_ids = [torch.tensor(x, dtype=torch.long) for x in pos_code_ids]
    pos_verus_ids = [torch.tensor(x, dtype=torch.long) for x in pos_verus_ids]
    pos_fail_type_ids = [torch.tensor(x, dtype=torch.long) for x in pos_fail_type_ids]
    pos_target_fail_type_ids = [torch.tensor(x, dtype=torch.long) for x in pos_target_fail_type_ids]
    pos_targets = torch.tensor(pos_target, dtype=torch.long)
    neg_code_ids = [torch.tensor(x, dtype=torch.long) for x in neg_code_ids]
    neg_verus_ids = [torch.tensor(x, dtype=torch.long) for x in neg_verus_ids]
    neg_fail_type_ids = [torch.tensor(x, dtype=torch.long) for x in neg_fail_type_ids]
    neg_targets = torch.tensor(neg_target, dtype=torch.long)


    pos_code_ids_padded, pos_len_seq = right_pad_sequence(pos_code_ids, batch_first=True, padding_value=0, max_len=MAX_SEQ_LEN)
    pos_verus_ids_padded, _ = right_pad_sequence(pos_verus_ids, batch_first=True, padding_value=0, max_len=MAX_SEQ_LEN)
    pos_fail_type_ids_padded, _ = right_pad_sequence(pos_fail_type_ids, batch_first=True, padding_value=0, max_len=MAX_SEQ_LEN)
    pos_target_fail_type_ids_padded, _ = right_pad_sequence(pos_target_fail_type_ids, batch_first=True, padding_value=0, max_len=MAX_SEQ_LEN)
    pos_len_seq = torch.tensor(pos_len_seq, dtype=torch.long)

    batch_size = pos_code_ids_padded.shape[0]
    max_len = MAX_SEQ_LEN
    fail_type_num = 36
    pos_candidate_mask = torch.zeros(
        batch_size,
        max_len,
        fail_type_num,
        dtype=torch.bool
    )
    for b, candidate_seq in enumerate(pos_candidate_fail_type_ids):
        candidate_seq = candidate_seq[-max_len:]
        for t, ids in enumerate(candidate_seq):
            pos_candidate_mask[b, t, ids] = True

    neg_code_ids_padded, neg_len_seq = right_pad_sequence(neg_code_ids, batch_first=True, padding_value=0, max_len=MAX_SEQ_LEN)
    neg_verus_ids_padded, _ = right_pad_sequence(neg_verus_ids, batch_first=True, padding_value=0, max_len=MAX_SEQ_LEN)
    neg_fail_type_ids_padded, _ = right_pad_sequence(neg_fail_type_ids, batch_first=True, padding_value=0, max_len=MAX_SEQ_LEN)
    neg_len_seq = torch.tensor(neg_len_seq, dtype=torch.long)

    return class_id, pos_code_ids_padded, pos_verus_ids_padded, pos_fail_type_ids_padded, pos_target_fail_type_ids_padded, pos_len_seq, pos_targets, pos_candidate_mask, neg_code_ids_padded, neg_verus_ids_padded, neg_fail_type_ids_padded, neg_len_seq, neg_targets

def collate_fn_te(batch):     
    class_id, code_ids, verus_ids, fail_type_ids, target, candidate_fail_type_ids = zip(*batch)
    code_ids = [torch.tensor(x, dtype=torch.long) for x in code_ids]
    verus_ids = [torch.tensor(x, dtype=torch.long) for x in verus_ids]
    fail_type_ids = [torch.tensor(x, dtype=torch.long) for x in fail_type_ids]
    class_id = torch.tensor(class_id, dtype=torch.long)
    targets = torch.tensor(target, dtype=torch.long)
    candidate_fail_type_ids = [
        candidate_seq[-MAX_SEQ_LEN:]
        for candidate_seq in candidate_fail_type_ids
    ]

    code_ids_padded, len_seq = right_pad_sequence(code_ids, batch_first=True, padding_value=0, max_len=MAX_SEQ_LEN)
    verus_ids_padded, _ = right_pad_sequence(verus_ids, batch_first=True, padding_value=0, max_len=MAX_SEQ_LEN)
    fail_type_ids_padded, _ = right_pad_sequence(fail_type_ids, batch_first=True, padding_value=0, max_len=MAX_SEQ_LEN)
    len_seq = torch.tensor(len_seq, dtype=torch.long)


    return class_id, code_ids_padded, verus_ids_padded, fail_type_ids_padded, len_seq, targets, candidate_fail_type_ids




def evaluate(model, test_file):
    model.eval()
    te_df = pd.read_csv(test_file)
    te_df['code_ids'] = te_df['code_ids'].apply(ast.literal_eval)
    te_df['verus_ids'] = te_df['verus_ids'].apply(ast.literal_eval)
    te_df['fail_type_ids'] = te_df['fail_type_ids'].apply(ast.literal_eval)
    te_df['candidate_fail_type_ids'] = te_df['candidate_fail_type_ids'].apply(ast.literal_eval)
    test_dataset = SeqDataset_te(te_df)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_te)
    
    #max_len = 5
    max_len = model.state_size
    N = [0] * max_len
    P = [0] * max_len
    n1, n2, n3, p1, p2, p3 = 0, 0, 0, 0, 0, 0
    padding_value = 0

    with torch.no_grad():
        for class_id, code_ids, verus_ids, fail_type_ids, len_seq, target, candidate_fail_type_ids in tqdm(test_loader):
            code_ids, verus_ids, fail_type_ids = code_ids.squeeze(0).to(model.device), verus_ids.squeeze(0).to(model.device), fail_type_ids.squeeze(0).to(model.device)
            if class_id.item() == 1:
                n1 += 1
            elif class_id.item() == 2:
                n2 += 1
            else:
                n3 += 1

            flag = True
            for i in range(len_seq.item()):
                k = i + 1
                pad_len = max_len - k
                padding = torch.full((pad_len,), padding_value, dtype=code_ids.dtype).to(model.device)
                code_ids_pad = torch.cat([code_ids[:k], padding], dim=0)
                verus_ids_pad = torch.cat([verus_ids[:k], padding], dim=0)
                pad_len = max_len - k
                padding = torch.full((pad_len,), padding_value, dtype=code_ids.dtype).to(model.device)
                fail_type_ids_pad = torch.cat([fail_type_ids[:k], padding], dim=0).to(model.device)
                code_ids_pad, verus_ids_pad, fail_type_ids_pad = code_ids_pad.unsqueeze(0), verus_ids_pad.unsqueeze(0), fail_type_ids_pad.unsqueeze(0)
              
                code_emb = model.code_embeddings(code_ids_pad)
                verus_emb = model.verus_embeddings(verus_ids_pad)
                fail_type_emb = model.fail_types_embeddings(fail_type_ids_pad)
                fail_type_emb = model.fail_map(fail_type_emb)

                last_logits = model.infer(code_emb, verus_emb, fail_type_emb, k)

                candidate_fail_type_id = candidate_fail_type_ids[0][i]
                if len(candidate_fail_type_id) == 0:
                    flag = False
                    break
                candidate_fail_type_id = torch.tensor(candidate_fail_type_id, dtype=torch.long).to(model.device)
                
                scores = last_logits[candidate_fail_type_id]
                top_1_index = torch.topk(scores, k=1, dim=-1)[1]
                N[i] += 1
                if k == len_seq.item():
                    cur_target = target.item()
                else:
                    cur_target = fail_type_ids[k].item()
                
                if candidate_fail_type_id[top_1_index].item() != cur_target:
                    flag = False
                    # break
                else:
                    P[i] += 1
            
            if flag:
                if class_id.item() == 1:
                    p1 += 1
                elif class_id.item() == 2:
                    p2 += 1
                else:
                    p3 += 1

        class_result = [
            p1, p2, p3,
            p1 / (n1 + 1e-8),
            p2 / (n2 + 1e-8),
            p3 / (n3 + 1e-8),
        ]
        
        position_acc = [
            p / n if n > 0 else float("nan")
            for p, n in zip(P, N)
        ]
        
        print(*class_result)
        print(position_acc)
    
    return [class_result, position_acc]

tr_df = pd.read_csv(train_file)
tr_df['class_id'] = tr_df['class_id'].apply(int)
tr_df['pos_code_ids'] = tr_df['pos_code_ids'].apply(ast.literal_eval)
tr_df['pos_verus_ids'] = tr_df['pos_verus_ids'].apply(ast.literal_eval)
tr_df['pos_fail_type_ids'] = tr_df['pos_fail_type_ids'].apply(ast.literal_eval)
tr_df['pos_candidate_fail_type_ids'] = tr_df['pos_candidate_fail_type_ids'].apply(ast.literal_eval)
tr_df['neg_code_ids'] = tr_df['neg_code_ids'].apply(ast.literal_eval)
tr_df['neg_verus_ids'] = tr_df['neg_verus_ids'].apply(ast.literal_eval)
tr_df['neg_fail_type_ids'] = tr_df['neg_fail_type_ids'].apply(ast.literal_eval)
tr_df = tr_df[['class_id', 'pos_code_ids', 'pos_verus_ids',  'pos_fail_type_ids', 'pos_candidate_fail_type_ids', 'neg_code_ids', 'neg_verus_ids', 'neg_fail_type_ids']]

fail_type_num = 35+1
dataset = SeqDataset(tr_df)
dataloader = DataLoader(dataset, batch_size=args.bsize, shuffle=True, collate_fn=collate_fn)

if args.cuda < 0 or not torch.cuda.is_available():
    device = torch.device("cpu")
else:
    if args.cuda >= torch.cuda.device_count():
        raise ValueError(
            f"Requested cuda:{args.cuda}, but only "
            f"{torch.cuda.device_count()} CUDA devices are visible"
        )
    device = torch.device(f"cuda:{args.cuda}")
    
model = SModel(hidden_size=args.hidden_size, fail_type_num=fail_type_num, state_size=MAX_SEQ_LEN, dropout=args.dropout, device=device, emb_path=args.emb_dir.as_posix().rstrip("/") + "/").to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
criterion = nn.CrossEntropyLoss(ignore_index=0, reduction="none")

#model.to(device)
metric_result = {}
best_epoch = -1
cur_c = float("-inf")
early_stop = args.patience
best_state = None
for epoch in range(args.max_epoch):
    model.train()
    total_loss = 0
    for class_id, pos_code_ids, pos_verus_ids, pos_fail_type_ids, pos_target_fail_type_ids, pos_len_seq, pos_target, pos_candidate_mask, neg_code_ids, neg_verus_ids, neg_fail_type_ids, neg_len_seq, neg_target in dataloader:
        pos_seq = {'code_ids': pos_code_ids.to(device), 'verus_ids': pos_verus_ids.to(device),
                   'fail_type_ids': pos_fail_type_ids.to(device), 'len_seq': pos_len_seq.to(device)}
        neg_seq = {'code_ids': neg_code_ids.to(device), 'verus_ids': neg_verus_ids.to(device),
                   'fail_type_ids': neg_fail_type_ids.to(device), 'len_seq': neg_len_seq.to(device)}
        pos_candidate_mask = pos_candidate_mask.to(device)
        batch_size = pos_target.shape[0]
        pos_target = pos_target.to(device)
        neg_target = neg_target.to(device)
        pos_target_fail_type_ids = pos_target_fail_type_ids.to(device)
        logits, pos_scores, neg_scores = model.forward(pos_seq, pos_target, neg_seq, neg_target)
        # print(output.shape)

        logits = logits.masked_fill(~pos_candidate_mask, -1e9)
        
        rand_mask = torch.rand(logits.shape[0], logits.shape[1], device=logits.device) > args.mask_prob
        rand_mask = rand_mask.view(-1)

        logits = logits.view(-1, logits.size(-1))
        targets = pos_target_fail_type_ids.view(-1)
        
        loss = torch.mean(rand_mask * criterion(logits, targets))
        loss += args.bpr_weight * (-torch.log(torch.sigmoid(pos_scores - neg_scores))).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # break
    print("*"*100)
    print(f"Epoch {epoch}: Loss = {total_loss / len(dataloader):.4f}")
    re = evaluate(model, val_file)
    metric_result[epoch] = re
    A = re[0][3]
    B = re[0][4]
    C = re[0][5]
    if (B + 0.5 * (A + C)) > cur_c:
        cur_c = (B + 0.5 * (A + C))
        best_epoch = epoch
        early_stop = args.patience
        torch.save(model.state_dict(), checkpoint_path)
        best_state = copy.deepcopy(model.state_dict())
    else:
        early_stop -= 1
        if early_stop == 0:
            break
print("training end!")
print(best_epoch)
print(metric_result[best_epoch])
print("*"*50)
model.load_state_dict(best_state)
re = evaluate(model, test_file)
print(re)

