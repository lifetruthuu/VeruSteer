#!/usr/bin/env python3
import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import os
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **kwargs):
        _ = kwargs
        return iterable


_printed_selector_example = True




def normalize_binary(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, int):
        return value if value in (0, 1) else None
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "t", "yes", "y"}:
            return 1
        if v in {"0", "false", "f", "no", "n"}:
            return 0
    return None


def extract_final_score_tuple(path_obj: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    path_states = path_obj.get("path_states")
    if not isinstance(path_states, list) or not path_states:
        return None
    last_state = path_states[-1]
    if not isinstance(last_state, dict):
        return None
    score = last_state.get("score_tuple")
    if (
        isinstance(score, list)
        and len(score) >= 2
        and isinstance(score[0], (int, float))
        and isinstance(score[1], (int, float))
    ):
        return float(score[0]), float(score[1])
    return None


def extract_final_round_id(path_obj: Dict[str, Any]) -> Optional[int]:
    path_states = path_obj.get("path_states")
    if not isinstance(path_states, list) or not path_states:
        return None
    last_state = path_states[-1]
    if not isinstance(last_state, dict):
        return None
    round_id = last_state.get("round_id")
    if isinstance(round_id, int):
        return round_id
    if isinstance(round_id, float) and round_id.is_integer():
        return int(round_id)
    return None


def second_score_comparable(default_second: float, nondefault_second: float) -> bool:
    return default_second != -1 and nondefault_second != -1


def is_nondefault_better_by_eval(
    nondefault_score: Tuple[float, float], default_score: Tuple[float, float]
) -> bool:
    nd_first, nd_second = nondefault_score
    d_first, d_second = default_score
    if nd_first > d_first:
        return True
    if nd_first < d_first:
        return False
    return second_score_comparable(d_second, nd_second) and nd_second < d_second


def normalize_tree_class_id(value: Any) -> Optional[int]:
    valid = {1, -1, 2, -2, 3, -3}
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value in valid else None
    if isinstance(value, float) and value.is_integer():
        iv = int(value)
        return iv if iv in valid else None
    if isinstance(value, str):
        text = value.strip()
        if text in {"1", "-1", "2", "-2", "3", "-3"}:
            return int(text)
    return None


@dataclass
class SelectorInput:
    tree_id: str
    current_state_id: str
    step_index: int
    history_state_ids: List[str]
    candidate_names: List[str]
    code_ids: List[int]
    verus_ids: List[int]
    fail_type_ids: List[int]
    candidate_fail_type_ids: List[int]
    len_seq: int
    state_size: int
    fail_type_history_size: int
    padding_value: int
    code_ids_padded: List[int]
    verus_ids_padded: List[int]
    fail_type_ids_padded: List[int]
    code_texts: List[str]
    verus_texts: List[str]
    code_texts_padded: List[str]
    verus_texts_padded: List[str]


def pad_right(seq: List[Any], max_len: int, pad_value: Any = 0) -> List[Any]:
    seq = list(seq)
    if len(seq) > max_len:
        seq = seq[-max_len:]
    if len(seq) < max_len:
        seq = seq + [pad_value] * (max_len - len(seq))
    return seq


def text_to_pseudo_id(text: str, modulo: int = 1_000_000_007) -> int:
    text = text or ""
    if not text:
        return 0
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % modulo + 1


def build_fail_type_to_id(tree_rows: Dict[str, List[Dict[str, Any]]]) -> Dict[str, int]:
    names = set()
    for rows in tree_rows.values():
        for row in rows:
            states = row.get("path_states") or []
            for state in states:
                if not isinstance(state, dict):
                    continue
                chosen = state.get("chosen_failure_type")
                if isinstance(chosen, str) and chosen:
                    names.add(chosen)
                for name in (state.get("failure_types") or []):
                    if isinstance(name, str) and name:
                        names.add(name)
                repaired = state.get("repaired_failure_list") or []
                for item in repaired:
                    if isinstance(item, dict):
                        name = item.get("error_type")
                        if isinstance(name, str) and name:
                            names.add(name)

    ordered = sorted(names)
    return {name: i + 1 for i, name in enumerate(ordered)}


def build_selector_input(
    tree_id: str,
    current_state_id: str,
    step_index: int,
    prefix_states: List[Dict[str, Any]],
    history_state_ids: List[str],
    candidate_names: List[str],
    fail_type_to_id: Dict[str, int],
    state_size: int,
    fail_type_history_size: int,
    padding_value: int,
    retrieve_code_id: Optional[Callable[[str], int]] = None,
    retrieve_verus_id: Optional[Callable[[str], int]] = None,
    include_current_fail_type: bool = False,
    history_fail_type_ids: Optional[List[int]] = None,
) -> SelectorInput:
    '''
    args:
    tree_id:树的唯一标识
    current_state_id:当前状态的id
    step_index:当前步索引
    prefix_states:前缀状态列表
    history_state_ids:历史状态id列表
    candidate_names:候选错误类型列表
    fail_type_to_id:错误类型到id的映射
    state_size:状态大小
    fail_type_history_size:错误类型历史大小
    padding_value:填充值

    return:
    SelectorInput对象，包含选择器输入信息
    '''
    window_states = list(prefix_states)[-state_size:]
    code_texts = [str(s.get("code_text", "") or "") for s in window_states]
    verus_texts = [str(s.get("verus_output_full", "") or "") for s in window_states]

    if retrieve_code_id is None:
        code_ids = [text_to_pseudo_id(text) for text in code_texts]
    else:
        code_ids = [retrieve_code_id(text) for text in code_texts]

    if retrieve_verus_id is None:
        verus_ids = [text_to_pseudo_id(text) for text in verus_texts]
    else:
        verus_ids = [retrieve_verus_id(text) for text in verus_texts]

    fail_type_ids_full: List[int] = []
    if history_fail_type_ids is not None:
        full_hist: List[int] = []
        for x in history_fail_type_ids:
            if isinstance(x, int):
                full_hist.append(x)
            else:
                try:
                    full_hist.append(int(x))
                except Exception:
                    full_hist.append(0)
        if len(full_hist) >= len(prefix_states):
            fail_type_ids_full = full_hist[-len(window_states):]
        else:
            fail_type_ids_full = [0] * len(window_states)
    else:
        for s in window_states:
            fail_name = s.get("chosen_failure_type")
            if isinstance(fail_name, str) and fail_name in fail_type_to_id:
                fail_type_ids_full.append(fail_type_to_id[fail_name])
            else:
                fail_type_ids_full.append(0)
    if include_current_fail_type:
        # 新模型中 fail_type 序列与 state 序列等长，当前步未知类型由上游填充为 0。
        fail_type_ids = fail_type_ids_full
    else:
        fail_type_ids = fail_type_ids_full[:-1]

    candidate_fail_type_ids = [fail_type_to_id.get(name, 0) for name in candidate_names]
    candidate_fail_type_ids = [x for x in candidate_fail_type_ids if x != 0]

    len_seq = min(len(window_states), state_size)
    code_ids_padded = pad_right(code_ids, state_size, padding_value)
    verus_ids_padded = pad_right(verus_ids, state_size, padding_value)
    code_texts_padded = pad_right(code_texts, state_size, "")
    verus_texts_padded = pad_right(verus_texts, state_size, "")
    fail_type_pad_size = state_size if include_current_fail_type else fail_type_history_size
    fail_type_ids_padded = pad_right(fail_type_ids, fail_type_pad_size, padding_value)

    return SelectorInput(
        tree_id=tree_id,
        current_state_id=current_state_id,
        step_index=step_index,
        history_state_ids=list(history_state_ids),
        candidate_names=list(candidate_names),
        code_ids=code_ids,
        verus_ids=verus_ids,
        fail_type_ids=fail_type_ids,
        candidate_fail_type_ids=candidate_fail_type_ids,
        len_seq=len_seq,
        state_size=state_size,
        fail_type_history_size=fail_type_history_size,
        padding_value=padding_value,
        code_ids_padded=code_ids_padded,
        verus_ids_padded=verus_ids_padded,
        fail_type_ids_padded=fail_type_ids_padded,
        code_texts=code_texts,
        verus_texts=verus_texts,
        code_texts_padded=code_texts_padded,
        verus_texts_padded=verus_texts_padded,
    )
    
def _select_error_type_random(selector_model: Any, selector_input: SelectorInput, rng: random.Random) -> Dict[str, Any]:
    global _printed_selector_example
    candidate_names = selector_input.candidate_names
    if not candidate_names:
        raise ValueError("candidate_names 不能为空")

    if not _printed_selector_example:
        print("[Debug] selector_input 示例:")
        print(json.dumps(asdict(selector_input), ensure_ascii=False, indent=2))
        _printed_selector_example = True

    _ = selector_model
    prob = 1.0 / len(candidate_names)
    prob_map = {name: prob for name in candidate_names}
    selected_name = rng.choice(candidate_names)

    return {
        "selected_name": selected_name,
        "candidate_names": candidate_names,
        "prob_map": prob_map,
    }


def _select_error_type_first(selector_model: Any, selector_input: SelectorInput, rng: random.Random) -> Dict[str, Any]:
    global _printed_selector_example
    candidate_names = selector_input.candidate_names
    if not candidate_names:
        raise ValueError("candidate_names 不能为空")

    if not _printed_selector_example:
        print("[Debug] selector_input 示例:")
        print(json.dumps(asdict(selector_input), ensure_ascii=False, indent=2))
        _printed_selector_example = True

    _ = selector_model
    _ = rng
    selected_name = candidate_names[0]
    prob_map = {name: 0.0 for name in candidate_names}
    prob_map[selected_name] = 1.0

    return {
        "selected_name": selected_name,
        "candidate_names": candidate_names,
        "prob_map": prob_map,
    }


def _select_error_type_model(selector_model: Any, selector_input: SelectorInput, rng: random.Random) -> Dict[str, Any]:
    _ = rng
    if not isinstance(selector_model, dict):
        raise ValueError("selector_mode=model 需要 dict 结构的 selector_model 运行时上下文")

    model = selector_model["model"]
    device = selector_model["device"]
    fail_type_to_id = selector_model["fail_type_to_id"]
    encode_text = selector_model.get("encode_text")
    model_kind = selector_model.get("model_kind", "sasrec")
    input_channels = selector_model.get("input_channels", "both")

    candidate_names = selector_input.candidate_names
    if not candidate_names:
        raise ValueError("candidate_names 不能为空")

    candidate_pairs: List[Tuple[str, int]] = []
    for name in candidate_names:
        cid = fail_type_to_id.get(name, 0)
        if isinstance(cid, int) and cid > 0:
            candidate_pairs.append((name, cid))
        elif selector_model.get("strict_loading"):
            raise ValueError(f"Unknown candidate error type: {name}")

    if not candidate_pairs:
        selected_name = candidate_names[0]
        prob_map = {name: 0.0 for name in candidate_names}
        prob_map[selected_name] = 1.0
        return {
            "selected_name": selected_name,
            "candidate_names": candidate_names,
            "prob_map": prob_map,
        }

    known_candidate_names = [x[0] for x in candidate_pairs]
    known_candidate_ids = [x[1] for x in candidate_pairs]
    vocab_size = (int(model.net[-1].out_features) if model_kind == "candidate-mlp"
                  else int(model.fail_types_embeddings.num_embeddings))
    filtered_ids: List[int] = []
    filtered_names: List[str] = []
    for n, cid in zip(known_candidate_names, known_candidate_ids):
        if 0 <= cid < vocab_size:
            filtered_names.append(n)
            filtered_ids.append(cid)
    if not filtered_ids:
        selected_name = candidate_names[0]
        prob_map = {name: 0.0 for name in candidate_names}
        prob_map[selected_name] = 1.0
        return {
            "selected_name": selected_name,
            "candidate_names": candidate_names,
            "prob_map": prob_map,
        }
    known_candidate_names = filtered_names
    known_candidate_ids = filtered_ids
    candidate_tensor = torch.tensor(known_candidate_ids, dtype=torch.long, device=device)

    if model_kind == "candidate-mlp":
        candidate_mask = torch.zeros((1, vocab_size), dtype=torch.bool, device=device)
        candidate_mask[0, candidate_tensor] = True
        model.eval()
        with torch.no_grad():
            scores = model(candidate_mask)[0, candidate_tensor]
            probs = torch.softmax(scores, dim=-1).cpu().tolist()
            selected_name = known_candidate_names[int(scores.argmax().item())]
        prob_map = dict.fromkeys(candidate_names, 0.0)
        prob_map.update(zip(known_candidate_names, probs))
        return {"selected_name": selected_name, "candidate_names": candidate_names, "prob_map": prob_map}

    def _infer_emb_dim_from_model(m: Any) -> int:
        fail_map = getattr(m, "fail_map", None)
        in_features = getattr(fail_map, "in_features", None)
        if isinstance(in_features, int) and in_features > 0:
            return int(in_features)

        raise ValueError("infer2 模型缺少 fail_map.in_features，无法推断文本 embedding 维度")

    model.eval()
    with torch.no_grad():
        if not callable(encode_text):
            raise ValueError("infer2 推理需要可用的 encode_text")
        emb_dim = _infer_emb_dim_from_model(model)
        hidden_size = int(model.hidden_size)
        seq_len = max(1, min(selector_input.len_seq, selector_input.state_size))
        # The current-state classifier must not encode historical states/actions.
        source_indices = [seq_len - 1] if model_kind == "mlp" else list(range(seq_len))
        if model_kind == "mlp":
            seq_len = 1

        code_emb_seq: List[torch.Tensor] = []
        verus_emb_seq: List[torch.Tensor] = []
        fail_type_emb_seq: List[torch.Tensor] = []

        for i in source_indices:
            code_text = selector_input.code_texts_padded[i]
            verus_text = selector_input.verus_texts_padded[i]

            if code_text and input_channels != "verus":
                code_vec_np = np.asarray(encode_text(code_text), dtype=np.float32)
                code_vec = torch.tensor(code_vec_np, dtype=torch.float32, device=device)
            else:
                code_vec = torch.zeros(emb_dim, dtype=torch.float32, device=device)

            if verus_text and input_channels != "code":
                verus_vec_np = np.asarray(encode_text(verus_text), dtype=np.float32)
                verus_vec = torch.tensor(verus_vec_np, dtype=torch.float32, device=device)
            else:
                verus_vec = torch.zeros(emb_dim, dtype=torch.float32, device=device)

            ft_id = int(selector_input.fail_type_ids_padded[i]) if i < len(selector_input.fail_type_ids_padded) else 0
            if model_kind == "mlp":
                ft_id = 0
            fail_type_id = torch.tensor([ft_id], dtype=torch.long, device=device)
            fail_type_vec = model.fail_types_embeddings(fail_type_id)
            fail_type_vec = model.fail_map(fail_type_vec).squeeze(0)

            code_emb_seq.append(code_vec)
            verus_emb_seq.append(verus_vec)
            fail_type_emb_seq.append(fail_type_vec)

        pad_len = 0 if model_kind == "mlp" else max(selector_input.state_size - seq_len, 0)
        if pad_len > 0:
            for _ in range(pad_len):
                code_emb_seq.append(torch.zeros(emb_dim, dtype=torch.float32, device=device))
                verus_emb_seq.append(torch.zeros(emb_dim, dtype=torch.float32, device=device))
                fail_type_emb_seq.append(torch.zeros(hidden_size, dtype=torch.float32, device=device))

        code_emb_tensor = torch.stack(code_emb_seq, dim=0).unsqueeze(0)
        verus_emb_tensor = torch.stack(verus_emb_seq, dim=0).unsqueeze(0)
        fail_type_emb_tensor = torch.stack(fail_type_emb_seq, dim=0).unsqueeze(0)

        # 与 infer2.py 保持一致：infer 直接返回 fail_type logits。
        last_logits = model.infer(
            code_emb_tensor,
            verus_emb_tensor,
            fail_type_emb_tensor,
            seq_len,
        )
        if last_logits.dim() == 2:
            logits_vec = last_logits[0]
        else:
            logits_vec = last_logits
        if logits_vec.dim() != 1:
            logits_vec = logits_vec.reshape(-1)
        scores = logits_vec[candidate_tensor]

        probs = torch.softmax(scores, dim=-1)
        top_idx = int(torch.argmax(scores).item())

    selected_name = known_candidate_names[top_idx]
    prob_map = {name: 0.0 for name in candidate_names}
    for idx, name in enumerate(known_candidate_names):
        prob_map[name] = float(probs[idx].item())

    return {
        "selected_name": selected_name,
        "candidate_names": candidate_names,
        "prob_map": prob_map,
    }


def l2_normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / denom


def build_text_encoder(embedding_model_path: str, device: Optional[torch.device] = None) -> Callable[[str], np.ndarray]:
    encoder_mode = None
    bge_model = None
    st_model = None
    tokenizer = None
    hf_model = None
    device_str = str(device) if device is not None else None

    try:
        from FlagEmbedding import BGEM3FlagModel

        bge_model = BGEM3FlagModel(
            embedding_model_path,
            use_fp16=torch.cuda.is_available() and device is not None and device.type == "cuda",
            devices=device_str,
        )
        encoder_mode = "flagembedding"
    except Exception:
        try:
            from sentence_transformers import SentenceTransformer

            st_model = SentenceTransformer(embedding_model_path, device=device_str)
            encoder_mode = "sentence_transformers"
        except Exception:
            from transformers import AutoModel, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(embedding_model_path, trust_remote_code=True)
            hf_model = AutoModel.from_pretrained(embedding_model_path, trust_remote_code=True)
            if device is not None:
                hf_model.to(device)
            hf_model.eval()
            encoder_mode = "transformers"

    text_cache: Dict[str, np.ndarray] = {}

    def encode_text(text: str) -> np.ndarray:
        text = text or ""
        if text in text_cache:
            return text_cache[text]

        if encoder_mode == "flagembedding":
            out = bge_model.encode([text], batch_size=1, max_length=8192)
            vec = np.asarray(out["dense_vecs"][0], dtype=np.float32)
            text_cache[text] = vec
            return vec

        if encoder_mode == "sentence_transformers":
            vec = np.asarray(st_model.encode([text], normalize_embeddings=False)[0], dtype=np.float32)
            text_cache[text] = vec
            return vec

        with torch.no_grad():
            inputs = tokenizer(
                [text],
                padding=True,
                truncation=True,
                max_length=1024,
                return_tensors="pt",
            )
            if device is not None:
                inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = hf_model(**inputs)
            last_hidden = outputs.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (last_hidden * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-6)
            vec = pooled[0].cpu().numpy().astype(np.float32)
            text_cache[text] = vec
            return vec

    return encode_text


def build_text_retriever(
    embedding_model_path: str,
    emb_matrix_path: str,
    device: Optional[torch.device] = None,
) -> Callable[[str], int]:
    emb_matrix = np.load(emb_matrix_path).astype(np.float32)
    emb_matrix = l2_normalize(emb_matrix)

    encoder_mode = None
    bge_model = None
    st_model = None
    tokenizer = None
    hf_model = None
    device_str = str(device) if device is not None else None

    try:
        from FlagEmbedding import BGEM3FlagModel

        bge_model = BGEM3FlagModel(
            embedding_model_path,
            use_fp16=torch.cuda.is_available() and device is not None and device.type == "cuda",
            devices=device_str,
        )
        encoder_mode = "flagembedding"
    except Exception:
        try:
            from sentence_transformers import SentenceTransformer

            st_model = SentenceTransformer(embedding_model_path, device=device_str)
            encoder_mode = "sentence_transformers"
        except Exception:
            from transformers import AutoModel, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(embedding_model_path, trust_remote_code=True)
            hf_model = AutoModel.from_pretrained(embedding_model_path, trust_remote_code=True)
            if device is not None:
                hf_model.to(device)
            hf_model.eval()
            encoder_mode = "transformers"

    text_cache: Dict[str, int] = {}

    def encode_text(text: str) -> np.ndarray:
        if encoder_mode == "flagembedding":
            out = bge_model.encode([text], batch_size=1, max_length=8192)
            vec = np.asarray(out["dense_vecs"][0], dtype=np.float32)
            denom = float(np.linalg.norm(vec) + 1e-12)
            return vec / denom

        if encoder_mode == "sentence_transformers":
            vec = st_model.encode([text], normalize_embeddings=True)
            return vec[0]

        with torch.no_grad():
            inputs = tokenizer(
                [text],
                padding=True,
                truncation=True,
                max_length=1024,
                return_tensors="pt",
            )
            if device is not None:
                inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = hf_model(**inputs)
            last_hidden = outputs.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (last_hidden * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-6)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
            return pooled[0].cpu().numpy()

    def retrieve_id(text: str) -> int:
        text = text or ""
        if not text:
            return 0
        if text in text_cache:
            return text_cache[text]

        q = encode_text(text)
        scores = np.matmul(emb_matrix, q)
        idx = int(np.argmax(scores))
        text_cache[text] = idx
        return idx

    return retrieve_id


def load_fail_type_mapping(path: str) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"错误类型映射文件格式非法: {path}")

    mapped: Dict[str, int] = {}
    for k, v in raw.items():
        if not isinstance(k, str):
            continue
        try:
            mapped[k] = int(v)
        except Exception:
            continue
    if not mapped:
        raise ValueError(f"错误类型映射为空: {path}")
    return mapped


def load_sasrec_class(model_definition_path: str, model_kind: str = "sasrec"):
    if not os.path.isfile(model_definition_path):
        raise FileNotFoundError(f"模型定义文件不存在: {model_definition_path}")

    module_name = f"sasrec_def_{abs(hash(model_definition_path))}"
    spec = importlib.util.spec_from_file_location(module_name, model_definition_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模型定义文件: {model_definition_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    class_name = {"sasrec": "SModel", "mlp": "MLP", "history-mlp": "MLP",
                  "candidate-mlp": "CandidateMLP"}[model_kind]
    if not hasattr(module, class_name):
        raise AttributeError(f"模型定义文件中未找到 {class_name} 类: {model_definition_path}")
    return getattr(module, class_name)


def load_model_weights_compatible(model: Any, model_weights_path: str, device: torch.device,
                                  strict: bool = False) -> None:
    if not os.path.isfile(model_weights_path):
        raise FileNotFoundError(f"模型权重不存在: {model_weights_path}")

    checkpoint = torch.load(model_weights_path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise ValueError("模型权重格式非法，无法解析 state_dict")

    if strict:
        model.load_state_dict(state_dict, strict=True)
        print(f"[Info] strict model load: {len(state_dict)} tensors")
        return

    model_state = model.state_dict()
    compatible_state = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    incompatible_keys = [
        key
        for key, value in state_dict.items()
        if key in model_state and model_state[key].shape != value.shape
    ]
    missing, unexpected = model.load_state_dict(compatible_state, strict=False)
    print(
        f"[Info] model state loaded: compatible={len(compatible_state)}/{len(state_dict)}, "
        f"incompatible={len(incompatible_keys)}, missing={len(missing)}, unexpected={len(unexpected)}"
    )


def build_selector_model_runtime(
    model_definition_path: str,
    model_weights_path: str,
    emb_path: Optional[str],
    embedding_model_path: str,
    fail_type_to_id: Dict[str, int],
    state_size: int,
    hidden_size: int,
    fail_type_num: int,
    dropout: float,
    device_name: Optional[str],
    cuda_id: int,
    model_kind: str = "sasrec",
    input_channels: str = "both",
    strict_loading: bool = False,
    encode_text_override: Optional[Callable[[str], np.ndarray]] = None,
    embedding_cache: Optional[str] = None,
    encoder_backend: str = "auto",
) -> Dict[str, Any]:
    if device_name:
        device = torch.device(device_name)
    else:
        device = torch.device(f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu")

    sasrec_cls = load_sasrec_class(model_definition_path, model_kind)
    model_init_kwargs: Dict[str, Any] = {
        "hidden_size": hidden_size,
        "fail_type_num": fail_type_num,
        "state_size": state_size,
        "dropout": dropout,
        "device": device,
    }
    emb_path_for_log: Optional[str] = None
    if isinstance(emb_path, str) and emb_path.strip():
        emb_path_abs = os.path.abspath(os.path.expanduser(emb_path))
        if not emb_path_abs.endswith(os.sep):
            emb_path_abs = emb_path_abs + os.sep
        model_init_kwargs["emb_path"] = emb_path_abs
        emb_path_for_log = emb_path_abs
    if model_kind == "candidate-mlp":
        model_init_kwargs = {"hidden_size": hidden_size, "fail_type_num": fail_type_num, "dropout": dropout}

    # Resolve resources relative to the selected model definition.
    model_definition_dir = os.path.dirname(os.path.abspath(model_definition_path)) or "."
    old_cwd = os.getcwd()
    try:
        os.chdir(model_definition_dir)
        model = sasrec_cls(**model_init_kwargs)
    finally:
        os.chdir(old_cwd)
    load_model_weights_compatible(model, model_weights_path, device, strict=strict_loading)
    model.to(device)
    model.eval()

    encode_text = encode_text_override
    if model_kind != "candidate-mlp" and encode_text is None:
        if encoder_backend == "flagembedding":
            from ablation_encoder import CachedBGEEncoder
            encode_text = CachedBGEEncoder(embedding_model_path, device, embedding_cache)
        else:
            if embedding_cache:
                raise ValueError("--embedding-cache requires --encoder-backend flagembedding")
            encode_text = build_text_encoder(embedding_model_path, device=device)
    id_to_fail_type = {int(v): k for k, v in fail_type_to_id.items()}

    print(f"[Info] selector model ready on {device}")
    print(f"[Info] emb_path={emb_path_for_log if emb_path_for_log else '<model default>'}")
    print(f"[Info] embedding_model={embedding_model_path}")
    print("[Info] model-step-backend=infer")

    return {
        "model": model,
        "device": device,
        "fail_type_to_id": fail_type_to_id,
        "id_to_fail_type": id_to_fail_type,
        "retrieve_code_id": None,
        "retrieve_verus_id": None,
        "encode_text": encode_text,
        "use_full_fail_type_seq": True,
        "model_kind": model_kind,
        "input_channels": input_channels,
        "strict_loading": strict_loading,
    }


def select_error_type(
    selector_model: Any,
    selector_input: SelectorInput,
    rng: random.Random,
    selector_mode: str,
) -> Dict[str, Any]:
    if selector_mode == "random":
        return _select_error_type_random(selector_model, selector_input, rng)
    if selector_mode == "first":
        return _select_error_type_first(selector_model, selector_input, rng)
    if selector_mode == "model":
        return _select_error_type_model(selector_model, selector_input, rng)
    raise ValueError(f"不支持的 selector_mode: {selector_mode}")


@dataclass
class TreeBuild:
    tree_id: str
    paths: List[Dict[str, Any]]
    state_by_id: Dict[str, Dict[str, Any]]
    children_by_state_and_action: Dict[Tuple[str, str], str]
    valid_actions_by_state: Dict[str, List[str]]
    children_actions: Dict[str, List[str]]
    path_id_by_state_tuple: Dict[Tuple[str, ...], str]
    better_path_ids: set
    better_classes: Dict[str, int]
    tree_class_id: Optional[int]


def build_tree_structures(tree_id: str, rows: List[Dict[str, Any]]) -> TreeBuild:
    '''
    args:
    tree_id:树的唯一标识
    rows:path.jsonl中的每行数据

    return:
    TreeBuild对象，包含树的结构信息
    '''
    state_by_id: Dict[str, Dict[str, Any]] = {}
    children_by_state_and_action: Dict[Tuple[str, str], str] = {}
    valid_actions_by_state_set: Dict[str, Set[str]] = defaultdict(set)
    path_id_by_state_tuple: Dict[Tuple[str, ...], str] = {}
    # state_by_id:收集节点(state_id -> state对象)
    # children_by_state_and_action:收集边((当前state_id, action) -> 下一state_id)
    # valid_actions_by_state:记录每个节点有哪些“实际可转移”的 action
    # path_id_by_state_tuple:把整条状态序列映射回 path_id

    for row in rows:
        state_ids = row.get("path_state_ids") or []
        states = row.get("path_states") or []
        if not state_ids or not states or len(state_ids) != len(states):
            continue

        for sid, sobj in zip(state_ids, states):
            if isinstance(sid, str) and sid and sid not in state_by_id and isinstance(sobj, dict):
                state_by_id[sid] = sobj

        for i in range(len(state_ids) - 1):
            cur_sid = state_ids[i]
            nxt_sid = state_ids[i + 1]
            cur_state = states[i]
            action = cur_state.get("chosen_failure_type")
            if not isinstance(cur_sid, str) or not isinstance(nxt_sid, str) or not isinstance(action, str):
                continue
            children_by_state_and_action[(cur_sid, action)] = nxt_sid
            valid_actions_by_state_set[cur_sid].add(action)

        path_id = row.get("path_id")
        if isinstance(path_id, str):
            path_id_by_state_tuple[tuple(state_ids)] = path_id

    better_path_ids: set = set()
    better_classes: Dict[str, int] = {}

    default_rows = [r for r in rows if normalize_binary(r.get("default")) == 1]
    nondefault_rows = [r for r in rows if normalize_binary(r.get("default")) == 0]

    tree_class_id: Optional[int] = None

    class_values = {
        normalize_tree_class_id(r.get("tree_class_id"))
        for r in rows
        if normalize_tree_class_id(r.get("tree_class_id")) is not None
    }
    if len(class_values) == 1:
        tree_class_id = next(iter(class_values))

    if len(default_rows) == 1 and nondefault_rows:
        default_row = default_rows[0]
        default_passed = normalize_binary(default_row.get("label")) == 1
        default_pid = default_row.get("path_id") if isinstance(default_row.get("path_id"), str) else None
        passed_nondefault_rows = [
            r for r in nondefault_rows if normalize_binary(r.get("label")) == 1
        ]

        if tree_class_id is None:
            if not default_passed and passed_nondefault_rows:
                tree_class_id = 2
            elif not default_passed and not passed_nondefault_rows:
                tree_class_id = 1
            elif default_passed and passed_nondefault_rows:
                tree_class_id = 3

        if tree_class_id == 1:
            default_score = extract_final_score_tuple(default_row)
            if default_score is not None:
                for r in nondefault_rows:
                    nd_score = extract_final_score_tuple(r)
                    pid = r.get("path_id")
                    if nd_score is None or not isinstance(pid, str):
                        continue
                    if is_nondefault_better_by_eval(nd_score, default_score):
                        better_path_ids.add(pid)
                        better_classes[pid] = 1

        elif tree_class_id == -1:
            default_score = extract_final_score_tuple(default_row)
            default_failed = normalize_binary(default_row.get("label")) == 0
            if default_score is not None and default_failed and default_pid is not None:
                has_default_better = False
                for r in nondefault_rows:
                    if normalize_binary(r.get("label")) != 0:
                        continue
                    nd_score = extract_final_score_tuple(r)
                    if nd_score is None:
                        continue
                    if is_nondefault_better_by_eval(default_score, nd_score):
                        has_default_better = True
                        break
                if has_default_better:
                    better_path_ids.add(default_pid)
                    better_classes[default_pid] = -1

        elif tree_class_id == 2:
            for r in passed_nondefault_rows:
                pid = r.get("path_id")
                if isinstance(pid, str):
                    better_path_ids.add(pid)
                    better_classes[pid] = 2

        elif tree_class_id == -2:
            if default_passed and default_pid is not None:
                has_nondefault_failed = any(
                    normalize_binary(r.get("label")) == 0 for r in nondefault_rows
                )
                if has_nondefault_failed:
                    better_path_ids.add(default_pid)
                    better_classes[default_pid] = -2

        elif tree_class_id == 3:
            default_round = extract_final_round_id(default_row)
            if default_round is not None:
                for r in passed_nondefault_rows:
                    pid = r.get("path_id")
                    nd_round = extract_final_round_id(r)
                    if isinstance(pid, str) and nd_round is not None and nd_round < default_round:
                        better_path_ids.add(pid)
                        better_classes[pid] = 3

        elif tree_class_id == -3:
            default_round = extract_final_round_id(default_row)
            if default_round is not None and default_pid is not None and default_passed:
                has_default_faster = False
                for r in passed_nondefault_rows:
                    nd_round = extract_final_round_id(r)
                    if nd_round is not None and default_round < nd_round:
                        has_default_faster = True
                        break
                if has_default_faster:
                    better_path_ids.add(default_pid)
                    better_classes[default_pid] = -3

    valid_actions_by_state: Dict[str, List[str]] = {
        sid: sorted(list(actions))
        for sid, actions in valid_actions_by_state_set.items()
    }

    return TreeBuild(
        tree_id=tree_id,
        paths=rows,
        state_by_id=state_by_id,
        children_by_state_and_action=children_by_state_and_action,
        valid_actions_by_state=valid_actions_by_state,
        # 兼容旧字段：保留相同含义的数据。
        children_actions=valid_actions_by_state,
        path_id_by_state_tuple=path_id_by_state_tuple,
        better_path_ids=better_path_ids,
        better_classes=better_classes,
        tree_class_id=tree_class_id,
    )


def greedy_rollout_to_leaf(
    tree: TreeBuild,
    selector_model: Any,
    rng: random.Random,
    selector_mode: str,
    fail_type_to_id: Dict[str, int],
    state_size: int,
    fail_type_history_size: int,
    padding_value: int,
) -> Dict[str, Any]:
    if not tree.paths:
        return {"status": "empty_tree"}

    root_state_id = tree.paths[0].get("root_state_id")
    if not isinstance(root_state_id, str) or root_state_id not in tree.state_by_id:
        return {"status": "no_root"}

    state_ids = [root_state_id]
    states = [tree.state_by_id[root_state_id]]
    history_fail_type_ids = [0]
    steps = 0
    dead_end = False
    dead_end_reason: Optional[str] = None
    selection_trace: List[Dict[str, Any]] = []

    retrieve_code_id = None
    retrieve_verus_id = None
    include_current_fail_type = False
    if selector_mode == "model" and isinstance(selector_model, dict):
        retrieve_code_id = selector_model.get("retrieve_code_id")
        retrieve_verus_id = selector_model.get("retrieve_verus_id")
        include_current_fail_type = bool(selector_model.get("use_full_fail_type_seq", False))

    while True:
        current_state_id = state_ids[-1]
        current_state = states[-1]

        candidate_names = list(current_state.get("failure_types", []) or [])
        if not candidate_names:
            repaired = current_state.get("repaired_failure_list", []) or []
            fallback_names = [x.get("error_type", "") for x in repaired if x.get("error_type")]
            candidate_names = fallback_names

        # 所有 selector 统一收敛到“实际可转移动作”：
        # candidate_names = failure_types(或fallback) ∩ valid_actions_by_state
        valid_actions = set(tree.valid_actions_by_state.get(current_state_id, []))
        if candidate_names:
            candidate_names = [name for name in candidate_names if name in valid_actions]

        if not candidate_names:
            selection_trace.append(
                {
                    "step": steps,
                    "current_state_id": current_state_id,
                    "candidate_names": [],
                    "selected_name": None,
                    "selected_prob": None,
                    "prob_map": {},
                    "next_state_id": None,
                    "dead_end_reason": "no_candidates",
                }
            )
            break

        selector_input = build_selector_input(
            tree_id=tree.tree_id,
            current_state_id=current_state_id,
            step_index=steps,
            prefix_states=list(states),
            history_state_ids=list(state_ids),
            candidate_names=list(candidate_names),
            fail_type_to_id=fail_type_to_id,
            state_size=state_size,
            fail_type_history_size=fail_type_history_size,
            padding_value=padding_value,
            retrieve_code_id=retrieve_code_id,
            retrieve_verus_id=retrieve_verus_id,
            include_current_fail_type=include_current_fail_type,
            history_fail_type_ids=history_fail_type_ids,
        )
        selected = select_error_type(selector_model, selector_input, rng, selector_mode)
        pred_name = selected["selected_name"]
        prob_map = selected.get("prob_map", {}) or {}
        step_log = {
            "step": steps,
            "current_state_id": current_state_id,
            "candidate_names": list(candidate_names),
            "selected_name": pred_name,
            "selected_prob": float(prob_map.get(pred_name, 0.0)) if pred_name is not None else None,
            "prob_map": {str(k): float(v) for k, v in prob_map.items()},
            "next_state_id": None,
            "next_state_score_tuple": None,
            "dead_end_reason": None,
        }
        steps += 1

        if pred_name is None:
            dead_end = True
            dead_end_reason = "selector_returned_none"
            step_log["dead_end_reason"] = dead_end_reason
            selection_trace.append(step_log)
            break

        nxt = tree.children_by_state_and_action.get((current_state_id, pred_name))
        if nxt is None:
            dead_end = True
            dead_end_reason = "no_transition_for_selected_action"
            step_log["dead_end_reason"] = dead_end_reason
            selection_trace.append(step_log)
            break

        if nxt in state_ids:
            dead_end = True
            dead_end_reason = "loop_detected"
            step_log["next_state_id"] = nxt
            step_log["dead_end_reason"] = dead_end_reason
            selection_trace.append(step_log)
            break

        if nxt not in tree.state_by_id:
            dead_end = True
            dead_end_reason = "next_state_missing"
            step_log["next_state_id"] = nxt
            step_log["dead_end_reason"] = dead_end_reason
            selection_trace.append(step_log)
            break

        step_log["next_state_id"] = nxt
        next_state = tree.state_by_id.get(nxt, {})
        if isinstance(next_state, dict):
            next_score = next_state.get("score_tuple")
            if isinstance(next_score, list) and len(next_score) >= 2:
                step_log["next_state_score_tuple"] = [next_score[0], next_score[1]]
        selection_trace.append(step_log)

        pred_fail_type_id = int(fail_type_to_id.get(pred_name, 0)) if isinstance(pred_name, str) else 0
        history_fail_type_ids.append(pred_fail_type_id)
        state_ids.append(nxt)
        states.append(tree.state_by_id[nxt])

    state_tuple = tuple(state_ids)
    reached_path_id = tree.path_id_by_state_tuple.get(state_tuple)
    reached_leaf_state_id = state_ids[-1]

    return {
        "status": "ok",
        "steps": steps,
        "dead_end": dead_end,
        "reached_path_id": reached_path_id,
        "reached_leaf_state_id": reached_leaf_state_id,
        "state_ids": state_ids,
        "dead_end_reason": dead_end_reason,
        "selection_trace": selection_trace,
    }


def better_path_probability(
    tree: TreeBuild,
    path_row: Dict[str, Any],
    selector_model: Any,
    rng: random.Random,
    selector_mode: str,
    fail_type_to_id: Dict[str, int],
    state_size: int,
    fail_type_history_size: int,
    padding_value: int,
) -> Dict[str, Any]:
    path_states = path_row.get("path_states") or []
    if not path_states:
        return {"path_prob": 0.0, "greedy_match": False, "valid_steps": 0, "total_steps": 0}

    total_steps = max(len(path_states) - 1, 0)
    if total_steps == 0:
        return {"path_prob": 1.0, "greedy_match": True, "valid_steps": 0, "total_steps": 0}

    path_prob = 1.0
    greedy_match = True
    valid_steps = 0
    history_fail_type_ids = [0]

    retrieve_code_id = None
    retrieve_verus_id = None
    include_current_fail_type = False
    if selector_mode == "model" and isinstance(selector_model, dict):
        retrieve_code_id = selector_model.get("retrieve_code_id")
        retrieve_verus_id = selector_model.get("retrieve_verus_id")
        include_current_fail_type = bool(selector_model.get("use_full_fail_type_seq", False))

    for t in range(total_steps):
        current_state = path_states[t]
        prefix_states = path_states[: t + 1]
        target_name = current_state.get("chosen_failure_type")
        target_fail_type_id = int(fail_type_to_id.get(target_name, 0)) if isinstance(target_name, str) else 0
        if not isinstance(target_name, str) or not target_name:
            path_prob = 0.0
            greedy_match = False
            history_fail_type_ids.append(target_fail_type_id)
            continue

        candidate_names = list(current_state.get("failure_types", []) or [])
        if not candidate_names:
            repaired = current_state.get("repaired_failure_list", []) or []
            fallback_names = [x.get("error_type", "") for x in repaired if x.get("error_type")]
            candidate_names = fallback_names

        if not candidate_names:
            path_prob = 0.0
            greedy_match = False
            history_fail_type_ids.append(target_fail_type_id)
            continue

        cur_state_id = current_state.get("state_id", f"unknown_{t}")
        selector_input = build_selector_input(
            tree_id=tree.tree_id,
            current_state_id=str(cur_state_id),
            step_index=t,
            prefix_states=list(prefix_states),
            history_state_ids=[str(s.get("state_id", "")) for s in prefix_states],
            candidate_names=list(candidate_names),
            fail_type_to_id=fail_type_to_id,
            state_size=state_size,
            fail_type_history_size=fail_type_history_size,
            padding_value=padding_value,
            retrieve_code_id=retrieve_code_id,
            retrieve_verus_id=retrieve_verus_id,
            include_current_fail_type=include_current_fail_type,
            history_fail_type_ids=history_fail_type_ids,
        )
        selected = select_error_type(selector_model, selector_input, rng, selector_mode)
        pred_name = selected["selected_name"]
        prob_map = selected["prob_map"]

        if target_name not in prob_map:
            path_prob = 0.0
            greedy_match = False
            history_fail_type_ids.append(target_fail_type_id)
            continue

        step_prob = prob_map.get(target_name, 0.0)
        path_prob *= step_prob
        valid_steps += 1
        if pred_name != target_name:
            greedy_match = False
        history_fail_type_ids.append(target_fail_type_id)

    return {
        "path_prob": float(path_prob),
        "greedy_match": bool(greedy_match),
        "valid_steps": int(valid_steps),
        "total_steps": int(total_steps),
    }

# 指定 path_id 的逐步预测模式
def parse_predict_result_text(text: str) -> List[str]:
    raw = str(text or "").strip()
    if not raw:
        return []
    if not (raw.startswith("[") and raw.endswith("]")):
        return [x.strip() for x in raw.split(",") if x.strip()]

    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            out: List[str] = []
            for x in parsed:
                if isinstance(x, str):
                    sx = x.strip()
                    if sx:
                        out.append(sx)
                elif x is not None:
                    out.append(str(x))
            return out
    except Exception:
        pass

    inner = raw[1:-1].strip()
    if not inner:
        return []
    return [x.strip() for x in inner.split(",") if x.strip()]


def predict_stepwise_for_path(
    path_row: Dict[str, Any],
    selector_model: Any,
    rng: random.Random,
    selector_mode: str,
    fail_type_to_id: Dict[str, int],
    state_size: int,
    fail_type_history_size: int,
    padding_value: int,
    history_mode: str,
) -> Dict[str, Any]:
    path_states = path_row.get("path_states") or []
    if not isinstance(path_states, list) or not path_states:
        return {
            "predicted_result": [],
            "target_result": [],
            "steps": [],
        }

    total_steps = max(len(path_states) - 1, 0)
    history_fail_type_ids = [0]
    predicted_result: List[str] = []
    target_result: List[str] = []
    steps: List[Dict[str, Any]] = []

    retrieve_code_id = None
    retrieve_verus_id = None
    include_current_fail_type = False
    if selector_mode == "model" and isinstance(selector_model, dict):
        retrieve_code_id = selector_model.get("retrieve_code_id")
        retrieve_verus_id = selector_model.get("retrieve_verus_id")
        include_current_fail_type = bool(selector_model.get("use_full_fail_type_seq", False))

    for t in range(total_steps):
        current_state = path_states[t]
        prefix_states = path_states[: t + 1]
        target_name = current_state.get("chosen_failure_type")
        target_name = target_name if isinstance(target_name, str) else ""
        target_result.append(target_name)
        target_fail_type_id = int(fail_type_to_id.get(target_name, 0))

        candidate_names = list(current_state.get("failure_types", []) or [])
        if not candidate_names:
            repaired = current_state.get("repaired_failure_list", []) or []
            candidate_names = [x.get("error_type", "") for x in repaired if x.get("error_type")]

        pred_name = ""
        prob_map: Dict[str, float] = {}
        selected_prob: Optional[float] = None
        if candidate_names:
            cur_state_id = current_state.get("state_id", f"unknown_{t}")
            selector_input = build_selector_input(
                tree_id=str(path_row.get("tree_id", "")),
                current_state_id=str(cur_state_id),
                step_index=t,
                prefix_states=list(prefix_states),
                history_state_ids=[str(s.get("state_id", "")) for s in prefix_states],
                candidate_names=list(candidate_names),
                fail_type_to_id=fail_type_to_id,
                state_size=state_size,
                fail_type_history_size=fail_type_history_size,
                padding_value=padding_value,
                retrieve_code_id=retrieve_code_id,
                retrieve_verus_id=retrieve_verus_id,
                include_current_fail_type=include_current_fail_type,
                history_fail_type_ids=history_fail_type_ids,
            )
            selected = select_error_type(selector_model, selector_input, rng, selector_mode)
            sel = selected.get("selected_name")
            pred_name = sel if isinstance(sel, str) else ""
            raw_map = selected.get("prob_map", {}) or {}
            prob_map = {str(k): float(v) for k, v in raw_map.items()}
            selected_prob = float(prob_map.get(pred_name, 0.0)) if pred_name else None

        predicted_result.append(pred_name)
        steps.append(
            {
                "step": t,
                "state_id": str(current_state.get("state_id", f"unknown_{t}")),
                "candidate_names": candidate_names,
                "predicted_name": pred_name,
                "target_name": target_name,
                "selected_prob": selected_prob,
                "match_target": bool(pred_name == target_name),
                "prob_map": prob_map,
            }
        )

        if history_mode == "pred":
            history_fail_type_ids.append(int(fail_type_to_id.get(pred_name, 0)))
        else:
            history_fail_type_ids.append(target_fail_type_id)

    return {
        "predicted_result": predicted_result,
        "target_result": target_result,
        "steps": steps,
    }


def parse_target_path_ids(ids_text: Optional[str], ids_file: Optional[str]) -> List[str]:
    ordered: List[str] = []
    seen = set()

    if isinstance(ids_text, str) and ids_text.strip():
        for item in ids_text.split(","):
            pid = item.strip()
            if pid and pid not in seen:
                seen.add(pid)
                ordered.append(pid)

    if isinstance(ids_file, str) and ids_file.strip():
        with open(ids_file, "r", encoding="utf-8") as f:
            for line in f:
                pid = line.strip()
                if pid and pid not in seen:
                    seen.add(pid)
                    ordered.append(pid)

    return ordered


def load_expected_triplets(expected_csv: Optional[str]) -> Dict[str, List[str]]:
    if not isinstance(expected_csv, str) or not expected_csv.strip():
        return {}
    mapping: Dict[str, List[str]] = {}
    with open(expected_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not isinstance(row, dict):
                continue
            pid = row.get("path_id")
            if not isinstance(pid, str) or not pid.strip():
                continue
            preds = parse_predict_result_text(row.get("predict_result", ""))
            mapping[pid.strip()] = preds
    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="评估 SASRec 在 path.jsonl 上的整树下行能力与更优路径到达概率"
    )
    parser.add_argument("--test-path-jsonl", required=True, help="测试集 path jsonl(每行一条根到叶路径)")
    parser.add_argument(
        "--selector-mode",
        choices=["random", "first", "model"],
        default="random",
        help="选择策略：random=随机, first=每次取第一个, model=使用SASRec模型推理",
    )
    parser.add_argument("--random-seed", type=int, default=2024, help="随机策略种子，保证可复现")
    parser.add_argument("--state-size", type=int, default=5, help="对齐 SASRec：状态序列长度")
    parser.add_argument("--fail-type-history-size", type=int, default=4, help="对齐 SASRec：错误类型历史长度")
    parser.add_argument("--padding-value", type=int, default=0, help="对齐 SASRec：padding 值")
    parser.add_argument("--model-weights", default=None, help="SASRec 权重路径(.pth)")
    parser.add_argument("--model-kind", choices=["sasrec"], default="sasrec",
                        help="mlp uses the current state; history-mlp preserves the state/action sequence")
    parser.add_argument("--input-channels", choices=["both", "code", "verus"], default="both")
    parser.add_argument("--strict-model-loading", action="store_true", help="拒绝缺失或尺寸不匹配的权重")
    parser.add_argument("--encoder-backend", choices=["auto", "flagembedding"], default="auto")
    parser.add_argument("--embedding-cache", default=None, help="固定 BGE 配置的 SQLite 文本向量缓存")
    parser.add_argument("--model-definition", default=str(ROOT / "selector/SModel.py"), help="SASRec 定义文件路径")
    parser.add_argument("--emb-path", default=str(ROOT / "selector/embedding_table"), help="可选：SASRec 初始化 embedding 目录(需包含 *.npy)；不传则走模型定义默认值")
    parser.add_argument("--embedding-model", default="BAAI/bge-m3", help="文本编码模型路径(BGE)")
    parser.add_argument("--fail-type-map", default=str(ROOT / "selector/idmap/failure_types2id.json"), help="failure_types2id.json 路径")
    parser.add_argument("--hidden-size", type=int, default=64, help="SASRec hidden size")
    parser.add_argument("--fail-type-num", type=int, default=36, help="SASRec fail_type_num(含padding)")
    parser.add_argument("--dropout", type=float, default=0.0, help="SASRec dropout")
    parser.add_argument("--device", default=None, help="推理设备，如 cuda:0 / cpu；默认自动")
    parser.add_argument("--cuda-id", type=int, default=0, help="默认 CUDA 设备编号（当 --device 未设置时生效，默认: 1）")
    parser.add_argument(
        "--save-tree-details",
        default=None,
        help="可选：保存每棵树明细到 JSON 文件",
    )
    parser.add_argument(
        "--save-selection-log",
        default=None,
        help="可选：保存每棵树选择轨迹日志到 JSONL 文件",
    )
    parser.add_argument(
        "--path-step-mode",
        action="store_true",
        help="开启指定 path_id 的逐步预测模式(用于与 infer 逐步对齐验证)",
    )
    parser.add_argument(
        "--target-path-ids",
        default=None,
        help="逗号分隔的 path_id 列表；仅在 --path-step-mode 下生效",
    )
    parser.add_argument(
        "--target-path-ids-file",
        default=None,
        help="每行一个 path_id 的文本文件；仅在 --path-step-mode 下生效",
    )
    parser.add_argument(
        "--path-step-history-mode",
        choices=["target", "pred"],
        default="target",
        help="逐步预测时 fail_type 历史来源：target=路径真实动作, pred=模型上一步预测",
    )
    parser.add_argument(
        "--expected-triplets-csv",
        default=None,
        help="可选：tree_id,path_id,predict_result 的期望 CSV，用于逐 path 对比",
    )
    parser.add_argument(
        "--save-path-step-csv",
        default=None,
        help="可选：保存 path-step 结果到 CSV",
    )
    parser.add_argument(
        "--save-path-step-details",
        default=None,
        help="可选：保存 path-step 逐步明细到 JSON 文件",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(args.random_seed)
    selector_model = None

    print(f"[Info] selector={args.selector_mode}, seed={args.random_seed}")

    tree_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with open(args.test_path_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            tree_id = obj.get("tree_id")
            if isinstance(tree_id, str) and tree_id:
                tree_rows[tree_id].append(obj)

    if not tree_rows:
        print("[Error] 未读取到有效树")
        return 1

    if args.selector_mode == "model":
        if not args.model_weights:
            print("[Error] selector_mode=model 时必须提供 --model-weights")
            return 1
        if not args.model_definition:
            print("[Error] selector_mode=model 时必须提供 --model-definition")
            return 1
        if not args.fail_type_map:
            print("[Error] selector_mode=model 时必须提供 --fail-type-map")
            return 1

        try:
            fail_type_to_id = load_fail_type_mapping(args.fail_type_map)
            selector_model = build_selector_model_runtime(
                model_definition_path=args.model_definition,
                model_weights_path=args.model_weights,
                emb_path=args.emb_path,
                embedding_model_path=args.embedding_model,
                fail_type_to_id=fail_type_to_id,
                state_size=args.state_size,
                hidden_size=args.hidden_size,
                fail_type_num=args.fail_type_num,
                dropout=args.dropout,
                device_name=args.device,
                cuda_id=args.cuda_id,
                model_kind=args.model_kind,
                input_channels=args.input_channels,
                strict_loading=args.strict_model_loading,
                embedding_cache=args.embedding_cache,
                encoder_backend=args.encoder_backend,
            )
        except Exception as e:
            print(f"[Error] 初始化 selector 模型失败: {e}")
            return 1
    else:
        fail_type_to_id = build_fail_type_to_id(tree_rows)

    # 指定 path_id 的逐步预测模式
    if args.path_step_mode:
        target_path_ids = parse_target_path_ids(args.target_path_ids, args.target_path_ids_file)
        if not target_path_ids:
            print("[Error] --path-step-mode 下必须提供 --target-path-ids 或 --target-path-ids-file")
            return 1

        rows_by_path_id: Dict[str, Dict[str, Any]] = {}
        for rows in tree_rows.values():
            for row in rows:
                pid = row.get("path_id")
                if isinstance(pid, str) and pid and pid not in rows_by_path_id:
                    rows_by_path_id[pid] = row

        expected_map = load_expected_triplets(args.expected_triplets_csv)
        step_results: List[Dict[str, Any]] = []
        found_count = 0
        exact_expected_match_count = 0

        for pid in target_path_ids:
            row = rows_by_path_id.get(pid)
            if row is None:
                step_results.append(
                    {
                        "path_id": pid,
                        "tree_id": "",
                        "status": "path_not_found",
                        "predict_result": [],
                        "target_result": [],
                        "expected_result": expected_map.get(pid),
                        "expected_match": False,
                        "all_steps_match_target": False,
                        "steps": [],
                    }
                )
                continue

            found_count += 1
            ret = predict_stepwise_for_path(
                path_row=row,
                selector_model=selector_model,
                rng=rng,
                selector_mode=args.selector_mode,
                fail_type_to_id=fail_type_to_id,
                state_size=args.state_size,
                fail_type_history_size=args.fail_type_history_size,
                padding_value=args.padding_value,
                history_mode=args.path_step_history_mode,
            )
            predicted_result = ret["predicted_result"]
            target_result = ret["target_result"]
            expected_result = expected_map.get(pid)
            expected_match = (
                expected_result is not None
                and isinstance(expected_result, list)
                and predicted_result == expected_result
            )
            all_steps_match_target = bool(predicted_result == target_result)
            if expected_match:
                exact_expected_match_count += 1

            step_results.append(
                {
                    "path_id": pid,
                    "tree_id": str(row.get("tree_id", "")),
                    "status": "ok",
                    "predict_result": predicted_result,
                    "target_result": target_result,
                    "expected_result": expected_result,
                    "expected_match": expected_match,
                    "all_steps_match_target": all_steps_match_target,
                    "steps": ret["steps"],
                }
            )

        print("=" * 100)
        print("[Path-Step] 指定 path_id 逐步预测结果")
        print(f"target_path_ids: {len(target_path_ids)}")
        print(f"found_path_ids: {found_count}")
        if expected_map:
            print(f"expected_paths: {len(expected_map)}")
            print(f"exact_match_with_expected: {exact_expected_match_count}")
        for rec in step_results:
            pid = rec["path_id"]
            status = rec["status"]
            if status != "ok":
                print(f"- path_id={pid} status={status}")
                continue
            pred = rec["predict_result"]
            tgt = rec["target_result"]
            expected = rec["expected_result"]
            print(
                f"- path_id={pid} "
                f"all_steps_match_target={rec['all_steps_match_target']} "
                f"expected_match={rec['expected_match'] if expected is not None else 'N/A'}"
            )
            print(f"  predict_result={pred}")
            if expected is not None:
                print(f"  expected_result={expected}")
            else:
                print(f"  target_result={tgt}")

        if args.save_path_step_csv:
            os.makedirs(os.path.dirname(args.save_path_step_csv) or ".", exist_ok=True)
            with open(args.save_path_step_csv, "w", encoding="utf-8", newline="") as wf:
                writer = csv.writer(wf)
                writer.writerow(
                    [
                        "tree_id",
                        "path_id",
                        "status",
                        "predict_result",
                        "target_result",
                        "expected_result",
                        "expected_match",
                        "all_steps_match_target",
                    ]
                )
                for rec in step_results:
                    writer.writerow(
                        [
                            rec.get("tree_id", ""),
                            rec.get("path_id", ""),
                            rec.get("status", ""),
                            "[" + ", ".join(rec.get("predict_result", [])) + "]",
                            "[" + ", ".join(rec.get("target_result", [])) + "]",
                            (
                                "[" + ", ".join(rec["expected_result"]) + "]"
                                if isinstance(rec.get("expected_result"), list)
                                else ""
                            ),
                            str(bool(rec.get("expected_match", False))),
                            str(bool(rec.get("all_steps_match_target", False))),
                        ]
                    )
            print(f"[Info] path-step CSV 已保存到: {args.save_path_step_csv}")

        if args.save_path_step_details:
            os.makedirs(os.path.dirname(args.save_path_step_details) or ".", exist_ok=True)
            with open(args.save_path_step_details, "w", encoding="utf-8") as wf:
                json.dump(step_results, wf, ensure_ascii=False, indent=2)
            print(f"[Info] path-step 明细已保存到: {args.save_path_step_details}")

        return 0

    total_trees = len(tree_rows)
    trees_with_better = 0
    rollout_hit_trees = 0
    class_order = [1, -1, 2, -2, 3, -3]
    class_hit = {c: 0 for c in class_order}
    class_total = {c: 0 for c in class_order}

    tree_best_probs: List[float] = []
    tree_mean_probs: List[float] = []
    better_path_prob_all: List[float] = []
    better_path_greedy_match_all: List[int] = []

    tree_details: List[Dict[str, Any]] = []
    selection_log_wf = None
    if args.save_selection_log:
        os.makedirs(os.path.dirname(args.save_selection_log) or ".", exist_ok=True)
        selection_log_wf = open(args.save_selection_log, "w", encoding="utf-8")

    for tree_id in tqdm(
        sorted(tree_rows.keys()),
        total=total_trees,
        desc="Evaluating trees",
        unit="tree",
    ):
        rows = tree_rows[tree_id]
        tree = build_tree_structures(tree_id, rows)

        present_classes = set(tree.better_classes.values())
        for c in present_classes:
            if c in class_total:
                class_total[c] += 1

        has_better = len(tree.better_path_ids) > 0
        if has_better:
            trees_with_better += 1

        rollout = greedy_rollout_to_leaf(
            tree=tree,
            selector_model=selector_model,
            rng=rng,
            selector_mode=args.selector_mode,
            fail_type_to_id=fail_type_to_id,
            state_size=args.state_size,
            fail_type_history_size=args.fail_type_history_size,
            padding_value=args.padding_value,
        )

        reached_path_id = rollout.get("reached_path_id")
        rollout_hit = bool(has_better and isinstance(reached_path_id, str) and reached_path_id in tree.better_path_ids)
        if rollout_hit:
            rollout_hit_trees += 1
            hit_class = tree.better_classes.get(reached_path_id)
            if hit_class in class_hit:
                class_hit[hit_class] += 1

        probs_this_tree: List[float] = []
        greedy_match_this_tree: List[int] = []
        better_paths_detail: List[Dict[str, Any]] = []

        if has_better:
            rows_by_pid = {
                r.get("path_id"): r for r in rows if isinstance(r.get("path_id"), str)
            }
            for pid in sorted(tree.better_path_ids):
                row = rows_by_pid.get(pid)
                if row is None:
                    continue
                ret = better_path_probability(
                    tree=tree,
                    path_row=row,
                    selector_model=selector_model,
                    rng=rng,
                    selector_mode=args.selector_mode,
                    fail_type_to_id=fail_type_to_id,
                    state_size=args.state_size,
                    fail_type_history_size=args.fail_type_history_size,
                    padding_value=args.padding_value,
                )
                probs_this_tree.append(ret["path_prob"])
                greedy_match_this_tree.append(1 if ret["greedy_match"] else 0)
                better_path_prob_all.append(ret["path_prob"])
                better_path_greedy_match_all.append(1 if ret["greedy_match"] else 0)
                better_paths_detail.append(
                    {
                        "path_id": pid,
                        "class_id": tree.better_classes.get(pid),
                        **ret,
                    }
                )

        if probs_this_tree:
            tree_best_probs.append(max(probs_this_tree))
            tree_mean_probs.append(float(np.mean(probs_this_tree)))

        if args.save_tree_details:
            tree_details.append(
                {
                    "tree_id": tree_id,
                    "num_paths": len(rows),
                    "num_better_paths": len(tree.better_path_ids),
                    "better_path_ids": sorted(tree.better_path_ids),
                    "rollout": rollout,
                    "rollout_hit_better": rollout_hit,
                    "better_paths_eval": better_paths_detail,
                }
            )

        if selection_log_wf is not None:
            reached_path_better_class = tree.better_classes.get(reached_path_id)
            selection_record = {
                "tree_id": tree_id,
                "tree_class_id": tree.tree_class_id,
                "selector_mode": args.selector_mode,
                "num_paths": len(rows),
                "num_better_paths": len(tree.better_path_ids),
                "better_path_ids": sorted(tree.better_path_ids),
                "rollout_hit_better": rollout_hit,
                "reached_path_id": rollout.get("reached_path_id"),
                "reached_path_better_class": reached_path_better_class if reached_path_better_class in (1, -1, 2, -2, 3, -3) else "none",
                "reached_leaf_state_id": rollout.get("reached_leaf_state_id"),
                "dead_end": rollout.get("dead_end"),
                "dead_end_reason": rollout.get("dead_end_reason"),
                "steps": rollout.get("steps"),
                "selection_trace": rollout.get("selection_trace", []),
            }
            selection_log_wf.write(json.dumps(selection_record, ensure_ascii=False) + "\n")

    print("=" * 100)
    print("[Metric-1] 整树从根逐步下行(greedy rollout)")
    print(f"总树数: {total_trees}")
    print(f"含更优路径(class 1/-1/2/-2/3/-3)的树数: {trees_with_better}")
    print(f"命中更优路径的树数: {rollout_hit_trees}")
    print(f"命中率(对所有树): {rollout_hit_trees / total_trees:.4f}")
    print(
        f"命中率(对含更优路径的树): "
        f"{(rollout_hit_trees / trees_with_better) if trees_with_better else 0.0:.4f}"
    )
    for c in class_order:
        denom = class_total[c]
        rate = class_hit[c] / denom if denom else 0.0
        print(f"class {c}: hit_tree={class_hit[c]} / total_tree={denom} = {rate:.4f}")

    print("=" * 100)
    print("[Metric-2] 更优路径逐步到达概率(teacher-forcing path probability)")
    print(f"更优路径总数: {len(better_path_prob_all)}")
    if better_path_prob_all:
        print(f"所有更优路径 平均到达概率: {float(np.mean(better_path_prob_all)):.6f}")
        print(f"所有更优路径 中位数到达概率: {float(np.median(better_path_prob_all)):.6f}")
        print(
            f"所有更优路径 greedy 全步匹配率: "
            f"{float(np.mean(better_path_greedy_match_all)):.4f}"
        )
    else:
        print("所有更优路径 平均到达概率: 0.000000")
        print("所有更优路径 中位数到达概率: 0.000000")
        print("所有更优路径 greedy 全步匹配率: 0.0000")

    if tree_best_probs:
        print(f"按树聚合(每棵树 max 更优路径概率) 平均值: {float(np.mean(tree_best_probs)):.6f}")
    else:
        print("按树聚合(每棵树 max 更优路径概率) 平均值: 0.000000")

    if tree_mean_probs:
        print(f"按树聚合(每棵树 mean 更优路径概率) 平均值: {float(np.mean(tree_mean_probs)):.6f}")
    else:
        print("按树聚合(每棵树 mean 更优路径概率) 平均值: 0.000000")

    if args.save_tree_details:
        os.makedirs(os.path.dirname(args.save_tree_details) or ".", exist_ok=True)
        with open(args.save_tree_details, "w", encoding="utf-8") as wf:
            json.dump(tree_details, wf, ensure_ascii=False, indent=2)
        print(f"[Info] per-tree 结果已保存到: {args.save_tree_details}")

    if selection_log_wf is not None:
        selection_log_wf.close()
        print(f"[Info] per-tree 选择日志已保存到: {args.save_selection_log}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
