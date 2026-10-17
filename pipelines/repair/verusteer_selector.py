"""Select a reported error type using the shared SModel and visited repair history."""

from __future__ import annotations

import importlib.util
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _cfg_get(config: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(config, name)
    except Exception:
        pass
    try:
        return config.get(name, default)
    except Exception:
        return default


@dataclass
class _HistoryItem:
    code: str
    verus_output: str
    fail_type_id: int


class VeruSteerSelector:
    def __init__(self, config: Any, logger: Any):
        self.config = config
        self.logger = logger
        self.enabled = bool(_cfg_get(config, "use_verusteer", False))
        self.history: list[_HistoryItem] = []
        self.max_len = int(_cfg_get(config, "verusteer_state_size", 5))
        self.device = None
        self.torch = None
        self.model = None
        self.emb_model = None
        self.failure_types2id: dict[str, int] = {}
        self.id2failure_types: dict[int, str] = {}
        self.code_text2id: dict[str, int] = {}
        self.verus_output2id: dict[str, int] = {}
        self.model_dir = Path(
            _cfg_get(config, "verusteer_model_dir", "")
            or self._default_model_dir()
        ).resolve()
        self.model_path = Path(
            _cfg_get(config, "verusteer_model_path", "")
            or (Path(__file__).resolve().parents[2] / "outputs/checkpoints/verusage.pth")
        )
        self.bge_model_path = str(
            _cfg_get(config, "verusteer_bge_model_path", "")
            or self._default_bge_path()
        )
        self._load_error: str | None = None

        if self.enabled:
            self._initialize()

    def _default_model_dir(self) -> str:
        return str(Path(__file__).resolve().parents[2] / "selector")

    def _default_bge_path(self) -> str:
        return "BAAI/bge-m3"

    def _initialize(self) -> None:
        try:
            import torch  # type: ignore

            self.torch = torch
            self.device = torch.device(_cfg_get(self.config, "verusteer_device", None) or ("cuda:0" if torch.cuda.is_available() else "cpu"))

            idmap_dir = self.model_dir / "idmap"
            with (idmap_dir / "failure_types2id.json").open("r", encoding="utf-8") as f:
                self.failure_types2id = {str(k): int(v) for k, v in json.load(f).items()}
            self.id2failure_types = {v: k for k, v in self.failure_types2id.items()}
            for name, attr in [
                ("code_text2id.json", "code_text2id"),
                ("verus_output2id.json", "verus_output2id"),
            ]:
                path = idmap_dir / name
                if path.exists():
                    with path.open("r", encoding="utf-8") as f:
                        setattr(self, attr, {str(k): int(v) for k, v in json.load(f).items()})

            spec = importlib.util.spec_from_file_location(
                "verusteer_code_sasrec3",
                self.model_dir / "SModel.py",
            )
            if spec is None or spec.loader is None:
                raise RuntimeError("cannot load SModel.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            fail_type_num = max(self.failure_types2id.values(), default=35) + 1
            self.model = module.SModel(
                hidden_size=int(_cfg_get(self.config, "verusteer_hidden_size", 64)),
                fail_type_num=fail_type_num,
                state_size=self.max_len,
                dropout=0.0,
                device=self.device,
                emb_path=str(self.model_dir / "embedding_table") + os.sep,
            )
            state = torch.load(self.model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)
            self.model.to(self.device)
            self.model.eval()

            self.emb_model = None  # Load BGE only for text absent from the stored maps.
            self.logger.info(f"[VeruSteer] loaded selector model: {self.model_path}")
        except Exception as exc:
            self._load_error = str(exc)
            self.enabled = False
            raise RuntimeError(f"Cannot initialize the requested VeruSteer selector: {exc}") from exc

    def _try_load_bge(self) -> None:
        from FlagEmbedding import BGEM3FlagModel
        self.emb_model = BGEM3FlagModel(self.bge_model_path, use_fp16=self.device.type == "cuda",
                                       devices=str(self.device))

    def select(self, code: str, verus_output: str, failures: list[Any]) -> Any | None:
        if not failures:
            return None
        if not self.enabled or self.model is None or self.torch is None:
            return None

        candidate_pairs: list[tuple[Any, int, str]] = []
        for failure in failures:
            name = self._failure_name(failure)
            type_id = self.failure_types2id.get(name)
            if type_id is not None:
                candidate_pairs.append((failure, type_id, name))

        if not candidate_pairs:
            raise ValueError("None of the reported error types is in the published vocabulary")

        try:
            start = max(0, len(self.history) - (self.max_len - 1))
            seq = [_HistoryItem(item.code, item.verus_output,
                                self.history[i - 1].fail_type_id if i else 0)
                   for i, item in enumerate(self.history) if i >= start]
            seq.append(_HistoryItem(code, verus_output,
                                    self.history[-1].fail_type_id if self.history else 0))
            logits = self._infer_logits(seq)
            cand_ids = self.torch.tensor([p[1] for p in candidate_pairs], dtype=self.torch.long).to(self.device)
            scores = logits[cand_ids]
            top_idx = int(self.torch.topk(scores, k=1, dim=-1)[1].item())
            selected, selected_id, selected_name = candidate_pairs[top_idx]
            score_items = []
            for (failure, type_id, name), score in zip(candidate_pairs, scores.detach().cpu().tolist()):
                score_items.append(f"{name}:{score:.4f}")
            self.logger.info(
                f"[VeruSteer] candidates=[{', '.join(score_items)}] selected={selected_name}"
            )
            self.history.append(
                _HistoryItem(code=code, verus_output=verus_output, fail_type_id=selected_id)
            )
            # Keep the visited prefix so a truncated window retains its preceding action.
            return selected
        except Exception as exc:
            raise RuntimeError(f"VeruSteer inference failed: {exc}") from exc

    def _failure_name(self, failure: Any) -> str:
        error = getattr(failure, "error", failure)
        return str(getattr(error, "name", error))

    def _infer_logits(self, seq: list[_HistoryItem]):
        torch = self.torch
        assert torch is not None
        code_embs = []
        verus_embs = []
        fail_embs = []
        for item in seq:
            code_embs.append(self._text_embedding(item.code, "code"))
            verus_embs.append(self._text_embedding(item.verus_output, "verus"))
            fail_id = torch.tensor([item.fail_type_id], dtype=torch.long).to(self.device)
            fail_emb = self.model.fail_types_embeddings(fail_id)
            fail_emb = self.model.fail_map(fail_emb).squeeze(0)
            fail_embs.append(fail_emb)

        seq_len = len(seq)
        code_tensor = torch.stack(code_embs)
        verus_tensor = torch.stack(verus_embs)
        fail_tensor = torch.stack(fail_embs)

        if seq_len < self.max_len:
            pad_len = self.max_len - seq_len
            code_tensor = torch.cat([code_tensor, torch.zeros(pad_len, code_tensor.shape[-1]).to(self.device)], dim=0)
            verus_tensor = torch.cat([verus_tensor, torch.zeros(pad_len, verus_tensor.shape[-1]).to(self.device)], dim=0)
            fail_tensor = torch.cat([fail_tensor, torch.zeros(pad_len, fail_tensor.shape[-1]).to(self.device)], dim=0)

        with torch.no_grad():
            return self.model.infer(
                code_tensor.unsqueeze(0),
                verus_tensor.unsqueeze(0),
                fail_tensor.unsqueeze(0),
                seq_len,
            )

    def _text_embedding(self, text: str, kind: str):
        torch = self.torch
        assert torch is not None
        text = text or ""
        exact_id = None
        if kind == "code":
            exact_id = self.code_text2id.get(text)
            emb_table = self.model.code_embeddings.weight
        else:
            exact_id = self.verus_output2id.get(text)
            emb_table = self.model.verus_embeddings.weight
        if exact_id is not None and 0 <= exact_id < emb_table.shape[0]:
            return emb_table[exact_id].detach().to(self.device).float()
        if self.emb_model is None:
            self._try_load_bge()
        if self.emb_model is not None:
            vec = self.emb_model.encode([text], batch_size=1, max_length=8192)["dense_vecs"][0]
            return torch.tensor(vec, dtype=torch.float32).to(self.device)
        raise RuntimeError("No text encoder available for an unseen state")


def get_selector(config: Any, logger: Any) -> VeruSteerSelector | None:
    if not bool(_cfg_get(config, "use_verusteer", False)):
        return None
    selector = _cfg_get(config, "_verusteer_selector", None)
    if selector is None:
        selector = VeruSteerSelector(config, logger)
        try:
            setattr(config, "_verusteer_selector", selector)
        except Exception:
            pass
    return selector if selector.enabled else None
