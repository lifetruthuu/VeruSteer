"""
Main Repair Loop - Simplified
Contains the main repair_veval function without checkpointing overhead.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from global_config import GlobalConfig
from llm_utils import insert_loop_isolation
from veval import VerusError, VerusErrorType, VEval

from utils import proof_completion_code_change_is_safe, verusfmt_code

from .failure_history import FailureHistoryManager
from .preprocessing import CodeAnalysis, CodePreprocessor
from .shared_types import write_to_temp_dir


class RepairTraceRecorder:
    """Persist tree states and transitions for downstream dataset processing."""

    def __init__(self, output_dir: Path, run_id: str):
        self.output_dir = Path(output_dir)
        self.run_id = run_id
        self.trace_path = self.output_dir / f"repair-trace-{run_id}.jsonl"
        self.trace_log_path = self.output_dir / f"repair-trace-{run_id}.log"
        self.dataset_path = self.output_dir / f"repair-dataset-{run_id}.jsonl"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_code(self, relative_path: str, code_text: str) -> str:
        full_path = self.output_dir / relative_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(code_text, encoding="utf-8")
        return str(full_path.resolve())

    def log_event(self, event: dict[str, Any]) -> None:
        payload = {"run_id": self.run_id, "timestamp": time.time(), **event}
        with self.trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._append_readable_trace(payload)

    def log_dataset_entry(self, entry: dict[str, Any]) -> None:
        with self.dataset_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _append_readable_trace(self, payload: dict[str, Any]) -> None:
        ts = time.strftime(
            "%Y-%m-%d %H:%M:%S",
            time.localtime(payload.get("timestamp", time.time())),
        )
        event = payload.get("event")
        line = f"[{ts}] event={event}"
        if event == "run_start":
            line = (
                f"[{ts}] run_start max_attempt={payload.get('max_attempt')} "
                f"func_name={payload.get('func_name')}"
            )
        elif event == "state_evaluated":
            line = (
                f"[{ts}] round={payload.get('round_id')} state={payload.get('state_id')} "
                f"score={payload.get('score_tuple')} terminal={payload.get('is_terminal')}"
            )
        elif event == "branch_start":
            chosen = payload.get("chosen_failure", {})
            line = (
                f"[{ts}] round={payload.get('round_id')} state={payload.get('state_id')} "
                f"branch_start error={chosen.get('error_type')}"
            )
        elif event == "branch_selected":
            line = (
                f"[{ts}] round={payload.get('round_id')} state={payload.get('state_id')} "
                f"winner_state={payload.get('winner_state_id')} source={payload.get('winner_source')} "
                f"score={payload.get('candidate_score_tuple')}"
            )
        elif event == "run_end":
            line = (
                f"[{ts}] run_end finished={payload.get('finished_state_count')} "
                f"remaining_active={payload.get('remaining_active_count')} "
                f"best={payload.get('best_state_score')}"
            )
        with self.trace_log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


class RepairMainLoop:
    """Main repair loop - simplified without checkpointing"""

    def __init__(self, agent_orchestrator=None):
        """
        Initialize with agent orchestrator.

        Args:
            agent_orchestrator: AgentOrchestrator instance
        """
        self.agent_orchestrator = agent_orchestrator

        # Get global resources
        self.config = GlobalConfig.get_config()
        self.logger = GlobalConfig.get_logger()
        self.llm = GlobalConfig.get_llm()

        # Preprocessing
        self.preprocessor: CodePreprocessor | None = None
        self.code_analysis: CodeAnalysis | None = None

        # Failure history management
        self.failure_manager = FailureHistoryManager(max_attempts_per_action=3)

    def setup_failure_manager(self):
        """Set up failure manager connection with agent orchestrator"""
        if self.agent_orchestrator:
            self.agent_orchestrator.set_failure_manager(self.failure_manager)
            self.logger.info("Failure manager connected to agent orchestrator")
        else:
            self.logger.warning("Could not connect failure manager - agent orchestrator not found")

    def setup_preprocessing(self):
        """Setup the preprocessor for code analysis"""
        if not self.preprocessor:
            self.preprocessor = CodePreprocessor(self.logger)
            self.logger.info("Preprocessor initialized")

    def perform_preprocessing(self, code: str, func_name: str | None = None) -> CodeAnalysis:
        """Perform static analysis on the code"""
        if not self.preprocessor:
            self.setup_preprocessing()

        self.code_analysis = self.preprocessor.analyze_code(code, func_name)
        self.logger.info(f"Code analysis completed: {self.code_analysis.function_count} functions")
        return self.code_analysis

    def share_preprocessing_with_agents(self):
        """Share preprocessing results with all agents in the orchestrator"""
        if not self.code_analysis or not self.agent_orchestrator:
            self.logger.warning(
                "Cannot share preprocessing - missing code_analysis or agent_orchestrator"
            )
            return

        # Share with all agents
        for agent in self.agent_orchestrator.agents:
            agent.code_analysis = self.code_analysis
            self.logger.debug(f"Shared code analysis with {agent.__class__.__name__}")

    def repair_veval(
        self,
        code: str,
        max_attempt: int = 5,
        func_name: str | None = None,
        temp_dir: Path | None = None,
        temp: float = 1.0,
        tree_search: bool = False,
        args: dict | None = None,
    ) -> tuple[str, bool]:
        """
        Main repair function - simplified version without checkpointing.

        Args:
            code: Source code to repair
            max_attempt: Maximum number of repair attempts
            func_name: Optional specific function to focus on
            temp_dir: Directory for temporary files
            temp: LLM temperature
            tree_search: Whether to use tree search
            args: Additional arguments

        Returns:
            Tuple of (repaired_code, is_verified)
        """
        args = args or {}
        self.logger.info(f"Starting repair with max_attempt={max_attempt}, func_name={func_name}")

        # Setup failure manager and preprocessing
        self.setup_failure_manager()
        self.setup_preprocessing()

        # Perform preprocessing
        self.perform_preprocessing(code, func_name)
        self.share_preprocessing_with_agents()

        code_version = 0

        # Ensure loop isolation is present
        if "loop_isolation(false)" not in code:
            code = insert_loop_isolation(code)

        # Set up temp directory - use provided or get from GlobalConfig
        if temp_dir:
            temp_dir = Path(temp_dir)
            GlobalConfig.set_temp_dir(temp_dir)
        else:
            temp_dir = GlobalConfig.get_temp_dir()

        code = verusfmt_code(code)
        # Write initial input if temp_dir is set
        write_to_temp_dir(temp_dir, f"fix-v{code_version}-input.rs", code)

        run_id = f"{int(time.time() * 1000)}-{os.getpid()}"
        source_metadata = self._extract_source_metadata_from_args(args)
        recorder = RepairTraceRecorder(temp_dir, run_id)
        recorder.log_event(
            {
                "event": "run_start",
                "max_attempt": max_attempt,
                "func_name": func_name,
                "temp": temp,
                "tree_search_enabled": True,
            }
        )

        active_states: list[dict[str, Any]] = [
            {
                "state_id": "r0:root",
                "parent_state_id": None,
                "code": code,
                "failed_last_time": 0,
            }
        ]
        # Keep a globally increasing attempt id for agent metadata/history.
        # Round id is for tree depth, while repair_attempt_id is for each actual repair call.
        repair_attempt_id = 0
        finished_states: list[dict[str, Any]] = []
        best_state: dict[str, Any] | None = None

        self.logger.info("Tree repair mode is enabled by default.")
        for attempt in range(1, max_attempt + 1):
            if len(active_states) == 0:
                break

            self.logger.info(f"\n{'='*60}")
            self.logger.info(
                f"Repair attempt {attempt}/{max_attempt} (active states: {len(active_states)})"
            )
            self.logger.info(f"{'='*60}")

            next_states: list[dict[str, Any]] = []

            for state in active_states:
                state_id = state["state_id"]
                parent_state_id = state["parent_state_id"]
                state_code = state["code"]
                failed_last_time = state["failed_last_time"]
                state_id_fs = self._state_id_for_fs(state_id)

                state_code_relpath = f"states/round-{attempt}/{state_id_fs}.rs"
                state_code_path = recorder.write_code(state_code_relpath, state_code)

                _, score, failures, full_output = self._eval_with_details(
                    state_code,
                    func_name=func_name,
                )
                self.logger.info(f"Current score ({state_id}): {score}")

                if best_state is None or score > best_state["score"]:
                    best_state = {"code": state_code, "score": score, "state_id": state_id}

                dedup_failures = self._dedupe_failures(failures)
                branch_failures = self._dedupe_failures_by_type(dedup_failures)
                default_failure = self._get_one_failure(failures) if failures else None

                recorder.log_event(
                    {
                        "event": "state_evaluated",
                        "round_id": attempt,
                        "state_id": state_id,
                        "parent_state_id": parent_state_id,
                        "code_path": state_code_path,
                        "code_text": state_code,
                        "score_tuple": self._score_tuple(score),
                        "raw_score": str(score),
                        "verus_output_full": full_output,
                        "failures_raw": [
                            self._serialize_failure(f, idx=i) for i, f in enumerate(failures)
                        ],
                        "failures_dedup": [
                            self._serialize_failure(f, idx=i)
                            for i, f in enumerate(dedup_failures)
                        ],
                        "is_terminal": score.is_correct(),
                    }
                )

                if score.is_correct():
                    finished_states.append(
                        {"code": state_code, "score": score, "state_id": state_id}
                    )
                    continue

                if not branch_failures:
                    self.logger.info(f"No fixable failures found for state {state_id}")
                    continue

                for failure_idx, current_failure in enumerate(branch_failures):
                    branch_state_id = self._child_state_id(
                        parent_state_id=state_id, child_round_id=attempt, branch_idx=failure_idx
                    )
                    self.logger.info(
                        f"State {state_id}, branch {failure_idx}, target error: {current_failure.error}"
                    )
                    self.logger.info(f"Error text: {current_failure.get_text()[:200]}...")
                    self.logger.info(f"History scope state: {branch_state_id}")

                    recorder.log_event(
                        {
                            "event": "branch_start",
                            "round_id": attempt,
                            "state_id": state_id,
                            "parent_state_id": parent_state_id,
                            "history_scope_state_id": branch_state_id,
                            "chosen_failure": self._serialize_failure(
                                current_failure, idx=failure_idx
                            ),
                        }
                    )

                    repair_attempt_id += 1
                    metadata_store = GlobalConfig.get_metadata_store()
                    metadata_store.set_history_scope(branch_state_id)
                    if self.failure_manager:
                        self.failure_manager.set_branch_scope(branch_state_id)
                    try:
                        success, new_code, metadata = self._try_agent_repair(
                            state_code,
                            current_failure,
                            repair_attempt_id,
                            args,
                            state_id=branch_state_id,
                        )
                    finally:
                        metadata_store.clear_history_scope()
                        if self.failure_manager:
                            self.failure_manager.clear_branch_scope()

                    action_name = metadata.get("success_action", "unknown_action")
                    accepted_for_branch = False
                    winner_source = "carry_forward_original"
                    fallback_reason = "no_accepted_candidate_keep_original"
                    winner_code = state_code
                    branch_failed_last_time = failed_last_time + 1
                    candidate_code_path = None

                    # Persist all branch candidates (if available) using round/branch layout
                    # aligned with autoverus/refinement.py.
                    raw_candidates = metadata.get("candidates", [])
                    branch_candidates = (
                        raw_candidates
                        if isinstance(raw_candidates, list)
                        else ([raw_candidates] if raw_candidates else [])
                    )
                    for cand_idx, cand_code in enumerate(branch_candidates):
                        if not cand_code or not str(cand_code).strip():
                            continue
                        candidate_relpath = (
                            f"candidates/round-{attempt}/{state_id_fs}/"
                            f"failure-{failure_idx}-{current_failure.error.name}/cand-{cand_idx}.rs"
                        )
                        cand_path = recorder.write_code(candidate_relpath, str(cand_code))
                        if (
                            success
                            and new_code
                            and candidate_code_path is None
                            and str(cand_code) == new_code
                        ):
                            candidate_code_path = cand_path

                    if success and new_code and new_code != state_code:
                        # Fallback write for pipelines that don't return metadata["candidates"].
                        if candidate_code_path is None:
                            candidate_relpath = (
                                f"candidates/round-{attempt}/{state_id_fs}/"
                                f"failure-{failure_idx}-{current_failure.error.name}/cand-0.rs"
                            )
                            candidate_code_path = recorder.write_code(candidate_relpath, new_code)

                        if self._validate_code_change(state_code, new_code, current_failure):
                            accepted_for_branch = True
                            winner_source = "accepted_candidate"
                            fallback_reason = None
                            winner_code = new_code
                            branch_failed_last_time = 0
                            code_version += 1

                            write_to_temp_dir(
                                temp_dir,
                                (
                                    f"fix-v{code_version}-a{attempt}-{state_id_fs}-"
                                    f"b{failure_idx}-success-{action_name}.rs"
                                ),
                                winner_code,
                            )
                            self.logger.info(
                                f"✓ Repair accepted ({state_id}, branch {failure_idx}, version {code_version})"
                            )
                            if self.agent_orchestrator.last_agent:
                                self.agent_orchestrator.last_agent.mark_last_action_accepted(
                                    True, "Code change validated and accepted"
                                )
                        else:
                            self.logger.warning("✗ Repair rejected (validation failed)")
                            if self.agent_orchestrator.last_agent:
                                self.agent_orchestrator.last_agent.mark_last_action_accepted(
                                    False, "Code change validation failed"
                                )
                    else:
                        self.logger.warning("✗ Repair failed to generate new code")

                    if branch_failed_last_time >= 3:
                        self.logger.warning(
                            f"State {state_id}, branch {failure_idx} failed "
                            f"{branch_failed_last_time} times in a row, may be stuck"
                        )

                    new_state_id = branch_state_id
                    winner_code_relpath = (
                        f"states/round-{attempt + 1}/{self._state_id_for_fs(new_state_id)}.rs"
                    )
                    winner_code_path = recorder.write_code(winner_code_relpath, winner_code)
                    _, winner_score, winner_failures, winner_output = self._eval_with_details(
                        winner_code, func_name=func_name
                    )
                    winner_dedup_failures = self._dedupe_failures(winner_failures)
                    winner_branch_failures = self._dedupe_failures_by_type(winner_dedup_failures)

                    recorder.log_event(
                        {
                            "event": "branch_selected",
                            "round_id": attempt,
                            "state_id": state_id,
                            "parent_state_id": parent_state_id,
                            "chosen_failure": self._serialize_failure(
                                current_failure, idx=failure_idx
                            ),
                            "winner_state_id": new_state_id,
                            "winner_source": winner_source,
                            "fallback_reason": fallback_reason,
                            "candidate_code_path": winner_code_path,
                            "candidate_score_tuple": self._score_tuple(winner_score),
                            "candidate_passed": winner_score.is_correct(),
                            "candidate_raw_score": str(winner_score),
                            "accepted_for_branch": accepted_for_branch,
                        }
                    )

                    is_default_chose = bool(
                        default_failure is not None
                        and current_failure.error == default_failure.error
                    )
                    dataset_id = (
                        f"verusage-{source_metadata.get('task_name')}-"
                        f"{source_metadata.get('llm_name')}-"
                        f"{source_metadata.get('shot')}-"
                        f"{source_metadata.get('generation_candidate_id')}-"
                        f"{state_id}-{new_state_id}"
                    )
                    default_path = None
                    if is_default_chose:
                        default_path = (
                            f"verusage-{source_metadata.get('task_name')}-"
                            f"{source_metadata.get('llm_name')}-"
                            f"{source_metadata.get('shot')}-"
                            f"{source_metadata.get('generation_candidate_id')}-"
                            f"{state_id}"
                        )

                    recorder.log_dataset_entry(
                        {
                            "id": dataset_id,
                            "method": "verusage",
                            "dataset_name": source_metadata.get("dataset_name"),
                            "task_name": source_metadata.get("task_name"),
                            "llm_name": source_metadata.get("llm_name"),
                            "shot": source_metadata.get("shot"),
                            "generation_candidate_id": source_metadata.get(
                                "generation_candidate_id"
                            ),
                            "repair_llm_name": source_metadata.get("repair_llm_name"),
                            "round_id": attempt,
                            "source_state_id": state_id,
                            "parent_state_id": parent_state_id,
                            "source_code_path": state_code_path,
                            "source_code_text": state_code,
                            "source_score_tuple": self._score_tuple(score),
                            "source_verus_output_full": full_output,
                            "source_failure_list": [
                                self._serialize_failure(f, idx=i)
                                for i, f in enumerate(dedup_failures)
                            ],
                            "source_failure_types": [f.error.name for f in branch_failures],
                            "chosen_failure_type": current_failure.error.name,
                            "is_default_chose": is_default_chose,
                            "default_path": default_path,
                            "is_carry_forward_original": winner_source
                            == "carry_forward_original",
                            "repaired_state_id": new_state_id,
                            "repaired_code_path": winner_code_path,
                            "repaired_code_text": winner_code,
                            "repaired_score_tuple": self._score_tuple(winner_score),
                            "repaired_verus_output_full": winner_output,
                            "repaired_failure_list": [
                                self._serialize_failure(f, idx=i)
                                for i, f in enumerate(winner_dedup_failures)
                            ],
                            "repaired_failure_types": [
                                f.error.name for f in winner_branch_failures
                            ]
                        }
                    )

                    if best_state is None or winner_score > best_state["score"]:
                        best_state = {
                            "code": winner_code,
                            "score": winner_score,
                            "state_id": new_state_id,
                        }

                    if winner_score.is_correct():
                        finished_states.append(
                            {
                                "code": winner_code,
                                "score": winner_score,
                                "state_id": new_state_id,
                            }
                        )
                    else:
                        next_states.append(
                            {
                                "state_id": new_state_id,
                                "parent_state_id": state_id,
                                "code": winner_code,
                                "failed_last_time": branch_failed_last_time,
                            }
                        )

            self.logger.info(
                f"Round {attempt} done. next_states={len(next_states)}, "
                f"finished={len(finished_states)}"
            )
            active_states = next_states

        recorder.log_event(
            {
                "event": "run_end",
                "finished_state_count": len(finished_states),
                "remaining_active_count": len(active_states),
                "best_state_score": str(best_state["score"]) if best_state else None,
            }
        )

        if len(finished_states) > 0:
            finished_states.sort(key=lambda x: x["score"], reverse=True)
            best_finished = finished_states[0]
            self.logger.info(
                f"Repair finished with {len(finished_states)} terminal states, choose best one."
            )
            write_to_temp_dir(temp_dir, "fix-final-success.rs", best_finished["code"])
            return best_finished["code"], True

        self.logger.warning(f"Max attempts ({max_attempt}) reached without full success")
        if best_state is not None:
            write_to_temp_dir(temp_dir, "fix-final-best-effort.rs", best_state["code"])
            return best_state["code"], False
        return code, False

    def _get_one_failure(
        self,
        verus_errors: list[VerusError],
    ) -> VerusError | None:
        """
        Get one failure to fix, with prioritization.

        Args:
            failures: List of failures
            func_name: Optional function name to focus on
            args: Additional arguments

        Returns:
            Selected failure or None
        """
        # type error gets first priority
        for verus_error in verus_errors:
            if verus_error.error == VerusErrorType.MismatchedType:
                return verus_error

        # array-length precondition gets second priority
        for verus_error in verus_errors:
            if verus_error.error == VerusErrorType.PreCondFailVecLen:
                return verus_error

        # arith overflow 3rd priority
        for verus_error in verus_errors:
            if verus_error.error == VerusErrorType.ArithmeticFlow:
                return verus_error

        # inv-fail before loop 4th priority
        for verus_error in verus_errors:
            if verus_error.error == VerusErrorType.InvFailFront:
                return verus_error

        for verus_error in verus_errors:
            if verus_error.error == VerusErrorType.InvFailEnd:
                return verus_error

        # default
        return verus_errors[0]

    def _failure_signature(self, failure: VerusError) -> tuple[str, str, str]:
        return (
            failure.error.name,
            (failure.error_text or "").strip(),
            (failure.get_text(snippet=False) or "").strip(),
        )

    def _dedupe_failures(self, failures: list[VerusError]) -> list[VerusError]:
        seen: set[tuple[str, str, str]] = set()
        deduped: list[VerusError] = []
        for failure in failures:
            sig = self._failure_signature(failure)
            if sig in seen:
                continue
            seen.add(sig)
            deduped.append(failure)
        return deduped

    def _dedupe_failures_by_type(self, failures: list[VerusError]) -> list[VerusError]:
        seen_types: set[str] = set()
        deduped: list[VerusError] = []
        for failure in failures:
            error_type = failure.error.name
            if error_type in seen_types:
                continue
            seen_types.add(error_type)
            deduped.append(failure)
        return deduped

    def _serialize_failure(self, failure: VerusError, idx: int | None = None) -> dict[str, Any]:
        return {
            "idx": idx,
            "error_type": failure.error.name,
            "error_text": failure.error_text,
            "trace_text": failure.get_text(snippet=False),
        }

    def _score_tuple(self, score) -> list[int]:
        if score.compilation_error:
            return [-1, -1]
        return [score.verified, score.errors]

    def _state_path_part(self, state_id: str) -> str:
        if isinstance(state_id, str) and ":" in state_id:
            return state_id.split(":", 1)[1]
        return str(state_id)

    def _child_state_id(self, parent_state_id: str, child_round_id: int, branch_idx: int) -> str:
        parent_path = self._state_path_part(parent_state_id)
        return f"r{child_round_id}:{parent_path}.b{branch_idx}"

    def _state_id_for_fs(self, state_id: str) -> str:
        return str(state_id).replace(":", "_")

    def _eval_with_details(
        self, code: str, func_name: str | None = None
    ) -> tuple[VEval, Any, list[VerusError], str]:
        veval = VEval(code, self.logger)
        veval.eval(func_name=func_name)
        score = veval.get_score()
        failures = veval.get_failures()

        veval_text = VEval(code, self.logger)
        veval_text.eval(func_name=func_name, json_mode=False)
        full_output = (veval_text.rustc_out or "") + (veval_text.verus_out or "")
        return veval, score, failures, full_output

    def _extract_source_metadata_from_args(self, args: dict[str, Any]) -> dict[str, Any]:
        metadata = {
            "dataset_name": None,
            "task_name": None,
            "llm_name": None,
            "shot": None,
            "generation_candidate_id": None,
            "repair_llm_name": None,
        }
        metadata["repair_llm_name"] = (
            getattr(self.config, "aoai_generation_model", None)
            or getattr(self.config, "generation_model", None)
            or getattr(self.config, "model", None)
        )

        input_path = args.get("input")
        if not input_path:
            return metadata

        p = Path(str(input_path))
        stem = p.stem
        metadata["task_name"] = stem
        parts = list(p.parts)
        shot_tokens = ("few-shot", "zero-shot")

        # Prefer generic directory parsing:
        # .../<shot>/<dataset>/<llm>/<file>.rs
        for idx, token in enumerate(parts[:-1]):
            if token not in shot_tokens:
                continue
            metadata["shot"] = token
            if idx + 1 < len(parts) - 1:
                metadata["dataset_name"] = parts[idx + 1]
            if idx + 2 < len(parts) - 1:
                metadata["llm_name"] = parts[idx + 2]
            break

        # Backward-compatible fallback for historical repair_candidate layout.
        if (metadata["dataset_name"] is None or metadata["llm_name"] is None) and (
            "repair_candidate" in parts
        ):
            idx = len(parts) - 1 - parts[::-1].index("repair_candidate")
            rel_dirs = parts[idx + 1 : -1]
            if len(rel_dirs) >= 1 and rel_dirs[0] in shot_tokens and metadata["shot"] is None:
                metadata["shot"] = rel_dirs[0]
            if metadata["dataset_name"] is None and len(rel_dirs) >= 2:
                metadata["dataset_name"] = rel_dirs[1] if metadata["shot"] else rel_dirs[0]
            if metadata["llm_name"] is None and len(rel_dirs) >= 3:
                metadata["llm_name"] = rel_dirs[2] if metadata["shot"] else rel_dirs[1]

        result_match = re.search(r"-result_(\d+)$", stem)
        if result_match:
            metadata["generation_candidate_id"] = int(result_match.group(1))
            task_name = re.sub(r"-result_\d+$", "", stem)
            if metadata["shot"] and task_name.endswith(f"-{metadata['shot']}"):
                task_name = task_name[: -(len(metadata["shot"]) + 1)]
            if metadata["llm_name"] and task_name.endswith(f"-{metadata['llm_name']}"):
                task_name = task_name[: -(len(metadata["llm_name"]) + 1)]
            metadata["task_name"] = task_name

        if metadata["shot"] is None:
            for token in shot_tokens:
                if token in stem:
                    metadata["shot"] = token
                    break

        # Fallback for paths without explicit dataset directory.
        if metadata["dataset_name"] is None and metadata["task_name"]:
            task_name = metadata["task_name"]
            if "_task_" in task_name:
                metadata["dataset_name"] = task_name.split("_task_", 1)[0]
            elif "_" in task_name:
                metadata["dataset_name"] = task_name.split("_", 1)[0]
        return metadata

    def _try_agent_repair(
        self,
        code: str,
        error: VerusError,
        attempt: int,
        args: dict | None = None,
        state_id: str | None = None,
    ) -> tuple[bool, str, dict[str, Any]]:
        """
        Try to repair using agent framework.

        Returns:
            (success, new_code, metadata)
        """
        if not self.agent_orchestrator:
            self.logger.error("Agent orchestrator not available")
            return False, code, {}

        try:
            success, new_code, metadata = self.agent_orchestrator.repair_with_agent(
                code, error, attempt, state_id=state_id
            )
            return success, new_code, metadata
        except Exception as e:
            self.logger.error(f"Agent repair failed with exception: {e}")
            import traceback

            self.logger.error(traceback.format_exc())
            return False, code, {"error": str(e)}

    def _validate_code_change(
        self,
        old_code: str,
        new_code: str,
        error: VerusError,
    ) -> bool:
        """
        Validate that a code change is safe and reasonable.

        Args:
            old_code: Original code
            new_code: Modified code
            error: The error being fixed

        Returns:
            True if change is valid, False otherwise
        """
        # Check if code actually changed
        if old_code == new_code:
            self.logger.warning("Code didn't change")
            return False

        is_safe = proof_completion_code_change_is_safe(old_code, new_code, self.logger)
        if not is_safe:
            self.logger.warning("Code change failed safety check")
            return False

        self.logger.info("✓ Code change passed validation")
        return True
