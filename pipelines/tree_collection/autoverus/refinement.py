# Copyright (c) Microsoft Corporation. #
# Licensed under the MIT license.      #

'''
autoverus的接受条件：
先要 is_safe_change == True
再满足以下任一：
    当前目标错误类型的实例数减少
    当前目标错误类型实例数不变，但整体 score 更好
        compilation_error：是否编译错误（False 比 True 好）
        verified：验证通过数量（越大越好）
        errors：验证错误数量（越小越好）
        verus_errors：Verus 解析出的错误条数（越小越好）
    在连续失败后（failed_last_time > 0），同类型同文本的细粒度实例减少，且 score 不变差

特例：如果验证通过就不要检查safe
'''


import json
import os
import time
from pathlib import Path

from houdini import houdini
from infer import LLM
from veval import VerusError, VerusErrorLabel, VerusErrorType, VEval

from utils import (
    clean_code,
    code_change_is_safe,
    fix_one_type_error_in_code,
    get_nonlinear_lines,
    insert_lemma_func,
    insert_loop_isolation,
)


class RepairTraceRecorder:
    def __init__(self, output_dir, run_id):
        self.output_dir = output_dir
        self.run_id = run_id
        self.trace_path = os.path.join(output_dir, f"repair-trace-{run_id}.jsonl")
        self.trace_log_path = os.path.join(output_dir, f"repair-trace-{run_id}.log")
        self.dataset_path = os.path.join(output_dir, f"repair-dataset-{run_id}.jsonl")
        os.makedirs(self.output_dir, exist_ok=True)

    def write_code(self, relative_path, code_text):
        full_path = os.path.join(self.output_dir, relative_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(code_text)
        return os.path.abspath(full_path)

    def log_event(self, event):
        payload = {"run_id": self.run_id, "timestamp": time.time(), **event}
        with open(self.trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
        self._append_readable_trace(payload)

    def log_dataset_entry(self, entry):
        payload = {**entry}
        with open(self.dataset_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    def _append_readable_trace(self, payload):
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(payload.get("timestamp", time.time())))
        event = payload.get("event")
        line = f"[{ts}] event={event}"

        if event == "run_start":
            line = (
                f"[{ts}] run_start max_attempt={payload.get('max_attempt')} "
                f"func_name={payload.get('func_name')} temp={payload.get('temp')}"
            )
        elif event == "state_evaluated":
            failures = payload.get("failures_dedup", [])
            failure_types = [f.get("error_type", "unknown") for f in failures]
            line = (
                f"[{ts}] round={payload.get('round_id')} state={payload.get('state_id')} "
                f"parent_state={payload.get('parent_state_id')} score={payload.get('score_tuple')} "
                f"is_terminal={payload.get('is_terminal')} dedup_failures={failure_types}"
            )
        elif event == "branch_start":
            chosen = payload.get("chosen_failure", {})
            line = (
                f"[{ts}] round={payload.get('round_id')} state={payload.get('state_id')} "
                f"branch_start main_error={chosen.get('error_type')} "
                f"repair_func={payload.get('repair_func')} candidate_count={payload.get('repair_num')}"
            )
        elif event == "candidate_evaluated":
            chosen = payload.get("chosen_failure", {})
            line = (
                f"[{ts}] round={payload.get('round_id')} state={payload.get('state_id')} "
                f"candidate={payload.get('candidate_id')} main_error={chosen.get('error_type')} "
                f"score={payload.get('candidate_score_tuple')} passed={payload.get('candidate_passed')} "
                f"is_safe_change={payload.get('is_safe_change')} accepted_for_branch={payload.get('accepted_for_branch')}"
            )
        elif event == "branch_selected":
            chosen = payload.get("chosen_failure", {})
            line = (
                f"[{ts}] round={payload.get('round_id')} state={payload.get('state_id')} "
                f"branch_selected main_error={chosen.get('error_type')} "
                f"winner_exists={payload.get('winner_exists')} "
                f"winner_state={payload.get('winner_state_id')} "
                f"winner_source={payload.get('winner_source')} "
                f"score={payload.get('candidate_score_tuple')} passed={payload.get('candidate_passed')}"
            )
        elif event == "run_end":
            line = (
                f"[{ts}] run_end finished_state_count={payload.get('finished_state_count')} "
                f"remaining_active_count={payload.get('remaining_active_count')} "
                f"best_state_score={payload.get('best_state_score')}"
            )

        with open(self.trace_log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


class Refinement:
    def __init__(self, config, logger):
        self.config = config
        self.llm = LLM(config, logger)
        self.logger = logger
        self.hdn = houdini(config)
        self.debug_dump_dir = "."
        self.default_system = "You are an experienced formal language programmer. You are very familiar with Verus, which is a tool for verifying the correctness of code written in Rust."

        # Proof block knowledge
        self.proof_block_info = """The proof block looks like this:
```
proof {
    // your proof code here
    // assert(...)
    // LEMMA_FUNCTION(...)
    // ...
} // Added by AI
```
Note, please add the assertion directly for the `proof fn` function and DO NOT use proof block.
You can only use the proof block for the `fn` and `pub fn` functions.

The ghost variable looks like this:
```
let ghost ...; // Added by AI
```

Note, please DO NOT modify all other proof blocks that are not related to the error. Just leave them as they are."""
        # Seq knowledge.
        _seq_examples = self.get_text_examples("seq")
        self.seq_knowledge = (
            "Here is the usage for Seq in Verus you can refer:\n```\n{}\n```\n".format(
                "\n".join(_seq_examples)
            )
        )

    def _extract_source_metadata(self, input_file):
        p = Path(input_file)
        stem = p.stem
        shot_tokens = {"few-shot", "zero-shot"}
        llm_name = None
        shot_from_dir = None
        metadata = {
            "dataset_name": None,
            "task_name": None,
            "llm_name": None,
            "shot": None,
            "generation_candidate_id": None,
        }

        # Supported directory patterns:
        # 1) .../repair_dataset/<dataset_name>/<llm_name>/<file>.rs
        # 2) .../repair_dataset/<dataset_name>/<llm_name>/<shot>/<file>.rs
        # 3) .../repair_candidate/<shot>/<dataset_name>/<llm_name>/<file>.rs
        parts = list(p.parts)
        if "repair_candidate" in parts:
            idx = len(parts) - 1 - parts[::-1].index("repair_candidate")
            rel_dirs = parts[idx + 1 : -1]
            if len(rel_dirs) >= 1 and rel_dirs[0] in shot_tokens:
                shot_from_dir = rel_dirs[0]
            if len(rel_dirs) >= 2:
                metadata["dataset_name"] = rel_dirs[1]
            if len(rel_dirs) >= 3:
                llm_name = rel_dirs[2]
        elif "repair_dataset" in parts:
            idx = len(parts) - 1 - parts[::-1].index("repair_dataset")
            rel_dirs = parts[idx + 1 : -1]
            if len(rel_dirs) >= 1:
                metadata["dataset_name"] = rel_dirs[0]
            if len(rel_dirs) >= 2 and rel_dirs[1] not in shot_tokens:
                llm_name = rel_dirs[1]
            if len(rel_dirs) >= 3 and rel_dirs[2] in shot_tokens:
                shot_from_dir = rel_dirs[2]
            elif len(rel_dirs) >= 2 and rel_dirs[1] in shot_tokens:
                shot_from_dir = rel_dirs[1]

        if metadata["dataset_name"] is None and len(parts) >= 3:
            # Fallback: use grandparent directory name.
            metadata["dataset_name"] = p.parent.parent.name

        # Generic shot fallback: scan near ancestors for few-shot/zero-shot.
        if shot_from_dir is None:
            for depth, parent in enumerate(p.parents):
                if depth > 4:
                    break
                if parent.name in shot_tokens:
                    shot_from_dir = parent.name
                    break

        if not llm_name:
            parent_name = p.parent.name
            if parent_name in shot_tokens and p.parent.parent.name:
                llm_name = p.parent.parent.name
                shot_from_dir = shot_from_dir or parent_name
            else:
                llm_name = parent_name
        metadata["llm_name"] = llm_name if llm_name else None

        # Expected pattern:
        # <task_name>-<llm_name>-<shot>-<generation_candidate_id>.rs
        # e.g. DAFNY2VERUS-COLLECTION_1-deepseek-chat-few-shot-result_0.rs
        shot_tag = None
        if "-few-shot-" in stem:
            shot_tag = "few-shot"
            prefix, suffix = stem.split("-few-shot-", 1)
        elif "-zero-shot-" in stem:
            shot_tag = "zero-shot"
            prefix, suffix = stem.split("-zero-shot-", 1)
        else:
            prefix, suffix = stem, None

        metadata["shot"] = shot_from_dir or shot_tag
        if shot_tag is not None:
            metadata["generation_candidate_id"] = suffix if suffix else None

        llm_marker = f"-{llm_name}" if llm_name else ""
        if llm_marker and prefix.endswith(llm_marker):
            task_name = prefix[: -len(llm_marker)]
        else:
            task_name = prefix

        metadata["task_name"] = task_name if task_name else None
        return metadata

    def _set_debug_dump_dir(self, debug_dir):
        if debug_dir is None:
            return
        self.debug_dump_dir = os.fspath(debug_dir)
        os.makedirs(self.debug_dump_dir, exist_ok=True)

    def _write_debug_dump(self, filename, content):
        output_path = os.path.join(self.debug_dump_dir, filename)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        return output_path

    def add_seq_knowledge(self, code, instruction) -> str:
        """Check whether the code contains the usage of Seq/Vec and add the Seq knowledge to the instruction."""
        _possible_usage = ["Seq", "Vec", "array", "nums"]
        for usage in _possible_usage:
            if usage in code:
                instruction += "\n\n" + self.seq_knowledge
                break
        return instruction

    def debug_type_error(self, code: str, verus_error: VerusError = None, num=1) -> str:
        """
        self debug to fix type error
        """
        del num

        rnd = 0
        max_rnd = 10

        if verus_error:
            # fix the reported one
            if verus_error.error != VerusErrorType.MismatchedType:
                print("Warning: a non type error is passed to debug_type_error")
            else:
                newcode = fix_one_type_error_in_code(code, verus_error.trace[0], verbose=False)
                if newcode:
                    code = newcode

        # check if there is any type errors in the code; if so, fix
        while rnd < max_rnd:
            rnd = rnd + 1

            veval = VEval(code, self.logger)
            veval.eval()
            failures = veval.get_failures()
            if len(failures) == 0:
                self.logger.info("Verus has succeeded.")
                return code, 0

            has_typeerr = False
            fixed_typeerr = False
            for cur_failure in failures:
                if cur_failure.error == VerusErrorType.MismatchedType:
                    has_typeerr = True
                    newcode = fix_one_type_error_in_code(code, cur_failure.trace[0], verbose=False)
                    # when newcode is "", the above function failed to fix any type error
                    if newcode:
                        fixed_typeerr = True
                        code = newcode
                        break
                    else:
                        # this type error is unfixable, let's move on to next error
                        continue
                if not fixed_typeerr:
                    # not able to fix any type error in this program, no need to try again
                    break

            if not has_typeerr:
                return code, 0

            if not fixed_typeerr:
                self.logger.info("Remaining type errors are unfixable.")
                self.logger.info(cur_failure.trace[0].get_text())
                return "", len(failures)

        return code, len(failures)

    def get_examples(self, example_dir_name):
        examples = []
        for f in sorted(
            os.listdir(os.path.join(self.config.example_path, f"input-{example_dir_name}"))
        ):
            if f.endswith(".rs") and f.startswith("ex"):
                input_file = os.path.join(self.config.example_path, f"input-{example_dir_name}", f)
                output_file = os.path.join(
                    self.config.example_path, f"output-{example_dir_name}", f
                )
                input_content = open(input_file).read()
                output_content = open(output_file).read()
                examples.append({"query": input_content, "answer": output_content})
        dump_content = []
        for example in examples:
            dump_content.append("Query:\n" + example["query"])
            dump_content.append("\n\nAnswer:\n" + example["answer"])
            dump_content.append("\n\n")
        self._write_debug_dump(f"example-{example_dir_name}.txt", "".join(dump_content))
        return examples

    def get_text_examples(self, example_dir_name):
        examples = []
        for f in sorted(os.listdir(os.path.join(self.config.example_path, example_dir_name))):
            if f.endswith(".rs") and f.startswith("ex"):
                input_file = os.path.join(self.config.example_path, example_dir_name, f)
                input_content = open(input_file).read()
                examples.append(input_content)
        return examples

    def repair_SeqSyntax_error(self, code: str, verus_error: VerusError, num=1, temp=1.0) -> str:
        system = self.default_system
        print("in seq syntax error fixing")

        error_trace = verus_error.trace[0]
        errline = error_trace.get_text().strip()
        errlinenum = error_trace.lines[0]

        instruction = (
            f"This code contains a syntax error on line {errlinenum}"
            + "in the expression ` "
            + errline
            + "'. Your mission is to rewrite this expression `"
            + errline
            + "' to fix the syntax error. Please make sure to change that wrong expression and do not change any other part of the code. Response with the Rust code only, do not include any explanation. Please use a comment to explain what changes you have made to fix this syntax error."
        )

        seq_knowledge = "Here is the usage for Seq in Verus you can refer:\n```\n{}\n```\n"
        seq_examples = self.get_text_examples("seq")
        seq_knowledge = seq_knowledge.format("\n".join(seq_examples))
        instruction += "\n\n" + seq_knowledge

        examples = []
        query_template = "Incorrect line \n```\n{}```\n"
        query_template += "\nCode\n```\n{}```\n"

        query = query_template.format(errline, code)

        self._write_debug_dump("seqsyntax-query.txt", query)

        return self.llm.infer_llm(
            self.config.aoai_debug_model,
            instruction,
            examples,
            query,
            system,
            answer_num=num,
            max_tokens=4096,
            temp=temp,
        )

    def repair_special_assertion_error(
        self, code: str, verus_error: VerusError, num=1, temp=1.0
    ) -> str:
        """
        Some assertions contain certain data structure / APIs that have a routine solution
        It is a bit ad-hoc now. Eventually, this should become a dedicated lemma-lookup function
        """
        assertion_info = verus_error.trace[0].get_text()

        newcode = ""
        did_special_fix = False

        # conducting these special fixes sequentially
        if ".filter(" in assertion_info:
            print("special fix: add reveal")
            instruction = """Please add `reveal(Seq::filter);' at the beginning of the function where the failed assert line is located. This will help Verus understand filter and hence prove anything related to filter."""
            examples = []
            query_template = "Failed assertion\n```\n{}```\n"
            query_template += "\nCode\n```\n{}```\n"
            query = query_template.format(assertion_info, code)
            output = self.llm.infer_llm(
                self.config.aoai_debug_model,
                instruction,
                examples,
                query,
                self.default_system,
                answer_num=num,
                max_tokens=4096,
                temp=temp,
            )[0]
            newcode = clean_code(output)
            newcode, _ = self.debug_type_error(newcode)
            if newcode:
                did_special_fix = True
                code = newcode

        # we may need to help a subrange-to-all assert to help Seq reasoning
        if (
            ".filter(" in assertion_info
            and ".subrange(" in code
            and ".subrange(" not in assertion_info
        ):
            print("special fix: add subrange all")
            if "lemma_seq_subrange_all" not in code:
                newcode = insert_lemma_func(code, ["seq_subrange_all"], self.config.lemma_path)
            newcode = self.repair_assertion_error_with_lemma_func(
                newcode, verus_error, 1, ["seq_subrange_all"]
            )[0]
            newcode = clean_code(newcode)
            newcode, _ = self.debug_type_error(newcode)
            if newcode:
                did_special_fix = True
                code = newcode

        if ".take(" in assertion_info:
            if "lemma_seq_take_ascend" not in code and "lemma_seq_take_all" not in code:
                newcode = insert_lemma_func(
                    code, ["seq_take_ascend", "seq_take_all"], self.config.lemma_path
                )
            elif "lemma_seq_take_all" not in code:
                newcode = insert_lemma_func(code, ["seq_take_all"], self.config.lemma_path)
            elif "lemma_seq_take_ascend" not in code:
                newcode = insert_lemma_func(code, ["seq_take_ascend"], self.config.lemma_path)
            else:
                newcode = code

            newcode = self.repair_assertion_error_with_lemma_func(
                newcode, verus_error, 1, ["seq_take_ascend", "seq_take_all"]
            )[0]
            newcode = clean_code(newcode)
            newcode, _ = self.debug_type_error(newcode)
            if newcode:
                did_special_fix = True
                code = newcode

        if ".subrange(" in assertion_info:
            newcode = insert_lemma_func(
                code,
                ["seq_subrange_ascend", "seq_subrange_all"],
                self.config.lemma_path,
            )
            newcode = self.repair_assertion_error_with_lemma_func(
                newcode, verus_error, 1, ["seq_subrange_ascend", "seq_subrange_all"]
            )[0]
            newcode = clean_code(newcode)
            if newcode:
                self.logger.info("subrange lemma fix")
                did_special_fix = True
                code = newcode

        if ".contains(" in assertion_info:
            newcode = insert_lemma_func(code, ["vec_push", "vec_remove"], self.config.lemma_path)
            newcode = self.repair_assertion_error_with_lemma_func(
                newcode, verus_error, 1, ["vec_push", "vec_remove"]
            )[0]
            newcode = clean_code(newcode)
            newcode, _ = self.debug_type_error(newcode)
            if newcode:
                did_special_fix = True
                code = newcode

        if ".subrange(" in assertion_info:
            if "lemma_seq_subrange_ascend" not in code and "lemma_seq_subrange_all" not in code:
                newcode = insert_lemma_func(
                    code,
                    ["seq_subrange_ascend", "seq_subrange_all"],
                    self.config.lemma_path,
                )
            elif "lemma_seq_subrange_all" not in code:
                newcode = insert_lemma_func(code, ["seq_subrange_all"], self.config.lemma_path)
            elif "lemma_seq_subrange_ascend" not in code:
                newcode = insert_lemma_func(code, ["seq_subrange_ascend"], self.config.lemma_path)
            else:
                newcode = code

            newcode = self.repair_assertion_error_with_lemma_func(
                newcode, verus_error, 1, ["seq_subrange_ascend", "seq_subrange_all"]
            )[0]
            newcode = clean_code(newcode)
            newcode, _ = self.debug_type_error(newcode)
            if newcode:
                did_special_fix = True
                print(newcode)
                code = newcode

        if did_special_fix:
            return code
        else:
            return ""

    def repair_nonlinear_arith_error(
        self, code: str, verus_error: VerusError, num=1, temp=1.0
    ) -> str:
        """
        This function is used to repair the nonlinear arithmetic error
        """
        system = self.default_system
        instruction = """Your mission is to add assert statements into the given Rust function to help Verus prove non-linear properties.

Here are some principles that you have to follow:
Response with the Rust code only, do not include any explanation.
You should only add assertions with non-linear property if necessary in the following ways, and you should not make any other changes to the program.

#### 1. Nonlinear Arithmetic
Nonlinear arithmetic involves equations that multiply, divide, or take the remainder of integer variables (e.g., x * (y * z) == (x * y) * z). Verus can reason about nonlinear arithmetic, but it needs to be told when to do so. To do this, you need to add a special keyword `nonlinear_arith' to the assert statement.
For example, if we know that variable X equals k*k+2*k and that variable Y equals (k+1)*(k+1), to prove that X+1 equals Y, we have to write the following statement to help Verus:

    assert(X+1 == Y) by (nonlinear_arith)
        requires
            X == k*k+2*k,
            Y == (k+1)*(k+1),
            0 < k,
            k < N,
            N <= 300,
            {}

In this example, the `nonlinear_arith' would enable Verus to use its non-linear reasoning to prove X+1 equals Y. The requires statements should include all the information that is needed to reason about the assert statement, including any variable bound information that is need to prove no-arithmetic overflow.

#### 2. Nonlinear Arithmetic Overflow
Verus cannot prove that a non-linear expression does not overflow unless you tell it the range of the expression.
For example, if a non-linear expression x*x*x is used in the program, only tell Verus 0 <= x <= 10 is not enough, we have to write the following statement to help Verus prove no arithmetic overflow for x*x*x:

    assert(0 < x*x*x <= 10 * 10 * 10) by (nonlinear_arith)
        requires
            0 < x,
            x <= 10,
            {}

In this example, the `nonlinear_arith' keyword enables Verus to use its non-linear reasoning, and
the `requires' statements should include all the variable bound information needed to prove no-arithmetic overflow.

#### Task
Please check the given program, and add nonlinear_arith assertion for the following assertions:
"""
        nl_lines = get_nonlinear_lines(code, self.logger)
        if not nl_lines:
            return []
        filtered_nl_lines = []
        for _, (st, ed, text) in enumerate(nl_lines):
            if text in verus_error.get_text():
                filtered_nl_lines.append((st, ed, text))
        if not filtered_nl_lines:
            return []
        for i, (st, ed, text) in enumerate(filtered_nl_lines):
            instruction += f"{i + 1}. Lines {st}-{ed}:\n{text}\n"

        examples = []
        return self.llm.infer_llm(
            self.config.aoai_debug_model,
            instruction,
            examples,
            code,
            system,
            answer_num=num,
            max_tokens=4096,
            temp=temp,
        )

    def repair_assertion_error(self, code: str, verus_error: VerusError, num=1, temp=1.0) -> str:
        # TODO: if we plan to deal with non-linear algebra, this might be needed
        # nl_repairs = self.repair_nonlinear_arith_error(code, verus_error, num=num, temp=temp)
        # if nl_repairs:
        #    return nl_repairs

        # Check if this assertion error needs special API treatment
        newcode = self.repair_special_assertion_error(code, verus_error, num=num, temp=temp)
        if newcode:
            return [newcode]

        # Normal route of assertion fixing
        system = self.default_system
        instruction = """Your mission is to fix the assertion error for the following code. Basically, you should either introduce the necessary proof blocks before the location where the assertion fails, or, if the assertion is within a loop or after a loop, you may need to add appropriate loop invariants to ensure the assertion holds true.

Response with the Rust code only, do not include any explanation."""

        instruction = self.add_seq_knowledge(code, instruction)
        examples = self.get_examples("assert")
        query_template = "Failed assertion\n```\n{}```\n"
        query_template += "\nCode\n```\n{}```\n"

        error_trace = verus_error.trace[0]
        assertion_info = error_trace.get_text() + "\n"

        query = query_template.format(assertion_info, code)

        self._write_debug_dump("assert-query.txt", query)

        return self.llm.infer_llm(
            self.config.aoai_debug_model,
            instruction,
            examples,
            query,
            system,
            answer_num=num,
            max_tokens=4096,
            temp=temp,
        )

    def repair_assertion_error_with_lemma_func(
        self, code: str, verus_error: VerusError, num=1, lemmas=None, temp=1.0
    ) -> str:
        """
        Here, we potentially know what lemma functions to use.
        And, no need to implement new proof functions
        """

        if lemmas is None:
            lemmas = []
        suggested_lemma = ",".join(lemmas)

        system = self.default_system
        instruction = (
            "Your mission is to fix the assertion error for the following code by using existing lemma functions"
            + suggested_lemma
            + "\n Do NOT change the lemma functions!"
        )

        instruction += (
            "Please read the comment right before lemma function"
            + suggested_lemma
            + " and add invocation to the suggested lemma functions at the right place accordingly to prove the assertion. \n You should NOT change lemma functions and you should NOT add any new proof function. \n Response with the Rust code only, do not include any explanation."
        )

        examples = ""
        query_template = "Failed assertion\n```\n{}```\n"
        query_template += "\nCode\n```\n{}```\n"

        error_trace = verus_error.trace[0]
        assertion_info = error_trace.get_text() + "\n"
        query = query_template.format(assertion_info, code)

        return self.llm.infer_llm(
            self.config.aoai_generation_model,
            instruction,
            examples,
            query,
            system,
            answer_num=num,
            max_tokens=4096,
            temp=temp,
        )

    def repair_assertion_error_with_proof_func(
        self, code: str, verus_error: VerusError, num=1, temp=1.0
    ) -> str:
        system = self.default_system
        instruction = """Your mission is to fix the assertion error for the following code by creating the helper proof functions.

        Basically, you should determine what proof functions are needed to prove the current failed assertion, based on the related invariants already had. Then generate them and their invocations in the code just before the assertion.

Response with the Rust code only, do not include any explanation."""
        instruction = self.add_seq_knowledge(code, instruction)

        examples = self.get_examples("proof-func-middle")
        query_template = "Failed assertion\n```\n{}```\n"
        query_template += "\nCode\n```\n{}```\n"

        error_trace = verus_error.trace[0]
        assertion_info = error_trace.get_text() + "\n"
        query = query_template.format(assertion_info, code)

        return self.llm.infer_llm(
            self.config.aoai_generation_model,
            instruction,
            examples,
            query,
            system,
            answer_num=num,
            max_tokens=4096,
            temp=temp,
        )

    def repair_precond_error(self, code: str, verus_error: VerusError, num=1, temp=1.0) -> str:
        system = self.default_system
        instruction = """Your mission is to fix the precondition not satisfied error for the following code. Basically, you should add the proof blocks related to the pre-condition check just before the invocation of the function. Note, DO NOT change the proof function whose pre-condition is not satisfied. You can use the pre-conditions of the current function, invariants of the current loop, and the pre-conditions of the called functions to fix the error.

Response with the Rust code only, do not include any explanation."""
        instruction += "\n\n" + self.proof_block_info
        instruction = self.add_seq_knowledge(code, instruction)

        examples = self.get_examples("precond")
        query_template = "Failed pre-condition\n```\n{}```\n"
        query_template += "Failed location\n```\n{}```\n"
        query_template += "\nCode\n```{}```\n"

        precond_trace, location_trace = verus_error.trace[0], verus_error.trace[1]
        if location_trace.label == VerusErrorLabel.FailedThisPreCond:
            precond_trace, location_trace = location_trace, precond_trace

        pre_cond_info = precond_trace.get_text() + "\n"
        location_info = f"LIne {location_trace.lines[0]}-{location_trace.lines[1]}:\n"
        location_info += location_trace.get_text() + "\n"
        query = query_template.format(pre_cond_info, location_info, code)

        self._write_debug_dump("precond-query.txt", query)

        return self.llm.infer_llm(
            self.config.aoai_generation_model,
            instruction,
            examples,
            query,
            system,
            answer_num=num,
            max_tokens=4096,
            temp=temp,
        )

    # a special type of precondition error: vec len
    def repair_precond_veclen_error(
        self, code: str, verus_error: VerusError, num=1, temp=1.0
    ) -> str:
        system = self.default_system

        error_line = verus_error.trace[1].lines[0]
        error_code = verus_error.trace[1].get_text().strip()

        instruction = (
            "Your mission is to help Verus prove the array access in the expression `"
            + error_code.strip()
            + f"' is always in bound --- this expression is on Line {error_line}"
            + " of the following program. Basically, you should identify all the arrays accessed (e.g., A[k] or A.set(k,..)) in this expression `"
            + error_code.strip()
            + "' and add the following loop invariants for EACH array: 1. an invariant that specify the array length (i.e., A.len() == ...); 2. an invariant about the array index not under bound (e.g., k >= 0). \n"
        )

        instruction += """
        Response requirements:
        Respond with the verus code only, do not include any explanation.
        Respond with the whole program, not just the invariants you added.
        You should only add loop invariants, and you should NOT make any other changes to the program.
        You should NOT change function's pre condition or post conditions.
        """
        instruction = self.add_seq_knowledge(code, instruction)

        examples = []
        query = code

        self._write_debug_dump("precond-query.txt", query)

        return self.llm.infer_llm(
            self.config.aoai_generation_model,
            instruction,
            examples,
            query,
            system,
            answer_num=num,
            max_tokens=4096,
            temp=temp,
        )

    def repair_postcond_error(self, code: str, verus_error: VerusError, num=1, temp=1.0) -> str:
        system = self.default_system
        instruction = f"""Your mission is to fix the post-condition not satisfied error for the following code. There are several general ways to fix the error:

1. Add the proof blocks related to the post-condition at or just before the exit point where the post-condition failure occurred.
2. Modify the existing loop invariants to make them work for the post-condition.
3. If the function ends with a loop, make sure there is a loop invariant in that loop that reflects the post-condition `{verus_error.trace[0].get_highlights()[0]}'.

Response with the Rust code only, do not include any explanation."""
        instruction += "\n\n" + self.proof_block_info
        instruction = self.add_seq_knowledge(code, instruction)

        examples = self.get_examples("postcond")
        query_template = "Failed post-condition\n```\n{}```\n"
        query_template += "Failed location\n```\n{}```\n"
        query_template += "\nCode\n```{}```\n"

        location_trace, postcond_trace = verus_error.trace[0], verus_error.trace[1]
        if location_trace.label == VerusErrorLabel.FailedThisPostCond:
            location_trace, postcond_trace = postcond_trace, location_trace

        post_cond_info = f"Line {postcond_trace.lines[0]}-{postcond_trace.lines[1]}:\n"
        post_cond_info += postcond_trace.get_text() + "\n"
        location_info = f"Line {location_trace.lines[0]}-{location_trace.lines[1]}:\n"
        location_info += location_trace.get_text() + "\n"
        query = query_template.format(post_cond_info, location_info, code)

        self._write_debug_dump("postcond-query.txt", query)

        return self.llm.infer_llm(
            self.config.aoai_generation_model,
            instruction,
            examples,
            query,
            system,
            answer_num=num,
            max_tokens=4096,
            temp=temp,
        )

    def repair_invfail_front(self, code: str, verus_error: VerusError, num=1, temp=1.0) -> str:
        system = self.default_system

        error_trace = verus_error.trace[0]
        error_highlight = error_trace.get_highlights()[0]
        query_template = "Failed invariant before the loop\n```\n{}```\n"
        query_template += "\nCode\n```\n{}```\n"

        line_info = f"Line {error_trace.lines[0]}-{error_trace.lines[1]}:\n"
        inv_info = line_info + error_trace.get_text() + "\n"
        query = query_template.format(inv_info, code)

        # let's try some quick fixes first
        # Quick fix 1: array length is only specified in the loop where the array is used, but not in earlier arrays

        if ".len() ==" in error_highlight or ".len()==" in error_highlight:
            instruction = f"""Verus verification engine feels that loop invariant `{error_highlight}' in the following program does not hold at the beginning of its loop. If there are multiple loops in the program, please add `{error_highlight}' as a loop invariant to all preceeding loops. Otherwise, please correct this loop invariant or add it as an assert right before the loop it belongs to."""

            examples = []

            return self.llm.infer_llm(
                self.config.aoai_debug_model,
                instruction,
                examples,
                query,
                system,
                answer_num=num,
                max_tokens=4096,
                temp=temp,
            )

        else:
            # Quick fix 2:
            instruction = f"""Verus verification engine feels that loop invariant `{error_trace.get_highlights()[0]}' in the following program does not hold at the beginning of its loop. Please modify this loop invariant to be conditioned on the loop index. For example, if the loop index variable INDEX starts from value A and increases for later iterations, please change this loop invariant to be `INDEX > A ==> {error_trace.get_highlights()[0]}'. Response with the Rust code only, do not include any explanation."""

            examples = []

            fix_code = self.llm.infer_llm(
                self.config.aoai_debug_model,
                instruction,
                examples,
                query,
                system,
                answer_num=num,
                max_tokens=4096,
                temp=temp,
            )[0]
            fix_code = clean_code(fix_code)

            self.logger.info("Here is the quick fix output")
            # DEBUG only
            print(fix_code)

            is_safe_change = code_change_is_safe(
                code, fix_code, self.config.verus_path, self.logger
            )
            if not is_safe_change:
                self.logger.info(
                    "[repair_invfail_front] Quick fix is not safe. Move on to more general fix."
                )
            else:
                hdn_err, hdn_fix_code = self.hdn.run(fix_code)

                if len(hdn_err) == 0:
                    self.logger.info(
                        "[repair_invfail_front] Quick fix solved all remaining verification errors!"
                    )
                    returns = []
                    returns.append(fix_code)
                    return returns
                else:
                    self.logger.info(
                        "[repair_invfail_front] Quick fix is not effective. Move on to more general fix attempts."
                    )

        # let's try more general fix now
        instruction = """Your mission is to fix the invariant not satisfied error before the loop for the following code. Here are several general and possible ways to fix the error:

1. Add the assertions related to the failed loop invariant before the loop body.
2. If there are multiple loops and you believe the failed invariant is also true in preceeding loops , you should add the failed invariant to those preceeding loops as well.
3. If you believe the failed invariant is incorrect or not needed, you can modify it or delete it.

Please think twice about which way is the best to fix the error!

Response with the Rust code only, do not include any explanation."""
        instruction += "\n\n" + self.proof_block_info
        instruction = self.add_seq_knowledge(code, instruction)

        examples = self.get_examples("inv-front")

        return self.llm.infer_llm(
            self.config.aoai_debug_model,
            instruction,
            examples,
            query,
            system,
            answer_num=num,
            max_tokens=4096,
            temp=temp,
        )

    def repair_invfail_end(self, code: str, verus_error: VerusError, num=1, temp=1.0) -> str:
        system = self.default_system
        instruction = """Your mission is to fix the invariant not satisfied error at end of the loop for the following code. Basically, you should add the assertion (in proof block) of the failed loop invariant at the end of the loop. DO NOT change the existing proof functions. If you think the failed invariant is incorrect, you can delete/correct it.

Response with the Rust code only, do not include any explanation."""
        instruction += "\n\n" + self.proof_block_info
        instruction = self.add_seq_knowledge(code, instruction)

        examples = self.get_examples("inv-end")
        query_template = "Failed invariant at end of the loop\n```\n{}```\n"
        query_template += "\nCode\n```\n{}```\n"

        error_trace = verus_error.trace[0]
        line_info = f"Line {error_trace.lines[0]}-{error_trace.lines[1]}:\n"
        inv_info = line_info + error_trace.get_text() + "\n"
        query = query_template.format(inv_info, code)

        return self.llm.infer_llm(
            self.config.aoai_debug_model,
            instruction,
            examples,
            query,
            system,
            answer_num=num,
            max_tokens=4096,
            temp=temp,
        )

    # TODO: this agent's instruction is a bit verbose
    def repair_arithmetic_flow(self, code: str, verus_error: VerusError, num=1, temp=1.0) -> str:
        system = self.default_system

        error_trace = verus_error.trace[0]

        instruction = f"""Your mission is to fix the arithmetic underflow/overflow error for the following code.
        Basically, for each variable involved in the expression ` {error_trace.get_highlights()[0]}' in line `{error_trace.get_text().strip()}' of the program, there are several general ways to fix the error:

0. Make sure the value of EVERY variable involved in this expression is specified as a loop invariant.
1. Add a bound for the whole expression `{error_trace.get_highlights()[0]}' as a loop invariant or as an assert. This
bound can be a constant value, or another expression whose bound has been specified through loop invariants or asserts.
2. Or, add BOTH a lower bound (i.e. x > ..., x >= ...) AND an upper bound (i.e., x < ..., x <= ...) as an assertion or a loop invariant if they are in a loop body for EACH variable involved in the expression {error_trace.get_highlights()[0]}. If the variable is a loop index variable, make sure that its lower bound (e.g., its initial value at the beginning of the loop) and upper bound (based on the loop-exit condition) are specified as loop invariants. You may use the loop index variable in the invariant.

Do not miss any variable in `{error_trace.get_highlights()[0]}', and do NOT add bound information related to any other variables. Please do not change function post-conditions.
        """

        instruction += """Response requirements:
Respond with the verus code only, do not include any explanation.
You should only add loop invariants, and you should NOT make any other changes to the program.

Hint for the upper bound:
1. For the lower/upper bound, you don't always need to find the exact or strict value. Your mission is to find a provable bound for Verus, which is usually based on the loop index, like `car <= CONSTANT * index`.
2. If the expression involves the loop index or is updated during each loop iteration, use the loop index variable as the upper or lower bound in the invariant instead of using the CONSTANT alone!
3. If there is a non-linear upper bound, you can use a constant to represent part of the expression (e.g., a * CONSTANT_RELATED_TO_b) to make it linear. However, ensure that at least one variable remains (DO NOT USE A CONSTANT TO REPLACE THE WHOLE NON-LINEAR). This approach makes it easier to prove.
4. You may use conditional loop invariants to specify the upper bound based on the loop index. For example, `i > 0 ==> x < 10 * i` means that if `i` is greater than 0, then `x` is less than 10 times `i`.
        """

        examples = self.get_examples("aritherr")

        # TODO: I probably should make this `if' condition more strict to capture the recursive expression bound situation
        if "decreases" in code:
            instruction = f"""Your mission is to fix the arithmetic underflow/overflow error for the following code.
        Basically, add an assertion about the bound of `{error_trace.get_highlights()[0]}' right BEFORE the line `{error_trace.get_text()}' in the code. Note that, if the value of this expression is related to a recursively defined spec function in the program, generate a lemma function that shows the monotonicity of this expression could help prove its bound. Please look at the example below to see how a monotonicity lemma function can help eliminate arithmetic underflow/overflow concerns.
        """
            examples = self.get_examples("aritherr-recur")

        query_template = "Arithmetic underflow/overflow \n```\n{}```\n"
        query_template += "\nCode\n```\n{}```\n"

        line_info = f"Line {error_trace.lines[0]}-{error_trace.lines[1]}:\n"
        inv_info = line_info + error_trace.get_text() + "\n"
        query = query_template.format(inv_info, code)

        self._write_debug_dump("arith-query.txt", instruction)

        return self.llm.infer_llm(
            self.config.aoai_generation_model,
            instruction,
            examples,
            query,
            system,
            answer_num=num,
            max_tokens=4096,
            temp=temp,
        )

    def repair_mismatched_type(self, code: str, verus_error: VerusError, num=1, temp=1.0) -> str:
        del temp  # unused
        new_code, errors = self.debug_type_error(code, verus_error, num)
        if errors == 0 and new_code:
            return [new_code]

        codes = self.repair_default(code, verus_error, num)
        for i, new_code in enumerate(codes):
            new_code = clean_code(new_code)
            new_code, _ = self.debug_type_error(new_code)
            codes[i] = new_code
        return codes

    def repair_plain_text(self, code: str, error_text: str, num=1, temp=1.0) -> str:
        system = self.default_system
        instruction = """Your mission is to fix the error for the following code. Basically, you should add/modify/delete the proof blocks, assertions and loop invariants related to the errors."""

        instruction += "\n\n" + self.proof_block_info
        instruction = self.add_seq_knowledge(code, instruction)

        examples = self.get_examples("all")
        query_template = "Errors\n```\n{}```\n"
        query_template += "\nCode\n```\n{}```\n"

        query = query_template.format(error_text, code)

        return self.llm.infer_llm(
            self.config.aoai_generation_model,
            instruction,
            examples,
            query,
            system,
            answer_num=num,
            max_tokens=4096,
            temp=temp,
        )

    def repair_default(self, code: str, verus_error: VerusError, num=1, temp=1.0) -> str:
        """
        The default function to repair the code.
        """
        system = self.default_system
        instruction = """Your mission is to fix the error for the following code. Basically, you should add/modify/delete the proof blocks, assertions and loop invariants related to the error.

Response with the Rust code only, do not include any explanation."""
        instruction += "\n\n" + self.proof_block_info
        instruction = self.add_seq_knowledge(code, instruction)

        examples = self.get_examples("default")
        query_template = "{}\n```\n{}```\n"
        query_template += "\nCode\n```\n{}```\n"

        error_text = verus_error.error_text
        if len(verus_error.trace) == 0:
            self.logger.warning("No trace information in the error.")
            return code
        trace = verus_error.trace[0]
        line_info = f"Line {trace.lines[0]}-{trace.lines[1]}:\n"
        error_info = line_info + verus_error.get_text() + "\n"

        query = query_template.format(error_text, error_info, code)
        self._write_debug_dump("default-query.txt", instruction + "\n" + query)

        return self.llm.infer_llm(
            self.config.aoai_generation_model,
            instruction,
            examples,
            query,
            system,
            answer_num=num,
            max_tokens=4096,
            temp=temp,
        )

    def repair_remove_lines(self, code: str, verus_error: VerusError, num=1) -> str:
        if verus_error in [VerusErrorType.PreCondFail, VerusErrorType.PostCondFail]:
            return code

        if len(verus_error.trace) == 0:
            self.logger.warning("No trace information in the error.")
            code = self.houdini.run(code)
            return code

        trace = verus_error.trace[0]
        code_lines = code.splitlines()
        new_code_lines = code_lines[: trace.lines[0] - 1] + code_lines[trace.lines[1] :]

        return "\n".join(new_code_lines)

    def show_all_failures(self, verus_errors):
        """
        for deubgging
        """
        self.logger.info(f"There are in total {len(verus_errors)} verus errors.")
        for verus_error in verus_errors:
            self.logger.info(verus_error.error)
            self.logger.info(verus_error.trace[0].get_highlights()[0])
            self.logger.info(f"on Line {verus_error.trace[0].lines[0]}")

    def _serialize_failure(self, failure, idx=None):
        return {
            "idx": idx,
            "error_type": failure.error.name,
            "error_text": failure.error_text,
            "trace_text": failure.get_text(snippet=False),
        }

    def _failure_signature(self, failure):
        # Deduplicate by concrete failure instance instead of error type only.
        # This keeps different failures of the same type as separate branches.
        return (
            failure.error.name,
            (failure.error_text or "").strip(),
            (failure.get_text(snippet=False) or "").strip(),
        )

    def _dedupe_failures(self, failures):
        seen = set()
        deduped = []
        for failure in failures:
            sig = self._failure_signature(failure)
            if sig in seen:
                continue
            seen.add(sig)
            deduped.append(failure)
        return deduped

    def _dedupe_failures_by_type(self, failures):
        seen_types = set()
        deduped = []
        for failure in failures:
            error_type = failure.error.name
            if error_type in seen_types:
                continue
            seen_types.add(error_type)
            deduped.append(failure)
        return deduped

    def _score_tuple(self, score):
        if score.compilation_error:
            return (-1, -1)
        return (score.verified, score.errors)

    def _state_path_part(self, state_id):
        if isinstance(state_id, str) and ":" in state_id:
            return state_id.split(":", 1)[1]
        return str(state_id)

    def _child_state_id(self, parent_state_id, child_round_id, branch_idx):
        parent_path = self._state_path_part(parent_state_id)
        return f"r{child_round_id}:{parent_path}.b{branch_idx}"

    def _state_id_for_fs(self, state_id):
        # Keep IDs readable in file paths while avoiding reserved separators.
        return str(state_id).replace(":", "_")

    def _eval_with_details(self, code, func_name=None):
        veval = VEval(code, self.logger)
        veval.eval(func_name=func_name)
        score = veval.get_score()
        failures = veval.get_failures()
        veval.eval(func_name=func_name, json_mode=False)
        full_output = veval.rustc_out + veval.verus_out
        return veval, score, failures, full_output

    def _pick_repair_func(
        self,
        cur_failure,
        failed_last_time,
        simple_repair_per_failure,
        label_repair_func,
        failed_repair_func,
    ):
        if (
            failed_last_time > simple_repair_per_failure
            and cur_failure.error in failed_repair_func
        ):
            return failed_repair_func[cur_failure.error]
        if cur_failure.error not in label_repair_func:
            return self.repair_default
        return label_repair_func[cur_failure.error]

    def _candidate_is_effective(
        self,
        base_score,
        candidate_score,
        base_failure_count,
        candidate_failure_count,
        candidate_failure_count_finer,
        failed_last_time,
    ):
        if candidate_score.is_correct():
            return True, 0
        if (
            candidate_failure_count < base_failure_count
            and candidate_score.is_good_repair(base_score)
        ):
            return True, 0
        if candidate_failure_count == base_failure_count and candidate_score > base_score:
            # Align with baseline single-path semantics:
            # this branch sets failed_last_time = max(-1, failed_last_time - 1),
            # then the round-end always does failed_last_time += 1, resulting in
            # the next-round counter staying unchanged.
            return True, failed_last_time
        if (
            failed_last_time > 0
            and candidate_failure_count_finer < base_failure_count
            and candidate_score.is_good_repair(base_score)
        ):
            return True, 0
        return False, failed_last_time + 1

    def get_one_failure(self, verus_errors):
        """
        This function tries to prioritize among a group of Verus errors
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

        # default
        return verus_errors[0]

    # Designed for Ablation Study
    def repair_veval_in_one(self, code, max_attempt=5, func_name=None, temp_dir=None, temp=1.0):
        self.logger.warning("All-in-one repair is used!!!")
        self._set_debug_dump_dir(temp_dir)

        if "loop_isolation(false)" not in code:
            self.logger.warning(
                "Loop isolation is not found in the code. Inserting loop isolation."
            )
            code = insert_loop_isolation(code)
        attempt = 0
        while attempt < max_attempt:
            attempt += 1

            veval = VEval(code, self.logger)
            veval.eval(func_name=func_name)
            score = veval.get_score()
            if score.is_correct():
                self.logger.info(f"All errors are fixed within {attempt - 1} steps!!!")
                break

            veval.eval(func_name=func_name, json_mode=False)
            error_text = veval.rustc_out + veval.verus_out

            self.logger.info(f"Step {attempt}")
            self.logger.info(f"Current score: {score}")
            repaired_candidates = self.repair_plain_text(code, error_text, num=5, temp=temp)

            for i, new_code in enumerate(repaired_candidates):
                if not new_code:
                    self.logger.warning("An unrepairable error was encountered.")
                    continue
                new_code = clean_code(new_code)
                new_code = new_code.replace("```", "")

                new_veval = VEval(new_code, self.logger)
                new_veval.eval(func_name=func_name)
                new_score = new_veval.get_score()

                if new_score.is_correct():
                    self.logger.info(f"All errors are fixed within {attempt} steps!!!")
                    return new_code

                is_safe_change = code_change_is_safe(
                    code, new_code, self.config.verus_path, self.logger
                )

                if temp_dir:
                    self.logger.info(f"repair-{attempt}-{i} file generated")
                    self.logger.info(f"{new_score}")
                    new_veval.eval(func_name=func_name, json_mode=False)
                    err_lines = (new_veval.rustc_out + new_veval.verus_out).splitlines()
                    with open(os.path.join(temp_dir, f"repair-{attempt}-{i}.rs"), "w") as f:
                        f.write(new_code)
                        f.write("\n\n// ")
                        f.write("\n// ".join(err_lines))
                        f.write("\n\n// " + str(new_score).replace("\n", "\n// "))
                        f.write("\n// Safe: " + str(is_safe_change))

                if not is_safe_change:
                    self.logger.warning("The repair is not safe.")
                    continue

                if new_score.is_correct():
                    self.logger.info(f"All errors are fixed within {attempt} steps!!!")
                    return new_code

                # Test: adding a houridni run after each repair, just in case a correct version actually already exists
                hdn_failures, hdn_code = self.hdn.run(new_code)
                if len(hdn_failures) == 0 and hdn_code:
                    self.logger.info("Verus succeeded with hdn!!")
                    return hdn_code

                code = new_code
        return code

    # Designed for ablation study where only default-repair agent is used
    def repair_veval_uniform(self, code, max_attempt=5, func_name=None, temp_dir=None, temp=1.0):
        self.logger.warning("Uniform repair is used!!!")
        self._set_debug_dump_dir(temp_dir)
        attempt = 0
        failed_last_time = 0
        while attempt < max_attempt:
            attempt += 1

            veval = VEval(code, self.logger)
            veval.eval(func_name=func_name)
            score = veval.get_score()
            if score.is_correct():
                self.logger.info(f"All errors are fixed within {attempt - 1} steps!!!")
                break

            failures = veval.get_failures()
            if len(failures) == 0:
                self.logger.info(code)
                raise Exception("No error found in the code, but the code is still incorrect.")

            cur_failure = self.get_one_failure(failures)
            num_cur_failure = len([f for f in failures if f.error == cur_failure.error])

            repair_func = self.repair_default
            self.logger.info(f"Step {attempt}: {cur_failure.error} with num={num_cur_failure}.")
            self.logger.info(f"Current score: {score}")

            repair_num = 5 if failed_last_time > 0 else 3
            repaired_candidates = repair_func(code, cur_failure, num=repair_num, temp=temp)
            for i, new_code in enumerate(repaired_candidates):
                if not new_code:
                    self.logger.warning("An unrepairable error was encountered.")
                    continue
                new_code = clean_code(new_code)
                new_code = new_code.replace("```", "")

                new_veval = VEval(new_code, self.logger)
                new_veval.eval(func_name=func_name)
                new_score = new_veval.get_score()

                if new_score.is_correct():
                    self.logger.info(f"All errors are fixed within {attempt} steps!!!")
                    return new_code

                is_safe_change = code_change_is_safe(
                    code, new_code, self.config.verus_path, self.logger
                )

                if temp_dir:
                    cur_failure_str = cur_failure.error.name
                    err_lines = cur_failure.get_text().splitlines()
                    self.logger.info(f"repair-{attempt}-{i} file generated")
                    self.logger.info(f"{new_score}")
                    with open(
                        os.path.join(temp_dir, f"repair-{attempt}-{i}-{cur_failure_str}.rs"),
                        "w",
                    ) as f:
                        f.write(new_code)
                        f.write("\n\n// ")
                        f.write("\n// ".join(err_lines))
                        f.write("\n\n// " + str(new_score).replace("\n", "\n// "))
                        f.write("\n// Safe: " + str(is_safe_change))

                if not is_safe_change:
                    self.logger.warning("The repair is not safe.")
                    continue

                if new_score.is_correct():
                    self.logger.info(f"All errors are fixed within {attempt} steps!!!")
                    return new_code

                # Test: adding a houridni run after each repair, just in case a correct version actually already exists
                hdn_failures, hdn_code = self.hdn.run(new_code)
                if len(hdn_failures) == 0 and hdn_code:
                    self.logger.info("Verus succeeded with hdn!!")
                    return hdn_code

                # We need to tell whether the repair is effective
                new_num_cur_failure = len(
                    [f for f in new_veval.get_failures() if f.error == cur_failure.error]
                )
                new_num_cur_failure_finer = len(
                    [
                        f
                        for f in new_veval.get_failures()
                        if f.error == cur_failure.error and f.get_text() == cur_failure.get_text()
                    ]
                )
                if new_num_cur_failure < num_cur_failure and new_score.is_good_repair(score):
                    code = new_code
                    self.logger.info(
                        f"Step {attempt}: {cur_failure.error} is fixed. Proof updated {attempt}-{i}."
                    )
                    failed_last_time = -1
                    break
                if new_num_cur_failure == num_cur_failure and new_score > score:
                    code = new_code
                    self.logger.info(
                        f"Step {attempt}: {cur_failure.error} is partially fixed. Proof updated {attempt}-{i}."
                    )
                    failed_last_time = max(-1, failed_last_time - 1)
                    break
                if (
                    failed_last_time > 0
                    and new_num_cur_failure_finer < num_cur_failure
                    and new_score.is_good_repair(score)
                ):
                    code = new_code
                    self.logger.info(
                        f"Step {attempt}: {cur_failure.error} is fixed based on finer-grained check."
                    )
                    failed_last_time = -1
                    break
            failed_last_time += 1
        return code

    # This is the default repair function
    def repair_veval(
        self,
        code,
        max_attempt=5,
        func_name=None,
        temp_dir=None,
        temp=1.0,
        source_metadata=None,
    ):
        label_repair_func = {
            VerusErrorType.PreCondFail: self.repair_precond_error,
            VerusErrorType.PostCondFail: self.repair_postcond_error,
            VerusErrorType.InvFailFront: self.repair_invfail_front,
            VerusErrorType.InvFailEnd: self.repair_invfail_end,
            VerusErrorType.AssertFail: self.repair_assertion_error,
            VerusErrorType.ArithmeticFlow: self.repair_arithmetic_flow,
            VerusErrorType.MismatchedType: self.repair_mismatched_type,
            VerusErrorType.PreCondFailVecLen: self.repair_precond_veclen_error,
        }
        failed_repair_func = {
            VerusErrorType.AssertFail: self.repair_assertion_error_with_proof_func,
        }

        # Let's first add loop-isolation to see if it solves all the problem
        if "loop_isolation(false)" not in code:
            code = insert_loop_isolation(code)

        if temp_dir is None:
            temp_dir = os.path.join("output-intermediate-temp-" + time.strftime("%Y%m%d-%H%M%S"))
            os.makedirs(temp_dir, exist_ok=True)
        self._set_debug_dump_dir(temp_dir)

        run_id = f"{int(time.time() * 1000)}-{os.getpid()}"
        if source_metadata is None:
            source_metadata = {}
        recorder = RepairTraceRecorder(temp_dir, run_id)
        recorder.log_event(
            {
                "event": "run_start",
                "max_attempt": max_attempt,
                "func_name": func_name,
                "temp": temp,
            }
        )

        self.logger.info("Start multi-state repair")
        simple_repair_per_failure = 3
        remove_lines_per_failure = 5

        active_states = [
            {
                "state_id": "r0:root",
                "parent_state_id": None,
                "code": code,
                "failed_last_time": 0,
            }
        ]
        finished_states = []
        best_state = None

        for attempt in range(1, max_attempt + 1):
            if len(active_states) == 0:
                break
            self.logger.info(f"Round {attempt}: active states = {len(active_states)}")
            next_states = []

            for state in active_states:
                state_id = state["state_id"]
                parent_state_id = state["parent_state_id"]
                state_code = state["code"]
                failed_last_time = state["failed_last_time"]
                state_id_fs = self._state_id_for_fs(state_id)

                state_code_relpath = f"states/round-{attempt}/{state_id_fs}.rs"
                state_code_path = recorder.write_code(state_code_relpath, state_code)
                _, score, failures, full_output = self._eval_with_details(state_code, func_name=func_name)

                if best_state is None or score > best_state["score"]:
                    best_state = {"code": state_code, "score": score}

                dedup_failures = self._dedupe_failures(failures)
                branch_failures = self._dedupe_failures_by_type(dedup_failures)
                default_failure = self.get_one_failure(failures) if len(failures) > 0 else None
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
                            self._serialize_failure(f, idx=i)
                            for i, f in enumerate(failures)
                        ],
                        "failures_dedup": [
                            self._serialize_failure(f, idx=i)
                            for i, f in enumerate(dedup_failures)
                        ],
                        "is_terminal": score.is_correct(),
                    }
                )

                if score.is_correct():
                    finished_states.append({"code": state_code, "score": score, "state_id": state_id})
                    continue

                if len(branch_failures) == 0:
                    continue

                for failure_idx, cur_failure in enumerate(branch_failures):
                    failed_last_time_for_branch = failed_last_time
                    num_cur_failure = len(
                        [f for f in failures if f.error == cur_failure.error]
                    )
                    all_failed_for_branch = True
                    branch_winner = None
                    branch_winner_score = None
                    branch_winner_state_failed_last_time = failed_last_time_for_branch + 1
                    repair_func_name = "repair_remove_lines"

                    # Align with baseline fallback behavior:
                    # after enough consecutive failures, try line-removal first.
                    if failed_last_time_for_branch > remove_lines_per_failure:
                        removed_code = self.repair_remove_lines(state_code, cur_failure)
                        failed_last_time_for_branch = 1
                        branch_winner_state_failed_last_time = failed_last_time_for_branch + 1
                        if removed_code:
                            removed_code = clean_code(removed_code).replace("```", "")
                            if removed_code:
                                _, removed_score, _, _ = self._eval_with_details(
                                    removed_code, func_name=func_name
                                )
                                if removed_score.is_correct():
                                    branch_winner = removed_code
                                    branch_winner_score = removed_score
                                    branch_winner_state_failed_last_time = 0

                    if branch_winner is None:
                        # Keep candidate generation count aligned with baseline AutoVerus:
                        # - normal round: generate 3 candidates
                        # - after previous failure: generate 5 candidates
                        repair_num = 5 if failed_last_time_for_branch > 0 else 3
                        # repair_num = 2
                        repair_func = self._pick_repair_func(
                            cur_failure,
                            failed_last_time_for_branch,
                            simple_repair_per_failure,
                            label_repair_func,
                            failed_repair_func,
                        )
                        repair_func_name = repair_func.__name__
                        self.logger.info(
                            f"Round {attempt} State {state_id} Branch {failure_idx}: "
                            f"{cur_failure.error.name} with num={repair_num} by {repair_func_name}"
                        )

                        recorder.log_event(
                            {
                                "event": "branch_start",
                                "round_id": attempt,
                                "state_id": state_id,
                                "parent_state_id": parent_state_id,
                                "chosen_failure": self._serialize_failure(
                                    cur_failure, idx=failure_idx
                                ),
                                "repair_func": repair_func_name,
                                "repair_num": repair_num,
                            }
                        )

                        repaired_candidates = repair_func(
                            state_code, cur_failure, num=repair_num, temp=temp
                        )

                        for cand_idx, candidate_code in enumerate(repaired_candidates):
                            if not candidate_code:
                                self.logger.warning("An unrepairable error was encountered.")
                                continue

                            candidate_code = clean_code(candidate_code)
                            candidate_code = candidate_code.replace("```", "")
                            if not candidate_code:
                                self.logger.warning("Empty candidate code.")
                                continue

                            _, cand_score, cand_failures, cand_full_output = self._eval_with_details(
                                candidate_code, func_name=func_name
                            )

                            if cand_score.compilation_error and len(cand_failures) > 0:
                                cand_prioritized_failure = self.get_one_failure(cand_failures)
                                self.logger.info(
                                    "Candidate has compilation issue, trying one-step type repair."
                                )
                                fixed_codes = self.repair_mismatched_type(
                                    candidate_code, cand_prioritized_failure, num=1
                                )
                                if len(fixed_codes) > 0:
                                    candidate_code = clean_code(fixed_codes[0]).replace("```", "")
                                    if candidate_code:
                                        _, cand_score, cand_failures, cand_full_output = (
                                            self._eval_with_details(
                                                candidate_code, func_name=func_name
                                            )
                                        )

                            is_safe_change = code_change_is_safe(
                                state_code, candidate_code, self.config.verus_path, self.logger
                            )

                            candidate_code_relpath = (
                                f"candidates/round-{attempt}/{state_id_fs}/"
                                f"failure-{failure_idx}-"
                                f"{cur_failure.error.name}/cand-{cand_idx}.rs"
                            )
                            candidate_code_path = recorder.write_code(
                                candidate_code_relpath, candidate_code
                            )

                            cand_failure_count = len(
                                [
                                    f
                                    for f in cand_failures
                                    if f.error == cur_failure.error
                                ]
                            )
                            cand_failure_count_finer = len(
                                [
                                    f
                                    for f in cand_failures
                                    if f.error == cur_failure.error
                                    and f.get_text() == cur_failure.get_text()
                                ]
                            )
                            accepted, next_failed_last_time = self._candidate_is_effective(
                                score,
                                cand_score,
                                num_cur_failure,
                                cand_failure_count,
                                cand_failure_count_finer,
                                failed_last_time_for_branch,
                            )

                            # Keep candidate acceptance aligned with baseline behavior:
                            # safe check must pass before considering correctness/effectiveness.
                            accepted_for_branch = False
                            hdn_succeeded = False
                            hdn_score = None
                            if not is_safe_change:
                                pass
                            else:
                                all_failed_for_branch = False
                                if cand_score.is_correct():
                                    accepted_for_branch = True
                                    branch_winner = candidate_code
                                    branch_winner_score = cand_score
                                    branch_winner_state_failed_last_time = 0
                                else:
                                    # Try Houdini after each safe repair candidate.
                                    hdn_failures, hdn_candidate = self.hdn.run(candidate_code)
                                    if len(hdn_failures) == 0 and hdn_candidate:
                                        hdn_succeeded = True
                                        _, hdn_score, _, _ = self._eval_with_details(
                                            hdn_candidate, func_name=func_name
                                        )
                                        accepted_for_branch = True
                                        branch_winner = hdn_candidate
                                        branch_winner_score = hdn_score
                                        branch_winner_state_failed_last_time = 0
                                    elif accepted:
                                        accepted_for_branch = True
                                        branch_winner = candidate_code
                                        branch_winner_score = cand_score
                                        branch_winner_state_failed_last_time = (
                                            next_failed_last_time
                                        )

                            recorder.log_event(
                                {
                                    "event": "candidate_evaluated",
                                    "round_id": attempt,
                                    "state_id": state_id,
                                    "parent_state_id": parent_state_id,
                                    "chosen_failure": self._serialize_failure(
                                        cur_failure, idx=failure_idx
                                    ),
                                    "repair_func": repair_func_name,
                                    "candidate_id": cand_idx,
                                    "code_path": state_code_path,
                                    "code_text": state_code,
                                    "score_tuple": self._score_tuple(score),
                                    "raw_score": str(score),
                                    "verus_output_full": full_output,
                                    "failures_dedup": [
                                        self._serialize_failure(f, idx=i)
                                        for i, f in enumerate(dedup_failures)
                                    ],
                                    "candidate_code_path": candidate_code_path,
                                    "candidate_code_text": candidate_code,
                                    "candidate_score_tuple": self._score_tuple(cand_score),
                                    "candidate_passed": cand_score.is_correct(),
                                    "candidate_raw_score": str(cand_score),
                                    "candidate_verus_output_full": cand_full_output,
                                    "is_safe_change": is_safe_change,
                                    "accepted_for_branch": accepted_for_branch,
                                    "hdn_succeeded": hdn_succeeded,
                                    "hdn_score_tuple": self._score_tuple(hdn_score)
                                    if hdn_score
                                    else None,
                                }
                            )

                            if accepted_for_branch:
                                break

                    winner_source = "accepted_candidate"
                    fallback_reason = None

                    # Path-level fallback/early-stop in tree search:
                    # - if all candidates fail repeatedly, prune this branch only
                    # - otherwise keep original code and continue this branch
                    if branch_winner is None:
                        next_failed_last_time_for_branch = failed_last_time_for_branch + 1
                        if (
                            all_failed_for_branch
                            and next_failed_last_time_for_branch >= remove_lines_per_failure
                        ):
                            winner_source = "pruned_path_early_stop"
                            fallback_reason = "path_early_stop_all_candidates_failed"
                            recorder.log_event(
                                {
                                    "event": "branch_selected",
                                    "round_id": attempt,
                                    "state_id": state_id,
                                    "parent_state_id": parent_state_id,
                                    "chosen_failure": self._serialize_failure(
                                        cur_failure, idx=failure_idx
                                    ),
                                    "repair_func": repair_func_name,
                                    "winner_exists": False,
                                    "winner_state_id": None,
                                    "winner_source": winner_source,
                                    "fallback_reason": fallback_reason,
                                }
                            )
                            continue

                        branch_winner = state_code
                        branch_winner_score = score
                        branch_winner_state_failed_last_time = next_failed_last_time_for_branch
                        winner_source = "carry_forward_original"
                        fallback_reason = "no_accepted_candidate_keep_original"

                    new_state_id = self._child_state_id(
                        parent_state_id=state_id, child_round_id=attempt, branch_idx=failure_idx
                    )
                    winner_code_relpath = (
                        f"states/round-{attempt + 1}/{self._state_id_for_fs(new_state_id)}.rs"
                    )
                    winner_code_path = recorder.write_code(winner_code_relpath, branch_winner)
                    _, winner_score, winner_failures, winner_output = self._eval_with_details(
                        branch_winner, func_name=func_name
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
                                cur_failure, idx=failure_idx
                            ),
                            "repair_func": repair_func_name,
                            "winner_exists": True,
                            "winner_state_id": new_state_id,
                            "winner_source": winner_source,
                            "fallback_reason": fallback_reason,
                            "candidate_code_path": winner_code_path,
                            "candidate_code_text": branch_winner,
                            "candidate_score_tuple": self._score_tuple(winner_score),
                            "candidate_passed": winner_score.is_correct(),
                            "candidate_raw_score": str(winner_score),
                            "candidate_verus_output_full": winner_output,
                        }
                    )

                    # Dataset-oriented sample: one row per successful repair change in any branch.
                    is_default_chose = bool(
                        default_failure is not None
                        and cur_failure.error == default_failure.error
                    )
                    dataset_id = (
                        f"autoverus-{source_metadata.get('task_name')}-"
                        f"{source_metadata.get('llm_name')}-"
                        f"{source_metadata.get('shot')}-"
                        f"{source_metadata.get('generation_candidate_id')}-"
                        f"{state_id}-{new_state_id}"
                    )
                    default_path = None
                    if is_default_chose:
                        default_path = (
                            f"autoverus-{source_metadata.get('task_name')}-"
                            f"{source_metadata.get('llm_name')}-"
                            f"{source_metadata.get('shot')}-"
                            f"{source_metadata.get('generation_candidate_id')}-"
                            f"{state_id}"
                        )
                    # Dataset 记录结构（每条表示一次“分支修复变化”）：
                    # - id: 本次修复变化的唯一标识
                    # - method: 修复方法/框架名称
                    # - dataset_name/task_name: 从输入路径/文件名解析出的数据集名与任务名
                    # - llm_name/shot/generation_candidate_id: 输入样本对应的生成元信息
                    # - round_id/source_state_id/parent_state_id: 本次修复在搜索树中的位置
                    # - source_*: 修复前的代码/分数/Verus输出/失败快照
                    # - source_failure_types: 修复前用于分支展开的错误类型列表
                    # - chosen_failure_type: 本条记录对应分支选择的错误类型
                    # - is_default_chose/default_path: 是否与 get_one_failure() 的默认选择一致
                    # - repaired_*: 分支 winner 修复后的代码/分数/Verus输出/失败快照
                    # - repaired_failure_types: 修复后的错误类型列表
                    # - fallback_reason: 兜底选优来源说明（若触发兜底）
                    # - is_carry_forward_original: 是否沿用原 state 进入下一状态
                    # - repair_llm_name: 本轮修复实际使用的模型名
                    '''
                    数据id
                    修复方法/框架名称
                    数据集名称
                    任务名称
                    模型名称
                    生成是fewshot还是zeroshot
                    生成候选id
                    修复模型名称

                    修复轮次
                    源状态id
                    父状态id
                    源代码路径
                    源代码文本
                    源分数（verify num, error num）
                    源Verus输出
                    源错误列表
                    源错误类型列表

                    选择错误类型
                    是否与autoverus的默认选择一致
                    默认选择路径

                    修复状态id
                    修复代码路径
                    修复代码文本
                    修复分数（verify num, error num）
                    修复Verus输出
                    修复错误列表
                    修复错误类型列表
                    是否沿用原state进入下一个状态
                    '''
                    recorder.log_dataset_entry(
                        {
                            "id": dataset_id,
                            "method": "autoverus",
                            "dataset_name": source_metadata.get("dataset_name"),
                            "task_name": source_metadata.get("task_name"),
                            "llm_name": source_metadata.get("llm_name"),
                            "shot": source_metadata.get("shot"),
                            "generation_candidate_id": source_metadata.get(
                                "generation_candidate_id"
                            ),
                            "repair_llm_name": self.config.aoai_generation_model,
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
                            "chosen_failure_type": cur_failure.error.name,
                            "is_default_chose": is_default_chose,
                            "default_path": default_path,
                            "is_carry_forward_original": winner_source
                            == "carry_forward_original",
                            "repaired_state_id": new_state_id,
                            "repaired_code_path": winner_code_path,
                            "repaired_code_text": branch_winner,
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
                        best_state = {"code": branch_winner, "score": winner_score}

                    if winner_score.is_correct():
                        finished_states.append(
                            {"code": branch_winner, "score": winner_score, "state_id": new_state_id}
                        )
                    else:
                        next_states.append(
                            {
                                "state_id": new_state_id,
                                "parent_state_id": state_id,
                                "code": branch_winner,
                                "failed_last_time": branch_winner_state_failed_last_time,
                            }
                        )

            self.logger.info(
                f"Round {attempt} done. next_states={len(next_states)}, finished={len(finished_states)}"
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
            self.logger.info(
                f"Repair finished with {len(finished_states)} terminal states, choose the best score one."
            )
            return finished_states[0]["code"]

        if best_state is not None:
            self.logger.info("Repair ended without terminal state, returning best explored state.")
            return best_state["code"]

        self.logger.info("Repair ended with no valid branch winner, returning original code.")
        return code

    def run(self, input_file, output_file, func_name=None, args: dict = None):
        if args is None:
            args = {}
        content = open(input_file).read()
        repair_steps = args.get("repair", 5)
        intermediate_dir = args.get("intermediate_dir")
        source_metadata = self._extract_source_metadata(input_file)
        code = self.run_code(
            content,
            repair_num=repair_steps,
            intermediate_dir=intermediate_dir,
            source_metadata=source_metadata,
        )

        with open(output_file, "w") as wf:
            wf.write(code)

    def run_code(
        self,
        code,
        func_name=None,
        repair_num=5,
        intermediate_dir=None,
        source_metadata=None,
    ):
        self.logger.info("self debugging...")

        if intermediate_dir:
            temp_dir = Path(intermediate_dir)
        else:
            temp_dir = Path("output-intermediate-temp-" + time.strftime("%Y%m%d-%H%M%S"))
        temp_dir.mkdir(parents=True, exist_ok=True)

        code = self.repair_veval(
            code,
            max_attempt=repair_num,
            func_name=func_name,
            temp_dir=temp_dir,
            source_metadata=source_metadata,
        )

        self.logger.info("finished!")
        return code
