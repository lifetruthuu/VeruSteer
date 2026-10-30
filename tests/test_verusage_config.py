"""Regression checks for the released VeruSAGE configuration crash.

Execute the actual source classes/methods in isolation to avoid importing the
entire agent registry and optional model SDKs. Only model calls, repair actions,
and the external safety checker are mocked; no model API or Verus is needed.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[1]
REPAIR = ROOT / "pipelines" / "repair" / "verusage"


def load_source(path, class_name, method_name=None, **namespace):
    """Load a production class or method without its module's import side effects."""
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    if method_name is not None:
        node = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[method_name or class_name]


class VeruSAGEConfigTests(unittest.TestCase):
    def setUp(self):
        self.AttrDict = load_source(REPAIR / "utils.py", "AttrDict")

    def test_example_explicitly_disables_fvt(self):
        config = json.loads((REPAIR.parent / "verusage.example.json").read_text())
        self.assertIs(config["fvt_mode"], False)
        self.assertEqual(config["fvt_prompt_supplement"], "")

    def test_missing_attributes_use_defaults_but_missing_keys_still_raise(self):
        config = self.AttrDict({"configured": False})
        self.assertIs(getattr(config, "fvt_mode", False), False)
        self.assertEqual(getattr(config, "fvt_prompt_supplement", ""), "")
        self.assertIs(config.configured, False)
        self.assertFalse(hasattr(config, "missing"))
        with self.assertRaises(AttributeError):
            config.missing
        with self.assertRaises(KeyError):
            config["missing"]

    def invoke(self, config):
        model = Mock(return_value=["MODEL_RESPONSE"])
        invoke = load_source(
            REPAIR / "agents" / "actions" / "base_action.py", "BaseAction", "invoke_llm",
            call_llm_with_diff_format=model,
        )
        action = SimpleNamespace(config=self.AttrDict(config))
        result = invoke(action, "model", "instruction", "query", "system", skip_history=True)
        self.assertEqual(result, ["MODEL_RESPONSE"])
        model.assert_called_once()
        return model.call_args.kwargs["query"]

    def test_missing_and_disabled_fvt_reach_model_with_same_restrictions(self):
        example = json.loads((REPAIR.parent / "verusage.example.json").read_text())
        without_optional_fields = dict(example)
        without_optional_fields.pop("fvt_mode", None)
        without_optional_fields.pop("fvt_prompt_supplement", None)
        query = self.invoke(example)
        self.assertEqual(query, self.invoke(without_optional_fields))
        self.assertIn("Do NOT modify the `requires` or `ensures`", query)
        self.assertNotIn("For this FVT task", query)

    def test_explicit_fvt_mode_is_preserved(self):
        query = self.invoke({"fvt_mode": True, "fvt_prompt_supplement": "CUSTOM_FVT_GUIDANCE"})
        self.assertIn("For this FVT task", query)
        self.assertIn("CUSTOM_FVT_GUIDANCE", query)

    def test_missing_and_disabled_fvt_still_enforce_safety_check(self):
        path = REPAIR / "agents" / "main_loop.py"
        tree = ast.parse(path.read_text())
        class_name = next(
            n.name for n in tree.body if isinstance(n, ast.ClassDef)
            and any(isinstance(m, ast.FunctionDef) and m.name == "_validate_code_change" for m in n.body)
        )
        for config in ({}, {"fvt_mode": False}):
            for safe in (False, True):
                with self.subTest(config=config, safe=safe):
                    checker = Mock(return_value=safe)
                    validate = load_source(
                        path, class_name, "_validate_code_change",
                        proof_completion_code_change_is_safe=checker,
                    )
                    agent = SimpleNamespace(config=self.AttrDict(config), logger=Mock())
                    self.assertIs(validate(agent, "original", "changed", None), safe)
                    checker.assert_called_once_with("original", "changed", agent.logger)

    def test_agent_exceptions_return_failure_without_secondary_crash(self):
        Result = load_source(REPAIR / "agents" / "shared_types.py", "ActionResult", dataclass=dataclass, field=field)
        checked = 0
        for path in sorted((REPAIR / "agents").glob("repair_*_agent.py")):
            # AssertFail's multi-action dispatcher is separate from these direct
            # dispatchers and did not use the obsolete constructor arguments.
            if path.name == "repair_assert_agent.py":
                continue
            tree = ast.parse(path.read_text())
            for node in (n for n in tree.body if isinstance(n, ast.ClassDef)):
                if not any(isinstance(m, ast.FunctionDef) and m.name == "act" for m in node.body):
                    continue
                with self.subTest(agent=path.name):
                    registry = SimpleNamespace(get_action_class=Mock(side_effect=RuntimeError("injected action failure")))
                    act = load_source(path, node.name, "act", ActionResult=Result, action_registry=registry)
                    result = act(
                        SimpleNamespace(logger=Mock()), SimpleNamespace(code="ORIGINAL"),
                        SimpleNamespace(primary_action="TEST", action_parameters={}),
                    )
                    self.assertIs(result.success, False)
                    self.assertEqual(result.candidates, [])
                    self.assertEqual(result.original_code, "ORIGINAL")
                    self.assertIn("injected action failure", result.explanation)
                    checked += 1
        self.assertEqual(checked, 14)


if __name__ == "__main__":
    unittest.main()
