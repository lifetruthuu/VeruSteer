"""Recorded-tree and request utilities for the published prompt selector."""
from __future__ import annotations
from collections import Counter, defaultdict
import json, os, re, time
from pathlib import Path
from urllib.parse import urlsplit
import yaml
ROOT = Path(__file__).resolve().parents[1]

def json_text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)

def load_config(path):
    config = {"temperature": 0.3, "max_tokens": None, "timeout": 120,
              "max_retries": 0, "extra_body": {}, "api_key_env": "PLANNER_API_KEY"}
    supplied = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    allowed = set(config) | {"base_url", "model", "api_key"}
    if not isinstance(supplied, dict) or set(supplied) - allowed:
        raise ValueError("Config must contain only the documented fields")
    inline_key = supplied.pop("api_key", None)
    if inline_key is not None and (not isinstance(inline_key, str) or not inline_key.strip()):
        raise ValueError("api_key must be a nonempty string when provided")
    config.update(supplied)
    for key in ("base_url", "model", "api_key_env"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(f"Missing configuration field: {key}")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", config["api_key_env"]):
        raise ValueError("api_key_env must be an environment variable name; put a literal credential in api_key")
    if not isinstance(config["temperature"], (float, int)) or not 0 <= config["temperature"] <= 2:
        raise ValueError("temperature must be between 0 and 2")
    if not isinstance(config["timeout"], (float, int)) or not 0 < config["timeout"] <= 3600:
        raise ValueError("timeout must be positive and at most 3600 seconds")
    if type(config["max_retries"]) is not int or not 0 <= config["max_retries"] <= 10:
        raise ValueError("max_retries must be an integer between 0 and 10")
    if config["max_tokens"] is not None and (type(config["max_tokens"]) is not int or config["max_tokens"] < 1):
        raise ValueError("max_tokens must be null or a positive integer")
    if not isinstance(config["extra_body"], dict):
        raise ValueError("extra_body must be a mapping")
    reserved = {"model", "messages", "temperature", "n", "max_tokens", "stream", "api_key", "authorization"}
    if reserved & set(config["extra_body"]):
        raise ValueError("extra_body cannot override request/configuration fields or contain credentials")
    url = urlsplit(config["base_url"])
    if url.username or url.password or url.query or url.fragment:
        raise ValueError("base_url must not contain credentials, query parameters, or a fragment")
    json_text(config)  # Also reject non-JSON values in extra_body.
    if inline_key is not None:
        # Resolve inline credentials through the process environment. Only the
        # variable name is returned to manifests, cache keys, and request logs.
        os.environ[config["api_key_env"]] = inline_key.strip()
    return config

def current_observation(state):
    code, errors, score = (state.get(k) for k in ("code_text", "verus_output_full", "score_tuple"))
    if not isinstance(code, str) or not code or not isinstance(errors, str) or not errors:
        raise ValueError("A planner decision requires current code and verifier diagnostics")
    if not isinstance(score, (list, tuple)) or len(score) != 2:
        raise ValueError("A planner decision requires a two-element current score")
    return code, errors, score

class Prompts:
    def __init__(self):
        self.history_mode = "path"
        self.system = (ROOT / "scripts/prompts/system.txt").read_text()
        self.template = (ROOT / "scripts/prompts/action.txt").read_text()
        self.priority = ['UnexpectedToken', 'ExpectedCurlyBraces', 'ExpectedComma', 'BadQuantifierSyntax', 'UnresolvedImport', 'CantFindValue', 'MismatchedType', 'IncompatibleTypes', 'TypeAnnotationNeeded', 'TriggerInferenceFail', 'UnknownProver', 'Other', 'ArithmeticFlow', 'PreCondFail', 'InvFailEnd', 'InvFailFront', 'AssertFail', 'PostCondFail']

    def messages(self, state, candidates, history=()):
        # Observations and executed actions are explicitly whitelisted.
        code, errors, score = current_observation(state)
        replacements = {"$REPAIRER_REGISTRY": "\n".join(f"- {name}: {name}Repairer" for name in sorted(candidates)),
                        "$SCORE": f"verified={score[0]}, errors={score[1]}", "$ERRORS": errors, "$CODE": code}
        template = self.template
        if self.history_mode == "path" and history:
            if len(history) > 4:
                raise ValueError("A planner decision can have at most four completed repairs")
            # Serialize each previous observation once. The next state supplies
            # each action's outcome; the last outcome is CURRENT CODE/ERRORS.
            entries = []
            for i, item in enumerate(history):
                past_code, past_errors, past_score = current_observation(item)
                action = item["selected_error_type"]
                if not isinstance(action, str) or not action:
                    raise ValueError("History requires an executed error-type selection")
                entries.append({"state": f"S{i}", "code_text": past_code,
                                "verus_output_full": past_errors, "score_tuple": list(past_score),
                                "selected_error_type": action, "outcome_state": f"S{i + 1}"})
            history_text = (
                "REPAIR HISTORY (chronological, executed path only):\n"
                "Use the prior code, diagnostics, selected error types, and their observed outcomes "
                "to inform your next selection from the current registry. Each entry records a "
                "state before a repair and the error type selected there. Its outcome_state points "
                "to the resulting code, diagnostics, and score in the next entry. "
                f"S{len(entries)} is the CURRENT CODE, VERUS ERRORS, and CURRENT SCORE above, "
                "which are the outcome of the last recorded repair. Single-candidate repairs are "
                "included. These are recorded state transitions; intermediate patch attempts "
                "are unavailable.\n" + json.dumps(entries, ensure_ascii=False, indent=2) + "\n\n")
            marker = "Return JSON only with these fields:"
            if template.count(marker) != 1:
                raise ValueError("Cannot locate the upstream planner output instructions")
            template = template.replace(marker, "$REPAIR_HISTORY" + marker)
            replacements["$REPAIR_HISTORY"] = history_text
        # One-pass substitution: source code containing $SCORE etc. stays literal.
        user = re.sub(r"\$REPAIRER_REGISTRY|\$REPAIR_HISTORY|\$SCORE|\$ERRORS|\$CODE",
                      lambda m: replacements.get(m[0], m[0]), template)
        return [{"role": "system", "content": self.system}, {"role": "user", "content": user}]

    def parse(self, content, candidates):
        """Use the upstream routing fields; fix_order never determines the edge."""
        try:
            if not isinstance(content, str):
                raise ValueError("missing_text_content")
            start, end = content.find("{"), content.rfind("}")
            if start < 0 or end < start:
                raise ValueError("missing_json_object")
            obj = json.loads(content[start:end + 1])
            if not isinstance(obj, dict):
                raise ValueError("not_json_object")
            selected = obj.get("root_cause_error_type")
            repairer = obj.get("selected_repairer")
            if not isinstance(selected, str) or selected.strip() not in candidates:
                raise ValueError("root_cause_outside_candidates")
            selected = selected.strip()
            if not isinstance(repairer, str) or repairer.strip() != f"{selected}Repairer":
                raise ValueError("repairer_mismatch")
            return {"selected_name": selected, "planner_decision": obj,
                    "used_fallback": False, "fallback_reason": None}
        except (ValueError, TypeError) as exc:
            rank = {name: i for i, name in enumerate(self.priority)}
            selected = min(candidates, key=lambda name: (rank.get(name, len(rank)), name))
            reason = "invalid_json" if isinstance(exc, json.JSONDecodeError) else str(exc)
            return {"selected_name": selected, "planner_decision": None,
                    "used_fallback": True, "fallback_reason": reason}

def candidates_at(tree, sid):
    state = tree.state_by_id[sid]
    reported = state.get("failure_types") or [e["error_type"] for e in state.get("repaired_failure_list", []) if e.get("error_type")]
    available = set(tree.valid_actions_by_state.get(sid, []))
    return list(dict.fromkeys(name for name in reported if name in available))

def path_history(tree, visited, trace):
    """Extract only observations and actual actions from a validated prefix."""
    if len(visited) != len(trace) + 1 or len(trace) > 4:
        raise ValueError("Invalid decision prefix length")
    history = []
    for i, step in enumerate(trace):
        sid, nxt = visited[i:i + 2]
        action = step["selected_name"]
        if (step["current_state_id"] != sid or step["next_state_id"] != nxt
                or tree.children_by_state_and_action.get((sid, action)) != nxt
                or action not in candidates_at(tree, sid)):
            raise ValueError("History does not match executed transitions")
        code, errors, score = current_observation(tree.state_by_id[sid])
        history.append({"code_text": code, "verus_output_full": errors,
                        "score_tuple": list(score), "selected_error_type": action})
    return history

def preview_prefixes(tree):
    """Enumerate legal prefixes by alphabetical actions, without reading outcomes."""
    def walk(visited, trace):
        sid = visited[-1]
        candidates = candidates_at(tree, sid)
        if len(candidates) > 1:
            yield visited, trace, candidates
        for action in sorted(candidates):
            nxt = tree.children_by_state_and_action[sid, action]
            step = {"current_state_id": sid, "selected_name": action, "next_state_id": nxt}
            yield from walk(visited + [nxt], trace + [step])

    yield from walk([tree.paths[0]["root_state_id"]], [])

def load_trees(path):
    from eval_tree_better_path import build_tree_structures

    grouped = defaultdict(list)
    with Path(path).open("rb") as stream:
        for raw in stream:
            if raw.strip():
                row = json.loads(raw)
                if not row["tree_id"].startswith("verusage-"):
                    raise ValueError("This experiment expects the VeruSAGE test split")
                grouped[row["tree_id"]].append(row)
    if not grouped:
        raise ValueError("Empty test corpus")
    trees = {}
    for tid in sorted(list(grouped)):
        rows = grouped.pop(tid)
        edges, observations = {}, {}
        for row in rows:
            ids, states = row["path_state_ids"], row["path_states"]
            if not ids or len(ids) != len(states) or ids[0] != row["root_state_id"] or len(set(ids)) != len(ids):
                raise ValueError(f"Invalid recorded path in {tid}")
            for i, (sid, state) in enumerate(zip(ids, states)):
                observation = tuple(state.get(k) for k in ("code_text", "verus_output_full", "score_tuple", "failure_types"))
                if observations.setdefault(sid, observation) != observation:
                    raise ValueError(f"Conflicting current-state observations: {tid}/{sid}")
                if i + 1 < len(ids):
                    edge = (sid, state["chosen_failure_type"])
                    if edges.setdefault(edge, ids[i + 1]) != ids[i + 1]:
                        raise ValueError(f"Ambiguous recorded transition: {tid}/{sid}")
        tree = build_tree_structures(tid, rows)
        # Keep one code/diagnostic payload per node rather than every path copy.
        tree.paths = [{k: v for k, v in row.items() if k != "path_states"} for row in rows]
        trees[tid] = tree
    return trees, {"bytes": Path(path).stat().st_size, "trees": len(trees)}

def audit(trees, history_mode="none"):
    counts, vocabulary = Counter(), set()
    low = high = branching_trees = 0
    for tree in trees.values():
        leaves = {ids[-1] for ids in tree.path_id_by_state_tuple}
        for sid in tree.state_by_id:
            candidates = candidates_at(tree, sid)
            if len(candidates) > 1 or (history_mode == "path" and candidates):
                current_observation(tree.state_by_id[sid])
            counts[len(candidates)] += 1
            vocabulary.update(candidates)
            if bool(candidates) == (sid in leaves):
                raise ValueError(f"Node is neither an unambiguous leaf nor an internal node: {tree.tree_id}/{sid}")
        memo, active = {}, set()

        def bounds(sid):
            if sid in active:
                raise ValueError(f"Cycle in {tree.tree_id}")
            if sid in memo:
                return memo[sid]
            active.add(sid)
            candidates = candidates_at(tree, sid)
            children = [bounds(tree.children_by_state_and_action[sid, name]) for name in candidates]
            here = int(len(candidates) > 1)
            result = ((here + min(x[0] for x in children), here + max(x[1] for x in children),
                       1 + max(x[2] for x in children)) if children else (0, 0, 0))
            memo[sid] = result
            active.remove(sid)
            return result

        a, b, depth = bounds(tree.paths[0]["root_state_id"])
        if depth > 5:
            raise ValueError(f"Recorded tree exceeds the five-round budget: {tree.tree_id}")
        low += a
        high += b
        branching_trees += int(b > 0)
    return {"trees": len(trees), "paths": sum(len(t.paths) for t in trees.values()),
            "branching_trees": branching_trees, "states_by_candidate_count": dict(sorted(counts.items())),
            "candidate_types": sorted(vocabulary), "requests_without_retries": {"min": low, "max": high}}

def make_requester(config):
    from openai import OpenAI, APIConnectionError, APIStatusError

    url = urlsplit(config["base_url"])
    if url.scheme not in ("http", "https") or not url.netloc:
        raise ValueError("Fill in base_url before a live run")
    key = os.environ.get(config["api_key_env"])
    if not key:
        raise ValueError(f"Set the environment variable named by api_key_env ({config['api_key_env']})")
    client = OpenAI(api_key=key, base_url=config["base_url"], timeout=config["timeout"], max_retries=0)

    def request(messages):
        kwargs = {"model": config["model"], "messages": messages, "temperature": config["temperature"], "n": 1}
        if config["max_tokens"] is not None:
            kwargs["max_tokens"] = config["max_tokens"]
        if config["extra_body"]:
            kwargs["extra_body"] = config["extra_body"]
        started = time.monotonic()
        for attempt in range(config["max_retries"] + 1):
            try:
                response = client.chat.completions.create(**kwargs)
                return {"raw_response": response.model_dump(mode="json"), "attempts": attempt + 1,
                        "latency_seconds": time.monotonic() - started}
            except (APIConnectionError, APIStatusError) as exc:
                status = getattr(exc, "status_code", None)
                transient = isinstance(exc, APIConnectionError) or status in (408, 429) or (status is not None and status >= 500)
                if not transient or attempt == config["max_retries"]:
                    # Provider error bodies can echo credentials; report only type/status.
                    raise RuntimeError(f"Planner API failed: {type(exc).__name__}, HTTP {status}; resume after fixing the service/configuration") from None
                time.sleep(min(2 ** attempt, 30))
        raise AssertionError("unreachable")

    return request

def rollout(tree, planner):
    visited = [tree.paths[0]["root_state_id"]]
    trace = []
    with_history = planner.prompts.history_mode == "path"
    while candidates := candidates_at(tree, visited[-1]):
        sid = visited[-1]
        if len(trace) >= 5:
            raise ValueError(f"Five-round budget exceeded: {tree.tree_id}")
        if len(candidates) == 1:
            decision = {"selected_name": candidates[0], "used_fallback": False,
                        "fallback_reason": None, "selected_reason": "single_candidate"}
        else:
            history = path_history(tree, visited, trace) if with_history else ()
            decision = planner.select(tree.tree_id, sid, tree.state_by_id[sid], candidates, history)
            decision["selected_reason"] = "planner_fallback" if decision["used_fallback"] else "planner"
            if with_history:
                decision["history_state_ids"] = list(visited)
        selected = decision["selected_name"]
        if selected not in candidates:
            raise ValueError("Selector returned an unavailable candidate")
        nxt = tree.children_by_state_and_action[sid, selected]
        if nxt in visited or nxt not in tree.state_by_id:
            raise ValueError(f"Invalid transition at {tree.tree_id}/{sid}")
        trace.append({"step": len(trace), "current_state_id": sid, "candidate_names": candidates,
                      **decision, "next_state_id": nxt})
        visited.append(nxt)
    path_id = tree.path_id_by_state_tuple.get(tuple(visited))
    if path_id is None:
        raise ValueError(f"Rollout did not reach a recorded path: {tree.tree_id}")
    mode = "prompt_selector"
    return {"tree_id": tree.tree_id, "selector_mode": mode, "num_paths": len(tree.paths),
            "reached_path_id": path_id, "reached_leaf_state_id": visited[-1], "dead_end": False,
            "dead_end_reason": None, "steps": len(trace), "selection_trace": trace}
