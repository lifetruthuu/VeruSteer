"""The final evidence-and-stagnation prompt, with six retrieved training examples."""
from collections import Counter
import copy, json, math, re
from pathlib import Path
from planner_common import ROOT, Prompts, current_observation

def tokens(state):
    code, errors, _ = current_observation(state)
    # Remove machine-specific paths and diagnostic line numbers. Keep Rust/Verus
    # identifiers, including specification vocabulary, without model embeddings.
    text = re.sub(r"(?:/[^\s:]+)+", " ", code + "\n" + errors)
    return Counter(re.findall(r"[A-Za-z_][A-Za-z_0-9]{2,}", text.lower()))

class Retriever:
    def __init__(self, examples):
        self.examples = examples
        frequency = Counter()
        documents = []
        for example in examples:
            document = tokens(example["observation"])
            documents.append(document)
            frequency.update(document.keys())
        self.idf = {term: math.log((1 + len(documents)) / (1 + n)) + 1
                    for term, n in frequency.items()}
        self.vectors = [self.vector(document) for document in documents]

    def vector(self, document):
        vector = {term: (1 + math.log(n)) * self.idf[term]
                  for term, n in document.items() if term in self.idf}
        norm = math.sqrt(sum(x * x for x in vector.values())) or 1
        return {term: value / norm for term, value in vector.items()}

    def retrieve(self, state, candidates, history, k):
        if not k:
            return []
        query, available = self.vector(tokens(state)), set(candidates)
        ranked = []
        for example, vector in zip(self.examples, self.vectors):
            types = set(example["candidates"])
            # Both the demonstrated successful choice and a comparison choice
            # must be relevant to the live registry.
            if len(types & available) < 2 or not (set(example["successful_actions"]) & available):
                continue
            similarity = sum(weight * vector.get(term, 0) for term, weight in query.items())
            jaccard = len(types & available) / len(types | available)
            depth = 1 / (1 + abs(example["completed_repairs"] - len(history)))
            last_match = int(bool(history and example["history_summary"])
                             and history[-1]["selected_error_type"]
                             == example["history_summary"][-1]["selected_error_type"])
            score = 0.55 * similarity + 0.35 * jaccard + 0.05 * depth + 0.05 * last_match
            ranked.append((-score, example["example_id"], example))
        chosen, tasks = [], set()
        for _, _, example in sorted(ranked):
            if example["task"] in tasks:
                continue
            chosen.append(example)
            tasks.add(example["task"])
            if len(chosen) == k:
                break
        return chosen

def example_payload(example):
    # Source IDs remain in the bank for audit, not in model input.
    return {key: example[key] for key in ("observation", "candidates", "completed_repairs",
                                         "history_summary", "action_outcomes")}

def normalized_diagnostic(text):
    return re.sub(r"(?:/[^\s:]+)+", "<path>", text).strip()

def stagnation_summary(state, history, candidates):
    observations = [*history, state]
    rows = {name: {"error_type": name, "attempts": 0, "unchanged_code": 0,
                   "unchanged_code_diagnostics_and_score": 0} for name in sorted(candidates)}
    for i, prior in enumerate(history):
        name = prior["selected_error_type"]
        if name not in rows:
            continue
        before = current_observation(prior)
        after = current_observation(observations[i + 1])
        rows[name]["attempts"] += 1
        same_code = before[0] == after[0]
        rows[name]["unchanged_code"] += same_code
        rows[name]["unchanged_code_diagnostics_and_score"] += (
            same_code and normalized_diagnostic(before[1]) == normalized_diagnostic(after[1])
            and list(before[2]) == list(after[2]))
    return list(rows.values())

def evidence_summary(prompts, state, candidates, history, examples):
    query = prompts.retriever.vector(tokens(state))
    remaining = 5 - len(history)
    result = []
    for name in sorted(candidates):
        total_weight = contrast = 0.0
        counts = Counter()
        for example in examples:
            outcome = next((row for row in example["action_outcomes"] if row["next_error_type"] == name), None)
            if outcome is None:
                continue
            vector = prompts.retriever.vector(tokens(example["observation"]))
            similarity = sum(value * vector.get(term, 0) for term, value in query.items())
            types, available = set(example["candidates"]), set(candidates)
            weight = 0.55 * similarity + 0.35 * len(types & available) / len(types | available)
            weight += 0.05 / (1 + abs(example["completed_repairs"] - len(history)))
            weight += 0.05 * int(bool(history and example["history_summary"]) and
                                history[-1]["selected_error_type"] == example["history_summary"][-1]["selected_error_type"])
            total_weight += weight
            counts["source_tasks"] += 1
            if not outcome["verified_continuation_observed"]:
                counts["no_verified_continuation_observed"] += 1
                contrast -= weight
            elif outcome["shortest_verified_calls"] <= remaining:
                counts["verified_within_remaining_calls"] += 1
                contrast += weight
            else:
                counts["verified_only_at_longer_horizon"] += 1
        result.append({"error_type": name, **{key: counts[key] for key in (
            "source_tasks", "verified_within_remaining_calls", "verified_only_at_longer_horizon",
            "no_verified_continuation_observed")},
            "support_score": round(contrast / total_weight, 4) if total_weight else None})
    return result

class PromptSelector(Prompts):
    def __init__(self, bank_path=None):
        super().__init__()
        bank_path = bank_path or ROOT / "data/prompt_examples/train_bank.json"
        self.bank = json.loads(Path(bank_path).read_text())
        train = set((ROOT / "data/splits/train_task_names.txt").read_text().splitlines())
        if not set(self.bank["support_tasks"]) <= train:
            raise ValueError("Example bank contains tasks outside the training partition")
        if any(e["task"] not in self.bank["support_tasks"] for e in self.bank["examples"]):
            raise ValueError("Example has an undeclared support task")
        self.retriever = Retriever(self.bank["examples"])
        self.prefix = (ROOT / "scripts/prompts/retrieval.txt").read_text()
        self.evidence_template = (ROOT / "scripts/prompts/evidence.txt").read_text()

    def messages(self, state, candidates, history=()):
        local = copy.copy(self)
        local.template = (self.template.replace("$COMPLETED", str(len(history)))
                          .replace("$REMAINING", str(5 - len(history)))
                          .replace("$TRANSITION_SUMMARY", ""))
        messages = Prompts.messages(local, state, candidates, history)
        examples = self.retriever.retrieve(state, candidates, history, 6)
        if examples:
            payload = json.dumps([example_payload(e) for e in examples], ensure_ascii=False, indent=2)
            messages[1]["content"] = self.prefix.replace("$EXAMPLES", payload) + messages[1]["content"]
        evidence = evidence_summary(self, state, candidates, history, examples)
        stagnation = (
            "EXECUTED REPAIR OUTCOMES IN THIS PATH:\n"
            "If the same selected type has repeatedly left code, diagnostics and score unchanged, "
            "explicitly consider another available target now, before the budget is exhausted. "
            "Repetition remains allowed when the observed code or obligation changed, or when "
            "concrete analogous outcomes support it. An unchanged score alone is insufficient.\n"
            + json.dumps(stagnation_summary(state, history, candidates), ensure_ascii=False, indent=2))
        suffix = self.evidence_template.replace("$EVIDENCE", json.dumps(evidence, ensure_ascii=False, indent=2))
        suffix = suffix.replace("$STAGNATION", stagnation)
        messages[1]["content"] += "\n\n" + suffix
        return messages
