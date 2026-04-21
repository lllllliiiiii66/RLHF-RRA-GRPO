import re
from collections import defaultdict
import torch

BOX_RE = re.compile(r"\\boxed\{([^{}]+)\}")
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")

DELTA = 0
ALPHA = 0.005

def normalize_text(x):
    return str(x).strip().replace(",", "").replace("−", "-")

def extract_answer(text):
    text = normalize_text(text)
    boxed = BOX_RE.findall(text)
    if boxed:
        return normalize_text(boxed[-1])
    nums = NUM_RE.findall(text)
    if nums:
        return normalize_text(nums[-1])
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""

def reward_func(queries, prompts, labels):
    batch_size = len(queries)

    completions = []
    preds = []
    golds = []
    exacts = []
    format_scores = []
    lengths = []
    boxed_hits = []
    pred_lens = []

    for query, prompt, label in zip(queries, prompts, labels):
        completion = query[len(prompt):] if isinstance(prompt, str) and query.startswith(prompt) else query
        pred = extract_answer(completion)
        gold = extract_answer(label)

        exact = 1.0 if pred != "" and pred == gold else 0.0
        boxed_hit = 1.0 if "\\boxed{" in completion else 0.0
        format_score = boxed_hit
        length = float(len(completion))

        completions.append(completion)
        preds.append(pred)
        golds.append(gold)
        exacts.append(exact)
        format_scores.append(format_score)
        lengths.append(length)
        boxed_hits.append(boxed_hit)
        pred_lens.append(float(len(pred)))

    # Group by prompt so we can compute per-prompt anchors.
    groups = defaultdict(list)
    for idx, prompt in enumerate(prompts):
        groups[prompt].append(idx)

    rewards = [0.0] * batch_size
    anchor_refs = [0.0] * batch_size
    length_penalties = [0.0] * batch_size

    for prompt, indices in groups.items():
        correct_indices = [i for i in indices if exacts[i] == 1.0]

        # No correct sample in the group: no anchor penalty.
        if not correct_indices:
            for i in indices:
                rewards[i] = exacts[i] + 0.05 * format_scores[i]
                anchor_refs[i] = -1.0
                length_penalties[i] = 0.0
            continue

        l_ref = min(lengths[i] for i in correct_indices)
        l_anchor = l_ref + DELTA

        for i in indices:
            base_reward = exacts[i] + 0.05 * format_scores[i]

            penalty = 0.0
            if exacts[i] == 1.0:
                penalty = ALPHA * max(0.0, lengths[i] - l_anchor)

            rewards[i] = base_reward - penalty
            anchor_refs[i] = l_ref
            length_penalties[i] = penalty

    rewards_t = torch.tensor(rewards, dtype=torch.float32)
    scores_t = torch.tensor(exacts, dtype=torch.float32)

    return {
        "rewards": rewards_t,
        "scores": scores_t,
        "extra_logs": {
            "exact_match": torch.tensor(exacts, dtype=torch.float32),
            "format_score": torch.tensor(format_scores, dtype=torch.float32),
            "boxed_hit": torch.tensor(boxed_hits, dtype=torch.float32),
            "pred_len": torch.tensor(pred_lens, dtype=torch.float32),
            "response_char_len": torch.tensor(lengths, dtype=torch.float32),
            "anchor_ref_len": torch.tensor(anchor_refs, dtype=torch.float32),
            "length_penalty": torch.tensor(length_penalties, dtype=torch.float32),
        },
    }
