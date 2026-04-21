import re
import torch

BOX_RE = re.compile(r"\\boxed\{([^{}]+)\}")
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")

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
    rewards = []
    scores = []
    exacts = []
    format_scores = []
    boxed_hits = []
    pred_lens = []

    for query, prompt, label in zip(queries, prompts, labels):
        completion = query[len(prompt):] if isinstance(prompt, str) and query.startswith(prompt) else query
        pred = extract_answer(completion)
        gold = extract_answer(label)

        boxed_hit = 1.0 if "\\boxed{" in completion else 0.0
        exact = 1.0 if pred != "" and pred == gold else 0.0
        format_score = boxed_hit

        reward = exact + 0.05 * format_score

        rewards.append(reward)
        scores.append(exact)
        exacts.append(exact)
        format_scores.append(format_score)
        boxed_hits.append(boxed_hit)
        pred_lens.append(float(len(pred)))

    rewards = torch.tensor(rewards, dtype=torch.float32)
    scores = torch.tensor(scores, dtype=torch.float32)

    return {
        "rewards": rewards,
        "scores": scores,
        "extra_logs": {
            "exact_match": torch.tensor(exacts, dtype=torch.float32),
            "format_score": torch.tensor(format_scores, dtype=torch.float32),
            "boxed_hit": torch.tensor(boxed_hits, dtype=torch.float32),
            "pred_len": torch.tensor(pred_lens, dtype=torch.float32),
        },
    }
