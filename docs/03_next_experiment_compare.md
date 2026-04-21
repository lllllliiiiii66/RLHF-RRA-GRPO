# Next Experiment: Baseline vs Soft Anchor

## Goal

Run one clean A/B comparison where the training setup stays the same and only the reward file changes:

- Baseline: `reward/baseline_reward_v3.py`
- Soft Anchor: `reward/soft_anchor_reward_v1.py`

## Keep Fixed

These settings should stay identical across the two runs:

- same base model
- same dataset
- same `max_new_tokens`
- same `n_samples_per_prompt`
- same PPO / GRPO training hyperparameters
- same seed policy unless explicitly testing variance

## Only Change

- reward file
- output directory
- tensorboard log directory

## Metrics To Compare

Read the final `Global step` record and compare:

- `exact_match`
- `reward`
- `score`
- `format_score`
- `boxed_hit`
- `pred_len`
- `response_char_len`
- `response_length`
- `anchor_ref_len`
- `length_penalty`
- `truncated`

## Current Interpretation Rule

Do not claim "shorter reasoning" from `response_length` alone when `truncated = 1.0`.

Prefer:

- `response_char_len` for observed output verbosity
- `anchor_ref_len` and `length_penalty` as evidence that the anchor mechanism actually fired

## Practical Note

Reuse the existing working server training command and only swap:

- `--reward.remote_url`
- output save path
- tensorboard path

## Log Processing

After each run, save the terminal log to a text file and summarize with:

```powershell
python scripts/summarize_openrlhf_logs.py --last-only <log1> <log2>
```

This gives a compact side-by-side metric table for the final step of each run.
