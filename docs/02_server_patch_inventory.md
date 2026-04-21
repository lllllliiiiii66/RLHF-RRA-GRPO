# Server Patch Inventory

This repo currently contains the experiment data and reward files copied back from the server, but it does **not**
contain the patched files that were edited directly inside the server Conda environment under
`/root/.conda/envs/rra_grpo/lib/python3.10/site-packages/openrlhf/`.

Those package-level patches are part of the working experiment setup and must be reapplied when moving to a new
environment.

## Patched OpenRLHF files on the server

### 1. Flash-attn compatibility

Files:

- `openrlhf/models/ring_attn_utils.py`
- `openrlhf/models/actor.py`

Purpose:

- Avoid hard failure when `flash_attn` is unavailable.
- Allow the actor path to fall back to `sdpa`.
- Make `python -m openrlhf.cli.train_ppo_ray --help` and PPO training startup succeed in the current environment.

### 2. Group-aware local reward batching

Files:

- `openrlhf/utils/agent.py`
- `openrlhf/trainer/ray/vllm_engine.py`

Purpose:

- Change the local custom reward `.py` path from per-sample reward calls to per-prompt batched reward calls.
- Let `reward_func(queries, prompts, labels)` receive all samples for the same prompt together.
- Support group-aware reward logic such as `Soft Dynamic Anchor`.

Observed behavior after patch:

- Debug logs showed repeated prompts inside the same reward batch.
- `len(queries)` matched `n_samples_per_prompt` for a single prompt group in the debug runs.

Implementation notes:

- `SingleTurnAgentExecutor` was extended so generation and reward attachment are separated.
- Reward attachment happens after collecting all samples for the prompt.
- `generate_responses(...)` in the Ray vLLM actor gathers all `num_samples` outputs first, then attaches rewards in
  batch.
- The executor signature also needs to accept `skip_reward=False` so the Ray actor can defer reward computation until
  the full prompt group has been generated.

## Important reminder

The local files in this repo under `reward/` are not enough by themselves to reproduce the server runs. Reproducing
the working setup also requires reapplying the above OpenRLHF package patches in the target environment.
