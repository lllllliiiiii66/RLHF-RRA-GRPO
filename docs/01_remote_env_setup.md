# 4x4090 Remote Environment Setup

This guide is for a Linux GPU container that you access from VS Code over SSH.
The goal is to get a stable `OpenRLHF + vLLM + Ray + DeepSpeed` environment running
with the fewest moving parts first, then add our own experiment code later.

## 0. Guiding principles

1. Do not start by cloning many repos and installing random versions.
2. First verify the container sees all 4 GPUs correctly.
3. Use one clean Python environment only for this project.
4. Install in small steps and verify after each step.
5. Do not mix `conda install torch ...` and `pip install torch ...` in the same env.

## 1. What we will use

- Python: `3.10`
- PyTorch: `2.4.1` with `CUDA 12.1`
- OpenRLHF: latest pip release first
- vLLM: installed via `openrlhf[vllm]`
- Training method for the first stable run: full fine-tuning with DeepSpeed ZeRO
- First target: run environment checks only

Why this choice:

- Python 3.10 is the safest common denominator for current RL tooling.
- PyTorch 2.4.1 with cu121 has an official wheel and is still broadly compatible.
- Installing `openrlhf[vllm]` from pip is lower risk than deep source edits on day 1.
- Current OpenRLHF documentation notes that `Ray + vLLM` does not currently support LoRA, so we should not rely on LoRA for the first stable setup.
- We want a stable baseline environment before touching custom rewards.

## 2. Enter the remote container

Open a terminal inside VS Code SSH and run:

```bash
whoami
pwd
python --version
nvidia-smi
```

Expected:

- You are inside the remote Linux container.
- `python --version` prints some existing system version.
- `nvidia-smi` shows `4` RTX 4090 GPUs.

If `nvidia-smi` fails, stop here. That means the container runtime or GPU mounting is not correct yet.

## 3. Create a clean conda environment

First check whether conda exists:

```bash
conda --version
```

If conda exists, create the env:

```bash
conda create -n rra_grpo python=3.10 -y
conda activate rra_grpo
python --version
```

If the shell says `conda activate` is unavailable, run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rra_grpo
```

If your platform uses another install path, replace `~/miniconda3` with the real conda root.

## 4. Upgrade base packaging tools

Inside the new environment:

```bash
python -m pip install --upgrade pip setuptools wheel
pip --version
```

Reason:

- Old `pip` is a common source of wheel resolution failures.

## 5. Install PyTorch first

Install the official CUDA 12.1 wheels:

```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
```

Then verify:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

Expected:

- `2.4.1`
- CUDA version string like `12.1`
- `True`
- `4`

If `torch.cuda.is_available()` is `False`, do not continue to install RL packages yet.

## 6. Install the RL stack

Install OpenRLHF with vLLM support:

```bash
pip install "openrlhf[vllm]"
```

Why:

- OpenRLHF documentation currently recommends `openrlhf[vllm]` and recent vLLM.
- This lets pip solve a tested dependency set instead of us manually guessing versions.

If the install fails because of preinstalled conflicting packages, try:

```bash
pip uninstall -y xgboost transformer_engine flash_attn pynvml opencv-python-headless
pip install "openrlhf[vllm]"
```

Do not run the uninstall first unless the install actually fails or those packages are already present and conflicting.

## 7. Install experiment helpers

These are lightweight packages we will likely need very early:

```bash
pip install datasets transformers accelerate sentencepiece wandb sympy math-verify
```

Notes:

- `datasets`: dataset loading and filtering
- `transformers`: tokenizer/model utilities
- `wandb`: metrics dashboard
- `sympy` and `math-verify`: math answer verification helpers

## 8. Run the environment self-check script

From the project root:

```bash
python scripts/check_env.py
```

This script checks:

- Python version
- Torch and CUDA visibility
- GPU count and names
- BF16 capability
- Import status of `transformers`, `datasets`, `ray`, `deepspeed`, `vllm`, `openrlhf`
- A tiny CUDA tensor operation

If this script fails, fix it before trying any Ray or training command.

## 9. Minimal distributed checks

Check Ray:

```bash
ray stop
ray start --head --node-ip-address 0.0.0.0 --num-gpus 4
ray status
```

Then stop it after verification:

```bash
ray stop
```

Why:

- We only want to confirm that Ray can see the machine and GPU resources.
- We do not start training yet.

## 10. Common fixes

### A. `torch.cuda.is_available()` is false

Possible causes:

- Wrong container image
- GPU not mounted into the container
- Installed CPU-only torch by mistake

Fix:

- Re-check `nvidia-smi`
- Reinstall the exact PyTorch command from section 5

### B. `deepspeed` import fails

Try:

```bash
pip install deepspeed
python -c "import deepspeed; print(deepspeed.__version__)"
```

If it still fails, send the full error log before trying random blog fixes.

### C. Ray starts but training later reports GPU index errors

Set:

```bash
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
```

OpenRLHF documents this as a troubleshooting option for DeepSpeed GPU device setup issues.

### D. `pip install openrlhf[vllm]` is very slow or fails halfway

Do not keep rerunning different install commands.
Instead:

1. Save the full error log.
2. Check whether the failure is network, build, or version conflict.
3. Fix the root cause once.

## 11. What we do next after environment success

Only after all checks pass, we move to:

1. clone or install OpenRLHF source for easier script inspection
2. run a tiny smoke test
3. prepare a very small math-style dataset
4. launch the first baseline pilot

## 12. Copy-paste command block

If you want the shortest clean setup path, use this block step by step:

```bash
conda create -n rra_grpo python=3.10 -y
conda activate rra_grpo
python -m pip install --upgrade pip setuptools wheel
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install "openrlhf[vllm]"
pip install datasets transformers accelerate sentencepiece wandb sympy math-verify
python scripts/check_env.py
ray stop
ray start --head --node-ip-address 0.0.0.0 --num-gpus 4
ray status
ray stop
```

## 13. References

- OpenRLHF Quick Start: https://openrlhf.readthedocs.io/en/latest/quick_start.html
- OpenRLHF GitHub: https://github.com/OpenRLHF/OpenRLHF
- PyTorch previous versions: https://docs.pytorch.org/get-started/previous-versions/
