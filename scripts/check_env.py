import importlib
import platform
import sys


def print_header(title: str) -> None:
    print(f"\n=== {title} ===")


def try_import(name: str) -> None:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "unknown")
        print(f"[OK] {name}: {version}")
    except Exception as exc:  # pragma: no cover
        print(f"[FAIL] {name}: {exc}")


def main() -> int:
    print_header("Python")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")

    print_header("Torch / CUDA")
    try:
        import torch
    except Exception as exc:
        print(f"[FAIL] torch import failed: {exc}")
        return 1

    print(f"torch.__version__: {torch.__version__}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    print(f"cuda available: {torch.cuda.is_available()}")
    print(f"cuda device count: {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            print(
                f"GPU {idx}: {props.name} | "
                f"total_memory={props.total_memory / 1024**3:.1f} GB | "
                f"cc={props.major}.{props.minor}"
            )

        try:
            bf16_supported = torch.cuda.is_bf16_supported()
        except Exception:
            bf16_supported = "unknown"
        print(f"bf16 supported: {bf16_supported}")

        try:
            x = torch.randn(4, 4, device="cuda")
            y = x @ x.T
            print(f"[OK] CUDA tensor op succeeded, mean={y.mean().item():.6f}")
        except Exception as exc:
            print(f"[FAIL] CUDA tensor op failed: {exc}")
            return 1
    else:
        print("[FAIL] CUDA is not available.")
        return 1

    print_header("Core packages")
    for pkg in [
        "transformers",
        "datasets",
        "accelerate",
        "ray",
        "deepspeed",
        "vllm",
        "openrlhf",
        "sympy",
        "math_verify",
        "wandb",
    ]:
        try_import(pkg)

    print_header("Result")
    print("Environment check finished.")
    print("If all key packages show [OK], we can move on to smoke testing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
