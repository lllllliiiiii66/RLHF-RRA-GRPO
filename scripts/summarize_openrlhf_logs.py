import argparse
import ast
import re
from pathlib import Path


STEP_RE = re.compile(r"Global step\s+(\d+):\s+(\{.*\})")

DEFAULT_KEYS = [
    "exact_match",
    "reward",
    "score",
    "format_score",
    "boxed_hit",
    "pred_len",
    "response_char_len",
    "response_length",
    "anchor_ref_len",
    "length_penalty",
    "truncated",
]


def parse_log(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = STEP_RE.search(line)
        if not match:
            continue
        step = int(match.group(1))
        metrics = ast.literal_eval(match.group(2))
        rows.append((step, metrics))
    return rows


def format_value(value):
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def main():
    parser = argparse.ArgumentParser(description="Summarize OpenRLHF Global step logs.")
    parser.add_argument("logs", nargs="+", help="One or more log files to summarize.")
    parser.add_argument(
        "--keys",
        nargs="+",
        default=DEFAULT_KEYS,
        help="Metric keys to print. Missing keys are shown as NA.",
    )
    parser.add_argument(
        "--last-only",
        action="store_true",
        help="Only show the last Global step record from each log file.",
    )
    args = parser.parse_args()

    header = ["run", "step"] + args.keys
    print("\t".join(header))

    for log_path in args.logs:
        path = Path(log_path)
        rows = parse_log(path)
        if not rows:
            print(f"{path.name}\tNA\t" + "\t".join(["NA"] * len(args.keys)))
            continue

        if args.last_only:
            rows = [rows[-1]]

        for step, metrics in rows:
            values = [format_value(metrics.get(key, "NA")) for key in args.keys]
            print("\t".join([path.name, str(step)] + values))


if __name__ == "__main__":
    main()
