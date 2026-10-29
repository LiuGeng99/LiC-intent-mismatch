
import json
from collections import defaultdict
from pathlib import Path

from data_split import HELD_OUT

ROOT = Path(__file__).resolve().parent
TASKS = ["code", "database", "actions", "math"]

def score_file(path, exclude_ids=None):
    by_id = defaultdict(list)
    total = 0.0
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if exclude_ids and str(row["task_id"]) in exclude_ids:
                continue
            value = row.get("score")
            if value is None:
                value = 0.0
            value = float(value)
            by_id[str(row["task_id"])].append(value)
            total += value
            n += 1
    if n == 0:
        return None
    performance = 100.0 * total / n
    reliability = 100.0 * sum(1.0 - (max(scores) - min(scores)) for scores in by_id.values()) / len(by_id)
    return n, len(by_id), performance, reliability

def main():
    logs = ROOT / "logs"
    files = sorted(logs.rglob("*.jsonl"))
    if not files:
        print(f"No logs under {logs}")
        return
    print(f"{'log':<72} {'n':>5} {'inst':>5} {'P':>6} {'R':>6}")
    for path in files:
        if path.stat().st_size == 0:
            continue
        name = str(path.relative_to(logs))
        if not any(f"_{task}_" in name for task in TASKS):
            continue
        result = score_file(path, HELD_OUT)
        if result is None:
            continue
        n, n_inst, performance, reliability = result
        print(f"{name:<72} {n:5d} {n_inst:5d} {performance:6.1f} {reliability:6.1f}")

if __name__ == "__main__":
    main()
