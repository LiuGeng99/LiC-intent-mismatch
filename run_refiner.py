import argparse
import json
from pathlib import Path

from model_openai import generate

ROOT = Path(__file__).resolve().parent


def build_examples(items):
    blocks = []
    for i, item in enumerate(items):
        conversation = item["conversation"]
        ground_truth = item["ground_truth"]
        blocks.append(
            f"[Conversation Transcript {i}]\n{conversation}\n\n"
            f"[Ground Truth Rewritten Request {i}]\n{ground_truth}\n"
        )
    return "\n".join(blocks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", required=True, help="JSON list of {conversation, ground_truth}")
    parser.add_argument("--model", default="deepseek-v3.2-thinking")
    parser.add_argument("--output", default=None, help="Where to write the rules. Default is stdout.")
    args = parser.parse_args()

    items = json.loads(Path(args.examples).read_text())
    if not items:
        raise SystemExit("examples is empty")

    prompt = (ROOT / "prompts" / "refiner.txt").read_text()
    message = prompt.replace("[[CONVERSATIONS]]", build_examples(items))
    response = generate(
        [{"role": "user", "content": message}],
        model=args.model,
        temperature=1.0,
        return_metadata=True,
    )
    rules = response["message"]
    if args.output:
        Path(args.output).write_text(rules)
    print(rules)


if __name__ == "__main__":
    main()
