from __future__ import annotations

import argparse

CASES = {
    "small": {"prompt_tokens": 300, "completion_tokens": 120, "latency_s": 0.8},
    "medium": {"prompt_tokens": 1800, "completion_tokens": 220, "latency_s": 2.4},
    "large": {"prompt_tokens": 9000, "completion_tokens": 350, "latency_s": 7.1},
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Token+latency cost proxy table.")
    parser.add_argument("--prompt-size", choices=["small", "medium", "large", "all"], default="all")
    args = parser.parse_args(argv)

    names = [args.prompt_size] if args.prompt_size != "all" else ["small", "medium", "large"]

    print("Prompt size cost proxy (tokens + latency)")
    print("size\tprompt\tcompletion\ttotal\tlatency_s")
    for name in names:
        row = CASES[name]
        total = row["prompt_tokens"] + row["completion_tokens"]
        print(f"{name}\t{row['prompt_tokens']}\t{row['completion_tokens']}\t{total}\t{row['latency_s']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
