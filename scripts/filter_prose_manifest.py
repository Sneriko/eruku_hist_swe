"""Exclude table leaders from an extracted JSONL manifest without changing crops."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re


DASH_RUN = re.compile(r"[-‐‑‒–—―−](?:\s*[-‐‑‒–—―−]){2,}")


def exclusion_reason(text):
    if DASH_RUN.search(text):
        return "repeated_dashes"
    if any(char in text for char in "—―⸺⸻"):
        return "long_dash_leader"
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.source.resolve().parent == args.output.resolve():
        raise ValueError("Use a separate output directory")
    counts = defaultdict(Counter)
    reasons, heights, splits = Counter(), Counter(), Counter()
    with args.source.open() as source, (args.output / "split.jsonl").open("x") as kept, (args.output / "excluded.jsonl").open("x") as excluded:
        for line in source:
            row = json.loads(line)
            century = str(row.get("year_start", ""))[:2]
            century = century + "00–" + century + "99" if len(century) == 2 else "unknown"
            reason = exclusion_reason(row["text"])
            counts[century]["total"] += 1
            counts[century]["excluded" if reason else "retained"] += 1
            if reason:
                reasons[reason] += 1
                excluded.write(json.dumps({**row, "exclusion_reason": reason}, ensure_ascii=False) + "\n")
            else:
                kept.write(line if line.endswith("\n") else line + "\n")
                heights[str(row["height"])] += 1
                splits[row["split"]] += 1
    report = dict(source=str(args.source.resolve()), by_century=dict(counts), exclusion_reasons=dict(reasons), retained_heights=dict(heights), retained_splits=dict(splits),
                  policy="Exclude complete lines with 3+ dash characters (optional intervening whitespace), or any em dash, horizontal bar, two-em dash or three-em dash. Preserve other single/double hyphens and ¬. Preserve original splits, keys, image paths, dimensions and text. Applies to both style and target pools when used by LinePairs.")
    (args.output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
