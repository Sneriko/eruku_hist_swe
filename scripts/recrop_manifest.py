"""Regenerate native-resolution crops from an existing line manifest."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from tqdm import tqdm
from historical_swedish.common import crop_polygon, normalize_background, to_line_height


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--max-width", type=int, default=4096)
    parser.add_argument("--padding", type=int, default=8)
    args = parser.parse_args()
    if args.height <= 0 or args.height % 8 or args.max_width <= 0 or args.padding < 0:
        parser.error("height must be a positive multiple of 8; max-width positive; padding nonnegative")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Refuse to overwrite an existing extraction, including an interrupted run.
    manifest_path = out / "split.jsonl"
    with manifest_path.open("x") as destination:
        raw_dir, normalized_dir = out / "lines/raw", out / "lines/normalized"
        raw_dir.mkdir(parents=True, exist_ok=True)
        normalized_dir.mkdir(parents=True, exist_ok=True)
        counts = Counter()
        seen = set()
        page_path, page = None, None
        try:
            with args.manifest.open() as source, (out / "excluded.jsonl").open("x") as excluded:
                for line in tqdm(source, desc="Recropping lines", unit=" lines"):
                    row = json.loads(line)
                    key = row["key"]
                    if key in seen or Path(key).name != key or key in {".", "..", ""}:
                        raise ValueError(f"Duplicate or unsafe line key: {key!r}")
                    seen.add(key)
                    if row["image_path"] != page_path:
                        if page is not None:
                            page.close()
                        with Image.open(row["image_path"]) as original:
                            page = original.convert("RGB")
                        page_path = row["image_path"]
                    crop = to_line_height(crop_polygon(page, row["polygon"], args.padding), args.height, args.max_width)
                    if crop is None:
                        excluded.write(json.dumps({**row, "exclusion_reason": "oversized_at_requested_height"}, ensure_ascii=False) + "\n")
                        counts["oversized"] += 1
                        continue
                    raw_path, normalized_path = raw_dir / f"{key}.png", normalized_dir / f"{key}.png"
                    crop.save(raw_path)
                    normalize_background(crop).save(normalized_path)
                    row.update(raw_path=str(raw_path), normalized_path=str(normalized_path), width=crop.width, height=crop.height)
                    destination.write(json.dumps(row, ensure_ascii=False) + "\n")
                    counts["retained"] += 1
                    counts[f"split/{row['split']}"] += 1
        finally:
            if page is not None:
                page.close()
    report = dict(source=str(args.manifest.resolve()), height=args.height, max_width=args.max_width,
                  padding=args.padding, counts=dict(counts), complete=True)
    (out / "extraction_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
