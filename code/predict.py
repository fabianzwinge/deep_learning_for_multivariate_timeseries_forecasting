"""Inference entrypoint for final private evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.tft_inference import predict


def typed_path(value: list[str] | None, expected_kind: str) -> Path | None:
    """Parse the alternate handout style: --input dir PATH / --output file PATH."""
    if value is None:
        return None
    if len(value) == 2 and value[0] == expected_kind:
        return Path(value[1])
    if len(value) == 1:
        return Path(value[0])
    raise SystemExit(f"Expected '--{expected_kind} {expected_kind} PATH', got: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TFT ensemble predictions.")
    parser.add_argument("--input_dir", type=Path, help="Input directory, e.g. /data/input")
    parser.add_argument("--output_file", type=Path, help="Output CSV, e.g. /output/predictions.csv")
    parser.add_argument("--input", nargs="+", help="Alternate style: --input dir /data/input")
    parser.add_argument("--output", nargs="+", help="Alternate style: --output file /output/predictions.csv")
    parser.add_argument("--checkpoint", required=True, type=Path)
    args = parser.parse_args()

    input_dir = typed_path(args.input, "dir") or args.input_dir
    output_file = typed_path(args.output, "file") or args.output_file
    if input_dir is None:
        raise SystemExit("Missing input directory. Use --input dir PATH or --input_dir PATH.")
    if output_file is None:
        raise SystemExit("Missing output file. Use --output file PATH or --output_file PATH.")

    predict(input_dir=input_dir, output_file=output_file, checkpoint_file=args.checkpoint)


if __name__ == "__main__":
    main()
