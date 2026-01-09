"""CLI utility for removing extra spaces between Khmer characters."""

import argparse
from pathlib import Path
from typing import Optional
import pandas as pd
import re

from src import config


def remove_space_between_khmer(text: Optional[str]):
    """Collapse spaces that appear between Khmer characters while keeping other spacing intact."""
    if pd.isna(text):
        return text

    cleaned = re.sub(r'([\u1780-\u17FF])\s+([\u1780-\u17FF])', r'\1\2', text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def clean_file(input_path: str, output_path: str) -> None:
    df = pd.read_csv(input_path)
    for col in df.columns:
        df[col] = df[col].apply(remove_space_between_khmer)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"✓ Khmer inter-word spaces removed: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Remove spaces between Khmer characters in a CSV file.")
    parser.add_argument(
        "--input",
        default=config.RAW_DATA_PATH,
        help="Input CSV path (default: config.RAW_DATA_PATH)"
    )
    parser.add_argument(
        "--output",
        default=config.CLEANED_DATA_PATH,
        help="Output CSV path (default: config.CLEANED_DATA_PATH)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    clean_file(args.input, args.output)


if __name__ == "__main__":
    main()
