#!/usr/bin/env python3
"""
Independent CLI for decrypting test result files.

The administrator uses this tool to read encrypted result files sent by
participants.  The decryption key is NOT embedded in this script; it must
be provided interactively or via command line.

Usage:
    python results_reader.py path/to/results.json.enc
    python results_reader.py path/to/results.json.enc --json
    python results_reader.py path/to/results.json.enc --csv output.csv
    python results_reader.py path/to/results.json.enc --key "YOUR_KEY"
"""

import argparse
import csv
import getpass
import json
import os
import sys

from cryptography.fernet import Fernet


def decrypt_file(enc_path: str, key: str) -> list:
    """Decrypt a results file and return the JSON array."""
    with open(enc_path, "rb") as f:
        data = f.read()
    raw = Fernet(key).decrypt(data)
    return json.loads(raw.decode("utf-8"))


def print_table(results: list):
    """Print results as a formatted table."""
    if not results:
        print("No results found.")
        return

    # Determine column widths
    headers = ["#", "Env", "Algo", "Pos", "Eps", "Avg Score", "Soups", "Timestamp"]
    rows = []
    for i, r in enumerate(results, 1):
        rows.append([
            str(i),
            r.get("env_name", "-"),
            r.get("algo", "-"),
            str(r.get("human_player", "-")),
            str(r.get("episodes", "-")),
            f"{r.get('avg_score', 0):.1f}",
            str(r.get("total_soups", "-")),
            r.get("timestamp", "-")[:19],
        ])

    # Calculate widths
    widths = [len(h) for h in headers]
    for row in rows:
        for j, cell in enumerate(row):
            widths[j] = max(widths[j], len(cell))

    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"

    def fmt(cells):
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"

    print(sep)
    print(fmt(headers))
    print(sep)
    for row in rows:
        print(fmt(row))
    print(sep)
    print(f"\nTotal: {len(results)} test(s)")


def print_json(results: list):
    """Print results as pretty-printed JSON."""
    print(json.dumps(results, indent=2, ensure_ascii=False))


def write_csv(results: list, csv_path: str):
    """Export results to a CSV file."""
    if not results:
        print("No results to export.")
        return

    fieldnames = [
        "test_index", "env_name", "algo", "human_player", "episodes",
        "avg_score", "total_soups", "scores", "durations", "soups_cooked",
        "timestamp",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: r.get(k, "") for k in fieldnames}
            # Convert lists to strings for CSV
            for k in ("scores", "durations", "soups_cooked"):
                if isinstance(row[k], list):
                    row[k] = json.dumps(row[k])
            writer.writerow(row)

    print(f"[OK] Exported to {csv_path}")


def prompt_key() -> str:
    """Prompt for decryption key interactively."""
    try:
        key = getpass.getpass("Decryption key: ")
    except Exception:
        key = input("Decryption key: ")
    return key.strip()


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="results_reader.py",
        description="Decrypt and read encrypted test result files.",
    )
    parser.add_argument("file", help="Path to the encrypted results file")
    parser.add_argument(
        "--key", type=str, default=None,
        help="Decryption key (if omitted, you will be prompted)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output as JSON instead of table",
    )
    parser.add_argument(
        "--csv", type=str, default=None, metavar="PATH",
        help="Export results to a CSV file",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.file):
        print(f"[ERR] File not found: {args.file}")
        return 1

    key = args.key or prompt_key()
    if not key:
        print("[ERR] Decryption key is required.")
        return 1

    # Validate key format
    try:
        Fernet(key)
    except Exception:
        print("[ERR] Invalid key format.  Expected a 32-byte base64-encoded Fernet key.")
        return 1

    try:
        results = decrypt_file(args.file, key)
    except Exception as e:
        print(f"[ERR] Decryption failed: {e}")
        return 1

    if args.csv:
        write_csv(results, args.csv)

    if args.json:
        print_json(results)
    else:
        print_table(results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
