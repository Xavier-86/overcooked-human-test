#!/usr/bin/env python3
"""
Independent CLI for decrypting test result files.

The administrator uses this tool to read encrypted result files sent by
participants.  Result files are encrypted with the administrator's RSA public
key; only the matching private key can decrypt them.

Usage:
    python results_reader.py path/to/results.json.enc
    python results_reader.py path/to/results.json.enc --private-key private_key.pem
    python results_reader.py path/to/results.json.enc --private-key private_key.pem --json
    python results_reader.py path/to/results.json.enc --private-key private_key.pem --csv output.csv
"""

import argparse
import csv
import getpass
import json
import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from human_test.crypto_utils import read_encrypted_results


def decrypt_file(enc_path: str, private_key_pem: bytes) -> list:
    """Decrypt a results file and return the JSON array."""
    return read_encrypted_results(enc_path, private_key_pem=private_key_pem)


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


def prompt_private_key_path() -> str:
    """Prompt for the private-key file path interactively."""
    try:
        path = getpass.getpass("Private key file path: ")
    except Exception:
        path = input("Private key file path: ")
    return path.strip()


def load_private_key_pem(path: str) -> bytes:
    """Load and validate the RSA private key file."""
    with open(path, "rb") as f:
        pem = f.read()
    # Validate by loading it (cryptography will raise on bad PEM).
    from cryptography.hazmat.primitives import serialization
    serialization.load_pem_private_key(pem, password=None)
    return pem


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="results_reader.py",
        description="Decrypt and read encrypted test result files with the administrator RSA private key.",
    )
    parser.add_argument("file", help="Path to the encrypted results file")
    parser.add_argument(
        "--private-key", type=str, default=None,
        help="Path to the RSA private key PEM file (if omitted, you will be prompted)",
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

    key_path = args.private_key or prompt_private_key_path()
    if not key_path:
        print("[ERR] Private key path is required.")
        return 1

    if not os.path.isfile(key_path):
        print(f"[ERR] Private key file not found: {key_path}")
        return 1

    try:
        private_pem = load_private_key_pem(key_path)
    except Exception as e:
        print(f"[ERR] Invalid private key: {e}")
        return 1

    try:
        results = decrypt_file(args.file, private_pem)
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
