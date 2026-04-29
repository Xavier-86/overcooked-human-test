#!/usr/bin/env python3
"""
Admin tool for encrypting / decrypting exp_configs/ files.

All files under exp_configs/ must be encrypted before the CLI is used.
The encryption key is embedded in crypto_utils.py (obfuscated).

Participants do not need to know the key; they interact with the CLI only.
The administrator uses results_reader.py to decrypt participant result files.
"""

import argparse
import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from human_test.crypto_utils import (
    encrypt_file,
    decrypt_file,
    generate_key,
)


def cmd_generate_key(_args):
    key = generate_key()
    print(key)


def cmd_encrypt(args):
    if not os.path.isfile(args.file):
        print(f"[ERR] File not found: {args.file}")
        sys.exit(1)
    enc_path = args.file + ".enc"
    encrypt_file(args.file, enc_path)
    print(f"[OK] Encrypted: {args.file} -> {enc_path}")
    if args.remove_plain:
        os.remove(args.file)
        print(f"[OK] Removed plaintext: {args.file}")


def cmd_decrypt(args):
    if not os.path.isfile(args.file):
        print(f"[ERR] File not found: {args.file}")
        sys.exit(1)
    if not args.file.endswith(".enc"):
        print("[WARN] File does not end with .enc; decrypting anyway.")
    plain_path = args.file.replace(".enc", "")
    decrypt_file(args.file, plain_path)
    print(f"[OK] Decrypted: {args.file} -> {plain_path}")


def cmd_encrypt_dir(args):
    target_dir = args.dir
    if not os.path.isdir(target_dir):
        print(f"[ERR] Directory not found: {target_dir}")
        sys.exit(1)

    encrypted = 0
    for root, _dirs, files in os.walk(target_dir):
        for fname in files:
            if fname.endswith(".enc"):
                continue
            if not fname.endswith(".json") and not fname.endswith(".jsonl"):
                continue
            plain_path = os.path.join(root, fname)
            enc_path = plain_path + ".enc"
            encrypt_file(plain_path, enc_path)
            print(f"  [OK] {plain_path} -> {enc_path}")
            if args.remove_plain:
                os.remove(plain_path)
            encrypted += 1

    print(f"[OK] Encrypted {encrypted} file(s) in {target_dir}")


def cmd_decrypt_dir(args):
    target_dir = args.dir
    if not os.path.isdir(target_dir):
        print(f"[ERR] Directory not found: {target_dir}")
        sys.exit(1)

    decrypted = 0
    for root, _dirs, files in os.walk(target_dir):
        for fname in files:
            if not fname.endswith(".enc"):
                continue
            enc_path = os.path.join(root, fname)
            plain_path = enc_path.replace(".enc", "")
            decrypt_file(enc_path, plain_path)
            print(f"  [OK] {enc_path} -> {plain_path}")
            decrypted += 1

    print(f"[OK] Decrypted {decrypted} file(s) in {target_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="admin_crypto.py",
        description="Administrator encryption/decryption tool for exp_configs/.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("generate-key", help="Generate a new Fernet key")

    p_enc = sub.add_parser("encrypt", help="Encrypt a single file")
    p_enc.add_argument("file", help="Path to plaintext file")
    p_enc.add_argument(
        "--remove-plain", action="store_true",
        help="Delete the original plaintext after encryption",
    )

    p_dec = sub.add_parser("decrypt", help="Decrypt a single .enc file")
    p_dec.add_argument("file", help="Path to encrypted file")

    p_enc_dir = sub.add_parser("encrypt-dir", help="Encrypt all JSON/JSONL files in a directory")
    p_enc_dir.add_argument("dir", help="Target directory")
    p_enc_dir.add_argument(
        "--remove-plain", action="store_true",
        help="Delete original plaintext files after encryption",
    )

    p_dec_dir = sub.add_parser("decrypt-dir", help="Decrypt all .enc files in a directory")
    p_dec_dir.add_argument("dir", help="Target directory")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    commands = {
        "generate-key": cmd_generate_key,
        "encrypt": cmd_encrypt,
        "decrypt": cmd_decrypt,
        "encrypt-dir": cmd_encrypt_dir,
        "decrypt-dir": cmd_decrypt_dir,
    }

    cmd = commands.get(args.command)
    if cmd is None:
        parser.print_help()
        sys.exit(1)

    cmd(args)


if __name__ == "__main__":
    main()
