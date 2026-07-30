#!/usr/bin/env python3
"""
Admin tool for encrypting / decrypting exp_configs/ files.

Files under exp_configs/ that are read by the local client (experiment config,
user accounts, progress) are encrypted with the Fernet config key.  The config
key is loaded from the `config.key` file or the `OVERCOOKED_CONFIG_KEY`
environment variable.

Test result files (`results.json.enc`) are encrypted with the administrator's
RSA public key and must be decrypted with the matching private key.  Use
`results_reader.py` or the `decrypt-results` subcommand for those files.
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
    generate_rsa_keypair,
    read_encrypted_results,
)


def cmd_generate_key(_args):
    key = generate_key()
    print(key)


def cmd_generate_rsa_keypair(_args):
    private_pem, public_pem = generate_rsa_keypair()
    private_path = os.path.join(_project_root, "private_key.pem")
    public_path = os.path.join(_project_root, "public_key.pem")
    with open(private_path, "wb") as f:
        f.write(private_pem)
    with open(public_path, "wb") as f:
        f.write(public_pem)
    print(f"[OK] Private key: {private_path}")
    print(f"[OK] Public key:  {public_path}")
    print("[WARN] Keep private_key.pem secret.  public_key.pem can be distributed.")


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


def cmd_decrypt_results(args):
    if not os.path.isfile(args.file):
        print(f"[ERR] File not found: {args.file}")
        sys.exit(1)
    if not os.path.isfile(args.private_key):
        print(f"[ERR] Private key file not found: {args.private_key}")
        sys.exit(1)

    with open(args.private_key, "rb") as f:
        private_pem = f.read()

    try:
        results = read_encrypted_results(args.file, private_key_pem=private_pem)
    except Exception as e:
        print(f"[ERR] Decryption failed: {e}")
        sys.exit(1)

    print(f"[OK] Decrypted {len(results)} record(s)")
    import json
    print(json.dumps(results, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="admin_crypto.py",
        description="Administrator encryption/decryption tool for exp_configs/ and result files.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("generate-key", help="Generate a new Fernet key for config files")

    sub.add_parser("generate-rsa-keypair", help="Generate a new RSA key pair for result files")

    p_enc = sub.add_parser("encrypt", help="Encrypt a single config file")
    p_enc.add_argument("file", help="Path to plaintext file")
    p_enc.add_argument(
        "--remove-plain", action="store_true",
        help="Delete the original plaintext after encryption",
    )

    p_dec = sub.add_parser("decrypt", help="Decrypt a single .enc config file")
    p_dec.add_argument("file", help="Path to encrypted file")

    p_enc_dir = sub.add_parser("encrypt-dir", help="Encrypt all JSON/JSONL files in a directory")
    p_enc_dir.add_argument("dir", help="Target directory")
    p_enc_dir.add_argument(
        "--remove-plain", action="store_true",
        help="Delete original plaintext files after encryption",
    )

    p_dec_dir = sub.add_parser("decrypt-dir", help="Decrypt all .enc config files in a directory")
    p_dec_dir.add_argument("dir", help="Target directory")

    p_dec_res = sub.add_parser("decrypt-results", help="Decrypt a results.json.enc file with the RSA private key")
    p_dec_res.add_argument("file", help="Path to the encrypted results file")
    p_dec_res.add_argument(
        "--private-key", type=str, default="private_key.pem",
        help="Path to the RSA private key PEM file (default: private_key.pem)",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    commands = {
        "generate-key": cmd_generate_key,
        "generate-rsa-keypair": cmd_generate_rsa_keypair,
        "encrypt": cmd_encrypt,
        "decrypt": cmd_decrypt,
        "encrypt-dir": cmd_encrypt_dir,
        "decrypt-dir": cmd_decrypt_dir,
        "decrypt-results": cmd_decrypt_results,
    }

    cmd = commands.get(args.command)
    if cmd is None:
        parser.print_help()
        sys.exit(1)

    cmd(args)


if __name__ == "__main__":
    main()
