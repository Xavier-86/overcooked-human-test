#!/usr/bin/env python3
"""
Transparent encryption layer for exp_configs/.

All files under exp_configs/ are stored encrypted on disk.
At runtime they are decrypted in memory only.

The encryption key is embedded in this module (obfuscated).
The administrator keeps the same raw key to decrypt files via
results_reader.py.  Participants cannot easily discover the key
from source code alone.
"""

import json
import os
import base64
from pathlib import Path
from cryptography.fernet import Fernet


# Embedded key (obfuscated via extra base64 layer).
# The administrator must save the raw key generated during first-run setup.
_ENCODED_KEY = "Smg1ZHdGVjFhQ2tfRWNyYlJzQVNNdXRFT0FSWnZSYURtcjVKblJlUURzWT0="

# Runtime override (used during first-run key setup).
_OVERRIDE_KEY = None


def _get_key() -> str:
    """Return the embedded encryption key."""
    if _OVERRIDE_KEY is not None:
        return _OVERRIDE_KEY
    return base64.b64decode(_ENCODED_KEY).decode()


def set_key(key: str):
    """Override the embedded key at runtime."""
    global _OVERRIDE_KEY
    _OVERRIDE_KEY = key


def _fernet() -> Fernet:
    return Fernet(_get_key())


def encrypt_file(plain_path: str, enc_path: str = None):
    """Encrypt a plaintext file and write the .enc version."""
    if enc_path is None:
        enc_path = plain_path + ".enc"
    with open(plain_path, "rb") as f:
        data = f.read()
    encrypted = _fernet().encrypt(data)
    with open(enc_path, "wb") as f:
        f.write(encrypted)


def decrypt_file(enc_path: str, plain_path: str = None):
    """Decrypt an encrypted file and write the plaintext version (admin tool)."""
    if plain_path is None:
        plain_path = enc_path.replace(".enc", "")
    with open(enc_path, "rb") as f:
        data = f.read()
    decrypted = _fernet().decrypt(data)
    with open(plain_path, "wb") as f:
        f.write(decrypted)


def read_encrypted(enc_path: str) -> bytes:
    """Read and decrypt a file, returning bytes (for JSON parsing)."""
    if not os.path.exists(enc_path):
        raise FileNotFoundError(f"Encrypted file not found: {enc_path}")
    with open(enc_path, "rb") as f:
        data = f.read()
    return _fernet().decrypt(data)


def write_encrypted(enc_path: str, data: bytes):
    """Encrypt and write data to a file."""
    encrypted = _fernet().encrypt(data)
    os.makedirs(os.path.dirname(enc_path), exist_ok=True)
    with open(enc_path, "wb") as f:
        f.write(encrypted)


def read_encrypted_json(enc_path: str) -> dict:
    """Convenience: read encrypted JSON file and return dict."""
    raw = read_encrypted(enc_path)
    return json.loads(raw.decode("utf-8"))


def write_encrypted_json(enc_path: str, obj: dict):
    """Convenience: write dict to encrypted JSON file."""
    data = json.dumps(obj, indent=2, ensure_ascii=False).encode("utf-8")
    write_encrypted(enc_path, data)


def append_encrypted_jsonl(enc_path: str, obj: dict):
    """Convenience: append a JSON line to an encrypted JSONL file."""
    line = json.dumps(obj, ensure_ascii=False) + "\n"
    existing = b""
    if os.path.exists(enc_path):
        existing = read_encrypted(enc_path)
    new_data = existing + line.encode("utf-8")
    write_encrypted(enc_path, new_data)


def read_encrypted_jsonl(enc_path: str) -> list:
    """Convenience: read encrypted JSONL file and return list of dicts."""
    if not os.path.exists(enc_path):
        return []
    raw = read_encrypted(enc_path)
    results = []
    for line in raw.decode("utf-8").splitlines():
        line = line.strip()
        if line:
            results.append(json.loads(line))
    return results


def generate_key() -> str:
    """Generate a new Fernet key for the administrator."""
    return Fernet.generate_key().decode()


# ---------------------------------------------------------------------------
# Test result helpers (same embedded key, admin decrypts with results_reader.py)
# ---------------------------------------------------------------------------


def read_encrypted_results(enc_path: str) -> list:
    """Read and decrypt a test results file (JSON array)."""
    if not os.path.exists(enc_path):
        return []
    with open(enc_path, "rb") as f:
        data = f.read()
    raw = _fernet().decrypt(data)
    return json.loads(raw.decode("utf-8"))


def write_encrypted_results(enc_path: str, results: list):
    """Encrypt and write a test results file (JSON array)."""
    data = json.dumps(results, indent=2, ensure_ascii=False).encode("utf-8")
    encrypted = _fernet().encrypt(data)
    os.makedirs(os.path.dirname(enc_path), exist_ok=True)
    with open(enc_path, "wb") as f:
        f.write(encrypted)
