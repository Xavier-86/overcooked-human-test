#!/usr/bin/env python3
"""
Transparent encryption layer for exp_configs/.

Files under exp_configs/ that are read by the local client (experiment config,
user accounts, progress) are encrypted with a Fernet key.  The Fernet key is no
longer hard-coded as the only source: it is loaded from the environment or an
external key file first, and only falls back to the embedded value for backward
compatibility.

Test result files are encrypted with the administrator's RSA **public key** so
participants can produce them but cannot read or tamper with them.  Only the
administrator, who holds the matching **private key**, can decrypt result files.

The RSA public key can safely be embedded in the distributed project.  The
private key must never be distributed.
"""

import json
import os
import base64
import struct
from pathlib import Path
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa


# ---------------------------------------------------------------------------
# Fernet key for config / users / progress files (client-readable)
# ---------------------------------------------------------------------------

# Embedded key (single base64 layer).  Kept only as a fallback for legacy setups.
_ENCODED_KEY = "Smg1ZHdGVjFhQ2tfRWNyYlJzQVNNdXRFT0FSWnZSYURtcjVKblJlUURzWT0="

# Runtime override (used during first-run key setup).
_OVERRIDE_KEY = None


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _get_key() -> str:
    """Return the Fernet key used for config/users/progress files.

    Resolution order:
      1. Runtime override via set_key().
      2. OVERCOOKED_CONFIG_KEY environment variable.
      3. config.key file in the project root.
      4. The embedded fallback key.
    """
    if _OVERRIDE_KEY is not None:
        return _OVERRIDE_KEY

    env_key = os.environ.get("OVERCOOKED_CONFIG_KEY")
    if env_key:
        return env_key.strip()

    key_file = os.path.join(_project_root(), "config.key")
    if os.path.isfile(key_file):
        with open(key_file, "r", encoding="utf-8") as f:
            key = f.read().strip()
        if key:
            return key

    return base64.b64decode(_ENCODED_KEY).decode()


def set_key(key: str):
    """Override the config Fernet key at runtime."""
    global _OVERRIDE_KEY
    _OVERRIDE_KEY = key


def _fernet() -> Fernet:
    return Fernet(_get_key())


# ---------------------------------------------------------------------------
# RSA public key for result files (admin-readable only)
# ---------------------------------------------------------------------------

# Runtime override for the RSA public key PEM bytes.
_OVERRIDE_PUBLIC_KEY = None


def get_public_key_pem() -> bytes:
    """Return the RSA public key PEM used to encrypt result files.

    Resolution order:
      1. Runtime override via set_public_key().
      2. OVERCOOKED_PUBLIC_KEY_PATH environment variable.
      3. public_key.pem file in the project root.
      4. The embedded fallback public key (if any).
    """
    if _OVERRIDE_PUBLIC_KEY is not None:
        return _OVERRIDE_PUBLIC_KEY

    env_path = os.environ.get("OVERCOOKED_PUBLIC_KEY_PATH")
    if env_path and os.path.isfile(env_path):
        with open(env_path, "rb") as f:
            return f.read()

    pem_file = os.path.join(_project_root(), "public_key.pem")
    if os.path.isfile(pem_file):
        with open(pem_file, "rb") as f:
            return f.read()

    return _EMBEDDED_PUBLIC_KEY.encode("utf-8")


def set_public_key(pem: bytes):
    """Override the RSA public key at runtime."""
    global _OVERRIDE_PUBLIC_KEY
    _OVERRIDE_PUBLIC_KEY = pem


# Embedded fallback public key.  During first-run setup the CLI rewrites this
# line with the generated public key.  It is safe to distribute.
_EMBEDDED_PUBLIC_KEY = """"""


def generate_rsa_keypair() -> tuple[bytes, bytes]:
    """Generate a new RSA key pair.  Returns (private_key_pem, public_key_pem)."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return private_pem, public_pem


def _load_public_key(pem: bytes):
    return serialization.load_pem_public_key(pem)


def _load_private_key(pem: bytes):
    return serialization.load_pem_private_key(pem, password=None)


# Format version marker for RSA-encrypted result files.
_RSA_FORMAT_MAGIC = b"RSA1"


def encrypt_results_for_admin(data: bytes, public_key_pem: Optional[bytes] = None) -> bytes:
    """Encrypt result data with RSA+OAEP hybrid encryption.

    A random Fernet key encrypts the payload; the Fernet key is then encrypted
    with the RSA public key.  This allows encrypting arbitrarily large payloads
    while keeping the public-key material safe to distribute.
    """
    pem = public_key_pem if public_key_pem is not None else get_public_key_pem()
    if not pem:
        raise RuntimeError("No RSA public key available for result encryption.")

    public_key = _load_public_key(pem)
    fernet_key = Fernet.generate_key()
    encrypted_data = Fernet(fernet_key).encrypt(data)
    encrypted_key = public_key.encrypt(
        fernet_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )

    # Format: magic(4) + key_len(2 BE) + encrypted_key + encrypted_data
    return (
        _RSA_FORMAT_MAGIC
        + struct.pack(">H", len(encrypted_key))
        + encrypted_key
        + encrypted_data
    )


def decrypt_results_with_private_key(data: bytes, private_key_pem: bytes) -> bytes:
    """Decrypt result data that was encrypted with the matching RSA private key."""
    if not data.startswith(_RSA_FORMAT_MAGIC):
        raise ValueError("Not an RSA-encrypted result file (missing magic marker).")

    private_key = _load_private_key(private_key_pem)
    key_len = struct.unpack(">H", data[4:6])[0]
    encrypted_key = data[6:6 + key_len]
    encrypted_data = data[6 + key_len:]

    fernet_key = private_key.decrypt(
        encrypted_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    return Fernet(fernet_key).decrypt(encrypted_data)


# ---------------------------------------------------------------------------
# Generic Fernet file helpers (config / users / progress)
# ---------------------------------------------------------------------------


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
    parent = os.path.dirname(enc_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
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
# Result file helpers
# ---------------------------------------------------------------------------


def _looks_like_rsa(data: bytes) -> bool:
    return data.startswith(_RSA_FORMAT_MAGIC)


def read_encrypted_results(
    enc_path: str,
    private_key_pem: Optional[bytes] = None,
) -> list:
    """Read and decrypt a test results file (JSON array).

    If the file is in the new RSA format, a private key must be supplied.
    Legacy Fernet-encrypted result files are still readable as a fallback when
    no private key is provided.
    """
    if not os.path.exists(enc_path):
        return []
    with open(enc_path, "rb") as f:
        data = f.read()

    if _looks_like_rsa(data):
        if private_key_pem is None:
            raise RuntimeError(
                "Result file is RSA-encrypted; provide the administrator private key."
            )
        raw = decrypt_results_with_private_key(data, private_key_pem)
        return json.loads(raw.decode("utf-8"))

    # Legacy Fernet-encrypted result file (for backward compatibility).
    try:
        raw = _fernet().decrypt(data)
        return json.loads(raw.decode("utf-8"))
    except InvalidToken as exc:
        raise RuntimeError(
            "Failed to decrypt result file.  It may be RSA-encrypted and require a private key."
        ) from exc


def write_encrypted_results(enc_path: str, results: list, public_key_pem: Optional[bytes] = None):
    """Encrypt and write a test results file with the admin RSA public key."""
    data = json.dumps(results, indent=2, ensure_ascii=False).encode("utf-8")
    encrypted = encrypt_results_for_admin(data, public_key_pem)
    parent = os.path.dirname(enc_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(enc_path, "wb") as f:
        f.write(encrypted)


# ---------------------------------------------------------------------------
# Local result cache helpers (client-readable, used for UI aggregation)
# ---------------------------------------------------------------------------


def read_local_results(enc_path: str) -> list:
    """Read a locally cached results file encrypted with the config Fernet key."""
    if not os.path.exists(enc_path):
        return []
    with open(enc_path, "rb") as f:
        data = f.read()
    raw = _fernet().decrypt(data)
    return json.loads(raw.decode("utf-8"))


def write_local_results(enc_path: str, results: list):
    """Write a locally cached results file encrypted with the config Fernet key."""
    data = json.dumps(results, indent=2, ensure_ascii=False).encode("utf-8")
    encrypted = _fernet().encrypt(data)
    parent = os.path.dirname(enc_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(enc_path, "wb") as f:
        f.write(encrypted)
