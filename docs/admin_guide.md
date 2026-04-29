# Admin Guide

> [README](../README.md) · [中文](admin_guide.zh.md) · [User Guide](user_guide.md) · [CLI Reference](cli.md)

---

## First-Time Setup

When the CLI is launched for the first time and no `.enc` files exist, the administrator must set up the encryption key.

```bash
overcooked-human-test
```

You will be prompted to either **generate a new key** or **enter an existing key**. The key is embedded into `human_test/crypto_utils.py` (obfuscated). **Save the raw key securely** -- you will need it to decrypt participant result files later.

After the key is set, any plaintext JSON files under `exp_configs/` are automatically encrypted.

---

## Distribution Workflow

To prepare the project for participants, follow these steps:

1. **Generate a new key** (or enter an existing one) on first launch:
   ```bash
   overcooked-human-test
   ```
   The key is embedded into `human_test/crypto_utils.py`.

2. **Save the raw key securely** (e.g. in a password manager). You will need it later to decrypt `results.json.enc` files sent by participants.

3. **Prepare the experiment config** (see below) and encrypt all files under `exp_configs/`.

4. **Remove any sensitive plaintext files** (logs, temporary JSONs, old keys).

5. **Distribute the project** to participants. They do **not** need the raw key -- the embedded obfuscated key is sufficient for encryption and local gameplay.

6. **After participants finish testing**, they send their `results.json.enc` file back to you. Decrypt it with `results_reader.py` using the raw key you saved in step 2.

---

## Experiment Configuration

Use `experiment_config_template.json` as a starting point. **Remember to remove `_template` from the filename** so it becomes `experiment_config.json` — the CLI only loads files without the `_template` suffix:

```json
{
  "experiment_name": "overcooked_human_ai_test",
  "environments": ["random1_m", "random3_m"],
  "positions": [0, 1],
  "episodes_per_config": 2,
  "algorithm_pool": ["bach", "fcp", "mep"]
}
```

| Field | Description |
|-------|-------------|
| `environments` | List of layout names to test |
| `positions` | Human player positions (0 or 1) |
| `episodes_per_config` | How many times each algorithm appears per test |
| `algorithm_pool` | AI algorithms in the pool (never shown to users) |

Place the config at `exp_configs/experiment_config.json`. The CLI will encrypt it automatically on first run if a key is already set.

---

## Admin CLI Menu

From the main menu (logged out), select **Admin** and enter the encryption key as the password. You can:

- View registered users
- Decrypt and inspect result files in-place

---

## File Structure

```
exp_configs/
├── experiment_config.json.enc      # Encrypted experiment schedule
├── experiment_config_template.json # Plaintext template (not loaded)
├── policy_pool/                    # Trained model files
└── user_data/
    ├── users.json.enc              # Encrypted user accounts
    └── <user_id>/
        ├── progress.json.enc       # Encrypted test progress
        └── results.json.enc        # Encrypted test results
```

All `.enc` files are encrypted with the key embedded in `crypto_utils.py`.

---

## Policy Network

Policies in `policy_pool/` use the **default network structure from [ZSC-Eval](https://github.com/sjtu-marl/ZSC-Eval)**. When training or replacing models, ensure they are compatible with this architecture.

---

## Command-Line Tools

For full command-line options of `admin_crypto.py` and `results_reader.py`, see the [CLI Reference](cli.md).
