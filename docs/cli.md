# CLI Reference

> [README](../README.md) · For user-facing instructions, see [User Guide](user_guide.md) | [中文](cli.zh.md) · [Admin Guide](admin_guide.md)

## Quick Start Examples

### Interactive Menu (default, no arguments)

```bash
overcooked-human-test
```

Navigate with **UP/DOWN arrow keys**, press **ENTER** to select.

### Terminal demo (no models needed)

```bash
overcooked-human-test --demo
```

### Random AI opponent (no policy pool needed)

```bash
overcooked-human-test -e <env_name> --random
```

### Headless AI-vs-AI test

```bash
overcooked-human-test -e <env_name> --random --headless
```

### User registration and login

```bash
# Register a new user (interactive password prompt)
overcooked-human-test --register --user-id alice

# Login (interactive password prompt)
overcooked-human-test --login --user-id alice

# Check current user
overcooked-human-test --whoami

# Logout
overcooked-human-test --logout
```

---

## Full Option List

```
overcooked-human-test [OPTIONS]

Options:
  -e, --env ENV              Environment / layout name (default: <env_name>)
  -p, --human-player {0,1}   Human player index (default: 1)
  -n, --episodes N           Number of episodes (default: 1)
  --tile-size SIZE           Pygame tile size in pixels (default: 75)
  --fps FPS                  Game speed in steps/sec (default: 4)
  --policy-pool PATH         Path to the policy pool directory
  --random                   Use a random AI instead of loading a trained model
  --demo                     Run terminal demo mode (no models needed)
  --headless                 Headless AI-only mode (no graphics)
  --user-id USER_ID          User ID for experiment tracking
  --exp-config PATH          Path to experiment config JSON
  --register                 Register a new user (requires --user-id)
  --login                    Login as a user (requires --user-id)
  --logout                   Logout current user
  --whoami                   Show current logged-in user
  --list-envs                List available environments and exit
  -v, --version              Show version and exit
  -h, --help                 Show help and exit
```

---

## User Registration & Login

Before running tests, users must be registered:

```bash
overcooked-human-test --register --user-id alice
```

This creates a local account in `exp_configs/user_data/users.json.enc` (encrypted on disk). After registering, log in:

```bash
overcooked-human-test --login --user-id alice
```

Once logged in, the interactive menu will automatically use the current user.

---

## How It Works

Algorithm names are **never revealed** to participants. The system works as follows:

1. Register and log in (see above).
2. The experiment configuration (`exp_configs/experiment_config.json`) defines the test schedule: environments, positions, and episode counts.
3. There are **4 tests** (2 envs x 2 positions). Each test contains every algorithm from `algorithm_pool` repeated `episodes_per_config` times. The internal episode order is shuffled randomly per user.
4. When a logged-in user runs an episode, the CLI picks the next unfinished episode from the selected test.
5. During gameplay, only the **environment name** and **player position** are shown. The algorithm is hidden.
6. After each **episode** completes, the user returns to the menu and can choose to continue, switch tests, or exit.
7. Results are saved after every episode.

### Experiment configuration format

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
| `environments` | List of environment/layout names to test |
| `positions` | List of player positions (0 or 1) the human will take |
| `episodes_per_config` | How many times each algorithm appears inside a test |
| `algorithm_pool` | List of AI algorithms in the pool (never shown to user) |

In the default config this produces **4 tests**, each with **6 episodes** (3 algos x 2). The 6 episodes are shuffled randomly per user.

### Progress tracking

Per-user progress is stored in `exp_configs/user_data/<user_id>/progress.json.enc` (encrypted on disk):

```json
{
  "user_id": "alice",
  "total_tests": 4,
  "completed_tests": 1,
  "current_test_index": 1,
  "sequence": [
    {
      "index": 0,
      "env_name": "random1_m",
      "human_player": 0,
      "algo_sequence": ["bach", "mep", "fcp", "mep", "bach", "fcp"],
      "completed_episodes": 3,
      "total_episodes": 6
    }
  ],
  "created_at": "2026-04-28T10:00:00",
  "updated_at": "2026-04-28T10:30:00"
}
```

### Result data

All test results for a single user are stored in **one encrypted file**:
`exp_configs/user_data/<user_id>/results.json.enc`

The file contains a JSON array. Each element is one completed **episode**:

```json
[
  {
    "test_index": 0,
    "env_name": "<env_name>",
    "algo": "<algo>",
    "human_player": 1,
    "episode_number": 1,
    "total_episodes": 6,
    "scores": [120],
    "durations": [45.2],
    "soups_cooked": [2],
    "avg_score": 120.0,
    "total_soups": 2,
    "timestamp": "2026-04-28T10:30:00"
  }
]
```

Result files are encrypted with an embedded key so participants cannot read them. After testing, the participant sends the `results.json.enc` file to the administrator, who decrypts it with `results_reader.py`.

#### Decrypting result files (admin only)

```bash
python results_reader.py path/to/results.json.enc
# You will be prompted for the decryption key

# Or pass the key directly
python results_reader.py path/to/results.json.enc --key "YOUR_KEY"

# Export to CSV
python results_reader.py path/to/results.json.enc --key "YOUR_KEY" --csv output.csv

# Output as JSON
python results_reader.py path/to/results.json.enc --key "YOUR_KEY" --json
```

### Inspecting progress programmatically

```python
from experiment_manager import print_progress, load_all_results

# Print progress to console
print_progress("alice")

# Load all results as a list of dicts
results = load_all_results("alice")
```

---

## Policy Pool Resolution

The CLI resolves the policy pool directory in the following order:

1. `--policy-pool PATH` (if provided)
2. `POLICY_POOL` environment variable
3. `exp_configs/policy_pool/` (relative to project root)
4. `zsceval/policy_pool/` (relative to project root)

---

## Encryption

All files under `exp_configs/` are stored **encrypted** on disk. The encryption key is embedded in `crypto_utils.py` (obfuscated). Participants cannot read raw data by opening files.

To prepare a project for distribution:

```bash
# Encrypt the experiment config (if it is still plaintext)
python admin_crypto.py encrypt exp_configs/experiment_config.json --remove-plain
```

To decrypt files as administrator:

```bash
python admin_crypto.py decrypt exp_configs/experiment_config.json.enc
```

---

## Available Environments

| Environment | Description |
|-------------|-------------|
| `<env_name>` | Basic kitchen (recommended for beginners) |
| `<env_name>` | Three-ingredient kitchen |

List all:
```bash
overcooked-human-test --list-envs
```

---

## Controls

| Key | Action |
|-----|--------|
| `W` / `↑` | Move Up |
| `S` / `↓` | Move Down |
| `A` / `←` | Move Left |
| `D` / `→` | Move Right |
| `Space` / `E` | Interact (pick up / put down / cook) |
| `Q` / `Esc` | Quit |

---

## Troubleshooting

**`Policy file not found`**
- Make sure `exp_configs/policy_pool/` contains the trained models for each environment.
- Or set `POLICY_POOL` to your model directory:
  ```bash
  export POLICY_POOL=/path/to/policy_pool
  ```
- Use `--random` or `--demo` to run without loading models.

**Pygame window does not respond to keyboard**
- Click on the Pygame window to give it focus.
- On macOS, grant **Input Monitoring** permission to your terminal app.

**Keyboard issues on Linux**
```bash
sudo usermod -a -G input $USER
# Log out and back in
```
