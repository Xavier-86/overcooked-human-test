#!/usr/bin/env python3
"""
Experiment manager for blind human-AI testing.

Handles:
- User registration / authentication
- Loading experiment configuration
- Generating per-user test sequences
- Tracking user progress
- Saving test results per user

All data under exp_configs/ is stored encrypted on disk.
Decryption happens in memory only via the embedded key in crypto_utils.py.
"""

import json
import os
import hashlib
import random
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

from .crypto_utils import (
    read_encrypted_json,
    write_encrypted_json,
    read_encrypted_results,
    write_encrypted_results,
)


# experiment_manager.py lives in human_test/, but exp_configs/ is at project root.
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXP_CONFIGS_DIR = os.path.join(_project_root, "exp_configs")
DEFAULT_CONFIG_PATH = os.path.join(EXP_CONFIGS_DIR, "experiment_config.json.enc")
USER_DATA_DIR = os.path.join(EXP_CONFIGS_DIR, "user_data")
USERS_FILE = os.path.join(USER_DATA_DIR, "users.json.enc")
SESSION_FILE = os.path.join(USER_DATA_DIR, ".session.json.enc")


def _ensure_user_dir(user_id: str) -> str:
    user_dir = os.path.join(USER_DATA_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)
    return user_dir


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def load_users() -> dict:
    try:
        return read_encrypted_json(USERS_FILE)
    except FileNotFoundError:
        return {}


def save_users(users: dict):
    os.makedirs(USER_DATA_DIR, exist_ok=True)
    write_encrypted_json(USERS_FILE, users)


def register_user(user_id: str, password: str) -> bool:
    """Register a new user. Returns False if user already exists."""
    users = load_users()
    if user_id in users:
        return False
    users[user_id] = {
        "password_hash": _hash_password(password),
        "created_at": datetime.now().isoformat(),
    }
    save_users(users)
    return True


def verify_user(user_id: str, password: str) -> bool:
    """Verify password for a user."""
    users = load_users()
    if user_id not in users:
        return False
    return users[user_id]["password_hash"] == _hash_password(password)


def login_user(user_id: str, password: str) -> bool:
    """Login user and write session. Returns False if credentials are wrong."""
    if not verify_user(user_id, password):
        return False
    session = {
        "user_id": user_id,
        "logged_in_at": datetime.now().isoformat(),
    }
    write_encrypted_json(SESSION_FILE, session)
    return True


def logout_user():
    """Clear session file."""
    if os.path.exists(SESSION_FILE):
        os.remove(SESSION_FILE)


def get_current_user() -> Optional[str]:
    """Get current logged-in user from session."""
    try:
        session = read_encrypted_json(SESSION_FILE)
        return session.get("user_id")
    except (FileNotFoundError, Exception):
        return None


def user_exists(user_id: str) -> bool:
    """Check if a user is registered."""
    users = load_users()
    return user_id in users


def _get_progress_path(user_id: str) -> str:
    return os.path.join(_ensure_user_dir(user_id), "progress.json.enc")


def _get_results_path(user_id: str) -> str:
    return os.path.join(_ensure_user_dir(user_id), "results.json.enc")


def load_experiment_config(config_path: str = None) -> dict:
    path = config_path or DEFAULT_CONFIG_PATH
    if not os.path.exists(path):
        plain_path = path.replace(".enc", "")
        if plain_path != path and os.path.exists(plain_path):
            from .crypto_utils import encrypt_file
            encrypt_file(plain_path, path)
            os.remove(plain_path)
            return read_encrypted_json(path)
        raise FileNotFoundError(
            f"Experiment config not found: {path}. "
            "Ensure the file is encrypted and placed by the administrator."
        )
    return read_encrypted_json(path)


def get_experiment_config(config: dict = None) -> dict:
    """Return the global experiment config (all users share the same schedule)."""
    cfg = config or load_experiment_config()
    return cfg


def get_user_config(user_id: str, config: dict = None) -> dict:
    """Get experiment config for a user. All users share the same global config."""
    return get_experiment_config(config)


def generate_test_sequence(user_id: str, config: dict = None) -> List[Dict[str, Any]]:
    """Generate the test sequence for a user.

    Each test is an environment.  Inside each test, every algorithm appears
    for every position exactly ``episodes_per_config`` times.  The episode
    order (algo + position) is shuffled using the current time as seed so
    every user gets a different order.
    """
    user_cfg = get_user_config(user_id, config)

    envs = user_cfg.get("environments", [])
    positions = user_cfg.get("positions", [0, 1])
    episodes = user_cfg.get("episodes_per_config", 2)
    algos = user_cfg.get("algorithm_pool", ["bach"])

    sequence = []
    for env in envs:
        # Build episode list: every (algo, position) combo repeated ``episodes`` times
        episodes_list = []
        for pos in positions:
            for algo in algos:
                for _ in range(episodes):
                    episodes_list.append({
                        "algo": algo,
                        "position": pos,
                    })

        # Shuffle with a time-based seed so each user gets a unique order
        rng = random.Random(time.time())
        rng.shuffle(episodes_list)

        sequence.append({
            "index": len(sequence),
            "env_name": env,
            "episodes": episodes_list,
            "completed_episodes": 0,
            "total_episodes": len(episodes_list),
        })

    random.shuffle(sequence)
    for i, item in enumerate(sequence):
        item["index"] = i
    return sequence


def load_progress(user_id: str) -> Optional[dict]:
    try:
        return read_encrypted_json(_get_progress_path(user_id))
    except FileNotFoundError:
        return None


def save_progress(user_id: str, progress: dict):
    write_encrypted_json(_get_progress_path(user_id), progress)


def init_user_progress(user_id: str, config: dict = None) -> dict:
    """Initialize or reset user progress."""
    sequence = generate_test_sequence(user_id, config)
    progress = {
        "user_id": user_id,
        "total_tests": len(sequence),
        "completed_tests": 0,
        "current_test_index": 0,
        "sequence": sequence,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }
    save_progress(user_id, progress)
    return progress


def _progress_needs_reset(progress: dict, config: dict = None) -> bool:
    """Detect whether the experiment config has changed enough to reset."""
    expected = generate_test_sequence("__checksum__", config)
    actual = progress.get("sequence", [])
    if len(actual) != len(expected):
        return True
    # Also check if total episodes per test match
    for a, e in zip(actual, expected):
        if a.get("total_episodes") != e.get("total_episodes"):
            return True
    return False


def get_next_episode(user_id: str, test_index: int = None, config: dict = None) -> Optional[dict]:
    """Get the next pending episode for a specific test.

    If ``test_index`` is None, use the current_test_index from progress.
    Returns None if the test (or all tests) is completed.
    """
    progress = load_progress(user_id)
    if progress is None:
        progress = init_user_progress(user_id, config)
    elif _progress_needs_reset(progress, config):
        progress = init_user_progress(user_id, config)

    sequence = progress.get("sequence", [])
    if not sequence:
        return None

    if test_index is None:
        test_index = progress.get("current_test_index", 0)

    if test_index >= len(sequence):
        return None

    test = sequence[test_index]
    ep_idx = test.get("completed_episodes", 0)
    episodes = test.get("episodes", [])

    if ep_idx >= len(episodes):
        # This test is fully done; auto-advance current_test_index if needed
        if progress.get("current_test_index", 0) == test_index:
            progress["current_test_index"] = test_index + 1
            progress["completed_tests"] = progress.get("completed_tests", 0) + 1
            save_progress(user_id, progress)
        return None

    ep = episodes[ep_idx]
    return {
        "test_index": test_index,
        "env_name": test["env_name"],
        "human_player": ep["position"],
        "algo": ep["algo"],
        "episode_number": ep_idx + 1,
        "total_episodes": len(episodes),
    }


def get_all_tests(user_id: str, config: dict = None) -> List[Dict[str, Any]]:
    """Return all tests with their current status."""
    progress = load_progress(user_id)
    if progress is None:
        progress = init_user_progress(user_id, config)
    return progress.get("sequence", [])


def record_episode_result(user_id: str, test_index: int, result: dict):
    """Record a single completed episode and advance progress."""
    progress = load_progress(user_id)
    if progress is None:
        raise RuntimeError(f"No progress found for user {user_id}. Initialize first.")

    sequence = progress.get("sequence", [])
    if test_index >= len(sequence):
        raise RuntimeError(f"Test index {test_index} out of range.")

    test = sequence[test_index]
    ep_idx = test.get("completed_episodes", 0)
    episodes = test.get("episodes", [])

    if ep_idx >= len(episodes):
        raise RuntimeError("All episodes in this test already completed.")

    ep = episodes[ep_idx]

    result_entry = {
        "test_index": test_index,
        "env_name": test["env_name"],
        "algo": ep["algo"],
        "human_player": ep["position"],
        "episode_number": ep_idx + 1,
        "total_episodes": len(episodes),
        "scores": result.get("scores", []),
        "durations": result.get("durations", []),
        "soups_cooked": result.get("soups_cooked", []),
        "avg_score": result.get("avg_score", 0.0),
        "total_soups": result.get("total_soups", 0),
        "timestamp": datetime.now().isoformat(),
    }

    results = load_all_results(user_id)
    results.append(result_entry)
    write_encrypted_results(_get_results_path(user_id), results)

    # Update progress
    test["completed_episodes"] = ep_idx + 1
    if test["completed_episodes"] >= len(episodes):
        progress["completed_tests"] = progress.get("completed_tests", 0) + 1
        if progress.get("current_test_index", 0) == test_index:
            progress["current_test_index"] = test_index + 1
    progress["updated_at"] = datetime.now().isoformat()
    save_progress(user_id, progress)

    return result_entry


def print_progress(user_id: str):
    """Print user progress to console. Algorithm names are never shown."""
    import shutil
    width = shutil.get_terminal_size().columns

    progress = load_progress(user_id)
    if progress is None:
        print(f"  No progress found for user: {user_id}")
        return

    total_tests = progress.get("total_tests", 0)
    completed_tests = progress.get("completed_tests", 0)
    sequence = progress.get("sequence", [])

    total_episodes = sum(t.get("total_episodes", 0) for t in sequence)
    completed_episodes = sum(t.get("completed_episodes", 0) for t in sequence)
    pct = (completed_episodes / total_episodes * 100) if total_episodes > 0 else 0

    total_score = calculate_user_score(user_id)

    print("\n" + "=" * width)
    print(f"  User: {user_id}")
    print(f"  Tests: {completed_tests}/{total_tests} completed")
    print(f"  Episodes: {completed_episodes}/{total_episodes} completed ({pct:.1f}%)")
    print(f"  Total Score: {total_score:.1f}")
    print("=" * width)

    for i, test in enumerate(sequence):
        ep_done = test.get("completed_episodes", 0)
        ep_total = test.get("total_episodes", 0)
        if ep_done >= ep_total:
            marker = "[DONE]"
        elif ep_done > 0:
            marker = f"[{ep_done}/{ep_total}]"
        else:
            marker = "[    ]"
        print(f"  {marker} Test {i+1}: env={test['env_name']}")

    if completed_episodes >= total_episodes:
        results_path = os.path.abspath(_get_results_path(user_id))
        print(f"\n  All episodes completed!")
        print(f"  Send the following file to the administrator to claim your reward:")
        print(f"  {results_path}")
    else:
        current = progress.get("current_test_index", 0)
        if current < len(sequence):
            next_test = sequence[current]
            print(f"\n  Suggested next: Test {current+1} "
                  f"(env={next_test['env_name']})")
    print("=" * width + "\n")


def calculate_user_score(user_id: str) -> float:
    """Calculate the user's total score.

    For every recorded episode score, compute ``max(score - 120, 0) * 0.2``.
    Unfinished episodes contribute 0. The total is divided by the fixed
    total episode count across all tests, so the score is monotonically
    non-decreasing as more episodes are completed.
    """
    progress = load_progress(user_id)
    if progress is None:
        return 0.0

    sequence = progress.get("sequence", [])
    total_episodes = sum(t.get("total_episodes", 0) for t in sequence)
    if total_episodes == 0:
        return 0.0

    results = load_all_results(user_id)
    total = 0.0
    for r in results:
        for score in r.get("scores", []):
            total += max(score - 120, 0) * 0.2

    return round(total / total_episodes, 1)


def load_all_results(user_id: str) -> List[dict]:
    """Load all recorded results for a user."""
    try:
        return read_encrypted_results(_get_results_path(user_id))
    except FileNotFoundError:
        return []
