#!/usr/bin/env python3
"""
Overcooked Human-AI Testing CLI
Unified entry point for human-AI interaction testing.

Algorithm names are never revealed to participants.
"""

import argparse
import sys
import os
import getpass
import shutil
import traceback
from typing import Optional

from .experiment_manager import (
    register_user,
    login_user,
    logout_user,
    get_current_user,
    user_exists,
    load_users,
    load_experiment_config,
    get_next_episode,
    get_all_tests,
    init_user_progress,
    record_episode_result,
    calculate_user_score,
)

__version__ = "1.0.0"
DEFAULT_FPS = 4

# ---------------------------------------------------------------------------
# i18n
# ---------------------------------------------------------------------------

_LANG = "en"

_TRANSLATIONS = {
    "Human-AI Interaction Testing": {"zh": "人机交互测试"},
    "First Time Setup - Encryption": {"zh": "首次设置 - 加密"},
    "No encrypted data found. Encryption keys are required.": {"zh": "未发现加密数据。需要设置加密密钥。"},
    "Save the private key shown below -- you will need it to decrypt results later.": {"zh": "请保存下方显示的私钥 -- 后续需要用它解密结果。"},
    "Generate new key pair": {"zh": "生成新的密钥对"},
    "Use existing key pair": {"zh": "使用已有密钥对"},
    "Exit": {"zh": "退出"},
    "NEW KEYS GENERATED": {"zh": "新密钥对已生成"},
    "Public key has been embedded into the project.": {"zh": "公钥已嵌入项目。"},
    "SAVE THE PRIVATE KEY SECURELY. You will need it to decrypt participant data.": {"zh": "请妥善保存私钥。您将需要它来解密参与者数据。"},
    "Press Enter to continue...": {"zh": "按回车键继续..."},
    "Use Existing Key Pair": {"zh": "使用已有密钥对"},
    "Enter private key PEM file path: ": {"zh": "输入私钥 PEM 文件路径: "},
    "[ERR] Private key file not found.": {"zh": "[错误] 未找到私钥文件。"},
    "[ERR] Invalid private key: ": {"zh": "[错误] 私钥无效: "},
    "[OK] Keys configured.": {"zh": "[成功] 密钥已配置。"},
    "Use UP/DOWN to move, ENTER to select": {"zh": "使用上下方向键移动，回车键选择"},
    "Admin Login": {"zh": "管理员登录"},
    "Admin password: ": {"zh": "管理员密码: "},
    "[ERR] Incorrect password.": {"zh": "[错误] 密码错误。"},
    "Set admin password: ": {"zh": "设置管理员密码: "},
    "Confirm admin password: ": {"zh": "确认管理员密码: "},
    "Admin passwords do not match.": {"zh": "两次输入的管理员密码不一致。"},
    "[ERR] Admin password cannot be empty.": {"zh": "[错误] 管理员密码不能为空。"},
    "Private key file path: ": {"zh": "私钥文件路径: "},
    "[ERR] Not an RSA-encrypted result file.": {"zh": "[错误] 不是 RSA 加密的结果文件。"},
    "Admin Mode": {"zh": "管理员模式"},
    "View registered users": {"zh": "查看注册用户"},
    "Decrypt results file": {"zh": "解密结果文件"},
    "Back to main menu": {"zh": "返回主菜单"},
    "Registered users: ": {"zh": "注册用户: "},
    "Results file path: ": {"zh": "结果文件路径: "},
    "[ERR] File not found.": {"zh": "[错误] 文件未找到。"},
    "Total records: ": {"zh": "总记录数: "},
    "Available environments:": {"zh": "可用环境:"},
    "Register": {"zh": "注册"},
    "Login": {"zh": "登录"},
    "Admin": {"zh": "管理员"},
    "Goodbye!": {"zh": "再见！"},
    "User ID: ": {"zh": "用户ID: "},
    "Set password: ": {"zh": "设置密码: "},
    "Confirm password: ": {"zh": "确认密码: "},
    "Passwords do not match.": {"zh": "两次输入的密码不一致。"},
    "Password cannot be empty.": {"zh": "密码不能为空。"},
    "User '{user_id}' registered successfully.": {"zh": "用户 '{user_id}' 注册成功。"},
    "User '{user_id}' already exists.": {"zh": "用户 '{user_id}' 已存在。"},
    "User '{user_id}' not found. Use Register to create an account.": {"zh": "用户 '{user_id}' 未找到。请使用注册功能创建账户。"},
    "Logged in as '{user_id}'.": {"zh": "已以 '{user_id}' 身份登录。"},
    "Incorrect password.": {"zh": "密码错误。"},
    "Run next episode (Test {test}: {env} pos={pos}, ep {ep}/{total})": {"zh": "运行下一局 (测试 {test}: {env} 位置={pos}, 第 {ep}/{total} 局)"},
    "Run next episode (all done)": {"zh": "运行下一局 (全部完成)"},
    "Open Test Menu": {"zh": "打开测试菜单"},
    "Practice": {"zh": "练习"},
    "Logout": {"zh": "退出登录"},
    "Reset Progress": {"zh": "重置进度"},
    "User: {user}  |  Total Score: {score}": {"zh": "用户: {user}  |  总得分: {score}"},
    "All tests completed! Send results.json.enc to the admin for your reward.": {"zh": "所有测试已完成！请将 results.json.enc 发送给管理员领取奖励。"},
    "Path: ": {"zh": "路径: "},
    "Progress reset. You can now start testing again.": {"zh": "进度已重置。您可以重新开始测试。"},
    "Open Test Menu": {"zh": "打开测试菜单"},
    "Test {i}: {env} pos={pos} [DONE]": {"zh": "测试 {i}: {env} 位置={pos} [已完成]"},
    "Test {i}: {env} pos={pos} ({done}/{total})": {"zh": "测试 {i}: {env} 位置={pos} ({done}/{total})"},
    "Back": {"zh": "返回"},
    "Starting Episode {ep}/{total}": {"zh": "开始第 {ep}/{total} 局"},
    "Environment: ": {"zh": "环境: "},
    "Your position: Player ": {"zh": "您的位置: 玩家 "},
    "Press Enter to start...": {"zh": "按回车键开始..."},
    "All episodes in this test are completed.": {"zh": "该测试的所有局数已完成。"},
    "Interrupted by user.": {"zh": "用户中断。"},
    "Error: ": {"zh": "错误: "},
    "Episode Complete": {"zh": "本局完成"},
    "Score: ": {"zh": "得分: "},
    "Soups: ": {"zh": "汤数: "},
    "Press Enter to return to menu...": {"zh": "按回车键返回菜单..."},
    "Practice Mode": {"zh": "练习模式"},
    "Practice {env} as Player {pos}": {"zh": "在 {env} 中以玩家 {pos} 身份练习"},
    "Practice Complete": {"zh": "练习完成"},
    "No user was logged in.": {"zh": "没有用户登录。"},
    "Logged out '{current}'.": {"zh": "已退出 '{current}'。"},
    "[ERR] Please log in first.": {"zh": "[错误] 请先登录。"},
    "[ERR] User '{user_id}' not found.": {"zh": "[错误] 用户 '{user_id}' 未找到。"},
    "[ERR] --register requires --user-id": {"zh": "[错误] --register 需要 --user-id"},
    "[ERR] Passwords do not match.": {"zh": "[错误] 两次输入的密码不一致。"},
    "[ERR] Password cannot be empty.": {"zh": "[错误] 密码不能为空。"},
    "[OK] User '{user_id}' registered successfully.": {"zh": "[成功] 用户 '{user_id}' 注册成功。"},
    "[ERR] User '{user_id}' already exists.": {"zh": "[错误] 用户 '{user_id}' 已存在。"},
    "[ERR] --login requires --user-id": {"zh": "[错误] --login 需要 --user-id"},
    "[ERR] User '{user_id}' not found. Use --register to create an account.": {"zh": "[错误] 用户 '{user_id}' 未找到。请使用 --register 创建账户。"},
    "[OK] Logged in as '{user_id}'.": {"zh": "[成功] 已以 '{user_id}' 身份登录。"},
    "[ERR] Incorrect password.": {"zh": "[错误] 密码错误。"},
    "[OK] Logged out '{current}'.": {"zh": "[成功] 已退出 '{current}'。"},
    "Logged in as: ": {"zh": "当前登录用户: "},
    "No user logged in.": {"zh": "没有用户登录。"},
    "All tests completed!": {"zh": "所有测试已完成！"},
    "Pygame not detected, falling back to terminal demo mode.": {"zh": "未检测到 Pygame，回退到终端演示模式。"},
    "Password: ": {"zh": "密码: "},
}


def _set_lang(lang: str):
    global _LANG
    _LANG = lang


def _T(key: str, **kwargs) -> str:
    """Translate a key. Falls back to the key itself if not found."""
    entry = _TRANSLATIONS.get(key, {})
    text = entry.get(_LANG, key)
    if kwargs:
        text = text.format(**kwargs)
    return text


def get_term_width() -> int:
    """Get terminal width, fallback to 80."""
    return shutil.get_terminal_size().columns


def print_bar(char: str = "=", text: str = ""):
    """Print a horizontal bar that adapts to terminal width."""
    width = get_term_width()
    if text:
        pad = (width - len(text) - 4) // 2
        left = char * max(1, pad)
        right = char * max(1, width - pad - len(text) - 4)
        print(f"{left}  {text}  {right}")
    else:
        print(char * width)


def print_banner():
    """Print ASCII art banner using pyfiglet."""
    try:
        import pyfiglet
        art = pyfiglet.Figlet(font="slant").renderText("Overcooked")
    except Exception:
        art = "  Overcooked\n"
    RED = "\033[31m"
    RESET = "\033[0m"
    print(RED + art + RESET)
    print_bar("=", _T("Human-AI Interaction Testing"))
    print()


# ---------------------------------------------------------------------------
# First-run key setup (administrator only)
# ---------------------------------------------------------------------------

def _has_encrypted_files() -> bool:
    """Check whether any .enc files already exist under exp_configs/."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    exp_dir = os.path.join(project_root, "exp_configs")
    if not os.path.isdir(exp_dir):
        return False
    for root, _dirs, files in os.walk(exp_dir):
        for f in files:
            if f.endswith(".enc"):
                return True
    return False


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _key_paths() -> tuple:
    """Return paths to the RSA key files."""
    root = _project_root()
    return (
        os.path.join(root, "private_key.pem"),
        os.path.join(root, "public_key.pem"),
    )


def _config_key_path() -> str:
    return os.path.join(_project_root(), "config.key")


def _admin_hash_path() -> str:
    return os.path.join(_project_root(), "exp_configs", "user_data", ".admin_hash")


def _load_admin_hash() -> Optional[str]:
    path = _admin_hash_path()
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _save_admin_hash(password: str):
    """Store a salted SHA-256 hash of the admin password."""
    import hashlib
    salt = os.urandom(16).hex()
    digest = hashlib.sha256((salt + password).encode("utf-8")).hexdigest()
    os.makedirs(os.path.dirname(_admin_hash_path()), exist_ok=True)
    with open(_admin_hash_path(), "w", encoding="utf-8") as f:
        f.write(f"{salt}:{digest}")


def _check_admin_password(password: str) -> bool:
    stored = _load_admin_hash()
    if not stored:
        return False
    import hashlib
    salt, digest = stored.split(":", 1)
    expected = hashlib.sha256((salt + password).encode("utf-8")).hexdigest()
    return digest == expected


def _setup_admin_password():
    """Prompt for and store a new admin password."""
    while True:
        pwd1 = getpass.getpass(_T("Set admin password: ")).strip()
        if not pwd1:
            print(_T("[ERR] Admin password cannot be empty."))
            input(_T("Press Enter to continue..."))
            continue
        pwd2 = getpass.getpass(_T("Confirm admin password: ")).strip()
        if pwd1 != pwd2:
            print(_T("Admin passwords do not match."))
            input(_T("Press Enter to continue..."))
            continue
        _save_admin_hash(pwd1)
        break


def _update_embedded_public_key(public_pem: bytes):
    """Rewrite crypto_utils.py so that _EMBEDDED_PUBLIC_KEY matches the new key."""
    project_root = _project_root()
    cu_path = os.path.join(project_root, "human_test", "crypto_utils.py")
    with open(cu_path, "r", encoding="utf-8") as f:
        content = f.read()

    import re
    # Embed the PEM as a triple-quoted string.
    escaped = public_pem.decode("utf-8").replace('\\', '\\\\').replace('"', '\\"')
    new_block = f'_EMBEDDED_PUBLIC_KEY = """{escaped}"""'
    content = re.sub(
        r'_EMBEDDED_PUBLIC_KEY\s*=\s*"""[\s\S]*?"""',
        new_block,
        content,
    )

    with open(cu_path, "w", encoding="utf-8") as f:
        f.write(content)

    # Update the in-memory key so the running process uses it immediately.
    from . import crypto_utils
    crypto_utils.set_public_key(public_pem)


def _write_key_files(private_pem: bytes, public_pem: bytes):
    """Write the RSA private/public key files to the project root."""
    private_path, public_path = _key_paths()
    with open(private_path, "wb") as f:
        f.write(private_pem)
    with open(public_path, "wb") as f:
        f.write(public_pem)


def _write_config_key(key: str):
    """Write the Fernet config key to config.key."""
    path = _config_key_path()
    with open(path, "w", encoding="utf-8") as f:
        f.write(key)
    from . import crypto_utils
    crypto_utils.set_key(key)


def _setup_key():
    """Interactive first-run key setup.  Generates/configures RSA key pair."""
    extra = [
        "No encrypted data found. Encryption keys are required.",
        "Save the private key shown below -- you will need it to decrypt results later.",
    ]
    options = [
        ("Generate new key pair", True),
        ("Use existing key pair", True),
        (_T("Exit"), True),
    ]
    while True:
        choice = _menu_select(options, title=_T("First Time Setup - Encryption"), extra_lines=extra)
        if choice == -1 or choice == 2:
            sys.exit(0)

        if choice == 0:
            from .crypto_utils import generate_rsa_keypair, generate_key
            private_pem, public_pem = generate_rsa_keypair()
            _write_key_files(private_pem, public_pem)
            _update_embedded_public_key(public_pem)

            # Generate a fresh Fernet key for config/users/progress files.
            config_key = generate_key()
            _write_config_key(config_key)

            # Prompt for an admin password used to access the in-app admin menu.
            _setup_admin_password()

            os.system("cls" if os.name == "nt" else "clear")
            print_banner()
            print_bar("=", _T("NEW KEYS GENERATED"))
            print(_T("Private key saved to: ") + _key_paths()[0])
            print(_T("Public key saved to: ") + _key_paths()[1])
            print(_T("Config key saved to: ") + _config_key_path())
            print_bar("=", "")
            print(f"  {_T('Public key has been embedded into the project.')}")
            print(f"  {_T('SAVE THE PRIVATE KEY SECURELY. You will need it to decrypt participant data.')}")
            print()
            input(_T("Press Enter to continue..."))
            return

        if choice == 1:
            os.system("cls" if os.name == "nt" else "clear")
            print_banner()
            print_bar("=", _T("Use Existing Key Pair"))
            private_path, public_path = _key_paths()
            path = input(_T("Enter private key PEM file path: ")).strip()
            if not path or not os.path.isfile(path):
                print(_T("[ERR] Private key file not found."))
                input(_T("Press Enter to continue..."))
                continue
            try:
                with open(path, "rb") as f:
                    private_pem = f.read()
                from cryptography.hazmat.primitives import serialization
                private_key = serialization.load_pem_private_key(private_pem, password=None)
                public_pem = private_key.public_key().public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo,
                )
                _write_key_files(private_pem, public_pem)
                _update_embedded_public_key(public_pem)

                # Reuse or create config key.
                if os.path.isfile(_config_key_path()):
                    with open(_config_key_path(), "r", encoding="utf-8") as f:
                        _write_config_key(f.read().strip())
                else:
                    from .crypto_utils import generate_key
                    _write_config_key(generate_key())

                _setup_admin_password()
                print(_T("[OK] Keys configured."))
                input(_T("Press Enter to continue..."))
                return
            except Exception as e:
                print(_T("[ERR] Invalid private key: ") + str(e))
                input(_T("Press Enter to continue..."))
                continue


def _encrypt_existing_configs():
    """Encrypt any plaintext JSON/JSONL files under exp_configs/."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    exp_dir = os.path.join(project_root, "exp_configs")
    if not os.path.isdir(exp_dir):
        return
    from .crypto_utils import encrypt_file
    for root, _dirs, files in os.walk(exp_dir):
        for f in files:
            # Skip template files and already-encrypted files
            if f.endswith(".enc"):
                continue
            if "_template" in f:
                continue
            if f.endswith(".json") or f.endswith(".jsonl"):
                plain = os.path.join(root, f)
                enc = plain + ".enc"
                encrypt_file(plain, enc)
                os.remove(plain)
                print(f"  [OK] Encrypted: {plain}")


# ---------------------------------------------------------------------------
# Cross-platform single-key reader (no echo)
# ---------------------------------------------------------------------------

def _read_key() -> str:
    """Read a single key press and return 'up', 'down', 'enter', or the char."""
    if os.name == "nt":
        import msvcrt
        ch = msvcrt.getch()
        if ch == b"\r" or ch == b"\n":
            return "enter"
        if ch == b"\x00" or ch == b"\xe0":
            ch2 = msvcrt.getch()
            if ch2 == b"H":
                return "up"
            if ch2 == b"P":
                return "down"
        try:
            return ch.decode("utf-8")
        except Exception:
            return ""
    else:
        import tty
        import termios
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            if ch == "\r" or ch == "\n":
                return "enter"
            if ch == "\x1b":
                seq = sys.stdin.read(2)
                if seq == "[A":
                    return "up"
                if seq == "[B":
                    return "down"
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ---------------------------------------------------------------------------
# Arrow-key menu helper
# ---------------------------------------------------------------------------

def _menu_select(options: list, title: str = "", extra_lines: list = None) -> int:
    """Display a menu and let the user navigate with arrow keys.

    ``options`` is a list of (label, enabled) tuples.
    Returns the index of the selected option, or -1 for quit.
    """
    idx = 0
    while True:
        os.system("cls" if os.name == "nt" else "clear")
        print_banner()
        if title:
            print_bar("=", title)
            print()

        if extra_lines:
            for line in extra_lines:
                print(f"  {line}")
            print()

        for i, (label, enabled) in enumerate(options):
            marker = "> " if i == idx else "  "
            print(f"{marker}{label}")

        if title:
            print_bar("=", "")
        else:
            print_bar("-", "")
        print(_T("Use UP/DOWN to move, ENTER to select"))

        key = _read_key()
        if key == "up":
            idx = (idx - 1) % len(options)
            # Skip disabled options
            while not options[idx][1]:
                idx = (idx - 1) % len(options)
        elif key == "down":
            idx = (idx + 1) % len(options)
            while not options[idx][1]:
                idx = (idx + 1) % len(options)
        elif key == "enter":
            if options[idx][1]:
                return idx
        elif key.lower() == "q":
            return -1


# ---------------------------------------------------------------------------
# Admin menu
# ---------------------------------------------------------------------------


def _verify_admin_password() -> bool:
    """Prompt for admin password. Returns True if it matches the stored hash."""
    pwd = _prompt_password(_T("Admin password: "))
    return _check_admin_password(pwd.strip())


def _admin_menu() -> int:
    """Admin interface for managing encrypted data."""
    os.system("cls" if os.name == "nt" else "clear")
    print_banner()
    print_bar("=", _T("Admin Login"))
    if not _verify_admin_password():
        print(_T("[ERR] Incorrect password."))
        input(_T("Press Enter to continue..."))
        return 0

    while True:
        options = [
            (_T("View registered users"), True),
            (_T("Decrypt results file"), True),
            (_T("Back to main menu"), True),
        ]
        choice = _menu_select(options, title=_T("Admin Mode"))
        if choice == -1 or choice == 2:
            return 0

        if choice == 0:
            try:
                users = load_users()
                print(f"\n{_T('Registered users: ')}{len(users)}")
                for uid in sorted(users.keys()):
                    print(f"  - {uid}")
            except Exception as e:
                print(f"[ERR] {e}")
            input("\nPress Enter to continue...")

        elif choice == 1:
            path = input(_T("Results file path: ")).strip()
            if not path or not os.path.exists(path):
                print(_T("[ERR] File not found."))
                input("\nPress Enter to continue...")
                continue

            key_path = input(_T("Private key file path: ")).strip()
            if not key_path or not os.path.exists(key_path):
                print(_T("[ERR] Private key file not found."))
                input("\nPress Enter to continue...")
                continue

            try:
                from .crypto_utils import read_encrypted_results
                with open(key_path, "rb") as f:
                    private_pem = f.read()
                results = read_encrypted_results(path, private_key_pem=private_pem)
                print(f"\n{_T('Total records: ')}{len(results)}")
                for r in results:
                    print(
                        f"  Test {r.get('test_index', '?'):>2} ep{r.get('episode_number', '?'):>2}: "
                        f"env={r.get('env_name', '?'):<10} "
                        f"algo={r.get('algo', '?'):<6} "
                        f"score={r.get('avg_score', 0):>7.1f} "
                        f"soups={r.get('total_soups', 0):>3}"
                    )
            except Exception as e:
                print(f"[ERR] {e}")
            input("\nPress Enter to continue...")


def get_available_envs() -> list:
    """Discover available environments from policy pools."""
    policy_roots = []
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    policy_roots.append(os.path.join(project_root, "exp_configs", "policy_pool"))
    policy_roots.append(os.path.join(project_root, "zsceval", "policy_pool"))
    if os.environ.get("POLICY_POOL"):
        policy_roots.insert(0, os.environ.get("POLICY_POOL"))

    envs = set()
    for root in policy_roots:
        if os.path.isdir(root):
            for item in os.listdir(root):
                if os.path.isdir(os.path.join(root, item)) and not item.startswith("."):
                    envs.add(item)

    defaults = ["random1_m", "random3_m"]
    for d in defaults:
        envs.add(d)
    return sorted(envs)


def print_version():
    print(f"overcooked-human-test {__version__}")


def print_envs():
    print(_T("Available environments:"))
    for env in get_available_envs():
        print(f"  - {env}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="overcooked-human-test",
        description="Human-AI interaction testing for Overcooked. "
                    "Algorithm names are never revealed to participants.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                         # Launch interactive menu
  %(prog)s --demo                  # Terminal demo mode
  %(prog)s -e random1_m --random   # Play against random AI
  %(prog)s --headless -e random1_m # AI-only headless test
  %(prog)s --register --user-id alice
  %(prog)s --login --user-id alice
        """.strip(),
    )

    parser.add_argument(
        "-e", "--env", type=str, default="random1_m",
        help="Environment / layout name (default: random1_m)",
    )
    parser.add_argument(
        "-p", "--human-player", type=int, default=1, choices=[0, 1],
        help="Human player index: 0 or 1 (default: 1)",
    )
    parser.add_argument(
        "-n", "--episodes", type=int, default=1,
        help="Number of episodes to play (default: 1)",
    )
    parser.add_argument(
        "--tile-size", type=int, default=75,
        help="Pygame tile size in pixels (default: 75)",
    )
    parser.add_argument(
        "--fps", type=int, default=DEFAULT_FPS,
        help=f"Game speed in steps per second (default: {DEFAULT_FPS})",
    )
    parser.add_argument(
        "--policy-pool", type=str, default=None,
        help="Path to the policy pool directory (default: auto-detect)",
    )
    parser.add_argument(
        "--random", action="store_true",
        help="Use a random AI instead of loading a trained model",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run terminal demo mode without Pygame",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Headless mode: AI plays without graphics",
    )
    parser.add_argument(
        "--user-id", type=str, default=None,
        help="User ID for experiment tracking",
    )
    parser.add_argument(
        "--exp-config", type=str, default=None,
        help="Path to experiment config JSON",
    )
    parser.add_argument(
        "--register", action="store_true",
        help="Register a new user (requires --user-id)",
    )
    parser.add_argument(
        "--login", action="store_true",
        help="Login as a user (requires --user-id)",
    )
    parser.add_argument(
        "--logout", action="store_true",
        help="Logout current user",
    )
    parser.add_argument(
        "--whoami", action="store_true",
        help="Show current logged-in user",
    )
    parser.add_argument(
        "--list-envs", action="store_true",
        help="List available environments and exit",
    )
    parser.add_argument(
        "-v", "--version", action="store_true",
        help="Show version and exit",
    )
    parser.add_argument(
        "--zh", action="store_true",
        help="Use Chinese interface",
    )

    return parser


def run_demo_mode(args):
    """Run demo mode. Prefer real Overcooked environment if pygame is available."""
    try:
        import pygame
        from .trained_policy_render import HumanTestWithPolicy
        test = HumanTestWithPolicy(
            env_name=args.env,
            algo="random",
            human_player=args.human_player,
            episodes=args.episodes,
            tile_size=args.tile_size,
            fps=args.fps,
            policy_pool_path=None,
            headless=False,
        )
        test.run()
    except ImportError:
        from .core import HumanTest
        print(_T("Pygame not detected, falling back to terminal demo mode."))
        test = HumanTest(
            env_name=args.env,
            algo="random",
            human_player=args.human_player,
            episodes=args.episodes,
            render_mode="cli",
        )
        test.run()
    return 0


def run_random_mode(args):
    """Run with a random AI opponent (no trained model needed)."""
    import pygame
    from .trained_policy_render import HumanTestWithPolicy

    test = HumanTestWithPolicy(
        env_name=args.env,
        algo="random",
        human_player=args.human_player,
        episodes=args.episodes,
        tile_size=args.tile_size,
        fps=args.fps,
        policy_pool_path=None,
        headless=args.headless,
    )
    test.run()
    return 0


def _prompt_password(prompt: str = _T("Password: ")) -> str:
    try:
        return getpass.getpass(prompt)
    except Exception:
        return input(prompt)


def run_register(args) -> int:
    """Register a new user."""
    user_id = args.user_id
    if not user_id:
        print(_T("[ERR] --register requires --user-id"))
        return 1

    password = _prompt_password(_T("Set password: "))
    confirm = _prompt_password(_T("Confirm password: "))
    if password != confirm:
        print(_T("[ERR] Passwords do not match."))
        return 1
    if not password:
        print(_T("[ERR] Password cannot be empty."))
        return 1

    if register_user(user_id, password):
        print(_T("[OK] User '{user_id}' registered successfully.", user_id=user_id))
        return 0
    else:
        print(_T("[ERR] User '{user_id}' already exists.", user_id=user_id))
        return 1


def run_login(args) -> int:
    """Login a user."""
    user_id = args.user_id
    if not user_id:
        print(_T("[ERR] --login requires --user-id"))
        return 1

    if not user_exists(user_id):
        print(_T("[ERR] User '{user_id}' not found. Use --register to create an account.", user_id=user_id))
        return 1

    password = _prompt_password(_T("Password: "))
    if login_user(user_id, password):
        print(_T("[OK] Logged in as '{user_id}'.", user_id=user_id))
        return 0
    else:
        print(_T("[ERR] Incorrect password."))
        return 1


def run_logout() -> int:
    """Logout current user."""
    current = get_current_user()
    logout_user()
    if current:
        print(_T("[OK] Logged out '{current}'.", current=current))
    else:
        print(_T("No user was logged in."))
    return 0


def run_whoami() -> int:
    """Show current logged-in user."""
    current = get_current_user()
    if current:
        print(f"{_T('Logged in as: ')}{current}")
    else:
        print(_T("No user logged in."))
    return 0


def _run_test_select(user_id: str) -> Optional[int]:
    """Let the user pick a test to run. Returns test_index or None.  -2 means reset."""
    tests = get_all_tests(user_id)
    if not tests:
        return None

    options = []
    has_pending = False
    for i, test in enumerate(tests):
        ep_done = test.get("completed_episodes", 0)
        ep_total = test.get("total_episodes", 0)
        if ep_done >= ep_total:
            label = _T("Test {i}: {env} [DONE]", i=i+1, env=test["env_name"])
            enabled = False
        else:
            label = _T("Test {i}: {env} ({done}/{total})", i=i+1, env=test["env_name"], done=ep_done, total=ep_total)
            enabled = True
            has_pending = True
        options.append((label, enabled))

    if not has_pending:
        options.append((_T("Reset Progress"), True))

    options.append((_T("Back"), True))

    choice = _menu_select(options, title=_T("Open Test Menu"))
    if choice == -1 or choice == len(options) - 1:
        return None

    if not has_pending and choice == len(options) - 2:
        init_user_progress(user_id)
        print(_T("Progress reset. You can now start testing again."))
        input(_T("Press Enter to continue..."))
        return -2

    return choice


def _run_single_episode(test_index: int) -> int:
    """Run a single episode for the given test index and return to menu."""
    import pygame
    from .trained_policy_render import HumanTestWithPolicy

    user_id = get_current_user()
    if not user_id:
        print(_T("[ERR] Please log in first."))
        return 1

    if not user_exists(user_id):
        print(_T("[ERR] User '{user_id}' not found.", user_id=user_id))
        return 1

    episode = get_next_episode(user_id, test_index)
    if episode is None:
        print(_T("All episodes in this test are completed."))
        return 0

    env_name = episode["env_name"]
    human_player = episode["human_player"]
    algo = episode["algo"]
    ep_num = episode["episode_number"]
    ep_total = episode["total_episodes"]

    print()
    print_bar("=", _T("Starting Episode {ep}/{total}", ep=ep_num, total=ep_total))
    print(f"  {_T('Environment: ')}{env_name}")
    print(f"  {_T('Your position: Player ')}{human_player}")
    print_bar("=", "")
    print()
    input(_T("Press Enter to start..."))

    # Auto-detect policy pool
    policy_pool_path = None
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    candidates = [
        os.path.join(project_root, "exp_configs", "policy_pool"),
        os.path.join(project_root, "zsceval", "policy_pool"),
    ]
    if os.environ.get("POLICY_POOL"):
        candidates.insert(0, os.environ.get("POLICY_POOL"))
    for c in candidates:
        if os.path.isdir(c):
            policy_pool_path = c
            break

    try:
        test = HumanTestWithPolicy(
            env_name=env_name,
            algo=algo,
            human_player=human_player,
            episodes=1,
            tile_size=75,
            fps=DEFAULT_FPS,
            policy_pool_path=policy_pool_path,
            headless=False,
        )
        test.run()
    except KeyboardInterrupt:
        print(_T("Interrupted by user."))
        try:
            pygame.quit()
        except Exception:
            pass
        return 130
    except Exception as e:
        print(f"\n{_T('Error: ')}{e}")
        try:
            traceback.print_exc()
        except Exception:
            pass
        try:
            pygame.quit()
        except Exception:
            pass
        return 1

    # Record results
    results = test.get_results()
    record_episode_result(user_id, test_index, results)

    print_bar("=", _T("Episode Complete"))
    print(f"  {_T('Score: ')}{results.get('avg_score', 0):.1f}")
    print(f"  {_T('Soups: ')}{results.get('total_soups', 0)}")
    print_bar("=", "")
    input(_T("Press Enter to return to menu..."))
    return 0


def _practice_select() -> Optional[tuple]:
    """Let the user pick an environment and position for practice.

    Returns (env_name, human_player) or None if cancelled.
    """
    cfg = load_experiment_config()
    envs = cfg.get("environments", [])
    positions = cfg.get("positions", [0, 1])

    options = []
    for env in envs:
        for pos in positions:
            label = _T("Practice {env} as Player {pos}", env=env, pos=pos)
            options.append((label, True))
    options.append((_T("Back"), True))

    choice = _menu_select(options, title=_T("Practice Mode"))
    if choice == -1 or choice == len(options) - 1:
        return None

    idx = 0
    for env in envs:
        for pos in positions:
            if idx == choice:
                return (env, pos)
            idx += 1
    return None


def _run_practice(env_name: str, human_player: int) -> int:
    """Run a practice episode. Results are displayed but not recorded."""
    import pygame
    from .trained_policy_render import HumanTestWithPolicy

    # Pick the first algorithm from the pool deterministically.
    cfg = load_experiment_config()
    algos = cfg.get("algorithm_pool", ["bach"])
    algo = algos[0] if algos else "bach"

    print()
    print_bar("=", _T("Practice Mode"))
    print(f"  {_T('Environment: ')}{env_name}")
    print(f"  {_T('Your position: Player ')}{human_player}")
    print_bar("=", "")
    print()
    input(_T("Press Enter to start..."))

    # Auto-detect policy pool
    policy_pool_path = None
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    candidates = [
        os.path.join(project_root, "exp_configs", "policy_pool"),
        os.path.join(project_root, "zsceval", "policy_pool"),
    ]
    if os.environ.get("POLICY_POOL"):
        candidates.insert(0, os.environ.get("POLICY_POOL"))
    for c in candidates:
        if os.path.isdir(c):
            policy_pool_path = c
            break

    try:
        test = HumanTestWithPolicy(
            env_name=env_name,
            algo=algo,
            human_player=human_player,
            episodes=1,
            tile_size=75,
            fps=DEFAULT_FPS,
            policy_pool_path=policy_pool_path,
            headless=False,
        )
        test.run()
    except KeyboardInterrupt:
        print(_T("Interrupted by user."))
        try:
            pygame.quit()
        except Exception:
            pass
        return 130
    except Exception as e:
        print(f"\n{_T('Error: ')}{e}")
        try:
            traceback.print_exc()
        except Exception:
            pass
        try:
            pygame.quit()
        except Exception:
            pass
        return 1

    results = test.get_results()

    print_bar("=", _T("Practice Complete"))
    print(f"  {_T('Score: ')}{results.get('avg_score', 0):.1f}")
    print(f"  {_T('Soups: ')}{results.get('total_soups', 0)}")
    print_bar("=", "")
    input(_T("Press Enter to return to menu..."))
    return 0


def interactive_menu() -> int:
    """Launch an interactive text menu."""
    # Always clear session on menu start so the user must log in every time.
    logout_user()

    # First-run key setup: if no encrypted files exist, force admin to set a key.
    if not _has_encrypted_files():
        _setup_key()
        _encrypt_existing_configs()

    while True:
        current_user = get_current_user()

        if not current_user:
            options = [
                (_T("Register"), True),
                (_T("Login"), True),
                (_T("Admin"), True),
                (_T("Exit"), True),
            ]
            choice = _menu_select(options)
            if choice == -1 or choice == 3:
                print(_T("Goodbye!"))
                return 0

            if choice == 0:
                os.system("cls" if os.name == "nt" else "clear")
                print_banner()
                user_id = input(_T("User ID: ")).strip()
                if not user_id:
                    input(_T("Press Enter to continue..."))
                    continue
                password = _prompt_password(_T("Set password: "))
                confirm = _prompt_password(_T("Confirm password: "))
                if password != confirm:
                    print(_T("Passwords do not match."))
                    input(_T("Press Enter to continue..."))
                    continue
                if not password:
                    print(_T("Password cannot be empty."))
                    input(_T("Press Enter to continue..."))
                    continue
                if register_user(user_id, password):
                    print(_T("User '{user_id}' registered successfully.", user_id=user_id))
                else:
                    print(_T("User '{user_id}' already exists.", user_id=user_id))
                input(_T("Press Enter to continue..."))

            elif choice == 1:
                os.system("cls" if os.name == "nt" else "clear")
                print_banner()
                user_id = input(_T("User ID: ")).strip()
                if not user_id:
                    input(_T("Press Enter to continue..."))
                    continue
                if not user_exists(user_id):
                    print(_T("User '{user_id}' not found. Use Register to create an account.", user_id=user_id))
                    input(_T("Press Enter to continue..."))
                    continue
                password = _prompt_password(_T("Password: "))
                if login_user(user_id, password):
                    print(_T("Logged in as '{user_id}'.", user_id=user_id))
                else:
                    print(_T("Incorrect password."))
                input(_T("Press Enter to continue..."))

            elif choice == 2:
                _admin_menu()

        else:
            # Logged-in menu
            next_ep = get_next_episode(current_user)
            total_score = calculate_user_score(current_user)
            if next_ep:
                run_label = _T(
                    "Run next episode (Test {test}: {env} pos={pos}, ep {ep}/{total})",
                    test=next_ep['test_index'] + 1,
                    env=next_ep['env_name'],
                    pos=next_ep['human_player'],
                    ep=next_ep['episode_number'],
                    total=next_ep['total_episodes'],
                )
            else:
                run_label = _T("Run next episode (all done)")

            extra = [_T("User: {user}  |  Total Score: {score}", user=current_user, score=f"{total_score:.1f}")]
            if next_ep is None:
                results_path = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "..", "exp_configs", "user_data", current_user, "results.json.enc")
                )
                extra.append(_T("All tests completed! Send results.json.enc to the admin for your reward."))
                extra.append(f"Path: {results_path}")

            if next_ep is not None:
                options = [
                    (run_label, True),
                    (_T("Open Test Menu"), True),
                    (_T("Practice"), True),
                    (_T("Logout"), True),
                    (_T("Exit"), True),
                ]
            else:
                options = [
                    (run_label, False),
                    (_T("Reset Progress"), True),
                    (_T("Practice"), True),
                    (_T("Logout"), True),
                    (_T("Exit"), True),
                ]
            choice = _menu_select(options, extra_lines=extra)
            if choice == -1 or choice == 4:
                print(_T("Goodbye!"))
                return 0

            if choice == 0 and next_ep:
                ret = _run_single_episode(next_ep["test_index"])
                if ret != 0:
                    return ret

            elif choice == 1:
                if next_ep is not None:
                    test_idx = _run_test_select(current_user)
                    if test_idx == -2:
                        continue
                    elif test_idx is not None:
                        ret = _run_single_episode(test_idx)
                        if ret != 0:
                            return ret
                else:
                    init_user_progress(current_user)
                    print(_T("Progress reset. You can now start testing again."))
                    input(_T("Press Enter to continue..."))

            elif choice == 2:
                pr = _practice_select()
                if pr:
                    env_name, human_player = pr
                    ret = _run_practice(env_name, human_player)
                    if ret != 0:
                        return ret

            elif choice == 3:
                logout_user()
                print(_T("Logged out '{current}'.", current=current_user))
                input(_T("Press Enter to continue..."))


def main() -> int:
    # Launch interactive menu if no arguments given (or only --zh)
    has_zh_only = len(sys.argv) == 2 and sys.argv[1] == "--zh"
    if len(sys.argv) == 1 or has_zh_only:
        if has_zh_only:
            _set_lang("zh")
        return interactive_menu()

    parser = build_parser()
    args = parser.parse_args()

    if args.zh:
        _set_lang("zh")

    if args.version:
        print_version()
        return 0

    if args.list_envs:
        print_envs()
        return 0

    # First-run key setup for CLI operations that touch encrypted data.
    if not _has_encrypted_files():
        _setup_key()
        _encrypt_existing_configs()

    if args.whoami:
        return run_whoami()

    if args.logout:
        return run_logout()

    if args.register:
        return run_register(args)

    if args.login:
        return run_login(args)

    if args.demo:
        try:
            return run_demo_mode(args)
        except KeyboardInterrupt:
            print(_T("Interrupted by user."))
            return 130
        except Exception as e:
            print(f"\n{_T('Error: ')}{e}")
            return 1

    if args.random:
        try:
            return run_random_mode(args)
        except KeyboardInterrupt:
            print(_T("Interrupted by user."))
            try:
                import pygame
                pygame.quit()
            except Exception:
                pass
            return 130
        except Exception as e:
            print(f"\n{_T('Error: ')}{e}")
            try:
                import pygame
                pygame.quit()
            except Exception:
                pass
            return 1

    # Default: run next test for logged-in user
    current = get_current_user()
    if not current:
        print(_T("[ERR] Please log in first."))
        return 1
    try:
        next_ep = get_next_episode(current)
        if next_ep is None:
            print(_T("All tests completed!"))
            return 0
        return _run_single_episode(next_ep["test_index"])
    except KeyboardInterrupt:
        print(_T("Interrupted by user."))
        try:
            import pygame
            pygame.quit()
        except Exception:
            pass
        return 130
    except Exception as e:
        print(f"\n{_T('Error: ')}{e}")
        try:
            import traceback
            traceback.print_exc()
        except Exception:
            pass
        try:
            import pygame
            pygame.quit()
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
