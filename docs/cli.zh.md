# CLI 参考

> [README](../README.zh.md) · 完整使用说明请见 [用户手册](user_guide.zh.md) | [English](cli.md) · [管理员手册](admin_guide.zh.md)

## 快速开始示例

### 交互式菜单（默认，无参数）

```bash
overcooked-human-test
```

使用 **上下方向键** 移动光标，**回车键** 选择。

### 终端演示模式（无需模型）

```bash
overcooked-human-test --demo
```

### 随机 AI 对手（无需策略池）

```bash
overcooked-human-test -e <env_name> --random
```

### 无头 AI 对战测试

```bash
overcooked-human-test -e <env_name> --random --headless
```

### 用户注册与登录

```bash
# 注册新用户（交互式输入密码）
overcooked-human-test --register --user-id alice

# 登录（交互式输入密码）
overcooked-human-test --login --user-id alice

# 查看当前用户
overcooked-human-test --whoami

# 登出
overcooked-human-test --logout
```

---

## 完整参数列表

```
overcooked-human-test [选项]

选项：
  -e, --env ENV              地图/环境名称（默认: <env_name>）
  -p, --human-player {0,1}   人类玩家序号（默认: 1）
  -n, --episodes N           测试局数（默认: 1）
  --tile-size SIZE           Pygame 图块像素大小（默认: 75）
  --fps FPS                  游戏速度，步/秒（默认: 4）
  --policy-pool PATH         策略池目录路径
  --random                   使用随机 AI，不加载训练模型
  --demo                     运行终端演示模式（无需模型）
  --headless                 无图形模式，仅 AI 运行
  --user-id USER_ID          用户 ID
  --exp-config PATH          实验配置文件路径
  --register                 注册新用户（需要 --user-id）
  --login                    登录用户（需要 --user-id）
  --logout                   登出当前用户
  --whoami                   显示当前登录用户
  --list-envs                列出可用地图并退出
  -v, --version              显示版本并退出
  -h, --help                 显示帮助并退出
```

---

## 用户注册与登录

运行测试前，用户需要先注册：

```bash
overcooked-human-test --register --user-id alice
```

这会在 `exp_configs/user_data/users.json.enc` 中创建本地账户（磁盘加密存储）。注册后登录：

```bash
overcooked-human-test --login --user-id alice
```

登录后，交互式菜单会自动使用当前用户。

---

## 工作原理

**算法名称永远不会向参与者透露**。系统按以下方式工作：

1. 注册并登录（见上文）。
2. 实验配置（`exp_configs/experiment_config.json`）定义了测试计划：环境、位置和局数。
3. 共有 **4 个测试**（2 环境 × 2 位置）。每个测试包含 `algorithm_pool` 中的每个算法重复 `episodes_per_config` 次，内部局顺序按随机种子打乱。
4. 已登录用户运行一局时，CLI 从选中的测试里选取下一个未完成的 episode。
5. 游戏过程中仅显示**环境名称**和**玩家位置**，算法被隐藏。
6. 每结束一个 **episode**，用户返回菜单，可选择继续、切换测试或退出。
7. 每局结束后自动保存结果。

### 实验配置格式

```json
{
  "experiment_name": "overcooked_human_ai_test",
  "environments": ["random1_m", "random3_m"],
  "positions": [0, 1],
  "episodes_per_config": 2,
  "algorithm_pool": ["bach", "fcp", "mep"]
}
```

| 字段 | 说明 |
|------|------|
| `environments` | 要测试的环境/地图列表 |
| `positions` | 人类玩家将使用的位置列表（0 或 1） |
| `episodes_per_config` | 每个算法在单个测试中出现几次 |
| `algorithm_pool` | 策略池中的 AI 算法列表（从不显示） |

在默认配置下产生 **4 个测试**，每个测试包含 **6 局**（3 算法 × 2 次）。6 局的顺序对每个用户随机打乱。

### 进度追踪

每个用户的进度保存在 `exp_configs/user_data/<user_id>/progress.json.enc`（磁盘加密存储）：

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

### 结果数据

单个用户的所有测试结果存放在**一个加密文件**中：
`exp_configs/user_data/<user_id>/results.json.enc`

文件内容为 JSON 数组，每个元素是一次完成的**单局**：

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

结果文件使用内嵌密钥自动加密，被试无法直接读取。测试结束后，被试将 `results.json.enc` 文件发给管理员，管理员使用 `results_reader.py` 解密查看。

#### 解密结果文件（仅管理员）

```bash
python results_reader.py path/to/results.json.enc
# 交互式输入解密密钥

# 或直接传入密钥
python results_reader.py path/to/results.json.enc --key "YOUR_KEY"

# 导出为 CSV
python results_reader.py path/to/results.json.enc --key "YOUR_KEY" --csv output.csv

# 以 JSON 格式输出
python results_reader.py path/to/results.json.enc --key "YOUR_KEY" --json
```

### 编程方式查看进度

```python
from experiment_manager import print_progress, load_all_results

# 打印进度到控制台
print_progress("alice")

# 加载所有结果为字典列表
results = load_all_results("alice")
```

---

## 策略池解析顺序

CLI 按以下顺序查找策略池目录：

1. `--policy-pool PATH`（如果提供了参数）
2. `POLICY_POOL` 环境变量
3. `exp_configs/policy_pool/`（项目根目录相对路径）
4. `zsceval/policy_pool/`（项目根目录相对路径）

---

## 加密

`exp_configs/` 下的所有文件均以**加密形式**存储在磁盘上。加密密钥内嵌在 `crypto_utils.py` 中（经过混淆）。被试无法通过直接打开文件查看原始数据。

分发项目前的准备：

```bash
# 加密实验配置文件（如果还是明文）
python admin_crypto.py encrypt exp_configs/experiment_config.json --remove-plain
```

管理员解密文件：

```bash
python admin_crypto.py decrypt exp_configs/experiment_config.json.enc
```

---

## 可用环境

| 环境 | 说明 |
|------|------|
| `<env_name>` | 基础厨房（新手推荐） |
| `<env_name>` | 三种食材厨房 |

列出所有：
```bash
overcooked-human-test --list-envs
```

---

## 控制方式

| 按键 | 动作 |
|------|------|
| `W` / `↑` | 向上移动 |
| `S` / `↓` | 向下移动 |
| `A` / `←` | 向左移动 |
| `D` / `→` | 向右移动 |
| `Space` / `E` | 交互（拾取 / 放下 / 烹饪） |
| `Q` / `Esc` | 退出 |

---

## 故障排除

**`Policy file not found`**
- 确保 `exp_configs/policy_pool/` 包含每个环境的训练模型。
- 或设置 `POLICY_POOL` 指向你的模型目录：
  ```bash
  export POLICY_POOL=/path/to/policy_pool
  ```
- 使用 `--random` 或 `--demo` 可在不加载模型的情况下运行。

**Pygame 窗口无法响应键盘**
- 点击 Pygame 窗口使其获得焦点。
- 在 macOS 上，为终端应用授予**输入监控**权限。

**Linux 键盘问题**
```bash
sudo usermod -a -G input $USER
# 注销并重新登录
```
