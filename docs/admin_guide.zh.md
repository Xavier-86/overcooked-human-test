# 管理员手册

> [README](../README.zh.md) · [English](admin_guide.md) · [用户手册](user_guide.zh.md) · [CLI 参考](cli.zh.md)

---

## 首次设置

当 CLI 首次启动且没有 `.enc` 文件时，管理员必须设置加密密钥。

```bash
overcooked-human-test
```

系统会提示您**生成新的 RSA 密钥对**或**使用已有私钥**。设置过程会创建以下文件：

- `private_key.pem` — **仅管理员持有**。用于解密参与者的结果文件。**切勿分发此文件。**
- `public_key.pem` — 嵌入项目中。参与者的客户端用它加密结果文件。可以安全分发。
- `config.key` — Fernet 密钥，用于加密磁盘上的配置/用户/进度文件。

系统还会要求您设置一个**管理员密码**，用于进入应用内的管理员菜单。

密钥设置完成后，`exp_configs/` 下的所有明文 JSON 文件会自动加密。

---

## 分发流程

向参与者分发项目前，请按以下步骤操作：

1. 首次启动时**生成新的密钥对**（或导入已有私钥）：
   ```bash
   overcooked-human-test
   ```
   这会在项目根目录写入 `private_key.pem`、`public_key.pem` 和 `config.key`。

2. **妥善保存 `private_key.pem`**（例如保存在密码管理器或离线存储中）。后续需要用它来解密参与者发回的 `results.json.enc` 文件。

3. **准备实验配置**（见下文），并加密 `exp_configs/` 下的所有文件。

4. **删除所有敏感的明文文件**（日志、临时 JSON、旧密钥等）。**确保 `private_key.pem` 不在分发包中。**

5. **将项目分发给参与者**。参与者**不需要** `private_key.pem` —— 本地运行只需要 `public_key.pem` 和 `config.key`。

6. **参与者完成测试后**，会将 `results.json.enc` 文件发回给您。使用 `results_reader.py` 和您的 `private_key.pem` 进行解密：
   ```bash
   python results_reader.py path/to/results.json.enc --private-key private_key.pem
   ```

---

## 实验配置

使用 `experiment_config_template.json` 作为起点。**记得去掉文件名中的 `_template`**，使其变为 `experiment_config.json` —— CLI 只会加载不含 `_template` 后缀的文件：

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
| `environments` | 要测试的地图名称列表 |
| `positions` | 人类玩家的位置（0 或 1） |
| `episodes_per_config` | 每个算法在单个测试中出现几次 |
| `algorithm_pool` | 策略池中的 AI 算法列表（从不向用户展示） |

将配置文件放在 `exp_configs/experiment_config.json`。首次运行时如果密钥已设置，CLI 会自动加密该文件。

---

## 管理员 CLI 菜单

在主菜单（未登录状态）选择 **管理员**，输入首次设置时设定的管理员密码。您可以：

- 查看注册用户
- 就地解密并查看结果文件（需要 `private_key.pem`）

---

## 文件结构

```
exp_configs/
├── experiment_config.json.enc      # 加密的实验计划
├── experiment_config_template.json # 明文模板（不会被加载）
├── policy_pool/                    # 训练好的模型文件
└── user_data/
    ├── users.json.enc              # 加密的用户账户
    └── <user_id>/
        ├── progress.json.enc       # 加密的测试进度
        ├── local_results.json.enc  # 客户端可读的结果缓存
        └── results.json.enc        # RSA 加密的测试结果（仅管理员可解密）
```

- 配置/用户/进度文件使用 `config.key` 中的 Fernet 密钥加密。
- `results.json.enc` 使用 RSA 公钥加密，只能用 `private_key.pem` 解密。

---

## 策略网络

`policy_pool/` 中的策略采用 **[ZSC-Eval](https://github.com/sjtu-marl/ZSC-Eval) 默认的网络结构**。在训练或替换模型时，请确保与此架构兼容。

---

## 命令行工具

`admin_crypto.py` 和 `results_reader.py` 的完整命令行选项请参见 [CLI 参考](cli.zh.md)。
