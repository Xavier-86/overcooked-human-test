# 管理员手册

> [README](../README.zh.md) · [English](admin_guide.md) · [用户手册](user_guide.zh.md) · [CLI 参考](cli.zh.md)

---

## 首次设置

当 CLI 首次启动且没有 `.enc` 文件时，管理员必须设置加密密钥。

```bash
overcooked-human-test
```

系统会提示您**生成新密钥**或**输入已有密钥**。密钥会被嵌入 `human_test/crypto_utils.py` 中（经过混淆处理）。**请妥善保存原始密钥** —— 后续需要用它来解密参与者的结果文件。

密钥设置完成后，`exp_configs/` 下的所有明文 JSON 文件会自动加密。

---

## 分发流程

向参与者分发项目前，请按以下步骤操作：

1. **生成新密钥**（首次启动时）或输入已有密钥：
   ```bash
   overcooked-human-test
   ```
   密钥会自动嵌入 `human_test/crypto_utils.py` 中。

2. **妥善保存原始密钥**（例如保存在密码管理器中）。后续需要用它来解密参与者发回的 `results.json.enc` 文件。

3. **准备实验配置**（见下文），并加密 `exp_configs/` 下的所有文件。

4. **删除所有敏感的明文文件**（日志、临时 JSON、旧密钥等）。

5. **将项目分发给参与者**。参与者**不需要**原始密钥 —— 嵌入的混淆密钥已足以进行加密和本地运行。

6. **参与者完成测试后**，会将 `results.json.enc` 文件发回给您。使用 `results_reader.py` 和第 2 步保存的原始密钥进行解密。

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

在主菜单（未登录状态）选择 **管理员**，输入加密密钥作为密码。您可以：

- 查看注册用户
- 就地解密并查看结果文件

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
        └── results.json.enc        # 加密的测试结果
```

所有 `.enc` 文件均使用 `crypto_utils.py` 中嵌入的密钥加密。

---

## 策略网络

`policy_pool/` 中的策略采用 **[ZSC-Eval](https://github.com/sjtu-marl/ZSC-Eval) 默认的网络结构**。在训练或替换模型时，请确保与此架构兼容。

---

## 命令行工具

`admin_crypto.py` 和 `results_reader.py` 的完整命令行选项请参见 [CLI 参考](cli.zh.md)。