<a id="readme-top"></a>

<div align="center">

# Overcooked 人机实验

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey.svg)]()

> 和 AI 队友一起烹饪。<br>盲测设计、本地加密、独立随机 —— 专为人机协作研究打造的测试平台。

[English](README.md) · [GitHub](https://github.com/Xavier-86/overcooked-human-test.git)

<img src="assets/image.png" alt="游戏截图" width="400">

</div>

---

## 特性

- 🎭 **盲测设计** — 玩家全程不知道自己面对的是哪种 AI 算法
- 🔐 **本地加密** — 所有数据在本地加密保存，安全可靠
- 🔀 **独立随机** — 每位玩家的测试顺序单独打乱，互不干扰
- 🎯 **练习模式** — 不计成绩的无限次自由练习
- 🛠️ **管理员工具** — 内置加解密与结果读取功能
- 🌐 **中英双语** — 支持中英文界面切换
- ⌨️ **交互式菜单** — 方向键菜单 + 命令行，两种操作方式

---

## 安装

```bash
git clone https://github.com/Xavier-86/overcooked-human-test.git
cd overcooked-human-test

conda create -n human-test python=3.9
conda activate human-test

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

然后启动：

```bash
overcooked-human-test
```

使用 `--zh` 进入中文界面：

```bash
overcooked-human-test --zh
```

---

## 文档

| 文档 | 说明 |
|------|------|
| [用户手册](docs/user_guide.zh.md) | 参与者指南：注册、游戏操作、控制方式、计分规则 |
| [管理员手册](docs/admin_guide.zh.md) | 管理员设置：加密密钥、配置文件、解密结果 |
| [CLI 参考](docs/cli.zh.md) | 完整的命令行选项和故障排除 |

---

## 许可证

基于 MIT 许可证分发。更多信息请参见 [`LICENSE`](LICENSE)。
