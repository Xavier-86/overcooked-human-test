<a id="readme-top"></a>

<div align="center">

# Overcooked Human-AI Testing

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux-lightgrey.svg)]()

> Cook with an AI teammate without knowing which algorithm it is.<br>Blind testing, encryption, and independent randomization — all designed for one purpose: **fair experiments with reliable incentive distribution**.

[中文](README.zh.md) · [GitHub](https://github.com/Xavier-86/overcooked-human-test.git)

<img src="assets/image.png" alt="Game Screenshot" width="400">

</div>

---

## Features

- 🎭 **Blind Testing** — Eliminates algorithm bias, ensuring incentives are awarded based on genuine performance
- 🔐 **Local Encryption** — Results are encrypted with the administrator's RSA public key; only the administrator can decrypt or verify them
- 🔀 **Independent Randomization** — Each player's test order is shuffled separately, so incentive evaluation is unaffected by others
- 🎯 **Practice Mode** — Unlimited practice runs that do not affect formal test scores or incentive calculations
- 🛠️ **Admin Tools** — Built-in encryption, decryption, and result reader
- 🌐 **Bilingual UI** — Switch between English and Chinese
- ⌨️ **Interactive Menu** — Arrow-key menu or command-line operation

---

## Installation

```bash
git clone https://github.com/Xavier-86/overcooked-human-test.git
cd overcooked-human-test

conda create -n human-test python=3.9
conda activate human-test

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

Then launch:

```bash
overcooked-human-test
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [User Guide](docs/user_guide.md) | Participant instructions: registration, gameplay, controls, scoring |
| [Admin Guide](docs/admin_guide.md) | Administrator setup: encryption key, config, decrypting results |
| [CLI Reference](docs/cli.md) | Complete command-line options and troubleshooting |

---

## License

Distributed under the MIT license. See [`LICENSE`](LICENSE) for more information.
