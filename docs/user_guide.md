# User Guide

> [README](../README.md) · [中文](user_guide.zh.md) · [CLI Reference](cli.md) · [Admin Guide](admin_guide.md)

---

## Getting Started

Launch the interactive menu:

```bash
overcooked-human-test
```

For the Chinese interface:

```bash
overcooked-human-test --zh
```

This opens a menu where you can **Register**, **Login**, **Open Test Menu**, **View Progress**, and **Logout**. Use **UP/DOWN arrow keys** to navigate and **ENTER** to select.

**Note:** You must log in every time you start the CLI.

---

## Practice Mode

You can enter **Practice** from the logged-in menu. Practice games let you try any environment and player position, but **results are not recorded**.

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

## Game Objective & Recipes

### Objective

Your goal is to **cook and serve as many orders as possible** before time runs out. Each order has a **time limit** and a **point value** -- serve it before the timer expires to earn the points.

### Recipes

| Order | Ingredients | Time Limit | Points |
|-------|-------------|------------|--------|
| Onion-Tomato Soup | 1 Onion + 1 Tomato | 10 | 10 |
| Onion Soup | 2 Onion | 10 | 10 |
| Triple-Onion-Tomato Soup | 1 Onion + 2 Tomato | 20 | 20 |
| Double-Onion-Tomato Soup | 2 Onion + 1 Tomato | 20 | 20 |
| Triple-Onion Soup | 3 Onion | 20 | 20 |

### Cooking Steps

1. **Gather** ingredients from the counter.
2. **Place** them into an empty pot.
3. **Interact with the stove** (press `Space` / `E`) to **start cooking**.
4. **Wait** for the soup to finish cooking.
5. **Pick up** a plate, then **interact with the pot** to fill it.
6. **Deliver** the plated soup to the serving window.

> **Important:** Placed ingredients will **not** cook automatically -- you must interact with the stove to begin cooking!

---

## Scoring

Your **Total Score** is computed as follows:

```
Total Score = sum over all episodes of max(episode_score - 100, 0) * 0.1
              divided by the total number of episodes (fixed)
```

- Only the portion **above 100** is counted per episode (scores below 100 contribute **0**, no penalty)
- Unfinished episodes count as **0** toward the total
- The score is **monotonically non-decreasing** -- it only goes up as you complete more episodes
- The total score is displayed in the menu header and in the progress view

---

## After Completing All Tests

When all tests are finished, send the encrypted results file to the administrator to claim your reward. The path is shown on screen:

```
exp_configs/user_data/<your_user_id>/results.json.enc
```

**Do not** rename or modify this file.
