#!/usr/bin/env python3
"""
overcooked-with-AI: Human-AI interaction testing tool
完全独立的工具，无需 ZSC-Eval
"""

import sys
import os
import time
import numpy as np
from typing import Optional, Dict, Any, List

from .renderer import CLIRenderer, ASCIIRenderer
from .keyboard import KeyboardHandler


class DummyMDP:
    """简化版 MDP 用于人机交互测试"""
    
    def __init__(self, env_name: str = "random1_m"):
        self.env_name = env_name
        # 预定义地图布局
        self.maps = {
            "random1_m": [
                ['#', '#', '#', '#', '#', '#', '#', '#', '#'],
                ['#', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '#'],
                ['#', ' ', 'O', ' ', ' ', ' ', 'S', ' ', '#'],
                ['#', ' ', ' ', ' ', 'D', ' ', ' ', ' ', '#'],
                ['#', '#', '#', '#', '#', '#', '#', '#', '#'],
            ],
            "random3_m": [
                ['#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#'],
                ['#', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '#'],
                ['#', ' ', 'O', ' ', 'T', ' ', ' ', ' ', 'S', ' ', '#'],
                ['#', ' ', ' ', 'D', ' ', ' ', ' ', ' ', ' ', ' ', '#'],
                ['#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#'],
            ],
        }
        # 默认使用 random1_m
        self.terrain_mtx = self.maps.get(env_name, self.maps["random1_m"])
        self.height = len(self.terrain_mtx)
        self.width = len(self.terrain_mtx[0])
    
    def get_standard_start_state(self):
        """返回初始状态"""
        return DummyState(self)
    
    def get_state_transition(self, state, actions):
        """模拟状态转移 - 处理玩家移动"""
        # 复制当前状态（保留位置）
        new_state = DummyState(self)
        
        # 复制玩家当前位置
        for i, (old_player, new_player) in enumerate(zip(state.players, new_state.players)):
            new_player.position = old_player.position
            new_player._held_object = old_player._held_object
        
        # 更新每个玩家的位置
        for i, (player, action) in enumerate(zip(new_state.players, actions)):
            x, y = player.position
            if action == 0:  # up
                new_y = max(0, y - 1)
                if self.terrain_mtx[new_y][x] != '#':
                    player.position = (x, new_y)
            elif action == 1:  # down
                new_y = min(self.height - 1, y + 1)
                if self.terrain_mtx[new_y][x] != '#':
                    player.position = (x, new_y)
            elif action == 2:  # left
                new_x = max(0, x - 1)
                if self.terrain_mtx[y][new_x] != '#':
                    player.position = (new_x, y)
            elif action == 3:  # right
                new_x = min(self.width - 1, x + 1)
                if self.terrain_mtx[y][new_x] != '#':
                    player.position = (new_x, y)
            # action 4 = interact, 5 = stay (不移动)
        
        # 模拟奖励
        reward = np.random.choice([0, 0, 0, 0, 5, 10], p=[0.6, 0.15, 0.15, 0.05, 0.03, 0.02])
        done = False
        info = {'episode_done': False}
        return new_state, reward, done, info
    
    def get_valid_player_positions(self):
        """获取有效位置"""
        positions = []
        for y, row in enumerate(self.terrain_mtx):
            for x, cell in enumerate(row):
                if cell != '#':
                    positions.append((x, y))
        return positions


class DummyState:
    """简化版游戏状态"""
    
    def __init__(self, mdp):
        self.mdp = mdp
        self.players = [DummyPlayer(0, mdp), DummyPlayer(1, mdp)]
    
    def to_dict(self):
        return {
            'players': [
                {'position': p.position, 'orientation': p.orientation, 'held_object': p._held_object}
                for p in self.players
            ],
            'objects': {}
        }


class DummyPlayer:
    """简化版玩家"""
    
    def __init__(self, idx: int, mdp):
        self.idx = idx
        self.mdp = mdp
        # 玩家0在左侧，玩家1在右侧
        self.position = (1, 2) if idx == 0 else (mdp.width - 2, 2)
        self.orientation = (0, -1)
        self._held_object = None
    
    def has_object(self):
        return self._held_object is not None
    
    def get_object(self):
        return self._held_object


class HumanTest:
    """人机交互测试主类"""
    
    # 动作映射
    ACTION_MAP = {
        'up': 0,
        'down': 1,
        'left': 2,
        'right': 3,
        'interact': 4,
        'stay': 5,
    }
    
    ACTION_NAMES = {
        0: '↑',
        1: '↓',
        2: '←',
        3: '→',
        4: 'Interact',
        5: 'Stay',
    }
    
    def __init__(
        self,
        env_name: str,
        algo: str,
        human_player: int = 0,
        episodes: int = 1,
        render_mode: str = 'cli',
        policy_path: Optional[str] = None,
        seed: Optional[int] = None,
        dry_run: bool = False
    ):
        self.env_name = env_name
        self.algo = algo
        self.human_player = human_player
        self.ai_player = 1 - human_player
        self.episodes = episodes
        self.render_mode = render_mode
        self.seed = seed
        self.dry_run = dry_run
        
        # 结果记录
        self.results = {
            'scores': [],
            'durations': [],
            'soups_cooked': []
        }
        
        # 初始化渲染器
        if render_mode == 'ascii':
            self.renderer = ASCIIRenderer()
        else:
            self.renderer = CLIRenderer()
        
        # 初始化键盘处理器（自动检测最佳模式）
        self.keyboard = KeyboardHandler()
        
        if not dry_run:
            self._setup_environment()
    
    def _setup_environment(self):
        """设置环境"""
        print(f"\n🎮 初始化游戏环境: {self.env_name}")
        self.mdp = DummyMDP(self.env_name)
        print(f"✓ 地图加载成功")
        print(f"  可用地图: random1_m, random3_m")
        print(f"  AI算法: {self.algo} (随机策略)")
    
    def run(self):
        """运行测试"""
        print(f"\n{'='*60}")
        print(f"  overcooked-with-AI: 人机交互测试")
        print(f"{'='*60}")
        print(f"  地图: {self.env_name}")
        print(f"  AI算法: {self.algo}")
        print(f"  你是: Player {self.human_player} (人类)")
        print(f"  AI是: Player {self.ai_player} (AI)")
        print(f"{'='*60}\n")
        
        print("控制说明:")
        print("  [WASD/方向键] 移动")
        print("  [空格] 交互 (拾取/放下/烹饪)")
        print("  [Q/ESC] 退出游戏")
        print()
        
        input("按 Enter 开始游戏...")
        
        for episode in range(self.episodes):
            self._run_episode(episode + 1)
        
        self._print_summary()
    
    def _run_episode(self, episode_num: int):
        """运行单局游戏"""
        print(f"\n--- 第 {episode_num}/{self.episodes} 局 ---")
        
        # 初始化游戏状态
        state = self.mdp.get_standard_start_state()
        
        # 开始监听键盘
        self.keyboard.start()
        
        try:
            step = 0
            max_steps = 500
            total_reward = 0
            soups_cooked = 0
            start_time = time.time()
            
            while step < max_steps:
                # 渲染当前状态
                self._render(state, step, total_reward, soups_cooked)
                
                # 获取人类动作
                human_action_name = self._get_human_action()
                
                if human_action_name == 'quit':
                    print("\n用户退出游戏")
                    break
                
                human_action = self.ACTION_MAP.get(human_action_name, 5)
                
                # 获取 AI 动作（随机）
                ai_action = np.random.randint(0, 6)
                
                # 组合动作
                actions = [0, 0]
                actions[self.human_player] = human_action
                actions[self.ai_player] = ai_action
                
                # 执行动作
                try:
                    new_state, reward, done, info = self.mdp.get_state_transition(
                        state, tuple(actions)
                    )
                    
                    total_reward += reward
                    if reward > 0:
                        soups_cooked += 1
                    
                    state = new_state
                    step += 1
                    
                    if done or info.get('episode_done', False):
                        print(f"\n游戏结束! 得分: {total_reward}, 步数: {step}")
                        break
                    
                    # 控制游戏速度
                    time.sleep(0.15)
                    
                except Exception as e:
                    print(f"执行动作出错: {e}")
                    break
            
            # 记录结果
            duration = time.time() - start_time
            self.results['scores'].append(total_reward)
            self.results['durations'].append(duration)
            self.results['soups_cooked'].append(soups_cooked)
            
        finally:
            self.keyboard.stop()
    
    def _get_human_action(self) -> str:
        """获取人类玩家的动作"""
        return self.keyboard.get_action()
    
    def _render(self, state, step: int, score: int, soups: int):
        """渲染游戏画面"""
        try:
            grid = self._state_to_grid(state)
            
            game_state = {
                'grid': grid,
                'score': score,
                'time': step,
                'soups': soups,
            }
            self.renderer.render(game_state)
        except Exception as e:
            pass
    
    def _state_to_grid(self, state) -> List[List[str]]:
        """将状态转换为可视化网格"""
        grid = []
        
        # 复制地形
        for y, row in enumerate(self.mdp.terrain_mtx):
            new_row = []
            for x, cell in enumerate(row):
                new_row.append(cell if cell != ' ' else ' ')
            grid.append(new_row)
        
        # 添加玩家位置
        for i, player in enumerate(state.players):
            x, y = player.position
            if 0 <= y < len(grid) and 0 <= x < len(grid[0]):
                symbol = 'P' if i == self.human_player else 'A'
                if player.has_object():
                    symbol = symbol.lower()
                grid[y][x] = symbol
        
        return grid
    
    def _print_summary(self):
        """打印测试总结"""
        print(f"\n{'='*60}")
        print("  测试完成!")
        print(f"{'='*60}")
        
        if self.results['scores']:
            avg_score = np.mean(self.results['scores'])
            avg_duration = np.mean(self.results['durations'])
            total_soups = sum(self.results['soups_cooked'])
            
            print(f"  总局数: {len(self.results['scores'])}")
            print(f"  平均得分: {avg_score:.2f}")
            print(f"  平均时长: {avg_duration:.1f}s")
            print(f"  总汤数: {total_soups}")
        
        print(f"{'='*60}\n")
    
    def get_results(self) -> Dict[str, Any]:
        """获取测试结果"""
        return {
            **self.results,
            'avg_score': np.mean(self.results['scores']) if self.results['scores'] else 0,
            'avg_duration': np.mean(self.results['durations']) if self.results['durations'] else 0,
            'total_soups': sum(self.results['soups_cooked']),
        }