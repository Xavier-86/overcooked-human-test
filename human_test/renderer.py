#!/usr/bin/env python3
"""
渲染器模块
支持 CLI (Unicode) 和 ASCII 两种渲染模式
"""

import os
import sys
from typing import Dict, Any, List

# 跨平台颜色支持
try:
    from colorama import init, Fore, Back, Style
    init()  # 初始化 colorama
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False
    # 定义空的颜色常量
    class _DummyColor:
        def __getattr__(self, name):
            return ''
    Fore = Back = Style = _DummyColor()


class BaseRenderer:
    """渲染器基类"""
    
    def __init__(self):
        self.width = 60
        self.height = 20
    
    def clear(self):
        """清屏"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def render(self, game_state: Dict[str, Any]):
        """渲染游戏状态"""
        raise NotImplementedError
    
    def _get_player_symbol(self, player_id: int, has_object: bool) -> str:
        """获取玩家符号"""
        symbols = ['P', 'A']  # P = Player (人类), A = AI
        symbol = symbols[player_id] if player_id < len(symbols) else str(player_id)
        if has_object:
            symbol = symbol.lower()
        return symbol


class CLIRenderer(BaseRenderer):
    """CLI 渲染器 (使用 Unicode 边框)"""
    
    def render(self, game_state: Dict[str, Any]):
        """渲染游戏画面"""
        self.clear()
        
        # 游戏标题
        print('╔' + '═' * (self.width - 2) + '╗')
        print('║' + ' Human vs AI - Overcooked '.center(self.width - 2) + '║')
        print('╠' + '═' * (self.width - 2) + '╣')
        
        # 状态栏
        score = game_state.get('score', 0)
        time_left = game_state.get('time', 0)
        status_line = f' Score: {score} | Time: {time_left}s '
        print('║' + status_line.ljust(self.width - 2) + '║')
        print('╠' + '═' * (self.width - 2) + '╣')
        
        # 游戏区域 - 左对齐显示网格
        grid = game_state.get('grid', [])
        if grid:
            # 计算网格宽度
            cell_width = 2  # 每个格子显示为2个字符宽度
            for row in grid:
                # 将每个格子格式化为固定宽度
                formatted_row = ''.join(self._format_cell(cell).ljust(cell_width) for cell in row)
                print('║ ' + formatted_row.ljust(self.width - 4) + ' ║')
        
        # 填充空白行
        grid_height = len(grid) if grid else 0
        for _ in range(self.height - grid_height - 6):
            print('║' + ' ' * (self.width - 2) + '║')
        
        # 控制说明
        print('╠' + '═' * (self.width - 2) + '╣')
        controls = ' [WASD]Move [E]Act [Q]Quit '
        print('║' + controls.center(self.width - 2) + '║')
        print('╚' + '═' * (self.width - 2) + '╝')
    
    def _format_cell(self, cell: str) -> str:
        """格式化单元格"""
        # 使用颜色
        color_map = {
            '#': Fore.WHITE + '█' + Style.RESET_ALL,  # 墙
            ' ': ' ',  # 空地
            'P': Fore.GREEN + '●' + Style.RESET_ALL,  # 人类玩家
            'A': Fore.RED + '●' + Style.RESET_ALL,    # AI玩家
            'p': Fore.GREEN + '◐' + Style.RESET_ALL,  # 人类携带物品
            'a': Fore.RED + '◐' + Style.RESET_ALL,    # AI携带物品
            'O': Fore.YELLOW + '○' + Style.RESET_ALL, # 洋葱
            'T': Fore.RED + '○' + Style.RESET_ALL,    # 番茄
            'D': Fore.BLUE + '□' + Style.RESET_ALL,   # 盘子
            'S': Fore.CYAN + '◆' + Style.RESET_ALL,   # 锅
            'C': Fore.MAGENTA + '☆' + Style.RESET_ALL,# 完成的食物
        }
        return color_map.get(cell, cell)


class ASCIIRenderer(BaseRenderer):
    """ASCII 渲染器 (兼容性更好)"""
    
    def render(self, game_state: Dict[str, Any]):
        """渲染游戏画面"""
        self.clear()
        
        # 游戏标题
        print('+' + '-' * (self.width - 2) + '+')
        print('|' + ' Human vs AI - Overcooked '.center(self.width - 2) + '|')
        print('+' + '-' * (self.width - 2) + '+')
        
        # 状态栏
        score = game_state.get('score', 0)
        time_left = game_state.get('time', 0)
        status_line = f' Score: {score}  |  Time: {time_left}s '
        print('|' + status_line.ljust(self.width - 2) + '|')
        print('+' + '-' * (self.width - 2) + '+')
        
        # 游戏区域
        grid = game_state.get('grid', [])
        for row in grid:
            line = ''.join(self._format_cell_ascii(cell) for cell in row)
            print('| ' + line.ljust(self.width - 4) + ' |')
        
        # 填充空白行
        grid_height = len(grid)
        for _ in range(self.height - grid_height - 6):
            print('|' + ' ' * (self.width - 2) + '|')
        
        # 控制说明
        print('+' + '-' * (self.width - 2) + '+')
        controls = ' [WASD]Move [Space]Act [Q]Quit '
        print('|' + controls.center(self.width - 2) + '|')
        print('+' + '-' * (self.width - 2) + '+')
    
    def _format_cell_ascii(self, cell: str) -> str:
        """格式化单元格 (ASCII 版本)"""
        ascii_map = {
            '#': '#',  # 墙
            ' ': ' ',  # 空地
            'P': 'P',  # 人类玩家
            'A': 'A',  # AI玩家
            'p': 'p',  # 人类携带物品
            'a': 'a',  # AI携带物品
            'O': 'o',  # 洋葱
            'T': 't',  # 番茄
            'D': 'd',  # 盘子
            'S': 'S',  # 锅
            'C': '*',  # 完成的食物
        }
        return ascii_map.get(cell, cell)
