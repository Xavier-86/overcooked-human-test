#!/usr/bin/env python3
"""
键盘输入处理模块
支持跨平台的键盘监听
提供三种模式：
1. pynput（异步，需要权限）
2. 简单输入（同步，需要Enter）
3. 终端原始模式（curses，无需Enter，适合SSH）
"""

import sys
import os
import threading
import queue
import platform
import select
import tty
import termios
from typing import Optional

# 尝试导入 pynput
try:
    from pynput import keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False


class KeyboardHandler:
    """键盘处理器 - 支持 Windows, Linux, macOS"""
    
    def __init__(self, mode='auto'):
        """
        初始化键盘处理器
        
        Args:
            mode: 'auto', 'pynput', 'simple', 'terminal'
        """
        self.action_queue = queue.Queue()
        self.listener = None
        self.running = False
        self.current_action = 'stay'
        self.system = platform.system()
        self.mode = mode
        
        # 自动选择模式
        if mode == 'auto':
            if self.system == 'Linux' and sys.stdin.isatty():
                self.mode = 'terminal'  # SSH/终端优先用原始模式
            elif PYNPUT_AVAILABLE:
                self.mode = 'pynput'
            else:
                self.mode = 'simple'
        
        # 终端原始模式设置
        self.old_terminal_settings = None
        
    def start(self):
        """开始监听键盘"""
        if self.mode == 'simple':
            print("使用简单输入模式（按Enter确认）")
            print("输入: w/a/s/d = 移动, space/e = 交互, q = 退出")
            self.running = True
            return
        
        if self.mode == 'terminal':
            print("使用终端原始模式（无需Enter，直接输入）")
            print("控制: w/a/s/d = 移动, e = 交互, q = 退出")
            self._enable_terminal_mode()
            self.running = True
            return
        
        # pynput 模式
        if self.system == 'Darwin':
            print("提示: macOS 用户请确保已授予终端'输入监控'权限")
        
        if not PYNPUT_AVAILABLE:
            print("警告: pynput 未安装，切换到终端模式")
            self.mode = 'terminal'
            self.start()
            return
        
        self.running = True
        try:
            self.listener = keyboard.Listener(
                on_press=self._on_press,
                on_release=self._on_release,
                suppress=False
            )
            self.listener.start()
        except Exception as e:
            print(f"键盘监听启动失败: {e}")
            print("切换到终端原始模式...")
            self.mode = 'terminal'
            self._enable_terminal_mode()
    
    def stop(self):
        """停止监听键盘"""
        self.running = False
        if self.mode == 'terminal':
            self._disable_terminal_mode()
        if self.listener:
            try:
                self.listener.stop()
            except Exception:
                pass
    
    def _enable_terminal_mode(self):
        """启用终端原始模式（无需按Enter）"""
        try:
            self.old_terminal_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        except Exception as e:
            print(f"终端模式设置失败: {e}")
            self.mode = 'simple'
    
    def _disable_terminal_mode(self):
        """恢复终端设置"""
        if self.old_terminal_settings:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_terminal_settings)
            except Exception:
                pass
    
    def _on_press(self, key):
        """按键按下回调"""
        try:
            if key == keyboard.Key.up:
                self.current_action = 'up'
            elif key == keyboard.Key.down:
                self.current_action = 'down'
            elif key == keyboard.Key.left:
                self.current_action = 'left'
            elif key == keyboard.Key.right:
                self.current_action = 'right'
            elif key == keyboard.Key.space:
                self.current_action = 'interact'
            elif key == keyboard.Key.esc:
                self.current_action = 'quit'
            elif hasattr(key, 'char') and key.char:
                char = key.char.lower()
                if char == 'w':
                    self.current_action = 'up'
                elif char == 's':
                    self.current_action = 'down'
                elif char == 'a':
                    self.current_action = 'left'
                elif char == 'd':
                    self.current_action = 'right'
                elif char == 'q':
                    self.current_action = 'quit'
        except AttributeError:
            pass
    
    def _on_release(self, key):
        """按键释放回调"""
        pass
    
    def get_action(self) -> str:
        """获取当前动作"""
        if self.mode == 'simple':
            return self._get_action_simple()
        elif self.mode == 'terminal':
            return self._get_action_terminal()
        
        # pynput 模式
        action = self.current_action
        self.current_action = 'stay'
        return action
    
    def _get_action_simple(self) -> str:
        """简单输入模式 - 需要Enter"""
        try:
            user_input = input("> ").strip().lower()
            return self._parse_input(user_input)
        except (EOFError, KeyboardInterrupt):
            return 'quit'
    
    def _get_action_terminal(self) -> str:
        """终端原始模式 - 无需Enter"""
        try:
            # 使用非阻塞读取
            import fcntl
            import os
            
            # 获取当前文件状态
            fd = sys.stdin.fileno()
            old_flags = fcntl.fcntl(fd, fcntl.F_GETFL)
            
            # 设置为非阻塞模式
            fcntl.fcntl(fd, fcntl.F_SETFL, old_flags | os.O_NONBLOCK)
            
            try:
                char = sys.stdin.read(1)
                if char:
                    return self._parse_input(char.lower())
            finally:
                # 恢复原来的设置
                fcntl.fcntl(fd, fcntl.F_SETFL, old_flags)
            
            return 'stay'
        except Exception:
            return 'stay'
    
    def _parse_input(self, char: str) -> str:
        """解析输入字符"""
        action_map = {
            'w': 'up',
            's': 'down',
            'a': 'left',
            'd': 'right',
            'e': 'interact',
            ' ': 'interact',
            'q': 'quit',
        }
        return action_map.get(char, 'stay')
    
    def is_available(self) -> bool:
        """检查键盘输入是否可用"""
        return self.running
