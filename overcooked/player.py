import vgamepad as vg
from typing import Dict


class Player:
    """
    玩家类
    """
    PLAYER_ID = 1  # 玩家 ID，从1开始

    def __init__(self, name):
        self.id = self.PLAYER_ID  # 全局ID（数字）
        self.PLAYER_ID += 1
        self.name = name

        self.gamepad = vg.VX360Gamepad()
        self.x_input = 0  # X 轴的初始位置（范围 -32768 到 32767）
        self.y_input = 0  # Y 轴的初始位置（范围 -32768 到 32767）

        self.last_left_hand_position = (0, 0)
        self.last_right_hand_position = (0, 0)

        self.button_timers: Dict[str, TimerManager.Timer] = {}  # 按键：计时器；给每个按键分配一个计时器

