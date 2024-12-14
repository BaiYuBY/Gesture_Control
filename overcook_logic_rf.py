import cv2
import mediapipe as mp
import numpy as np
import vgamepad as vg
from pynput import keyboard

from tk.tk_param_window import *
from overcooked.player import Player
from overcooked.timer import *

# 初始化 MediaPipe 姿势估计模型
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 初始化 OpenCV 视频捕获对象
cap = cv2.VideoCapture(0)

# 全局变量
is_active = False  # 标志是否正在录制
output_dir = None  # 当前输出文件夹
video_writer = None  # 视频写入对象
frame_index = 0  # 帧计数

# 姿态检测参数
left_right_threshold = 0.11
head_body_threshold = 0.08
move_threshold = 0.55
pack_threshold = 0.15

# 肢体对应的landmark索引
head_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
left_hand_indices = [15, 17, 19, 21]
right_hand_indices = [16, 18, 20, 22]
hand_indeices = [15, 16, 17, 18, 19, 20, 21, 22]
body_left_indices = [11, 23]
body_right_indices = [12, 24]
body_up_indices = [11, 12]
body_indices = [11, 12, 23, 24]

# 创建计时器对象和tk调参面板
timers = TimerManager()
tk_window = TKParamWindow()

# 调参面板参数
auto_press_duration = tk_window.get_scalar(TKDataType.FLOAT, "自动按键时间", default_value=0.1, range_min=0.001, range_max=1.0)
cam_cap_resize = tk_window.get_scalar(TKDataType.FLOAT, "识别图像缩放倍数", default_value=0.6, range_min=0.1, range_max=1.0)

# 创建玩家对象
p1 = Player("Player1")
p2 = Player("Player2")

# 按钮绑定：字符串 -> 手柄按键
button_mapping = {
    'throw': vg.XUSB_BUTTON.XUSB_GAMEPAD_A,  # H -> X 按键
    'cut': vg.XUSB_BUTTON.XUSB_GAMEPAD_X,  # J -> Y 按键
    'B': vg.XUSB_BUTTON.XUSB_GAMEPAD_B,  # K -> A 按键
    "Y": vg.XUSB_BUTTON.XUSB_GAMEPAD_Y,  # L -> B 按键
}

# 按钮绑定：键盘按键 -> 手柄按键
keyboard_mapping = {
    keyboard.Key.esc: vg.XUSB_BUTTON.XUSB_GAMEPAD_B,
}


def auto_press_button(button: str, player, duration):
    """自动模拟按下-等待-抬起的过程。玩家按下按钮，一段时间后释放按钮，如还没释放就不能再按"""

    def release_button():
        """释放按钮"""
        player.gamepad.release_button(button=button_mapping[button])  # 释放按键
        player.gamepad.update()  # 更新手柄状态
        print(f'player{player.id} release {button}')

    # 模拟按下按钮
    player.gamepad.press_button(button=button_mapping[button])
    player.gamepad.update()  # 更新手柄状态
    print(f'player{player.id} press {button}')

    # 找到player对应按键绑定的timer，设为duration的时间；如果没有timer就创建个新的
    timer = player.button_timers.get(button)
    if timer is None:  # 要么没有这个键，要么有键但值是None
        player.button_timers[button] = timers.start_new_timer(duration, release_button)
    else:
        timer.duration = duration  # 重置时间
        timer.callback = release_button
        timer.start()


def press_down_button(button: [str | vg.XUSB_BUTTON], player: Player):
    """玩家按下按钮"""
    if type(button) is str:
        player.gamepad.press_button(button=button_mapping[button])
    else:
        player.gamepad.press_button(button=button)
    player.gamepad.update()  # 更新手柄状态


def release_up_button(button: [str | vg.XUSB_BUTTON], player: Player):
    """玩家抬起按钮"""
    if type(button) is str:
        player.gamepad.release_button(button=button_mapping[button])
    else:
        player.gamepad.release_button(button=button)
    player.gamepad.update()  # 更新手柄状态


def on_keyboard_press(key: keyboard.Key):
    """按下键盘按键"""

    try:
        if key.char == 'q':  # 特殊处理，用于退出程序
            global is_running_demo
            is_running_demo = False
    except Exception:
        pass

    button = keyboard_mapping.get(key)
    if button is None:
        return
    press_down_button(button, p1)  # 不知道游戏里哪个是主玩家，所以两个玩家各模拟一次
    press_down_button(button, p2)


def on_keyboard_prelease(key: keyboard.Key):
    """抬起键盘按键"""
    button = keyboard_mapping.get(key)
    if button is None:
        return
    release_up_button(button, p1)
    release_up_button(button, p2)


# 开始keyboard后台键盘监听
keyboard_listener = keyboard.Listener(on_press=on_keyboard_press, on_release=on_keyboard_prelease, suppress=True)
keyboard_listener.start()

def update_axes(action, player: Player):
    """更新手柄的 X 和 Y 轴值"""
    if action == 'up':
        player.y_input = 32767 // 2
    elif action == 'down':
        player.y_input = -32767 // 2
    elif action == 'left':
        player.x_input = -32767 // 2
    elif action == 'right':
        player.x_input = 32766 // 2
    elif action == 'x_stop':
        player.x_input = 0
    elif action == 'y_stop':
        player.y_input = 0
    player.gamepad.left_joystick(player.x_input, player.y_input)
    player.gamepad.update()


def simple_logic(landmarks, player: Player):
    if not is_active:
        return

    # 提取关键点数据
    keypoints = []

    for landmark in landmarks:
        keypoints.append({
            "x": landmark.x,
            "y": landmark.y,
            "z": landmark.z,
            "visibility": landmark.visibility
        })

    left_hand_avg = {key: np.mean([keypoints[i][key] for i in left_hand_indices]) for key in keypoints[0].keys()}
    right_hand_avg = {key: np.mean([keypoints[i][key] for i in right_hand_indices]) for key in keypoints[0].keys()}
    hand_avg = {key: np.mean([keypoints[i][key] for i in hand_indeices]) for key in keypoints[0].keys()}
    body_left_avg = {key: np.mean([keypoints[i][key] for i in body_left_indices]) for key in keypoints[0].keys()}
    body_right_avg = {key: np.mean([keypoints[i][key] for i in body_right_indices]) for key in keypoints[0].keys()}
    body_up_avg = {key: np.mean([keypoints[i][key] for i in body_up_indices]) for key in keypoints[0].keys()}
    body_avg = {key: np.mean([keypoints[i][key] for i in body_indices]) for key in keypoints[0].keys()}
    left_shoulder = keypoints[11]
    right_shoulder = keypoints[12]

    # if frame_index % 60 == 0:
    #     # print(f'left_hand_avg: {left_hand_avg}')
    #     # print(f'right_hand_avg: {right_hand_avg}')
    #     # print(f'head_avg: {head_avg}')
    #     print(f'hand_avg: {hand_avg}')
    #     print(f'body_left_avg: {body_left_avg}')
    #     print(f'body_right_avg: {body_right_avg}')
    #     print(f'body_up_avg: {body_up_avg}')
    #     print(f'body_avg: {body_avg}')
    left_check = body_left_avg['x'] < hand_avg['x'] and body_avg['y'] >= hand_avg['y'] >= body_up_avg['y']
    right_check = body_right_avg['x'] > hand_avg['x'] and body_avg['y'] >= hand_avg['y'] >= body_up_avg['y']
    up_check = body_right_avg['x'] <= hand_avg['x'] <= body_left_avg['x'] and hand_avg['y'] < body_up_avg['y']
    down_check = body_right_avg['x'] <= hand_avg['x'] <= body_left_avg['x'] and hand_avg['y'] > body_avg['y']

    if body_left_avg['x'] < hand_avg['x'] and body_avg['y'] >= hand_avg['y'] >= body_up_avg['y']:
        update_axes('left', player)
        print(f'player{player.id} move left')
    elif body_right_avg['x'] > hand_avg['x'] and body_avg['y'] >= hand_avg['y'] >= body_up_avg['y']:
        update_axes('right', player)
        print(f'player{player.id} move right')
    else:
        update_axes('x_stop', player)
        # print(f'player{player_num} stop')

    if body_right_avg['x'] <= hand_avg['x'] <= body_left_avg['x'] and hand_avg['y'] < body_up_avg['y']:
        update_axes('up', player)
        print(f'player{player.id} move up')
    elif body_right_avg['x'] <= hand_avg['x'] <= body_left_avg['x'] and hand_avg['y'] > body_avg['y']:
        update_axes('down', player)
        print(f'player{player.id} move down')
    else:
        update_axes('y_stop', player)
        # print(f'player{player_num} stop')

    right_hand_position = (right_hand_avg['x'], right_hand_avg['y'])
    left_hand_position = (left_hand_avg['x'], left_hand_avg['y'])

    if (sum(abs(a - b) for a, b in zip(player.last_right_hand_position, right_hand_position)) +
        sum(abs(a - b) for a, b in zip(player.last_left_hand_position, left_hand_position))) > move_threshold:
        auto_press_button('cut', player, auto_press_duration)
        print(f'player{player.id} cut')
    elif sum(abs(a - b) for a, b in zip(right_hand_position, left_hand_position)) < pack_threshold:
        auto_press_button('throw', player, auto_press_duration)
        print(f'player{player.id} throw')
    player1_last_right_hand_position = right_hand_position
    player1_last_left_hand_position = left_hand_position

is_running_demo = True  # 标记是否正在运行Demo
while is_running_demo:
    ret, ori_frame = cap.read()
    if not ret:
        break

    # 缩放图像，提高识别速度
    height, width, _ = ori_frame.shape
    rescale = eval(cam_cap_resize.get())
    frame = cv2.resize(ori_frame, dsize=(int(width * rescale), int(height * rescale)), interpolation=cv2.INTER_NEAREST)

    # 获取帧的宽度和高度
    height, width, _ = frame.shape

    # 计算中间位置
    mid_x = width // 2

    # 分割帧为左半部分和右半部分
    left_frame = frame[:, :mid_x]
    right_frame = frame[:, mid_x:]

    # 将图像从 BGR 转换为 RGB
    left_rgb_frame = cv2.cvtColor(left_frame, cv2.COLOR_BGR2RGB)
    right_rgb_frame = cv2.cvtColor(right_frame, cv2.COLOR_BGR2RGB)

    # 进行姿势估计
    left_result = pose.process(left_rgb_frame)
    right_result = pose.process(right_rgb_frame)

    if right_result.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            right_frame, right_result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.putText(right_frame, "Player 2", (5, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        simple_logic(right_result.pose_landmarks.landmark, 2)  # 处理右手

    if left_result.pose_landmarks:
        # 在副本上绘制关键点
        mp.solutions.drawing_utils.draw_landmarks(
            left_frame, left_result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.putText(left_frame, "Player 1", (5, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        simple_logic(left_result.pose_landmarks.landmark, 1)  # 处理左手

    processed_frame = cv2.hconcat([left_frame, right_frame])  # 左右两边画面拼接
    cv2.line(processed_frame, (mid_x, 0), (mid_x, height), (255, 0, 0), 2)  # 绿色分割线，线宽为2
    # 在副本上绘制文字（显示录制状态）
    cv2.putText(processed_frame, "Recording: ON" if is_active else "Recording: OFF",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if is_active else (0, 0, 255), 2)

    # 显示带有标记的窗口
    cv2.imshow("MediaPipe Pose Demo", processed_frame)

    # 如果正在录制，将帧写入视频文件（这里写入的是原始视频帧，不包含标记）
    if is_active and video_writer:
        video_writer.write(frame)  # 录制原始画面，不包含标记

    # 按键处理
    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):  # 按 'r' 键开始新录制
        if not is_active:
            is_active = True

    elif key == ord('s'):  # 按 's' 键停止录制
        if is_active:
            is_active = False

    elif key == ord('q'):  # 按 'q' 键退出程序
        is_running_demo = False
        break

    timers.update()  # 更新定时器

# join keyboard listener线程
keyboard_listener.stop()

# join tk参数面板线程
tk_window.quit()
tk_window.join_loop_thread()

# 释放cv窗口资源
cap.release()
if video_writer:
    video_writer.release()
cv2.destroyAllWindows()

print("程序退出")
