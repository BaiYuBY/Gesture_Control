import cv2
import mediapipe as mp
import json
import os
import time
import numpy as np
import vgamepad as vg
import threading
from pynput import keyboard

# 初始化 MediaPipe 姿势估计模型
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 初始化 OpenCV 视频捕获对象
cap = cv2.VideoCapture(0)
gamepad_1 = vg.VX360Gamepad()
gamepad_2 = vg.VX360Gamepad()

# 全局变量
is_active = True  # 标志是否正在录制
output_dir = None  # 当前输出文件夹
video_writer = None  # 视频写入对象
frame_index = 0  # 帧计数

player1_x_value = 0  # X 轴的初始位置（范围 -32768 到 32767）
player1_y_value = 0  # Y 轴的初始位置（范围 -32768 到 32767）
player2_x_value = 0  # X 轴的初始位置（范围 -32768 到 32767）
player2_y_value = 0  # Y 轴的初始位置（范围 -32768 到 32767）

player1_last_left_hand_position = (0, 0)
player1_last_right_hand_position = (0, 0)
player2_last_left_hand_position = (0, 0)
player2_last_right_hand_position = (0, 0)

button_mapping = {
    'throw': vg.XUSB_BUTTON.XUSB_GAMEPAD_A,  # H -> X 按键
    'cut': vg.XUSB_BUTTON.XUSB_GAMEPAD_X,  # J -> Y 按键
    'B': vg.XUSB_BUTTON.XUSB_GAMEPAD_B,  # K -> A 按键
    "Y": vg.XUSB_BUTTON.XUSB_GAMEPAD_Y,  # L -> B 按键
}

def on_press(key):
    if key == keyboard.Key.esc:
        press_button('B', 1)
        press_button('B', 2)

def on_release(key):
    pass

def press_button(button, player_num=1):
    global button_mapping

    def release_button():
        """释放按钮"""
        if player_num == 1:
            gamepad_1.release_button(button=button_mapping[button])  # 释放按键
            gamepad_1.update()  # 更新手柄状态
        elif player_num == 2:
            gamepad_2.release_button(button=button_mapping[button])  # 释放按键
            gamepad_2.update()  # 更新手柄状态
            
    if player_num == 1:
        gamepad_1.press_button(button=button_mapping[button])  # 按下按键
        gamepad_1.update()  # 更新手柄状态
        
    elif player_num == 2:
        gamepad_2.press_button(button=button_mapping[button])  # 按下按键
        gamepad_2.update()  # 更新手柄状态
    
    threading.Timer(0.2, release_button).start()  # 100ms 后释放按键
    
def update_axes(action, player_num=1):
    """更新手柄的 X 和 Y 轴值"""
    global player1_x_value, player1_y_value, player2_x_value, player2_y_value
    if player_num == 1:
        if action == 'up':
            player1_y_value = 32767 // 2
        elif action == 'down':
            player1_y_value = -32767 // 2
        elif action == 'left':
            player1_x_value = -32767 // 2
        elif action == 'right':
            player1_x_value = 32766 // 2
        elif action =='x_stop':
            player1_x_value = 0
        elif action == 'y_stop':
            player1_y_value = 0
        gamepad_1.left_joystick(player1_x_value, player1_y_value)
        gamepad_1.update()
        
    elif player_num == 2:
        if action == 'up':
            player2_y_value = 32767 // 2
        elif action == 'down':
            player2_y_value = -32767 // 2
        elif action == 'left':
            player2_x_value = -32767 // 2
        elif action == 'right':
            player2_x_value = 32766 // 2
        elif action =='x_stop':
            player2_x_value = 0
        elif action == 'y_stop':
            player2_y_value = 0
        gamepad_2.left_joystick(player2_x_value, player2_y_value)
        gamepad_2.update()

def simple_logic(landmarks, player_num):
    global player1_x_value, player1_y_value, player2_x_value, player2_y_value, frame_index, player1_last_left_hand_position, \
        player1_last_right_hand_position, player2_last_left_hand_position, player2_last_right_hand_position
        
    if is_active:
        # 提取关键点数据
        keypoints = []
        left_right_threshold = 0.11
        head_body_threshold = 0.08
        move_threshold = 0.55
        pack_threshold = 0.15

        for landmark in landmarks:
            keypoints.append({
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z,
                "visibility": landmark.visibility
            })
        head_indices = [1,2,3,4,5,6,7,8,9,10]
        left_hand_indices = [15, 17, 19, 21]
        right_hand_indices = [16, 18, 20, 22]
        hand_indeices = [15, 16, 17, 18, 19, 20, 21, 22]
        body_left_indices = [11, 23]
        body_right_indices = [12, 24]
        body_up_indices = [11, 12]
        body_indices = [11, 12, 23, 24]

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
            update_axes('left', player_num)
            print(f'player{player_num} move left')
        elif body_right_avg['x'] > hand_avg['x'] and body_avg['y'] >= hand_avg['y'] >= body_up_avg['y']:
            update_axes('right', player_num)
            print(f'player{player_num} move right')
        else:
            update_axes('x_stop', player_num)
            # print(f'player{player_num} stop')
            
        if body_right_avg['x'] <= hand_avg['x'] <= body_left_avg['x'] and hand_avg['y'] < body_up_avg['y']:
            update_axes('up', player_num)
            print(f'player{player_num} move up')
        elif body_right_avg['x'] <= hand_avg['x'] <= body_left_avg['x'] and hand_avg['y'] > body_avg['y']:
            update_axes('down', player_num)
            print(f'player{player_num} move down')
        else:
            update_axes('y_stop', player_num)
            # print(f'player{player_num} stop')
        
        right_hand_position = (right_hand_avg['x'], right_hand_avg['y'])
        left_hand_position = (left_hand_avg['x'], left_hand_avg['y'])
        
        if player_num == 1:
            if (sum(abs(a - b) for a, b in zip(player1_last_right_hand_position, right_hand_position)) + \
                sum(abs(a - b) for a, b in zip(player1_last_left_hand_position, left_hand_position))) > move_threshold:
                press_button('cut', player_num=player_num)
                print(f'player{player_num} cut')
            elif sum(abs(a - b) for a, b in zip(right_hand_position, left_hand_position)) < pack_threshold:
                press_button('throw', player_num=player_num)
                print(f'player{player_num} throw')
            player1_last_right_hand_position = right_hand_position
            player1_last_left_hand_position = left_hand_position
            
        elif player_num == 2:
            if (sum(abs(a - b) for a, b in zip(player2_last_right_hand_position, right_hand_position)) + \
                sum(abs(a - b) for a, b in zip(player2_last_left_hand_position, left_hand_position))) > move_threshold:
                press_button('cut', player_num=player_num)
                print(f'player{player_num} cut')
            elif sum(abs(a - b) for a, b in zip(right_hand_position, left_hand_position)) < pack_threshold:
                press_button('throw', player_num=player_num)
                print(f'player{player_num} throw')
            player2_last_right_hand_position = right_hand_position
            player2_last_left_hand_position = left_hand_position
            
keyboard.Listener(on_press=on_press, on_release=on_release, suppress=True)
    
while True:
    ret, frame = cap.read()
    if not ret:
        break

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
        break


# 释放资源
cap.release()
if video_writer:
    video_writer.release()
cv2.destroyAllWindows()
