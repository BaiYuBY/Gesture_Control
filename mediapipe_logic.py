import cv2
import mediapipe as mp
import json
import os
import time
import numpy as np
import vgamepad as vg


# 初始化 MediaPipe 姿势估计模型
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 初始化 OpenCV 视频捕获对象
cap = cv2.VideoCapture(0)
gamepad = vg.VX360Gamepad()

# 全局变量
is_active = True  # 标志是否正在录制
output_dir = None  # 当前输出文件夹
video_writer = None  # 视频写入对象
frame_index = 0  # 帧计数

x_value = 0  # X 轴的初始位置（范围 -32768 到 32767）
y_value = 0  # Y 轴的初始位置（范围 -32768 到 32767）
left_triggers = 0
right_triggers = 0

button_mapping = {
    'h': vg.XUSB_BUTTON.XUSB_GAMEPAD_X,  # H -> X 按键
    'j': vg.XUSB_BUTTON.XUSB_GAMEPAD_Y,  # J -> Y 按键
    'k': vg.XUSB_BUTTON.XUSB_GAMEPAD_A,  # K -> A 按键
    'l': vg.XUSB_BUTTON.XUSB_GAMEPAD_B   # L -> B 按键
}

def update_axes():
    """更新手柄的 X 和 Y 轴值"""
    gamepad.left_joystick(x_value, y_value)
    gamepad.update()

def update_triggers():
    """更新手柄的左右摇杆值"""
    gamepad.left_trigger(left_triggers)
    gamepad.right_trigger(right_triggers)

    gamepad.update()



while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 将图像从 BGR 转换为 RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 进行姿势估计
    result = pose.process(rgb_frame)

    # 仅绘制关键点和文字用于显示，不影响视频录制
    frame_with_text = frame.copy()  # 创建一个副本用于显示文字和关键点

    if result.pose_landmarks:
        # 在副本上绘制关键点
        mp.solutions.drawing_utils.draw_landmarks(frame_with_text, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        if is_active:
            # 提取关键点数据
            keypoints = []
            left_right_threshold = 0.11
            head_body_threshold = 0.08

            for landmark in result.pose_landmarks.landmark:
                keypoints.append({
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z,
                    "visibility": landmark.visibility
                })
            left_hand_indices = [15, 17, 19, 21]
            right_hand_indices = [16, 18, 20, 22]
            head_indices = [1,2,3,4,5,6,7,8,9,10]
            body_up_inices = [11, 12]
            body_down_inices = [23, 24]

            left_hand_avg = {key: np.mean([keypoints[i][key] for i in left_hand_indices]) for key in keypoints[0].keys()}
            right_hand_avg = {key: np.mean([keypoints[i][key] for i in right_hand_indices]) for key in keypoints[0].keys()}
            # head_avg = {key: np.mean([keypoints[i][key] for i in head_indices]) for key in keypoints[0].keys()}
            body_up_avg = {key: np.mean([keypoints[i][key] for i in body_up_inices]) for key in keypoints[0].keys()}
            body_down_avg = {key: np.mean([keypoints[i][key] for i in body_down_inices]) for key in keypoints[0].keys()}
            body_down_avg['z'] = body_down_avg['z'] - 0.11

            # if frame_index % 60 == 0:
            #     print(f'left_hand_avg: {left_hand_avg}')
            #     print(f'right_hand_avg: {right_hand_avg}')
            #     # print(f'head_avg: {head_avg}')
            #     print(f'body_up_avg: {body_up_avg}')
            #     print(f'body_down_avg: {body_down_avg}')

            if (lr_amplitude := abs(left_hand_avg['y'] - right_hand_avg['y'])) > left_right_threshold and \
                all(0 <= x <= 1 for x in [left_hand_avg['x'], right_hand_avg['x'], left_hand_avg['y'], right_hand_avg['y']]):
                if left_hand_avg['y'] < right_hand_avg['y']:
                    x_value = min(32767, int(32767 * (lr_amplitude-left_right_threshold+0.03) * 2.1))
                    print(f"Frame {frame_index}: Left; amplitude: {x_value}, {lr_amplitude}")
                    update_axes()
                else:
                    x_value = -min(32768, int(32768 * (lr_amplitude-left_right_threshold+0.03) * 2.1))
                    print(f"Frame {frame_index}: Right; amplitude: {x_value}, {lr_amplitude}")
                    update_axes()
            else:
                if x_value > 0:
                    x_value -= 1000
                    x_value = max(0, x_value)
                if x_value < 0:
                    x_value += 1000
                    x_value = min(0, x_value)
                update_axes()
            
            if (fb_amplitude := abs(body_up_avg['z'] - body_down_avg['z'])) > head_body_threshold and \
                all(0 <= x <= 1 for x in [body_up_avg['x'], body_up_avg['y'], body_down_avg['x'], body_down_avg['y']]):
                if body_up_avg['z'] < body_down_avg['z']:
                    right_triggers = min(32767, int(32767 * (fb_amplitude-head_body_threshold+0.03) * 6.5))
                    left_triggers = 0
                    print(f"Frame {frame_index}: Forward; amplitude: {right_triggers}, {fb_amplitude}")
                    update_triggers()
                else:
                    left_triggers = min(32767, int(32767 * (fb_amplitude-head_body_threshold+0.03) * 13))
                    right_triggers = 0
                    print(f"Frame {frame_index}: Backward; amplitude: {left_triggers}, {fb_amplitude}")
                    update_triggers()
            else:
                left_triggers -= 1000
                right_triggers -= 1000

                left_triggers = max(0, left_triggers)
                right_triggers = max(0, right_triggers)
                update_triggers()


    # 在副本上绘制文字（显示录制状态）
    cv2.putText(frame_with_text, "Recording: ON" if is_active else "Recording: OFF",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if is_active else (0, 0, 255), 2)

    # 显示带有标记的窗口
    cv2.imshow("MediaPipe Pose Demo", frame_with_text)

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

    if is_active:
        frame_index += 1  # 增加帧计数

# 释放资源
cap.release()
if video_writer:
    video_writer.release()
cv2.destroyAllWindows()
