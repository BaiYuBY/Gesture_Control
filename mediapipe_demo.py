import cv2
import mediapipe as mp
import json
import os
import time

# 初始化 MediaPipe 姿势估计模型
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 初始化 OpenCV 视频捕获对象
cap = cv2.VideoCapture(0)

# 全局变量
is_recording = False  # 标志是否正在录制
output_dir = None  # 当前输出文件夹
video_writer = None  # 视频写入对象
frame_index = 0  # 帧计数

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

        test_keypoints = []
        for landmark in result.pose_landmarks.landmark:
            test_keypoints.append({
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z,
                "visibility": landmark.visibility
            })
        
        print(test_keypoints)

        # 如果正在录制，记录姿势数据
        if is_recording:
            # 提取关键点数据
            keypoints = []
            for landmark in result.pose_landmarks.landmark:
                keypoints.append({
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z,
                    "visibility": landmark.visibility
                })

            # 保存为 JSON 文件
            json_file_path = os.path.join(output_dir, f"frame_{frame_index:04d}.json")
            with open(json_file_path, 'w') as json_file:
                json.dump({"frame": frame_index, "pose_landmarks": keypoints}, json_file, indent=4)

    # 在副本上绘制文字（显示录制状态）
    cv2.putText(frame_with_text, "Recording: ON" if is_recording else "Recording: OFF",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if is_recording else (0, 0, 255), 2)

    # 显示带有标记的窗口
    cv2.imshow("MediaPipe Pose Demo", frame_with_text)

    # 如果正在录制，将帧写入视频文件（这里写入的是原始视频帧，不包含标记）
    if is_recording and video_writer:
        video_writer.write(frame)  # 录制原始画面，不包含标记

    # 按键处理
    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):  # 按 'r' 键开始新录制
        if not is_recording:
            # 创建新的输出文件夹
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_dir = f"./recordings/{timestamp}/"
            os.makedirs(output_dir, exist_ok=True)

            # 初始化视频写入对象
            video_file_path = os.path.join(output_dir, "video.avi")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(video_file_path, fourcc, 20.0, (frame.shape[1], frame.shape[0]))

            frame_index = 0  # 重置帧计数
            is_recording = True
            print(f"Recording started. Saving to {output_dir}")

    elif key == ord('s'):  # 按 's' 键停止录制
        if is_recording:
            is_recording = False
            video_writer.release()  # 关闭视频写入
            video_writer = None
            print(f"Recording stopped. Files saved to {output_dir}")

    elif key == ord('q'):  # 按 'q' 键退出程序
        break

    if is_recording:
        frame_index += 1  # 增加帧计数

# 释放资源
cap.release()
if video_writer:
    video_writer.release()
cv2.destroyAllWindows()
