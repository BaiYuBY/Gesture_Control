import cv2
import mediapipe as mp
import json
import os
import time
import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class PoseGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim1, output_dim2):
        super(PoseGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc1 = torch.nn.Linear(hidden_dim, output_dim1)  # 输出 label1
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim2)  # 输出 label2

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = x.mean(dim=0)  # 全局池化

        logits1 = self.fc1(x)  # 输出 logits
        logits2 = self.fc2(x)

        # 使用 softmax 激活函数将 logits 转为概率
        probs1 = F.softmax(logits1, dim=-1)
        probs2 = F.softmax(logits2, dim=-1)

        return probs1, probs2

# 初始化 MediaPipe 姿势估计模型
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

model = torch.load("./pose_gnn_model_1000.pth").to(device)  # 加载模型
model.eval()  # 设置模型为推理模式
print("完整模型已加载")

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

        # 提取节点特征
        keypoints = []
        for landmark in result.pose_landmarks.landmark:
            keypoints.append([landmark.x, landmark.y, landmark.z])

        # 转换为张量
        node_features = torch.tensor(keypoints, dtype=torch.float32).to(device)

        # 定义骨骼连接关系（固定）
        edge_index = torch.tensor([
            [0, 1], [0, 4], [1, 2], [2, 3], [3, 7], [4, 5], [5, 6], [6, 8], [9, 10], [11, 12], 
            [11, 13], [11, 23], [12, 14], [12, 24], [13, 15], [14, 16], [15, 21], [15, 17], [15, 19],
            [16, 18], [16, 20], [16, 22], [17, 19], [18, 20], [23, 24], [23, 25], [24, 26], [25, 27],
            [26, 28], [27, 29], [27, 31], [28, 30], [28, 32], [29, 31], [30, 32]
            # 添加其他骨骼连接关系...
        ]).t().to(device)  # 转置为 [2, num_edges]

        # 推理
        with torch.no_grad():
            output1, output2 = model(node_features, edge_index)
            pred_label1 = torch.argmax(output1).item()
            pred_label2 = torch.argmax(output2).item()

        # 显示推理结果
        cv2.putText(frame_with_text, f"Label1: {pred_label1}, Label2: {pred_label2}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        print(f"Frame {frame_index}: Label1={pred_label1}, Label2={pred_label2}")
    
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
