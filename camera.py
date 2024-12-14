import cv2
import os
import numpy as np
import sys
import argparse
import json
import time  # 用于记录时间

sys.path.append(r'A:\Desktop\small_projects\Gesture_control\openpose-1.7.0\build\python\openpose\Release')
import pyopenpose as op

params = {
    "model_folder": "A:/Desktop/small_projects/Gesture_control/openpose-1.7.0/models/",
    "hand": True,
    # "face": True,
    "write_json": "./json_output/"
}

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

cap = cv2.VideoCapture(0)

# 初始化计时变量
frame_count = 0
start_time = time.time()  # 获取程序开始时间
fps = 0

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        break
    
    # 记录处理前的时间
    frame_count += 1
    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))
    
    # 计算时间差（每帧处理所用时间）
    if frame_count % 5 == 0:  # 每30帧更新一次
        end_time = time.time()
        fps = frame_count / (end_time - start_time)  # 计算FPS
        start_time = end_time  # 重置开始时间
        frame_count = 0  # 重置帧计数器

    # 显示帧率
    fps_text = f"FPS: {fps:.2f}"
    print(fps)
    cv2.putText(datum.cvOutputData, fps_text, (frame_count, frame_count), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # Display the result
    # print("Body keypoints: \n" + str(datum.poseKeypoints))

    # Check if JSON file exists and read keypoints from JSON
    json_path = os.path.join(params["write_json"], f"test.json")
    if os.path.exists(json_path):
        with open(json_path, 'r') as json_file:
            json_data = json.load(json_file)
            keypoints = json_data["people"][0]["pose_keypoints_2d"]
            print("Body keypoints from JSON: \n", keypoints)

    # Show the output with FPS overlay
    cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
