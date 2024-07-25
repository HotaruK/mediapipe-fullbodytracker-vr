import mediapipe as mp
import cv2
import numpy as np
from pythonosc import udp_client

# ユーザーの身長（メートル単位）を入力
user_height = 1.60  # 例：1.75メートル

# MediapipeのPoseモジュールを初期化
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# OSCクライアントの設定
ip = "127.0.0.1"
port = 9000  # VRChatのOSCポート番号
client = udp_client.SimpleUDPClient(ip, port)

# VRChatのOSCアドレス
osc_addresses = {
    'hip': '/tracking/trackers/1',
    'chest': '/tracking/trackers/2',
    'left_foot': '/tracking/trackers/3',
    'right_foot': '/tracking/trackers/4',
    'left_knee': '/tracking/trackers/5',
    'right_knee': '/tracking/trackers/6',
    'left_elbow': '/tracking/trackers/7',
    'right_elbow': '/tracking/trackers/8',
    'head': '/tracking/trackers/head'
}

# カメラキャプチャの初期化
cap = cv2.VideoCapture(0)

def send_osc_message(address, position, rotation):
    client.send_message(f"{address}/position", position)
    # client.send_message(f"{address}/rotation", rotation)

def get_landmark_position(landmark, user_height):
    # Mediapipeの座標系をVRChatの座標系に変換し、スケーリングを適用
    return (landmark.x * user_height, (1.0 - landmark.y) * user_height, -landmark.z * user_height)

def main():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # MediapipeでPose推定
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # 各トラッカーの位置と回転を計算してOSCメッセージを送信
            hip_position = get_landmark_position(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value], user_height)
            left_foot_position = get_landmark_position(landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value], user_height)
            right_foot_position = get_landmark_position(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value], user_height)
            left_knee_position = get_landmark_position(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value], user_height)
            right_knee_position = get_landmark_position(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value], user_height)

            # 回転データを計算する場合、適宜算出する
            # 今回は回転データをゼロに設定
            zero_rotation = (0, 0, 0)

            send_osc_message(osc_addresses['hip'], hip_position, zero_rotation)
            send_osc_message(osc_addresses['left_foot'], left_foot_position, zero_rotation)
            send_osc_message(osc_addresses['right_foot'], right_foot_position, zero_rotation)
            send_osc_message(osc_addresses['left_knee'], left_knee_position, zero_rotation)
            send_osc_message(osc_addresses['right_knee'], right_knee_position, zero_rotation)

        cv2.imshow('Mediapipe Pose', frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
