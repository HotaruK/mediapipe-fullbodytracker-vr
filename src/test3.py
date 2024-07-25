import mediapipe as mp
import cv2
import numpy as np
from pythonosc import udp_client
import time

# ユーザーの身長（メートル単位）を入力
user_height = 1.60  # 例：1.60メートル

# 手動補正用のパラメーター
MANUAL_OFFSET_X = 0.1  # X軸の補正値（メートル単位）
MANUAL_OFFSET_Y = -0.6  # Y軸の補正値（メートル単位）
MANUAL_OFFSET_Z = 1.8  # Z軸の補正値（メートル単位）

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
    'left_foot': '/tracking/trackers/3',
    'right_foot': '/tracking/trackers/4',
}

# カメラキャプチャの初期化
cap = cv2.VideoCapture(0)

# キャリブレーションデータ
calibration_offset = np.array([0.0, 0.0, 0.0])
calibration_scale = 1.0

def send_osc_message(address, position, rotation):
    # 手動補正を適用
    adjusted_position = (position[0] + MANUAL_OFFSET_X,
                         position[1] + MANUAL_OFFSET_Y,
                         position[2] + MANUAL_OFFSET_Z)
    client.send_message(f"{address}/position", adjusted_position)
    client.send_message(f"{address}/rotation", rotation)

def get_landmark_position(landmark, user_height):
    # Mediapipeの座標系をVRChatの座標系に変換し、スケーリングを適用
    pos = np.array([
        landmark.x * user_height,
        (1 - landmark.y) * user_height,  # Y軸を反転
        -landmark.z * user_height  # Z軸を反転（カメラ方向が正）
    ])
    # キャリブレーションを適用
    return (pos - calibration_offset) * calibration_scale

def calculate_rotation(landmark1, landmark2):
    # 簡易的な回転計算（より正確な計算が必要な場合は改良が必要）
    direction = np.array([landmark2.x - landmark1.x, landmark1.y - landmark2.y, landmark1.z - landmark2.z])
    yaw = np.arctan2(direction[2], direction[0]) * 180 / np.pi
    pitch = np.arctan2(direction[1], np.sqrt(direction[0]**2 + direction[2]**2)) * 180 / np.pi
    return (pitch, 0, yaw)  # Roll is set to 0 for simplicity

def calibrate(landmarks):
    global calibration_offset, calibration_scale
    # 腰の位置をオフセットとして使用
    hip_pos = np.array([
        (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x + landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x) / 2,
        1 - (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y + landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y) / 2,
        -(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z + landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z) / 2
    ]) * user_height
    calibration_offset = hip_pos

    # 身長に基づいてスケールを調整
    actual_height = (landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y + landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y) / 2 - \
                    (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y + landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y) / 2
    calibration_scale = (user_height * 0.5) / (actual_height * user_height)  # ヒップから足までの距離を体の半分と仮定

    print(f"Calibration completed. Offset: {calibration_offset}, Scale: {calibration_scale}")

def main():
    start_time = time.time()
    calibration_done = False

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

            # キャリブレーション
            current_time = time.time()
            if not calibration_done and current_time - start_time >= 10:
                calibrate(landmarks)
                calibration_done = True

            # 各トラッカーの位置と回転を計算してOSCメッセージを送信
            hip_position = get_landmark_position(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value], user_height)
            left_foot_position = get_landmark_position(landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value], user_height)
            right_foot_position = get_landmark_position(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value], user_height)

            # 回転データを計算
            hip_rotation = calculate_rotation(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value], landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value])
            left_foot_rotation = calculate_rotation(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value], landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value])
            right_foot_rotation = calculate_rotation(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value], landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value])

            send_osc_message(osc_addresses['hip'], hip_position, hip_rotation)
            send_osc_message(osc_addresses['left_foot'], left_foot_position, left_foot_rotation)
            send_osc_message(osc_addresses['right_foot'], right_foot_position, right_foot_rotation)

        cv2.imshow('Mediapipe Pose', frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()