import mediapipe as mp
import cv2
import numpy as np
from pythonosc import udp_client

# ユーザーの身長（メートル単位）を入力
user_height = 1.60  # 例：1.60メートル

MANUAL_OFFSET_X = 0  # X軸の補正値（メートル単位）
MANUAL_OFFSET_Y = 0  # Y軸の補正値（メートル単位）
MANUAL_OFFSET_Z = 0  # Z軸の補正値（メートル単位）

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
    'head': '/tracking/trackers/head',
}

# カメラキャプチャの初期化
cap = cv2.VideoCapture(0)


def send_osc_message(address, position, rotation):
    # 手動補正を適用
    adjusted_position = (position[0] + MANUAL_OFFSET_X,
                         position[1] + MANUAL_OFFSET_Y,
                         position[2] + MANUAL_OFFSET_Z)
    client.send_message(f"{address}/position", adjusted_position)
    client.send_message(f"{address}/rotation", rotation)  # rotationはオイラー角（度数法）


def get_landmark_position(landmark, user_height):
    # Mediapipeの座標系をVRChatの座標系に変換し、スケーリングを適用
    return np.array([
        -landmark.x * user_height,  # X軸を反転
        (1 - landmark.y) * user_height,  # Y軸を反転（元の状態に戻す）
        landmark.z * user_height  # Z軸はそのまま
    ])


def calculate_rotation(landmark1, landmark2):
    direction = np.array([-(landmark2.x - landmark1.x), -(landmark2.y - landmark1.y), landmark2.z - landmark1.z])
    yaw = np.arctan2(direction[0], direction[2]) * 180 / np.pi
    pitch = np.arctan2(direction[1], np.sqrt(direction[0] ** 2 + direction[2] ** 2)) * 180 / np.pi
    roll = 0  # ロールは0と仮定

    return [pitch, yaw, roll]  # VRChatはZ, X, Yの順でオイラー角を適用


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
            left_foot_position = get_landmark_position(landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value],
                                                       user_height)
            right_foot_position = get_landmark_position(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value],
                                                        user_height)
            head_position = get_landmark_position(landmarks[mp_pose.PoseLandmark.NOSE.value], user_height)

            # 回転データを計算
            hip_rotation = calculate_rotation(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                              landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value])
            left_foot_rotation = calculate_rotation(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value],
                                                    landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value])
            right_foot_rotation = calculate_rotation(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value],
                                                     landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value])
            head_rotation = calculate_rotation(landmarks[mp_pose.PoseLandmark.NOSE.value],
                                               landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value])

            send_osc_message(osc_addresses['hip'], hip_position, hip_rotation)
            send_osc_message(osc_addresses['left_foot'], left_foot_position, left_foot_rotation)
            send_osc_message(osc_addresses['right_foot'], right_foot_position, right_foot_rotation)
            send_osc_message(osc_addresses['head'], head_position, head_rotation)

        cv2.imshow('Mediapipe Pose', frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# todo: いまのところこれが一番ましな動き