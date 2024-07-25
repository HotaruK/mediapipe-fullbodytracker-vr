import mediapipe as mp
import cv2
import numpy as np
from pythonosc import udp_client
from scipy.spatial.transform import Rotation as R
import time

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

# キャリブレーション用変数
calibration_data = {}
is_calibrating = False
calibration_start_time = None


def send_osc_message(address, position, rotation):
    client.send_message(f"{address}/position", position)
    client.send_message(f"{address}/rotation", rotation)


def get_landmark_position(landmark, user_height):
    # Mediapipeの座標系（右手系）をVRChatの座標系（左手系）に変換し、スケーリングを適用
    return (landmark.x * user_height, (1.0 - landmark.y) * user_height, -landmark.z * user_height)


def calculate_rotation(vec1, vec2):
    # 二つのベクトル間の回転を計算
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    cross_prod = np.cross(v1, v2)
    dot_prod = np.dot(v1, v2)
    angle = np.arccos(dot_prod / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    rotation_vector = cross_prod / np.linalg.norm(cross_prod) * angle
    return R.from_rotvec(rotation_vector).as_euler('xyz', degrees=True)


def calibrate_landmarks(landmarks):
    calibration_data.clear()
    for idx, landmark in enumerate(landmarks):
        position = get_landmark_position(landmark, user_height)
        calibration_data[idx] = position


def get_joint_rotation(joint, landmarks):
    # ジョイントの回転を計算
    if joint == 'left_foot':
        vec1 = np.array(
            [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x - landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y - landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
             landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z - landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z])
        vec2 = np.array(
            [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x - landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y - landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
             landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z - landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z])
    elif joint == 'right_foot':
        vec1 = np.array(
            [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x - landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
             landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y - landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
             landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z - landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z])
        vec2 = np.array(
            [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x - landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
             landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y - landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
             landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z - landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z])
    elif joint == 'left_elbow':
        vec1 = np.array(
            [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x - landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y - landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z - landmarks[
                 mp_pose.PoseLandmark.LEFT_ELBOW.value].z])
        vec2 = np.array(
            [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x - landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y - landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
             landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z - landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z])
    elif joint == 'right_elbow':
        vec1 = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x - landmarks[
            mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y - landmarks[
                             mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
                         landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z - landmarks[
                             mp_pose.PoseLandmark.RIGHT_ELBOW.value].z])
        vec2 = np.array(
            [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x - landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
             landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y - landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
             landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z - landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z])
    else:
        return (0, 0, 0)

    return calculate_rotation(vec1, vec2)


def main():
    global is_calibrating, calibration_start_time

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

            if is_calibrating:
                if time.time() - calibration_start_time > 10:
                    calibrate_landmarks(landmarks)
                    is_calibrating = False
                    print("Calibration complete.")

            # 各トラッカーの位置と回転を計算してOSCメッセージを送信
            for key, idx in mp_pose.PoseLandmark.__members__.items():
                if key in osc_addresses:
                    position = get_landmark_position(landmarks[idx.value], user_height)
                    if idx.value in calibration_data:
                        cal_position = calibration_data[idx.value]
                        relative_position = (
                            position[0] - cal_position[0],
                            position[1] - cal_position[1],
                            position[2] - cal_position[2]
                        )
                    else:
                        relative_position = position

                    # 回転の計算
                    rotation = get_joint_rotation(key.lower(), landmarks)

                    send_osc_message(osc_addresses[key], relative_position, rotation)

        cv2.imshow('Mediapipe Pose', frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

        # キャリブレーション開始
        if cv2.waitKey(5) & 0xFF == ord('c'):
            is_calibrating = True
            calibration_start_time = time.time()
            print("Calibration started. Please hold still for 10 seconds.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
