import argparse
import cv2
import mediapipe as mp
from pythonosc import udp_client

# capture video from camera
cap = cv2.VideoCapture(0)

# Mediapipe settings
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# settings for OSC
ip = "127.0.0.1"
port = 9000  # VRChat's default OS port
client = udp_client.SimpleUDPClient(ip, port)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--show-window', action='store_true')

    args = parser.parse_args()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            # Get the head center point and send VRC it
            head_center = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
            client.send_message('/tracking/trackers/head/position',
                                (head_center.z, head_center.x, head_center.y))
            if args.show_window:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # open window
        if args.show_window:
            cv2.imshow('Step 4', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
