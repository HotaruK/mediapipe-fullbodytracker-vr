import argparse
import cv2
import mediapipe as mp
from vrchatConnector import send_osc_message

# Mediapipe settings
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# capture video from camera
cap = cv2.VideoCapture(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VRChat Full Body Tracker using Mediapipe')
    parser.add_argument('--show-window', action='store_true', help='Show OpenCV window with skeleton')

    args = parser.parse_args()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            # draw skeleton on frame
            if args.show_window:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            landmarks = results.pose_landmarks.landmark
            positions = [(landmark.x, landmark.y, landmark.z) for landmark in landmarks]
            for idx, (x, y, z) in enumerate(positions):
                send_osc_message(f'/avatar/parameters/{idx}', [x, y, z])

        # open window
        if args.show_window:
            cv2.imshow('MediaPipe Pose', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
