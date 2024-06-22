import cv2
from time import sleep

# capture video from camera
cap = cv2.VideoCapture(0)

if __name__ == '__main__':
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # open window
        cv2.imshow('Step 1', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
