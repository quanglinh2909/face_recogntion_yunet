import time

import cv2

cap = cv2.VideoCapture("output_faces_video.mp4")
start = time.time()
while True:
    ret, frame = cap.read()
    if ret:
        cv2.imshow('frame', frame)

    sleep_time = 1 / 30 - (time.time() - start)
    if sleep_time > 0:
        time.sleep(sleep_time)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite('data/frame.jpg', frame)

    start = time.time()


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
