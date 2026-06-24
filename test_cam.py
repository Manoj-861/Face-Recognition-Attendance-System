import cv2
import time
import threading

def test_cam():
    print("Testing cv2.VideoCapture(0, cv2.CAP_DSHOW) in thread...")
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        print("cap.isOpened():", cap.isOpened())
        if cap.isOpened():
            ret, frame = cap.read()
            print("ret:", ret)
        cap.release()
    except Exception as e:
        print("Error:", e)
    print("Done DSHOW in thread")

t = threading.Thread(target=test_cam)
t.start()
t.join()
