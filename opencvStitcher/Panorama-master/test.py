import cv2


capture1 = cv2.VideoCapture(0+cv2.CAP_DSHOW)   # //打开电脑自带摄像头

capture1.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
capture1.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while True:
    ref1, frame1 = capture1.read()
    print(frame1.shape)
    if ref1:
        cv2.imshow("tu1", frame1)
    if cv2.waitKey(30) > 0:
        break


