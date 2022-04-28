
import cv2
import numpy as np

imgpath = r"D:\Project\PyProject\Ai\opencv\Panorama-master\data\building\building1.jpg"

img = cv2.imread(imgpath)


cv2.imshow("img1",img)

descriptor = cv2.xfeatures2d.SIFT_create()  # 在2004年，不列颠哥伦比亚大学的D.Lowe在他的论文中提出了一种新的算法，即尺度不变特征变换（SIFT）
(kps, features) = descriptor.detectAndCompute(img, None)
cv2.drawKeypoints(img, kps, img, (0, 255, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('img2', img)


while True:
    if cv2.waitKey(30) > 0: # 按任意键退出
        break











