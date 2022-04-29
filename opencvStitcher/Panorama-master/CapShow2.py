import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import stitch
import utils
import features



class capshow:
    def __init__(self):
        print("1254")
        self.capture1 = cv2.VideoCapture(0 + cv2.CAP_DSHOW)  # //打开电脑自带摄像头
        self.capture2 = cv2.VideoCapture(1 + cv2.CAP_DSHOW)
        self.capture1.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.capture1.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.capture2.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.capture2.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    def initHe(self):
        # 初始化融合带
        ref1, src_img = self.capture1.read()
        ref2, dst_img = self.capture2.read()
        if ref1 and ref2:
            # src_img = frame1
            # dst_img = frame2
            showstep = False



        pass

    def imageStitcher(self):
        biao = True
        #  图片扭曲融合
        img = []
        # img = np.array(pano, dtype=float) / float(255)
        return img, biao

    def showCap(self):
        sc = 1
        while True:
            try:
                ref1, frame1 = self.capture1.read()
                ref2, frame2 = self.capture2.read()

                if ref1 and ref2:
                    cv2.imshow("tu1", frame1)
                    cv2.imshow("tu2", frame2)
                    # img, biao = self.imageStitcher()
                    # if biao:
                    #     cv2.imshow("pano ", img)
                k = cv2.waitKey(30)
                if sc == 1:
                    print(type(ref1), ref1, type(frame1), type(ref2), ref2, type(frame2))
                    print("k : ",k) # q 113  w 119 a  97   d 100  s 115
                if k == 113:
                    break
                if k == 97:
                    print("即将初始化")
                if k == 115:
                    if sc == 1:
                        sc = 0
                    else:
                        sc = 1
            except Exception as e:
                print("摄像头被触碰 e ",e)


if __name__ == '__main__':
    s = capshow()
    s.showCap()