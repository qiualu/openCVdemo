
#!/usr/bin/python3

import _thread
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import stitch
import utils
import features

delay = 0.05

class showCap:

    def __init__(self):
        print(" 类多线程开启 ")
        self.count = 0


        self.capture1 = cv2.VideoCapture(0)
        self.capture2 = cv2.VideoCapture(1)

        self.thread1_frame1 = np.zeros(shape=(5,2))
        self.thread1_frame2 = np.zeros(shape=(5,2))

        self.thread2_frame1 = np.zeros(shape=(5, 2))
        self.thread2_frame2 = np.zeros(shape=(5, 2))

        self.thread3_frame1 = np.zeros(shape=(5, 2))
        self.thread3_frame2 = np.zeros(shape=(5, 2))

        self.thread1_pano =  np.zeros(shape=(5, 2))
        self.thread2_pano = np.zeros(shape=(5, 2))
        self.thread3_pano = np.zeros(shape=(5, 2))

        self.thread_pano = [0,0,0,0,0]

        self.play_index = 0

        self.play_info = [0, 0, 0, 0]
        self.play_zt = [1, 1, 1, 1]

        self.read_index = 0


        self.dexec = True

        self.func()

    # 为线程定义一个函数
    def print_time(self,threadName, delay):
        # global count

        print(threadName, delay,type(threadName),type(delay))

        while self.count < 500:
            time.sleep(delay)
            self.count += 1
            print("%s: %s %d" % (threadName, time.ctime(time.time()), self.count))

    def readCapStitcher(self,threadName, rindex):
        print(threadName, rindex, type(threadName), type(rindex))

        while self.dexec:
            # 等待 时间
            while True:
                time.sleep(delay)
                if self.read_index == rindex:
                    break

            self.play_info[rindex] = 0

            ref1, frame1 = self.capture1.read()
            ref2, frame2 = self.capture2.read()
            # 下一个线程读取
            if self.read_index == 3:
                self.read_index = 1
            else:
                self.read_index += 1

            # 暂停 标志
            while self.dexec:
                time.sleep(delay)
                if self.play_zt[rindex] == 0:
                    break

            if ref1 and ref2:
                print("threadName :  warpTwoImages ",end="")
                pano, non_blend, left_side, right_side = stitch.warpTwoImages(frame2, frame1, True)
                pano = np.array(pano, dtype=float) / float(255)
                if pano.shape[0] > 0 and pano.shape[1]:
                    print("done")
                    self.thread_pano[rindex] = pano
                    # if rindex == 1:
                    #     self.thread1_pano = pano
                    # elif rindex == 2:
                    #     self.thread2_pano = pano
                    # elif rindex == 3:
                    #     self.thread3_pano = pano
                    self.play_info[rindex] = 1
                else:
                    print("XXXXXXXXXXXXXXXXXXXXXXXXX")
            self.play_info[rindex] = 2

    def func(self):
        # 创建两个线程
        try:
            _thread.start_new_thread(self.readCapStitcher, ("Thread-1", 1,))
            _thread.start_new_thread(self.readCapStitcher, ("Thread-2", 2,))
            _thread.start_new_thread(self.readCapStitcher, ("Thread-2", 3,))
        except:
            print("Error: 无法启动线程")

        #  开始阅读 1
        self.read_index = 1

        self.play_zt = [0, 1, 0, 0]
        while True:
            if self.play_info[1] > 0:
                print(" ---------------Start ---------------- ",self.play_info[1])
            if self.play_info[1] == 1:
                print(" done -> ", self.play_info[1])
                break
            if self.play_info[1] == 2:
                self.read_index = 1
            time.sleep(0.5)
        self.play_zt = [1, 0, 1, 1]  # 暂停第一个线程

        while True:
            #time.sleep(5)
            # print("主线程 在线 count ", self.count)
            index = 1
            self.play_zt[index] = 0
            if self.play_info[index] > 0 and self.play_info[index] == 1:
                cv2.imshow("pano ", self.thread_pano[index])
                if cv2.waitKey(30) > 0:
                    break
            else:
                self.play_zt[index] = 1

            index = 2
            self.play_zt[index] = 0
            if self.play_info[index] > 0 and self.play_info[index] == 1:
                cv2.imshow("pano ", self.thread_pano[index])
                if cv2.waitKey(30) > 0:
                    break
            else:
                self.play_zt[index] = 1

            index = 3
            self.play_zt[index] = 0
            if self.play_info[index] > 0 and self.play_info[index] == 1:
                cv2.imshow("pano ", self.thread_pano[index])
                if cv2.waitKey(30) > 0:
                    break
            else:
                self.play_zt[index] = 1



if __name__ == '__main__':

    print("123")

    r = showCap()













