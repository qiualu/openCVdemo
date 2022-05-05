import cv2
import numpy as np
import timeit






class capshow:
    def __init__(self):
        print("1254")
        # self.capture1 = cv2.VideoCapture(0 + cv2.CAP_DSHOW)  # //打开电脑自带摄像头
        # self.capture2 = cv2.VideoCapture(1 + cv2.CAP_DSHOW)
        # self.capture1.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        # self.capture1.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        # self.capture2.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        # self.capture2.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        self.md_opt = "ORB"  # SURF  ORB  SIFT  匹配特征点
        self.ransacRep = 5.0  # ransac Rep是 RANSAC 算法允许的最大像素“摆动空间”
        self.H = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]) # 变换矩阵
        self.Ht = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]) # 变换矩阵

        self.height_src = 480   # 左边图像大小
        self.width_src = 640    #
        self.height_dst = 480   # 右边图像大小
        self.width_dst = 640    #

        self.height_pano = 480   # 合成图像大小
        self.width_pano = 640    #
        self.t = [0,0]  # 偏移距离

        # 遮罩
        self.mask1 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])  # 遮罩
        self.mask2 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])  # 遮罩

        # 最近邻近似匹配
        self.fm_opt = "FB"  # opt='FB', FlannBased opt='BF', BruteForce
        self.ratio = 0.75  # 这个比值就是罗氏比率检验

        # 拼接模式 左右
        self.side = "left"
        # 除去黑边
        self.pts = np.array([[0, 0], [0, 0],[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
        self.init_status = True

        # init
        self.cs_status = False
        self.cs_index = 0
    def print_class(self):
        print(" md_opt : ", self.md_opt)
        print(" ransacRep : ", self.ransacRep)
        print(" H : ", self.H)
        print("  ---------------------------  ")
        print(" height_src : ", self.height_src)
        print(" width_src : ", self.width_src)
        print(" height_dst : ", self.height_dst)
        print(" width_dst : ", self.width_dst)


    def initHe(self,src_imgIn = None,dst_imgIn = None, inStatus = 0):
        # 初始化融合带
        # ref1, src_img = self.capture1.read()
        # ref2, dst_img = self.capture2.read()
        print("显示图片")
        showstep = False
        # src_img = cv2.imread(r"C:\Users\yl\Desktop\Projece\python\openCVdemo\opencvStitcher\Panorama-master\data\building\building1.jpg")
        # dst_img = cv2.imread( r"C:\Users\yl\Desktop\Projece\python\openCVdemo\opencvStitcher\Panorama-master\data\building\building2.jpg")
        if inStatus == 0:
            src_img = cv2.imread(
                r"C:\Users\yl\Desktop\sucai\cd\TDMovieOut.0.jpg")
            dst_img = cv2.imread(
                r"C:\Users\yl\Desktop\sucai\cd\TDMovieOut.1.jpg")
        else:
            src_img = src_imgIn
            dst_img = dst_imgIn
        #

        # cv2.imshow("tu1", src_img)
        # cv2.imshow("tu2", dst_img)
        start = timeit.default_timer()
        self.get_H(src_img,dst_img)
        self.print_class()

        # stop = timeit.default_timer()
        # print("get_H time: ", stop - start)
        #
        # start = timeit.default_timer()
        # pano = self.imageStitcher(src_img,dst_img)
        # stop = timeit.default_timer()
        # print("imageStitcher time: ", stop - start)
        # cv2.imshow("tu2", pano)
        # try:
        #     while True:
        #         # print("1 show d")
        #         k = cv2.waitKey(30)
        #         # print(k)
        #         if k == 113:
        #             print("按下q 退出成功")
        #             break
        # except Exception as e:
        #     print(e)

    def matchFeatures(self,featuresA, featuresB):
        """matching features beetween 2 @features.
             If opt='FB', FlannBased algorithm is used.
             If opt='BF', BruteForce algorithm is used.
             @ratio is the Lowe's ratio test.
             @return matches"""
        if self.fm_opt == "BF":
            featureMatcher = cv2.DescriptorMatcher_create("BruteForce")
        if self.fm_opt == "FB":
            # featureMatcher = cv2.DescriptorMatcher_create("FlannBased")
            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            #  最近邻近似匹配
            featureMatcher = cv2.FlannBasedMatcher(index_params, search_params)

        # performs k-NN matching between the two feature vector sets using k=2
        # (indicating the top two matches for each feature vector are returned).
        # 使用k=2对两个特征向量集进行k- nn匹配
        # (表示返回每个特征向量的前两个匹配)。
        matches = featureMatcher.knnMatch(featuresA, featuresB, k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < self.ratio * n.distance:
                good.append(m)

        if len(good) > 4:
            status = True
            return good,status
        status = False
        return good,status
        # raise Exception("Not enought matches")

    def findAndDescribeFeatures(self,image):
        """find and describe features of @image,
            if opt='SURF', SURF algorithm is used.
            if opt='SIFT', SIFT algorithm is used.
            if opt='ORB', ORB algorithm is used.
            @Return keypoints and features of img"""
        # Getting gray image
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.md_opt == "SURF":
            md = cv2.xfeatures2d.SURF_create()
        if self.md_opt == "ORB":
            md = cv2.ORB_create(nfeatures=3000)
        if self.md_opt == "SIFT":
            md = cv2.xfeatures2d.SIFT_create()
        # Find interest points and Computing features.
        keypoints, features = md.detectAndCompute(grayImage, None)
        # Converting keypoints to numbers.
        # keypoints = np.float32(keypoints)
        features = np.float32(features)
        return keypoints, features

    def generateHomography(self,src_img, dst_img):
        """@Return Homography matrix, @param src_img is the image which is warped by homography,
            @param dst_img is the image which is choosing as pivot, @param ratio is the David Lowe’s ratio,
            @param ransacRep is the maximum pixel “wiggle room” allowed by the RANSAC algorithm
            """
        src_kp, src_features = self.findAndDescribeFeatures(src_img)
        dst_kp, dst_features = self.findAndDescribeFeatures(dst_img)

        good,status = self.matchFeatures(src_features, dst_features)
        if status:
            src_points = np.float32([src_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_points = np.float32([dst_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            # ＃返回值中H为变换矩阵.mask是掩模，在线的点
            H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, self.ransacRep)
            matchesMask = mask.ravel().tolist()
            return H, matchesMask, status
        else:
            return status, status, status


    # 获取拼接模版 遮罩
    def get_H(self,src_img, dst_img):
        side = self.side
        self.init_status = True
        H, _, status = self.generateHomography(src_img, dst_img)

        if status:
            print("变换矩阵 更新成功")
        else:
            self.init_status = False
            print("变换矩阵 保留原样")
            return False
        height_src, width_src = src_img.shape[:2]
        height_dst, width_dst = dst_img.shape[:2]

        pts1 = np.float32(
            [[0, 0], [0, height_src], [width_src, height_src], [width_src, 0]]
        ).reshape(-1, 1, 2)
        pts2 = np.float32(
            [[0, 0], [0, height_dst], [width_dst, height_dst], [width_dst, 0]]
        ).reshape(-1, 1, 2)

        try:
            # 将单应性应用于SRC img的角
            pts1_ = cv2.perspectiveTransform(pts1, H)
            pts = np.concatenate((pts1_, pts2), axis=0)

            # find max min of x,y coordinate
            [xmin, ymin] = np.int64(pts.min(axis=0).ravel() - 0.5)
            [xmax, ymax] = np.int64(pts.max(axis=0).ravel() + 0.5)
            t = [-xmin, -ymin]

            if pts[0][0][0] < 0:
                side = "left"
                width_pano = width_dst + t[0]
            else:
                width_pano = int(pts1_[3][0][0])
                side = "right"
            height_pano = ymax - ymin

            Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])

            dst_img_rz = np.zeros((height_pano, width_pano, 3))
            if side == "left":
                if height_pano >= height_src + t[1] and width_pano >= width_dst + t[0] :
                    print("left 合成的图片大小符合最小标准")
                else:
                    print("left 重合区范围出错")
                    return False
                #dst_img_rz[t[1]: height_src + t[1], t[0]: width_dst + t[0]] = dst_img
            else:
                if height_pano >= height_src + t[1] :
                    print("right 合成的图片大小符合最小标准")
                else:
                    print("right 重合区范围出错")
                    return False
                #dst_img_rz[t[1]: height_src + t[1], :width_dst] = dst_img

            print("width_pano",width_pano)
            print("height_pano", height_pano)
        except BaseException as e:
            # 转换失败
            self.init_status = False
            return False

        h, w = height_pano, width_pano
        smoothing_window = int(width_dst / 8)
        barrier = width_dst - int(smoothing_window / 2)
        if barrier < w:
            mask1 = self.blendingMask(
                h, w, barrier, smoothing_window=smoothing_window, left_biased=True
            )
            mask2 = self.blendingMask(
                h, w, barrier, smoothing_window=smoothing_window, left_biased=False
            )
        else:
            print("遮罩获取范围不足")
            return False

        self.side = side
        self.height_src, self.width_src = height_src, width_src
        self.height_dst, self.width_dst = height_dst, width_dst
        self.H = H
        self.Ht = Ht.dot(H)
        self.height_pano = height_pano
        self.width_pano = width_pano
        self.t = t
        self.pts = pts
        self.mask1 = mask1
        self.mask2 = mask2

        self.cs_status = True
        return True

    #  计算遮罩
    def blendingMask(self,height, width, barrier, smoothing_window, left_biased=True):
        assert barrier < width
        mask = np.zeros((height, width))

        offset = int(smoothing_window / 2)
        try:
            if left_biased:
                mask[:, barrier - offset: barrier + offset + 1] = np.tile(
                    np.linspace(1, 0, 2 * offset + 1).T, (height, 1)
                )
                mask[:, : barrier - offset] = 1
            else:
                mask[:, barrier - offset: barrier + offset + 1] = np.tile(
                    np.linspace(0, 1, 2 * offset + 1).T, (height, 1)
                )
                mask[:, barrier + offset:] = 1
        except BaseException:
            if left_biased:
                mask[:, barrier - offset: barrier + offset + 1] = np.tile(
                    np.linspace(1, 0, 2 * offset).T, (height, 1)
                )
                mask[:, : barrier - offset] = 1
            else:
                mask[:, barrier - offset: barrier + offset + 1] = np.tile(
                    np.linspace(0, 1, 2 * offset).T, (height, 1)
                )
                mask[:, barrier + offset:] = 1

        return cv2.merge([mask, mask, mask])

    def warpTwoImages(self,src_img, dst_img, showstep=False):

        # generate Homography matrix
        self.get_H(src_img, dst_img)
        H = self.H
        height_src, width_src = self.height_src, self.width_src
        height_dst, width_dst = self.height_dst, self.width_dst

        # get height and width of two images
        # height_src, width_src = src_img.shape[:2]
        # height_dst, width_dst = dst_img.shape[:2]

        # extract conners of two images: top-left, bottom-left, bottom-right, top-right
        pts1 = np.float32(
            [[0, 0], [0, height_src], [width_src, height_src], [width_src, 0]]
        ).reshape(-1, 1, 2)
        pts2 = np.float32(
            [[0, 0], [0, height_dst], [width_dst, height_dst], [width_dst, 0]]
        ).reshape(-1, 1, 2)

        try:
            # 将单应性应用于SRC img的角
            pts1_ = cv2.perspectiveTransform(pts1, H)
            pts = np.concatenate((pts1_, pts2), axis=0)

            # find max min of x,y coordinate
            [xmin, ymin] = np.int64(pts.min(axis=0).ravel() - 0.5)
            [_, ymax] = np.int64(pts.max(axis=0).ravel() + 0.5)
            t = [-xmin, -ymin]  # 假设识别有问题要不要给个 abs()

            # top left point of image which apply homography matrix, which has x coordinate < 0, has side=left
            # otherwise side=right
            # source image is merged to the left side or right side of destination image
            if pts[0][0][0] < 0:
                side = "left"
                width_pano = width_dst + t[0]
            else:
                width_pano = int(pts1_[3][0][0])
                side = "right"
            height_pano = ymax - ymin

            dst_img_rz_size_H = dst_img.shape[0]
            dst_img_rz_size_W = dst_img.shape[1]
            if height_pano < dst_img_rz_size_H:
                height_pano = dst_img_rz_size_H
            if width_pano < dst_img_rz_size_W:
                width_pano = dst_img_rz_size_W

            # Translation
            # https://stackoverflow.com/a/20355545
            Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])

            print(" Class Ht.dot(H) : ", type(Ht.dot(H)))
            print(" Class width_pano : ", type(width_pano))
            print(" Class height_pano : ", type(height_pano))

            src_img_warped = cv2.warpPerspective(
                src_img, Ht.dot(H), (width_pano, height_pano)
            )
            # generating size of dst_img_rz which has the same size as src_img_warped
            dst_img_rz = np.zeros((height_pano, width_pano, 3))
            if side == "left":
                dst_img_rz[t[1]: height_src + t[1], t[0]: width_dst + t[0]] = dst_img
            else:
                dst_img_rz[t[1]: height_src + t[1], :width_dst] = dst_img

            # blending panorama
            pano, nonblend, leftside, rightside = panoramaBlending(
                dst_img_rz, src_img_warped, width_dst, side, showstep=showstep
            )

            # croping black region
            pano = crop(pano, height_dst, pts)
            return pano, nonblend, leftside, rightside
        except BaseException as e:
            raise Exception("Please try again with another image set!", e)
    # 去除黑边
    def crop(self, panorama, h_dst, conners):
        """crop panorama based on destination.
        @param panorama is the panorama
        @param h_dst is the hight of destination image
        @param conner is the tuple which containing 4 conners of warped image and
        4 conners of destination image"""
        # find max min of x,y coordinate
        [xmin, ymin] = np.int32(conners.min(axis=0).ravel() - 0.5)
        t = [-xmin, -ymin]
        conners = conners.astype(int)

        # conners[0][0][0] is the X coordinate of top-left point of warped image
        # If it has value<0, warp image is merged to the left side of destination image
        # otherwise is merged to the right side of destination image
        if conners[0][0][0] < 0:
            n = abs(-conners[1][0][0] + conners[0][0][0])
            panorama = panorama[t[1]: h_dst + t[1], n:, :]
        else:
            if conners[2][0][0] < conners[3][0][0]:
                panorama = panorama[t[1]: h_dst + t[1], 0: conners[2][0][0], :]
            else:
                panorama = panorama[t[1]: h_dst + t[1], 0: conners[3][0][0], :]
        return panorama
    def imageStitcher(self,src_img, dst_img):

        # 进行初始化
        if self.cs_status == True:
            status = self.get_H(src_img, dst_img)
            if status == False:
                self.cs_index += 1
                if self.cs_index < 100:
                    self.imageStitcher(src_img, dst_img)
                else:
                    print("imageStitcher 次数过多")
                assert self.cs_index >= 100
        src_img_warped = cv2.warpPerspective(
            src_img, self.Ht, (self.width_pano, self.height_pano)
        )
        dst_img_rz = np.zeros((self.height_pano, self.width_pano, 3))
        if self.side == "left":
            dst_img_rz[self.t[1]: self.height_src + self.t[1], self.t[0]: self.width_dst + self.t[0]] = dst_img
        else:
            dst_img_rz[self.t[1]: self.height_src + self.t[1], :self.width_dst] = dst_img

        if self.side == "left":
            dst_img_rz = cv2.flip(dst_img_rz, 1)
            src_img_warped = cv2.flip(src_img_warped, 1)
            dst_img_rz = dst_img_rz * self.mask1
            src_img_warped = src_img_warped * self.mask2
            pano = src_img_warped + dst_img_rz
            pano = cv2.flip(pano, 1)
        else:
            dst_img_rz = dst_img_rz * self.mask1
            src_img_warped = src_img_warped * self.mask2
            pano = src_img_warped + dst_img_rz

        pano = self.crop(pano, self.height_dst, self.pts)
        pano = np.array(pano, dtype=float) / float(255)
        return pano

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
    # s.showCap()
    s.initHe()