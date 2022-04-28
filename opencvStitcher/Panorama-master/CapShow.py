import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import stitch
import utils
import features


def warpTwoImages2(src_img, dst_img, showstep=False):
    # generate Homography matrix
    H, _ = features.generateHomography(src_img, dst_img)

    # get height and width of two images
    height_src, width_src = src_img.shape[:2]
    height_dst, width_dst = dst_img.shape[:2]

    # extract conners of two images: top-left, bottom-left, bottom-right, top-right
    pts1 = np.float32(
        [[0, 0], [0, height_src], [width_src, height_src], [width_src, 0]]
    ).reshape(-1, 1, 2)
    pts2 = np.float32(
        [[0, 0], [0, height_dst], [width_dst, height_dst], [width_dst, 0]]
    ).reshape(-1, 1, 2)

    try:
        # print(" ------1------- ")
        # aply homography to conners of src_img
        pts1_ = cv2.perspectiveTransform(pts1, H)
        pts = np.concatenate((pts1_, pts2), axis=0)

        # find max min of x,y coordinate
        [xmin, ymin] = np.int64(pts.min(axis=0).ravel() - 0.5)
        [_, ymax] = np.int64(pts.max(axis=0).ravel() + 0.5)
        t = [-xmin, -ymin]
        # print(" ------2------- ")
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

        # print(" ------3------- ")
        # Translation
        # https://stackoverflow.com/a/20355545
        Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
        src_img_warped = cv2.warpPerspective(
            src_img, Ht.dot(H), (width_pano, height_pano)
        )
        # print(" ------4------- ")

        # generating size of dst_img_rz which has the same size as src_img_warped
        dst_img_rz = np.zeros((height_pano, width_pano, 3))

        if side == "left":
            dst_img_rz[t[1]: height_src + t[1], t[0]: width_dst + t[0]] = dst_img
        else:
            dst_img_rz[t[1]: height_src + t[1], :width_dst] = dst_img

        # print(" ------7------- ")
        # blending panorama
        pano, nonblend, leftside, rightside = stitch.panoramaBlending(
            dst_img_rz, src_img_warped, width_dst, side, showstep=showstep
        )
        # print(" ------8------- ")
        # croping black region
        pano = stitch.crop(pano, height_dst, pts)
        return pano, nonblend, leftside, rightside
    except BaseException as e:
        raise Exception("Please try again with another image set!", e)

capture1 = cv2.VideoCapture(0)
capture2 = cv2.VideoCapture(1)
def convertResult(img):
    '''Because of your images which were loaded by opencv,
    in order to display the correct output with matplotlib,
    you need to reduce the range of your floating point image from [0,255] to [0,1]
    and converting the image from BGR to RGB:'''
    img = np.array(img, dtype=float) / float(255)
    img = img[:,:,::-1]
    return img
while True:
    try:
        ref1, frame1 = capture1.read()
        ref2, frame2 = capture2.read()
        print(type(ref1), ref1, type(frame1), type(ref2), ref2, type(frame2))
        if ref1 and ref2:
            cv2.imshow("tu1", frame1)
            cv2.imshow("tu2", frame2)
            # wrap 2 image
            # choose list_images[0] as desination
            pano, non_blend, left_side, right_side = warpTwoImages2(frame2, frame1, True)
            img = np.array(pano, dtype=float) / float(255)
            if pano.shape[0] > 0 and pano.shape[1]:
                print("pano : ",pano.shape)
                cv2.imshow("pano ", img)
        if cv2.waitKey(30) > 0:
            break
    except Exception as e:
        print("摄像头被触碰 e ",e)










