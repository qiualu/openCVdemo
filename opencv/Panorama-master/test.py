
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import stitch
import utils
import features


def convertResult(img):
    '''Because of your images which were loaded by opencv,
    in order to display the correct output with matplotlib,
    you need to reduce the range of your floating point image from [0,255] to [0,1]
    and converting the image from BGR to RGB:'''
    img = np.array(img, dtype=float) / float(255)
    img = img[:, :, ::-1]
    return img

#load images
list_images=utils.loadImages('data/myhouse',resize=0)

#extract keypoints and descriptors using sift
k0,f0=features.findAndDescribeFeatures(list_images[0],opt='SIFT')
k1,f1=features.findAndDescribeFeatures(list_images[1],opt='SIFT')

#draw keypoints
img0_kp=features.drawKeypoints(list_images[0],k0)
img1_kp=features.drawKeypoints(list_images[1],k1)

plt_img = np.concatenate((img0_kp, img1_kp), axis=1)
plt.figure(figsize=(15,15))
plt.imshow(convertResult(plt_img))

#matching features using BruteForce
mat=features.matchFeatures(f0,f1,ratio=0.6,opt='BF')

#Computing Homography matrix and mask
H,matMask=features.generateHomography(list_images[0],list_images[1])

#draw matches
img=features.drawMatches(list_images[0],k0,list_images[1],k1,mat,matMask)
plt.figure(figsize=(15,15))
plt.imshow(convertResult(img))



