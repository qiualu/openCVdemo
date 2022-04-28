import cv2
import numpy as np


def findAndDescribeFeatures(image, opt="ORB"):
    """find and describe features of @image,
        if opt='SURF', SURF algorithm is used.
        if opt='SIFT', SIFT algorithm is used.
        if opt='ORB', ORB algorithm is used.
        @Return keypoints and features of img"""
    # Getting gray image
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if opt == "SURF":
        md = cv2.xfeatures2d.SURF_create()
    if opt == "ORB":
        md = cv2.ORB_create(nfeatures=3000)
    if opt == "SIFT":
        md = cv2.xfeatures2d.SIFT_create()
    # Find interest points and Computing features.
    keypoints, features = md.detectAndCompute(grayImage, None)
    # Converting keypoints to numbers.
    # keypoints = np.float32(keypoints)
    features = np.float32(features)
    return keypoints, features


def matchFeatures(featuresA, featuresB, ratio=0.75, opt="FB"):
    """matching features beetween 2 @features.
         If opt='FB', FlannBased algorithm is used.
         If opt='BF', BruteForce algorithm is used.
         @ratio is the Lowe's ratio test.
         @return matches"""
    if opt == "BF":
        featureMatcher = cv2.DescriptorMatcher_create("BruteForce")
    if opt == "FB":
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
        if m.distance < ratio * n.distance:
            good.append(m)
    if len(good) > 4:
        return good
    raise Exception("Not enought matches")


def generateHomography(src_img, dst_img, ransacRep=5.0):
    """@Return Homography matrix, @param src_img is the image which is warped by homography,
        @param dst_img is the image which is choosing as pivot, @param ratio is the David Lowe’s ratio,
        @param ransacRep is the maximum pixel “wiggle room” allowed by the RANSAC algorithm
        """

    src_kp, src_features = findAndDescribeFeatures(src_img)
    dst_kp, dst_features = findAndDescribeFeatures(dst_img)

    good = matchFeatures(src_features, dst_features)

    src_points = np.float32([src_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_points = np.float32([dst_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, ransacRep)
    matchesMask = mask.ravel().tolist()
    return H, matchesMask


def drawKeypoints(img, kp):
    """
    cv2.drawKeypoints(image, keypoints, outImage, color=None, flags=None)
    第一个参数image：原始图像，可以使三通道或单通道图像；
    第二个参数keypoints：特征点向量，向量内每一个元素是一个KeyPoint对象，包含了特征点的各种属性信息；
    第三个参数outImage：特征点绘制的画布图像，可以是原图像；
    第四个参数color：绘制的特征点的颜色信息，默认绘制的是随机彩色；
    第五个参数flags：特征点的绘制模式，其实就是设置特征点的那些信息需要绘制，那些不需要绘制，有以下几种模式可选：
    特征点的绘制模式，其实就是设置特征点的那些信息需要绘制，那些不需要绘制，有以下几种模式可选：
    DRAW_MATCHES_FLAGS_DEFAULT：只绘制特征点的坐标点，显示在图像上就是一个个小圆点，每个小圆点的圆心坐标都是特征点的坐标。
    DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG：函数不创建输出的图像，而是直接在输出图像变量空间绘制，要求本身输出图像变量就是一个初始化好了的，size与type都是已经初始化好的变量。
    DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS ：单点的特征点不被绘制。
    DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS：绘制特征点的时候绘制的是一个个带有方向的圆，这种方法同时显示图像的坐标，size和方向，是最能显示特征的一种绘制方式。

    """
    img1 = img
    cv2.drawKeypoints(img, kp, img1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return img1


def drawMatches(src_img, src_kp, dst_img, dst_kp, matches, matchesMask):
    """/*
    /*其中参数如下：
    * img1 – 源图像1
    * keypoints1 –源图像1的特征点.
    * img2 – 源图像2.
    * keypoints2 – 源图像2的特征点
    * matches1to2 – 源图像1的特征点匹配源图像2的特征点[matches[i]] .
    * outImg – 输出图像具体由flags决定.
    * matchColor – 匹配的颜色（特征点和连线),若matchColor==Scalar::all(-1)，颜色随机.
    * singlePointColor – 单个点的颜色，即未配对的特征点，若matchColor==Scalar::all(-1)，颜色随机.
    matchesMask – Mask决定哪些点将被画出，若为空，则画出所有匹配点.
    * flags – Fdefined by DrawMatchesFlags.

    */
    """
    jiequ100 = len(matches)  # 数量太多保留 最多100个点线
    if len(matchesMask) > jiequ100:
        if jiequ100 > 100:
            jiequ100 = 100
    else:
        if len(matchesMask) > 100:
            jiequ100 = 100
        else:
            jiequ100 = len(matchesMask)

    draw_params = dict(
        matchColor=(0, 255, 0),  # draw matches in green color
        singlePointColor=None,
        matchesMask=matchesMask[:jiequ100],  # draw only inliers
        flags=2,
    )
    return cv2.drawMatches(
        src_img, src_kp, dst_img, dst_kp, matches[:jiequ100], None, **draw_params
    )

"""


    drawMatches(srcImg1, keyPoints1, srcImg2, keyPoints2, good_matches, result, Scalar(0, 255, 0), vector<char>(), 2);
    img1 – 源图像1
    keypoints1 – 源图像1的特征点.
    img2 – 源图像2.
    keypoints2 – 源图像2的特征点
    matches1to2 – 源图像1的特征点匹配源图像2的特征点[matches[i]] .
    outImg – 输出图像具体由flags决定.
    matchColor – 匹配的颜色（特征点和连线),若matchColor==Scalar::all(-1)，颜色随机.
    singlePointColor – 单个点的颜色，即未配对的特征点，若matchColor==Scalar::all(-1)，颜色随机.
    matchesMask – Mask决定哪些点将被画出，若为空，则画出所有匹配点.
    flags – Fdefined by DrawMatchesFlags.



// Draws matches of keypints from two images on output image.
void drawMatches( const Mat& img1, const vector<KeyPoint>& keypoints1,
                  const Mat& img2, const vector<KeyPoint>& keypoints2,
                  const vector<vector<DMatch> >& matches1to2, Mat& outImg,
                  const Scalar& matchColor=Scalar::all(-1), const Scalar& singlePointColor=Scalar::all(-1),
                  const vector<vector<char> >& matchesMask=vector<vector<char> >(), int flags=DrawMatchesFlags::DEFAULT );

drawMatches(srcImg1, keyPoints1, srcImg2, keyPoints2, good_matches, result, Scalar(0, 255, 0), vector<char>(), 2);
drawMatches(srcImg1, keyPoints1, srcImg2, keyPoints2, good_matches, result, Scalar(0, 255, 0), Scalar::all(-1));
drawMatches(srcImg1, keyPoints1, srcImg2, keyPoints2, good_matches, result);

"""

