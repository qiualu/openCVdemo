
from Stitcher import Stitcher
import cv2


imageA = cv2.imread(r"/opencv/data/building/building1.jpg", 0)
imageB = cv2.imread(r"/opencv/data/building/building2.jpg", 0)

#  把图片拼接成全景图
stitcher = Stitcher()
(result, vis) = stitcher.stitch([imageA,imageB],showMatches=True)

# 显示所有图
cv2.imshow("  ", imageA)
cv2.imshow("  ", imageB)
cv2.imshow("  ", result)
cv2.imshow("  ", vis)
cv2.waitKey(0)
cv2.destroyAllWindows()

