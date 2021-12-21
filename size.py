# 改变图像大小
import cv2

im1 = cv2.imread('/Users/yusi/Desktop/cv_palm/pic/skeleton2.jpg')
im2 = cv2.resize(im1, (500, 500), )  # 为图片重新指定尺寸
cv2.imwrite("/Users/yusi/Desktop/cv_palm/pic2/skeleton2.jpg", im2)
