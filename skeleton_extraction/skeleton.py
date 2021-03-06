# -*- coding: utf-8 -*-
import cv2


# 细化函数，输入需要细化的图片（经过二值化处理的图片）和映射矩阵array
# 这个函数将根据算法，运算出中心点的对应值
def Thin(image, array):
    h, w = image.shape
    iThin = image

    for i in range(h):
        for j in range(w):
            if image[i, j] == 0:
                a = [1] * 9
                for k in range(3):
                    for l in range(3):
                        # 如果3*3矩阵的点不在边界且这些值为零，也就是黑色的点
                        if -1 < (i - 1 + k) < h and -1 < (j - 1 + l) < w and iThin[i - 1 + k, j - 1 + l] == 0:
                            a[k * 3 + l] = 0
                sum = a[0] * 1 + a[1] * 2 + a[2] * 4 + a[3] * 8 + a[5] * 16 + a[6] * 32 + a[7] * 64 + a[8] * 128
                iThin[i, j] = array[sum] * 255
    return iThin


# 映射表
array = [0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,\
         1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1,\
         0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,\
         1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1,\
         1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
         1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1,\
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
         0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,\
         1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1,\
         0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,\
         1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,\
         1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
         1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,\
         1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0,\
         1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0]




if __name__ == '__main__':
    src = cv2.imread('/Users/yusi/Desktop/cv_palm/similarity/test4/未标题-1.jpg', 0)
    Gauss_img = cv2.GaussianBlur(src, (3,3), 0)
    cv2.imshow('image', Gauss_img)
    cv2.waitKey(0)

    # 自适应二值化函数，需要修改的是55那个位置的数字，越小越精细，细节越好，噪点更多，最大不超过图片大小
    adap_binary = cv2.adaptiveThreshold(Gauss_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 55, 2)
    iThin = Thin(adap_binary.copy(), array)

    cv2.imshow('adaptive', adap_binary)
    cv2.imshow('adaptive_iThin', iThin)
    cv2.waitKey(0)

    # 获取简单二值化的细化图，并显示
    ret, binary = cv2.threshold(Gauss_img, 130, 255, cv2.THRESH_BINARY)
    iThin_2 = Thin(binary.copy(), array)

    cv2.imshow('binary', binary)
    cv2.imshow('binary_iThin', iThin_2)
    cv2.imwrite("../pic/skeleton2.jpg", iThin_2)
    skeleton = cv2.imread("skeleton.jpg")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

