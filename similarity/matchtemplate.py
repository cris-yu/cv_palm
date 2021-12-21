'''
import cv2 as cv
import numpy as np


def template_image():
    target = cv.imread("/Users/yusi/Desktop/cv_palm/skeleton_extraction/skeleton1.jpg")
    tpl = cv.imread("/Users/yusi/Desktop/cv_palm/skeleton_extraction/skeleton2.jpg")
    cv.imshow("modul", tpl)
    cv.imshow("yuan", target)
    methods = [cv.TM_SQDIFF_NORMED, cv.TM_CCORR_NORMED, cv.TM_CCOEFF_NORMED]
    th, tw = tpl.shape[:2]
    for md in methods:
        result = cv.matchTemplate(target, tpl, md)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
        if md == cv.TM_SQDIFF_NORMED:
            tl = min_loc
        else:
            tl = max_loc
        br = (tl[0] + tw, tl[1] + th)
        cv.rectangle(target, tl, br, [0, 0, 0])
        cv.imshow("pipei"+np.str(md), target)

template_image()
cv.waitKey(0)
cv.destroyAllWindows()
'''

'''
import cv2 as cv
import numpy as np


def template_demo():
    # tpl是模板图像
    tpl = cv.imread('/Users/yusi/Desktop/cv_palm/skeleton_extraction/skeleton1.jpg')
    # target是源图像
    target = cv.imread('/Users/yusi/Desktop/cv_palm/plam2line/palmlines.jpg')
    cv.imshow('template image', tpl)
    cv.imshow('target image', target)
    # 第一个是平方不同，第二个是相关性，第三个是相关性因子
    methods = [cv.TM_SQDIFF_NORMED, cv.TM_CCORR_NORMED, cv.TM_CCOEFF_NORMED]
    # 模板图像的宽、高
    th, tw = tpl.shape[:2]
    for md in methods:
        print(md)
        result = cv.matchTemplate(target, tpl, md)
        # 匹配程度最大的像素值，以及该像素的位置（不同的匹配方法，有不同的衡量标准）
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
        if md == cv.TM_SQDIFF_NORMED:
            tl = min_loc  # tl是矩形左上角坐标
        else:
            tl = max_loc
        br = (tl[0]+tw, tl[1]+th)  # 矩形框右下角坐标
        cv.rectangle(target, tl, br, (0, 0, 255), 2)  # 在target图像上绘制匹配的矩形框
        cv.imshow('match-'+np.str(md), target)
        # cv.imshow('match-'+np.str(md), result)


#src = cv.imread('C:/Users/Y/Pictures/Saved Pictures/demo.png')
# cv.namedWindow('input image', cv.WINDOW_AUTOSIZE)
# cv.imshow('input image', src)
template_demo()
cv.waitKey(0)
cv.destroyAllWindows()

'''
'''
#opencv模板匹配----单目标匹配
import cv2
#读取目标图片
target = cv2.imread("/Users/yusi/Desktop/cv_palm/plam2line/palmlines.jpg")
#读取模板图片
template = cv2.imread("/Users/yusi/Desktop/cv_palm/skeleton_extraction/skeleton1.jpg")
#获得模板图片的高宽尺寸
theight, twidth = template.shape[:2]
#执行模板匹配，采用的匹配方式cv2.TM_SQDIFF_NORMED
#平方差匹配method=CV_TM_SQDIFF
#标准平方差匹配method=CV_TM_SQDIFF_NORMED
#相关匹配method=CV_TM_CCORR
#标准相关匹配method=CV_TM_CCORR_NORMED
#相关匹配method=CV_TM_CCOEFF
#标准相关匹配method=CV_TM_CCOEFF_NORMED
result = cv2.matchTemplate(target,template,cv2.TM_CCOEFF_NORMED)
#归一化处理
cv2.normalize( result, result, 0, 1, cv2.NORM_MINMAX, -1 )
#寻找矩阵（一维数组当做向量，用Mat定义）中的最大值和最小值的匹配结果及其位置
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
#匹配值转换为字符串
#对于cv2.TM_SQDIFF及cv2.TM_SQDIFF_NORMED方法min_val越趋近与0匹配度越好，匹配位置取min_loc
#对于其他方法max_val越趋近于1匹配度越好，匹配位置取max_loc
strmin_val = str(min_val)
#绘制矩形边框，将匹配区域标注出来
#min_loc：矩形定点
#(min_loc[0]+twidth,min_loc[1]+theight)：矩形的宽高
#(0,0,225)：矩形的边框颜色；2：矩形边框宽度
cv2.rectangle(target,min_loc,(min_loc[0]+twidth,min_loc[1]+theight),(0,0,225),2)
#显示结果,并将匹配值显示在标题栏上
cv2.imshow("Mat="+strmin_val,target)
cv2.waitKey()
cv2.destroyAllWindows()

'''
'''
import numpy as np
import time
import cv2

def EM_EM2(temp):
    array = temp.reshape(1,-1)
    EM_sum = np.double(np.sum(array[0]))

    square_arr = np.square(array[0])
    EM2_sum = np.double(np.sum(square_arr))
    return EM_sum,EM2_sum


def EI_EI2(img, u, v,temp):
    height, width = temp.shape[:2]
    roi = img[v:v+height, u:u+width]
    array_roi = roi.reshape(1,-1)

    EI_sum = np.double(np.sum(array_roi[0]))

    square_arr = np.square(array_roi[0])
    EI2_sum = np.double(np.sum(square_arr))
    return EI_sum,EI2_sum


def EIM(img, u, v, temp):
    height, width = temp.shape[:2]
    roi = img[v:v+height, u:u+width]
    product = temp*roi*1.0
    product_array = product.reshape(1, -1)
    sum = np.double(np.sum(product_array[0]))
    return sum

def Match(img, temp):
    imgHt, imgWd = img.shape[:2]
    height, width = temp.shape[:2]

    uMax = imgWd-width
    vMax = imgHt-height
    temp_N = width*height
    match_len = (uMax+1)*(vMax+1)
    MatchRec = [0.0 for _ in range(0, match_len)]
    k = 0

    EM_sum, EM2_sum = EM_EM2(temp)
    for u in range(0, uMax+1):
        for v in range(0, vMax+1):
            EI_sum, EI2_sum = EI_EI2(img, u, v, temp)
            IM = EIM(img,u,v,temp)

            numerator=(  temp_N * IM - EI_sum*EM_sum)*(temp_N * IM - EI_sum * EM_sum)
            denominator=(temp_N * EI2_sum - EI_sum**2)*(temp_N * EM2_sum - EM_sum**2)

            ret = numerator/denominator
            MatchRec[k]=ret
            k+=1
        print('进度==》[{}]'.format(u/(vMax+1)))

    val = 0
    k = 0
    x = y = 0
    for p in range(0, uMax+1):
        for q in range(0, vMax+1):
            if MatchRec[k] > val:
                val = MatchRec[k]
                x = p
                y = q
            k+=1
    print ("val: %f"%val)
    return (x, y)

def main():
    img = cv2.imread('/Users/yusi/Desktop/cv_palm/plam2line/palmlines.jpg', cv2.IMREAD_GRAYSCALE)
    temp = cv2.imread('/Users/yusi/Desktop/cv_palm/similarity/test4/未标题-1.jpg', cv2.IMREAD_GRAYSCALE)

    tempHt, tempWd = temp.shape
    (x, y) = Match(img, temp)
    cv2.rectangle(img, (x, y), (x+tempWd, y+tempHt), (0,0,0), 2)
    cv2.imshow("temp", temp)
    cv2.imshow("result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print("Total Spend time：", str((end - start) / 60)[0:6] + "分钟")

# val: 1.000025
# Total Spend time： 0.0866分钟
'''

from skimage.measure import compare_ssim
import imutils
import cv2

image1 = "/Users/yusi/Desktop/cv_palm/pic2/palmlines1.jpg"
image2 = "/Users/yusi/Desktop/cv_palm/pic2/skeleton1.jpg"

image_a = cv2.imread(image1)
image_b = cv2.imread(image2)

gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)

(score, diff) = compare_ssim(gray_a, gray_b, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))

thresh = cv2.threshold(diff, 0, 255,
    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)

for contour in contours:
    (x, y, w, h) = cv2.boundingRect(contour)
    cv2.rectangle(image_a, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.rectangle(image_b, (x, y), (x + w, y + h), (0, 0, 255), 2)

cv2.imshow("Original", image_a)
cv2.imshow("Modified", image_b)
cv2.imshow("Diff", diff)
cv2.imshow("Thresh", thresh)
cv2.waitKey(0)

exit()