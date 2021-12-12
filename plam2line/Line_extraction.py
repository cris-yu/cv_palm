import cv2

image = cv2.imread("/Users/yusi/Desktop/cv_palm/palm_test.jpg")
cv2.imshow("palm",image) #to view the palm in python
#转化为灰度图
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#提取边缘
edges = cv2.Canny(gray,50,180,apertureSize = 3)

#转换成白底黑纹
lines = cv2.bitwise_not(edges)

#保存图片
# cv2.imwrite("palmlines.jpg", edges)
cv2.imshow("test",lines)
cv2.waitKey(0)