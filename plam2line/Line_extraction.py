import cv2
import photohash
#读取掌纹图片
image = cv2.imread("/Users/yusi/Desktop/cv_palm/palm_test.jpg")
cv2.imshow("palm",image) #to view the palm in python
#转化为灰度图
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#提取边缘
edges = cv2.Canny(gray,65,180,apertureSize = 3)

#转换成白底黑纹
lines = cv2.bitwise_not(edges)

#保存图片
cv2.imwrite("../pic/palmlines1.jpg", lines)

#感知哈希算法计算相似度
# hash_one = photohash.average_hash('palmlines.jpg')
# hash_two = photohash.average_hash('/Users/yusi/Desktop/cv_palm/skeleton_extraction/skeleton1.jpg')
# similar = photohash.hash_distance(hash_one, hash_two)
# print(similar)

# cv2.imshow("test",lines)
# cv2.waitKey(0)

