# 上一篇文章 https://blog.csdn.net/UssterXingyiLi/article/details/111312661 筛选出了测试流出的图片。
# 接下来对这些流出图片和标注图片进行两两比较相似度，取出最高相似度的图片，有针对性地比较混淆样本(修改NG或者OK)，再训练，提高判定准确率。
# 相似度比较的类和方法借鉴于 https://blog.csdn.net/alicia_n/article/details/107045273 ，scikit-image v0.15.0测试通过，v0.14.1、v0.16.1、v0.18.0均报错

from numpy import average, linalg, dot
from skimage.measure import compare_ssim
from PIL import Image
import cv2
import glob
import os
import pandas as pd

class CompareImage():

    # 通过图片像素点来比较，对使用场景有要求
    def calculate_pixel(self, image):
        image_his = image.histogram()
        sum_pixel = 0
        for i in range(0, len(image_his)):
            sum_pixel += image_his[i]
        return sum_pixel

    def pixel_compare(self, file_image1, file_image2):
        image1 = Image.open(file_image1)
        image2 = Image.open(file_image2)
        img1 = image1.resize((256,256)).convert('L')
        img2 = image2.resize((256,256)).convert('L')
        image1_pixel_sum = self.calculate_pixel(img1)
        image2_pixel_sum = self.calculate_pixel(img2)
        score = 1-(abs(image1_pixel_sum - image2_pixel_sum)/max(image1_pixel_sum, image2_pixel_sum))
        print('像素相似度指数：{}'.format(score))
        return score

    # 通过余弦方法来比较
    # 把图片表示一个向量，通过计算向量之间的余弦值来表征图片的相似度
    def get_thum(self, image, size=(640,480), grayscale=False):
        image = Image.open(image).resize(size, Image.ANTIALIAS)
        if grayscale:
            image = image.convert('L')
        return image

    def vector_compare(self, image1, image2):
        image1 = self.get_thum(image1)
        image2 = self.get_thum(image2)
        images = [image1, image2]
        vectors = []
        norms = []
        for image in images:
            vector = []
            for pixel_tuple in image.getdata():
                vector.append(average(pixel_tuple))
            vectors.append(vector)
            norms.append(linalg.norm(vector, 2))
        a, b = vectors
        a_norm, b_norm = norms
        score = dot(a/a_norm, b/b_norm)
        print('余弦相似度指数：{}'.format(score))

    # 通过结构相似性指数
    # 基于sklearn中的scikit-image中的ssim来计算的一种全参考性的图像质量评价指标
    # 分别从图像的亮度、对比度、结构三个方面度量图像的相似性
    def ssmi_compare(self, path_image1, path_image2):
        imageA = cv2.imread(path_image1)
        imageB = cv2.imread(path_image2)

        grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

        (score, diff) = compare_ssim(grayA, grayB, full=True)
        print("结构相似性指数: {}".format(score))
        return score

compare_image = CompareImage()

# 两两对比
annos_image_path = glob.glob("/Users/yusi/Desktop/cv_palm/plam2line/palmlines.jpg")
outflow_image_path = glob.glob('/pic/palmlines1.jpg')
score_list = []
annos_images = []
outflow_images = []
similar_images = []
similarity = []
for i in outflow_image_path:
    outflow_image_name = os.path.split(i)[1]
    for j in annos_image_path:
        annos_image_name = os.path.split(j)[1]
        print(outflow_image_name, annos_image_name)
        # compare_image.pixel_compare(i, j)
        # compare_image.vector_compare(i, j)
        score = compare_image.ssmi_compare(i, j)
        annos_images.append(annos_image_name)
        score_list.append(score)
        print('-------------------------------------')
    similar_index = score_list.index(max(score_list))
    similar_image = annos_images[similar_index]
    outflow_images.append(outflow_image_name)
    similar_images.append(similar_image)
    similarity.append(max(score_list))
similar_df = pd.DataFrame({'流出图片': outflow_images, '相似图片': similar_images, '相似度': similarity})
similar_df.to_csv('similar_image.csv')

