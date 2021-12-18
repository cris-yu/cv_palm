from skimage.measure import compare_ssim
import imutils
import cv2

image1 = "/Users/yusi/Desktop/cv_palm/skeleton_extraction/skeleton2cai.jpg"
image2 = "/Users/yusi/Desktop/cv_palm/skeleton_extraction/skeleton1.jpg"

image_a = cv2.imread(image1)
image_b = cv2.imread(image2)

gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)
#计算结构相似性
(score, diff) = compare_ssim(gray_a, gray_b, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))
#推敲
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