import cv2
#按q拍照并保存图片
def make_photo():
    """使用opencv拍照"""
cap = cv2.VideoCapture(0)  # 默认的摄像头
i=0
while(1):
    ret, frame = cap.read()
    k=cv2.waitKey(1)
    if k==5:
        break
    elif k==ord('q'):
        cv2.imwrite('/Users/yusi/Desktop/cv_palm/photo/'+str(i)+'.jpg', frame)
        i=i+1
    cv2.imshow("capture", frame)  # 弹窗口
cap.release()
cv2.destroyAllWindows()


'''
def make_photo():
"""使用opencv拍照"""
cap = cv2.VideoCapture(0)  # 默认的摄像头
while True:
    ret, frame = cap.read()
    if ret:
        cv2.imshow("capture", frame)  # 弹窗口
        # 等待按键q操作关闭摄像头
        if cv2.waitKey(1) & 0xFF == ord('q'):
            file_name = "photo.jpeg"
            cv2.imwrite(file_name, frame)
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

def make_video():
"""使用opencv录像"""
cap = cv2.VideoCapture(0)  # 默认的摄像头
# 指定视频代码
fourcc = cv2.VideoWriter_fourcc(*"DIVX")
out = cv2.VideoWriter('xieyang.avi', fourcc, 20.0, (640,480))
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        out.write(frame)
        #
        cv2.imshow('frame',frame)
        # 等待按键q操作关闭摄像头
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
out.release()
cv2.destroyAllWindows()
'''
if __name__ == '__main__':
    make_photo()