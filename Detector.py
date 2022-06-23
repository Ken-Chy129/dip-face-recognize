import cv2


# 鼻子检测
def nose_detection(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)  # 高斯滤波
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将图片转化成灰度
    nose_cascade = cv2.CascadeClassifier("haarcascade_mcs_nose.xml")
    nose_cascade.load("data/haarcascades/haarcascade_mcs_nose.xml")  # 文件所在的具体位置
    '''此文件是opencv的haar鼻子特征分类器'''
    noses = nose_cascade.detectMultiScale(gray, 1.3, 5)  # 鼻子检测
    for (x, y, w, h) in noses:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 画框标识鼻子
    flag = 0  # 检测到鼻子的标志位，如果监测到鼻子，则判断未带口罩
    if len(noses) > 0:
        flag = 1
    return img, flag


# 眼睛检测
def eye_detection(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)  # 高斯滤波
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将图片转化成灰度
    eyes_cascade = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")
    eyes_cascade.load("data/haarcascades/haarcascade_eye_tree_eyeglasses.xml")  # 文件所在的具体位置
    '''此文件是opencv的haar眼睛特征分类器'''
    eyes = eyes_cascade.detectMultiScale(gray, 1.3, 5)  # 眼睛检测
    for (x, y, w, h) in eyes:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 画框标识眼部
        # print("x y w h is", (x, y, w, h))
        # frame = cv2.rectangle(img, (x, y+h), (x + 3*w, y + 3*h), (255, 0, 0), 2)  # 画框标识眼部
    return img, eyes


def back(x):
    pass


# 嘴巴检测
def get_mouth(faces, predictor, img_gray):
    for i, face in enumerate(faces):
        # 人脸高度
        height = face.bottom() - face.top()
        # 人脸宽度
        width = face.right() - face.left()
        shape = predictor(img_gray, face)
        # 48-67 为嘴唇部分
        x = []
        y = []
        for j in range(48, 68):
            x.append(shape.part(j).x)
            y.append(shape.part(j).y)
        # 根据嘴唇位置和人脸大小推断口罩位置
        y_max = int(max(y) + 2 * height / 5 + 10)
        y_min = int(min(y) - 2 * height / 5 - 10)
        x_max = int(max(x) + 2 * width / 5)
        x_min = int(min(x) - 2 * width / 5)
        size = ((x_max - x_min), (y_max - y_min))
        return x_min, x_max, y_min, y_max, size
