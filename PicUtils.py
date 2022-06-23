import math

import cv2
import dlib
import face_recognition
import numpy as np
from PIL import Image

import Detector


def change_size(image, ratio):
    h = image.shape[0]
    w = image.shape[1]
    dh = int(ratio * h)
    dw = int(ratio * w)
    image = cv2.resize(image, (dw, dh))
    return image


# 计算灰度直方图
def get_gray_hist(image):
    h, w, c = image.shape
    gray_hist = np.zeros(256, np.uint32)
    for i in range(h):
        for j in range(w):
            for k in range(c):
                gray_hist[image[i][j][k]] += 1
    return gray_hist


# 直方图正规化
def regularize(image):
    i_max = np.max(image)
    i_min = np.min(image)
    o_min, o_max = 0, 255
    a = float(o_max - o_min) / (i_max - i_min)
    b = o_min - a * i_min
    image = a * image + b
    image = image.astype(np.uint8)
    return image


# 直方图均衡化
def equalize(image):
    h, w, c = image.shape
    gray_hist = get_gray_hist(image)
    cal_gray_hist = np.zeros(256, np.uint32)
    cal_gray_hist[0] = gray_hist[0]
    for i in range(1, 256):
        cal_gray_hist[i] = cal_gray_hist[i-1] + gray_hist[i]
    output = np.zeros(256, np.uint8)
    param = 256.0 / (h * w)
    for i in range(256):
        j = param * float(cal_gray_hist[i]) - 1
        if j > 0:
            output[i] = math.floor(j)
        else:
            output[i] = 0
    equal_hist = np.zeros(image.shape, np.uint8)
    for i in range(h):
        for j in range(w):
            for k in range(c):
                equal_hist[i][j][k] = output[image[i][j][k]]
    return equal_hist


def remove_background(img, col):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat')
    h, w = img.shape[:2]
    faces = detector(img, 0)
    mask = np.zeros(img.shape[:2], np.uint8)
    bg = np.zeros((1, 65), np.float64)
    fg = np.zeros((1, 65), np.float64)
    if len(faces) > 0:
        for k, d in enumerate(faces):
            left = max(int((3 * d.left() - d.right()) / 2), 1)
            top = max(int((3 * d.top() - d.bottom()) / 2) - 60, 1)
            right = min(int((3 * d.right() - d.left()) / 2), w)
            bottom = min(int((3 * d.bottom() - d.top()) / 2) + 60, h)
            rect = (left, top, right, bottom)
    else:
        exit(0)
    # 函数返回的mask中明显的背景像素为0，明显的前景像素为1，可能的背景像素为2，可能的前景像素为3
    cv2.grabCut(img, mask, rect, bg, fg, 10, cv2.GC_INIT_WITH_RECT)
    # mask为0或2(即为背景的)设置为0
    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    # 背景颜色(蓝/红/白)
    bg_color = [(225, 166, 23), (0, 0, 255), (255, 255, 255)]
    # 相乘则将背景的像素置为0，留下前景
    img = img * mask[:, :, np.newaxis]
    # 腐蚀操作
    img = cv2.erode(img, None, iterations=1)
    # 膨胀操作
    img = cv2.dilate(img, None, iterations=1)
    for i in range(h):  # 高
        for j in range(w):
            if max(img[i, j]) == 0:
                img[i, j] = bg_color[col]
    img = img[rect[1]:rect[3], rect[0]:rect[2]]
    img = cv2.resize(img, (164, 233))
    cv2.imwrite("images/tmp/certificate.png", img)


def paste_pic(img, col):
    remove_background(img, col)
    img = Image.open("images/tmp/certificate.png")
    background = Image.open("images/certificate.png")
    background.paste(img, (534, 196))
    background.save("result/certificate.png", quality=95)


def face_rec(img, face_characters, face_names):
    face_locations = face_recognition.face_locations(img)  # 获得所有人的人脸位置
    img_characters = face_recognition.face_encodings(img, face_locations)  # 获得所有人的人脸特征值
    pic_face_names = []  # 记录画面中的所有人名
    for img_character in img_characters:  # 和数据库人脸进行对比
        # 数据集中相似度超过0.5的则为true，否则为false,优先返回高匹配度
        tol = 0.05
        flag = 0
        while tol <= 0.5:
            res = face_recognition.compare_faces(face_characters, img_character, tolerance=tol)
            # 存在匹配的结果则记录下标
            if True in res:
                index = res.index(True)
                name = face_names[index]
                pic_face_names.append(name)
                flag = 1
                break
            tol += 0.02
        if flag == 0:
            pic_face_names.append("unknown")
    # 将捕捉到的人脸显示出来
    for (top, right, bottom, left), name in zip(face_locations, pic_face_names):
        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)  # 画人脸矩形框
        # 加上人名标签
        cv2.rectangle(img, (left, bottom - 30), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(img, name, (left + 5, bottom - 5), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
    # 保存图片并进行实时的显示
    h = img.shape[0]
    w = img.shape[1]
    ratio1 = 500 / h
    ratio2 = 800 / w
    img = change_size(img, min(ratio1, ratio2))
    cv2.imwrite("images/tmp/face_rec.jpg", img)
    if len(img_characters) == 0:
        return 0
    return 1


def mask_rec(img):
    img_nose, noses = Detector.nose_detection(img)  # 鼻子检测
    if noses == 1:  # 检测到鼻子说明未戴口罩
        cv2.putText(img_nose, "NO MASK", (15, 30), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 0, 255), 1)  # 图片上写字
        cv2.imwrite('images/tmp/mask_rec.jpg', img_nose)
    if noses == 0:  # 未检测到鼻子则进行眼睛检测
        img_eye, eyes = Detector.eye_detection(img)  # 进行眼睛检测，返回检测之后的图形以及标志位
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 将图片转化成HSV格式
        h, s, v = cv2.split(hsv)  #
        # h_min = cv2.getTrackbarPos("h_min", 'skin')  # 获取bar
        # h_max = cv2.getTrackbarPos("h_max", 'skin')
        # if h_min > h_max:
        #     h_max = h_min
        thresh = cv2.inRange(h, 0, 15)  # 提取人体肤色区域
        if len(eyes) > 1:  # 判断是否检测到两个眼睛，其中eyes[0]为左眼坐标
            # 确定口罩区域
            # 左眼的begin为口罩begin
            mask_x_begin = min(eyes[0][0], eyes[1][0])
            # 右眼begin+右眼宽度为口罩end
            mask_x_end = max(eyes[0][0], eyes[1][0]) + \
                         eyes[list([eyes[0][0], eyes[1][0]]).index(max(list([eyes[0][0], eyes[1][0]])))][2]
            # 越界处理
            if mask_x_end > img_eye.shape[0]:
                mask_x_end = img_eye.shape[0]
            # 眼睛高度为口罩begin
            mask_y_begin = max(eyes[0][1] + eyes[0][3], eyes[1][1] + eyes[1][3]) + 20
            # 越界处理
            if mask_y_begin > img_eye.shape[1]:
                mask_y_begin = img_eye.shape[1]
            mask_y_end = max(eyes[0][1] + 3 * eyes[0][3], eyes[1][1] + 3 * eyes[1][3]) + 20
            if mask_y_end > img_eye.shape[1]:
                mask_y_end = img_eye.shape[1]
            cv2.rectangle(img_eye, (mask_x_begin, mask_y_begin), (mask_x_end, mask_y_end), (255, 0, 0), 2)
            mask_scale = 0
            face_scale = 0
            # 遍历二值图，为0则total_mask_pixel+1，否则total_face_pixel+1
            for i in range(mask_x_begin, mask_x_end):
                for j in range(mask_y_begin, mask_y_end):
                    if thresh[i, j] == 0:
                        mask_scale += 1
                    else:
                        face_scale += 1
            if mask_scale > face_scale:
                cv2.putText(img_eye, "HAVE MASK", (mask_x_begin, mask_y_begin - 10),
                            cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 0, 255), 1)
            if mask_scale < face_scale:
                cv2.putText(img_eye, "NO MASK", (mask_x_begin, mask_y_begin - 10),
                            cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 0, 255), 1)
        cv2.imwrite("images/tmp/mask_rec.jpg", img_eye)
