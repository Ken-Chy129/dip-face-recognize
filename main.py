import webbrowser

import dlib
import numpy as np
import qtawesome
from PIL import Image
from PyQt5 import QtWidgets, QtCore
import threading
import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import face_recognition
import cv2
import os

from PyQt5.QtWidgets import QFileDialog, QDialog, QTabWidget, QMainWindow, QWidget, QMessageBox, \
    QInputDialog, QPushButton, QGridLayout, QSlider, QVBoxLayout

import Detector
import Filter
import NoiseGenerator
import PicUtils


# 自定义Label继承自QLabel
class MyQLabel(QtWidgets.QLabel):
    # 自定义信号
    button_clicked_signal = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super(MyQLabel, self).__init__(parent)

    def mouseReleaseEvent(self, e):
        self.button_clicked_signal.emit()

    # 重写父类的单击事件，以实现绑定外部函数
    def connect_customized_slot(self, func):
        self.button_clicked_signal.connect(func)


# 窗口类
class MainWindow(QTabWidget, QMainWindow, QDialog):

    def __init__(self):
        super().__init__()
        self.setFixedSize(1100, 700)
        self.setWindowTitle("线上证件制作")
        self.setWindowIcon(QIcon("images/favicon/favicon.ico"))
        self.video_capture = cv2.VideoCapture(0)
        self.stopEvent = threading.Event()  # 线程事件
        self.stopEvent.clear()
        self.img_path1 = ""  # 口罩图路径
        self.img_path2 = ""  # 润色图路径
        self.fun_num = 0  # 当前使用了进入了第几个按钮界面
        self.exist = False  # 是否已经生成了证件照
        self.face_names = []  # 人名
        self.face_characters = []  # 人脸特征
        self.load_faces()  # 加载人脸数据
        self.init_ui()  # 初始化界面

    # 初始化界面
    def init_ui(self):
        """
          1.人脸图像上传界面
        """
        face_upload_widget = QWidget()  # 创建人脸图像上传界面总部件
        face_upload_layout = QVBoxLayout()  # 创建布局方式
        face_upload_layout.setAlignment(Qt.AlignCenter)  # 部件居中

        self.face_upload_label = MyQLabel()
        self.face_upload_label.setText("数据收集")
        self.face_upload_label.setFixedSize(1050, 640)  # 设定部件大小
        self.face_upload_label.setStyleSheet("border: 2px solid black; font-size: 40px; font-weight: bolder;")
        self.face_upload_label.setAlignment(Qt.AlignCenter)  # 内容居中
        self.face_upload_label.connect_customized_slot(self.upload_face_pic)  # 绑定鼠标单击事件

        face_upload_layout.addWidget(self.face_upload_label)  # 添加部件
        face_upload_widget.setLayout(face_upload_layout)  # 设置该界面布局方式

        '''
          2.口罩检测界面
        '''
        mask_rec_widget = QWidget()
        mask_rec_layout = QGridLayout()

        # 设置显示的界面
        self.mask_rec_label = MyQLabel()
        self.mask_rec_label.setAlignment(Qt.AlignCenter)
        self.mask_rec_label.setText("口罩检测")
        self.mask_rec_label.setFixedSize(1050, 560)
        self.mask_rec_label.setStyleSheet("border: 2px solid black; font-size: 40px; font-weight: bolder;"
                                          "margin: 10px 30px 10px 50px")

        self.mask_rec_op_btn = QPushButton("开始检测")
        self.mask_rec_op_btn.setFixedSize(480, 70)
        self.mask_rec_op_btn.setStyleSheet("QPushButton{background-color:#d1d1d1; font-size: 20px; border:2px}"
                                           "QPushButton{border-radius:5px; margin-left: 120px}"
                                           "QPushButton:hover{background-color: #c2c2c2}")
        self.mask_rec_cl_btn = QtWidgets.QPushButton("结束检测")
        self.mask_rec_cl_btn.setFixedSize(480, 70)
        self.mask_rec_cl_btn.setEnabled(False)
        self.mask_rec_cl_btn.setStyleSheet("QPushButton{background-color:#d1d1d1; font-size: 20px; border:2px}"
                                           "QPushButton{border-radius:5px; margin-right: 120px}"
                                           "QPushButton:hover{background-color: #c2c2c2}")

        self.mask_rec_op_btn.clicked.connect(self.mask_rec_by_camera)
        self.mask_rec_cl_btn.clicked.connect(self.face_rec_close)
        mask_rec_layout.addWidget(self.mask_rec_label, 0, 0)
        mask_rec_layout.addWidget(self.mask_rec_op_btn, 1, 0)
        mask_rec_layout.addWidget(self.mask_rec_cl_btn, 1, 1)
        mask_rec_widget.setLayout(mask_rec_layout)

        '''
          3. 人脸识别界面 
        '''
        face_rec_widget = QWidget()
        face_rec_layout = QGridLayout()

        self.face_rec_label = MyQLabel()
        self.face_rec_label.setText("人脸识别")
        self.face_rec_label.setFixedSize(800, 640)
        self.face_rec_label.setStyleSheet("border: 2px solid black; font-size: 40px; font-weight: bolder; margin: 30px")
        self.face_rec_label.setAlignment(Qt.AlignCenter)

        self.image_btn = QPushButton("图片识别")
        self.image_btn.setFixedSize(280, 160)
        self.camera_btn = QPushButton("摄像头识别")
        self.camera_btn.setFixedSize(280, 160)
        self.video_btn = QPushButton("视频识别")
        self.video_btn.setFixedSize(280, 160)
        self.close_btn = QPushButton("结束识别")
        self.close_btn.setFixedSize(280, 160)
        self.close_btn.setEnabled(False)

        self.image_btn.setStyleSheet("QPushButton{background-color:#d1d1d1; font-size: 20px; border:2px}"
                                     "QPushButton{border-radius:5px; margin:20px}"
                                     "QPushButton:hover{background-color: #c2c2c2}")
        self.camera_btn.setStyleSheet("QPushButton{background-color:#d1d1d1; font-size: 20px; border:2px}"
                                      "QPushButton{border-radius:5px; margin:20px}"
                                      "QPushButton:hover{background-color: #c2c2c2}")
        self.video_btn.setStyleSheet("QPushButton{background-color:#d1d1d1; font-size: 20px; border:2px}"
                                     "QPushButton{border-radius:5px; margin:20px}"
                                     "QPushButton:hover{background-color: #c2c2c2}")
        self.close_btn.setStyleSheet("QPushButton{background-color:#d1d1d1; font-size: 20px; border:2px}"
                                     "QPushButton{border-radius:5px; margin:20px}"
                                     "QPushButton:hover{background-color: #c2c2c2}")

        self.image_btn.clicked.connect(self.img_face_rec)
        self.video_btn.clicked.connect(lambda: self.face_rec(0))
        self.camera_btn.clicked.connect(lambda: self.face_rec(1))
        self.close_btn.clicked.connect(self.face_rec_close)
        face_rec_layout.addWidget(self.face_rec_label, 0, 0)
        face_rec_layout.addWidget(self.image_btn, 0, 1)
        face_rec_layout.addWidget(self.camera_btn, 1, 1)
        face_rec_layout.addWidget(self.video_btn, 2, 1)
        face_rec_layout.addWidget(self.close_btn, 3, 1)
        face_rec_widget.setLayout(face_rec_layout)

        '''
          4. 佩戴口罩界面
        '''
        mask_wear_widget = QWidget()
        mask_wear_layout = QGridLayout()

        self.mask_wear_widget1 = MyQLabel()
        self.mask_wear_widget1.setFixedSize(460, 560)
        self.mask_wear_widget1.setAlignment(Qt.AlignCenter)
        self.mask_wear_widget1.setStyleSheet(
            "border: 2px solid black; text-align: center; font-size: 30px; font-weight: bolder; margin-left: 60px")
        self.mask_wear_widget1.setText("原始图片")
        self.mask_wear_widget1.setScaledContents(True)
        self.mask_wear_widget1.connect_customized_slot(self.show_pic_no_mask)

        self.mask_wear_widget2 = MyQLabel()
        self.mask_wear_widget2.setAlignment(Qt.AlignCenter)
        self.mask_wear_widget2.setFixedSize(400, 560)
        self.mask_wear_widget2.setStyleSheet(
            "border: 2px solid black; text-align: center; font-size: 30px; font-weight: bolder")
        self.mask_wear_widget2.setText("佩戴效果")
        self.mask_wear_widget2.setScaledContents(True)

        mask1 = MyQLabel()
        mask1.setFixedSize(109, 140)
        mask1.setStyleSheet("border: 2px solid black; margin-right: 9px; margin-top: 20px; margin-bottom: 20px")
        mask1.setPixmap(QPixmap("images/mask1.png"))
        mask1.setScaledContents(True)
        mask1.connect_customized_slot(lambda: self.wear_mask(0))

        mask2 = MyQLabel()
        mask2.setFixedSize(100, 120)
        mask2.setStyleSheet("border: 2px solid black; margin-bottom: 20px")
        mask2.setPixmap(QPixmap("images/mask2.png"))
        mask2.setScaledContents(True)
        mask2.connect_customized_slot(lambda: self.wear_mask(1))

        mask3 = MyQLabel()
        mask3.setFixedSize(100, 120)
        mask3.setStyleSheet("border: 2px solid black; margin-bottom: 20px")
        mask3.setPixmap(QPixmap("images/mask3.png"))
        mask3.setScaledContents(True)
        mask3.connect_customized_slot(lambda: self.wear_mask(2))

        mask4 = MyQLabel()
        mask4.setFixedSize(100, 120)
        mask4.setStyleSheet("border: 2px solid black; margin-bottom: 20px")
        mask4.setPixmap(QPixmap("images/mask4.png"))
        mask4.setScaledContents(True)
        mask4.connect_customized_slot(lambda: self.wear_mask(3))

        mask_wear_layout.addWidget(self.mask_wear_widget1, 0, 1, 8, 8)
        mask_wear_layout.addWidget(self.mask_wear_widget2, 0, 10, 8, 8)
        mask_wear_layout.addWidget(mask1, 2, 9)
        mask_wear_layout.addWidget(mask2, 3, 9)
        mask_wear_layout.addWidget(mask3, 4, 9)
        mask_wear_layout.addWidget(mask4, 5, 9)
        mask_wear_widget.setLayout(mask_wear_layout)

        '''
          5. 图片美化
        '''
        polish_pic_widget = QWidget()
        polish_pic_layout = QGridLayout()
        polish_pic_widget.setLayout(polish_pic_layout)

        left_widget = QWidget()  # 创建左侧部件
        left_widget.setFixedSize(180, 640)
        left_widget.setObjectName('left_widget')
        left_layout = QGridLayout()  # 创建左侧部件为网格布局型
        left_widget.setLayout(left_layout)
        right_widget = QWidget()  # 创建右侧部件
        right_widget.setObjectName('right_widget')
        right_layout = QGridLayout()  # 创建右侧部件为网格布局型
        right_widget.setLayout(right_layout)

        polish_pic_layout.addWidget(left_widget, 0, 0, 12, 2)  # 左侧部件在第0行第0列
        polish_pic_layout.addWidget(right_widget, 0, 2, 12, 10)  # 右侧部件在第0行第2列

        self.left_button_1 = QPushButton(qtawesome.icon('fa.volume-up', color='white'), "高斯噪声")
        self.left_button_1.setObjectName('left_button')
        self.left_button_2 = QPushButton(qtawesome.icon('fa.volume-up', color='white'), "椒盐噪声")
        self.left_button_2.setObjectName('left_button')
        self.left_button_3 = QPushButton(qtawesome.icon('fa.volume-up', color='white'), "伽马噪声")
        self.left_button_3.setObjectName('left_button')
        self.left_button_4 = QPushButton(qtawesome.icon('fa.volume-up', color='white'), "均匀噪声")
        self.left_button_4.setObjectName('left_button')
        self.left_button_5 = QPushButton(qtawesome.icon('fa.volume-up', color='white'), "瑞利噪声")
        self.left_button_5.setObjectName('left_button')
        self.left_button_6 = QPushButton(qtawesome.icon('fa.volume-down', color='white'), "双边滤波")
        self.left_button_6.setObjectName('left_button')
        self.left_button_7 = QPushButton(qtawesome.icon('fa.volume-down', color='white'), "中值滤波")
        self.left_button_7.setObjectName('left_button')
        self.left_button_8 = QPushButton(qtawesome.icon('fa.volume-down', color='white'), "均值滤波")
        self.left_button_8.setObjectName('left_button')
        self.left_button_9 = QPushButton(qtawesome.icon('fa.volume-down', color='white'), "高通滤波")
        self.left_button_9.setObjectName('left_button')
        self.left_button_10 = QPushButton(qtawesome.icon('fa.volume-down', color='white'), "低通滤波")
        self.left_button_10.setObjectName('left_button')
        self.left_button_11 = QPushButton(qtawesome.icon('fa.wrench', color='white'), "直方图正规化")
        self.left_button_11.setObjectName('left_button')
        self.left_button_12 = QPushButton(qtawesome.icon('fa.wrench', color='white'), "直方图均衡化")
        self.left_button_12.setObjectName('left_button')

        left_layout.addWidget(self.left_button_1, 1, 0, 1, 3)
        left_layout.addWidget(self.left_button_2, 2, 0, 1, 3)
        left_layout.addWidget(self.left_button_3, 3, 0, 1, 3)
        left_layout.addWidget(self.left_button_4, 4, 0, 1, 3)
        left_layout.addWidget(self.left_button_5, 5, 0, 1, 3)
        left_layout.addWidget(self.left_button_6, 6, 0, 1, 3)
        left_layout.addWidget(self.left_button_7, 7, 0, 1, 3)
        left_layout.addWidget(self.left_button_8, 8, 0, 1, 3)
        left_layout.addWidget(self.left_button_9, 9, 0, 1, 3)
        left_layout.addWidget(self.left_button_10, 10, 0, 1, 3)
        left_layout.addWidget(self.left_button_11, 11, 0, 1, 3)
        left_layout.addWidget(self.left_button_12, 12, 0, 1, 3)

        self.pic_widget1 = MyQLabel()
        self.pic_widget1.setLayout(QGridLayout())
        self.pic_widget1.setFixedSize(390, 500)
        self.pic_widget1.setAlignment(Qt.AlignCenter)
        self.pic_widget1.setStyleSheet(
            "border: 2px solid darkGray; text-align: center; font-size: 30px; font-weight: bolder;"
            "margin-left: 40px; margin-top: 10px")
        self.pic_widget1.setText("原始图")
        self.pic_widget1.setScaledContents(True)

        self.pic_widget2 = MyQLabel()
        self.pic_widget2.setLayout(QGridLayout())
        self.pic_widget2.setFixedSize(390, 500)
        self.pic_widget2.setAlignment(Qt.AlignCenter)
        self.pic_widget2.setStyleSheet(
            "border: 2px solid darkGray; text-align: center; font-size: 30px; font-weight: bolder;"
            "margin-right: 40px; margin-top: 10px")
        self.pic_widget2.setText("效果图")
        self.pic_widget2.setScaledContents(True)

        self.text_widget = MyQLabel()
        self.text_widget.setLayout(QGridLayout())
        self.text_widget.setText("当前参数: ")
        self.text_widget.setStyleSheet("font-size: 20px; margin-left: 14px; font-weight: bolder")

        self.bar_widget = QSlider(Qt.Horizontal)
        self.bar_widget.setLayout(QGridLayout())
        self.bar_widget.setTickPosition(QSlider.TicksBelow)
        self.bar_widget.setMinimum(0)
        self.bar_widget.setMaximum(50)
        self.bar_widget.setSingleStep(2)
        self.bar_widget.setTickInterval(5)

        right_layout.addWidget(self.pic_widget1, 0, 2, 10, 7)
        right_layout.addWidget(self.pic_widget2, 0, 10, 10, 7)
        right_layout.addWidget(self.text_widget, 10, 2, 2, 3)
        right_layout.addWidget(self.bar_widget, 10, 5, 2, 11)

        polish_pic_layout.setSpacing(0)  # 设置内部控件间距

        left_widget.setStyleSheet("\
                QPushButton{border: none; color: white; font-size: 22px}\
                QPushButton:hover{border-left: 4px solid red; font-weight: 700;}\
                QWidget#left_widget{\
                    background: #c2c2c2;\
                    border-top: 1px solid darkGray;\
                    border-bottom: 1px solid darkGray;\
                    border-left: 1px solid darkGray;\
                    border-top-left-radius: 6px;\
                    border-bottom-left-radius: 6px;\
                }")

        right_widget.setStyleSheet("\
                    QWidget#right_widget{\
                        color: #232C51;\
                        background: white;\
                        border-top: 1px solid darkGray;\
                        border-bottom: 1px solid darkGray;\
                        border-right: 1px solid darkGray;\
                        border-top-right-radius: 6px;\
                        border-bottom-right-radius: 6px;\
                    }")

        self.bar_widget.sliderReleased.connect(self.hand_out)
        self.pic_widget1.connect_customized_slot(self.polish_pic)
        self.left_button_1.clicked.connect(self.gauss_noise)
        self.left_button_2.clicked.connect(self.salt_noise)
        self.left_button_3.clicked.connect(self.gamma_noise)
        self.left_button_4.clicked.connect(self.uniform_noise)
        self.left_button_5.clicked.connect(self.rayleigh_noise)
        self.left_button_6.clicked.connect(self.bilateral_filter)
        self.left_button_7.clicked.connect(self.median_filter)
        self.left_button_8.clicked.connect(self.average_filter)
        self.left_button_9.clicked.connect(self.high_pass_filter)
        self.left_button_10.clicked.connect(self.low_pass_filter)
        self.left_button_11.clicked.connect(self.regularization)
        self.left_button_12.clicked.connect(self.equalization)

        '''
          6. 生成证件照
        '''
        certificate_widget = QWidget()
        certificate_layout = QGridLayout()
        certificate_widget.setLayout(certificate_layout)

        self.certificate_label = MyQLabel()
        self.certificate_label.setFixedSize(792, 480)
        self.certificate_label.setScaledContents(True)
        self.certificate_label.setPixmap(QPixmap("images/certificate.png"))
        self.certificate_label.setStyleSheet("border: 4px solid black; padding: 5px; margin-right: 85px")

        white = QPushButton(self)
        white.setFixedSize(280, 90)
        blue = QPushButton(self)
        blue.setFixedSize(256, 90)
        red = QPushButton(self)
        red.setFixedSize(280, 90)
        white.setStyleSheet("background: white; border: 2px solid black; margin-left: 50px; margin-bottom: 25px")
        blue.setStyleSheet("background: #24bcec; border: 2px solid black; margin-left: 26px; margin-bottom: 25px")
        red.setStyleSheet("background: red; border: 2px solid black; margin-right: 50px; margin-bottom: 25px")

        certificate_layout.addWidget(self.certificate_label, 1, 3, 5, 8)
        certificate_layout.addWidget(white, 6, 1, 1, 3)
        certificate_layout.addWidget(blue, 6, 5, 1, 3)
        certificate_layout.addWidget(red, 6, 9, 1, 3)

        self.certificate_label.connect_customized_slot(self.save_certificate)
        white.clicked.connect(lambda: self.show_certificate(2))
        blue.clicked.connect(lambda: self.show_certificate(0))
        red.clicked.connect(lambda: self.show_certificate(1))

        '''
          7. 关于界面
        '''
        about_widget = QWidget()
        about_layout = QGridLayout()
        about_widget.setLayout(about_layout)

        about_title = MyQLabel("欢迎开始线上证件制作")
        about_title.setFont(QFont("宋体", 22))
        about_title.setAlignment(Qt.AlignCenter)
        about_title.setFixedHeight(65)
        about_title.setStyleSheet("font-weight: bolder; margin-top: 25px;")

        about_tip = MyQLabel("(您需要提前上传自己的照片进行数据收集后才能开始使用！)")
        about_tip.setFont(QFont("楷体", 16))
        about_tip.setStyleSheet("color: red")
        about_tip.setAlignment(Qt.AlignCenter)
        about_tip.setFixedHeight(40)

        about_text = MyQLabel("一、防疫检测:检查您是否佩戴了口罩\n"
                              "二、身份验证:摘下口罩验证身份\n"
                              "三、润色照片:上传照片进行美化、修改\n"
                              "四、佩戴口罩:为您的照片选择一款喜欢的口罩\n"
                              "五、生成证件照:选择颜色得到自己的证件照")
        about_text.setFont(QFont("宋体", 15))
        about_text.setAlignment(Qt.AlignCenter)
        about_text.setFixedHeight(400)
        about_text.setStyleSheet("line-height: 30px")

        about_author = MyQLabel("点击联系作者")
        about_author.setLayout(QGridLayout())
        about_author.setFont(QFont("宋体", 12))
        about_author.setAlignment(Qt.AlignRight)
        about_author.setFixedHeight(30)
        about_author.setStyleSheet("border: none; margin-right: 20px; font-weight: bolder")

        about_layout.addWidget(about_title)
        about_layout.addWidget(about_tip)
        about_layout.addWidget(about_text)
        about_layout.addWidget(about_author)
        about_layout.setSpacing(0)
        about_author.connect_customized_slot(lambda: webbrowser.open("https:www.ken-chy129.cn"))

        # 分别添加子页面
        self.addTab(about_widget, "使用说明")
        self.addTab(face_upload_widget, "数据收集")
        self.addTab(mask_rec_widget, "防疫检测")
        self.addTab(face_rec_widget, "身份认证")
        self.addTab(polish_pic_widget, "润色照片")
        self.addTab(mask_wear_widget, "佩戴口罩")
        self.addTab(certificate_widget, "生成证件照")
        self.setTabIcon(0, qtawesome.icon('fa.rss', color='gray'))
        self.setTabIcon(1, qtawesome.icon('fa.steam', color='gray'))
        self.setTabIcon(2, qtawesome.icon('fa.ambulance', color='gray'))
        self.setTabIcon(3, qtawesome.icon('fa.user-o', color='gray'))
        self.setTabIcon(4, qtawesome.icon('fa.paint-brush', color='gray'))
        self.setTabIcon(5, qtawesome.icon('fa.user-md', color='gray'))
        self.setTabIcon(6, qtawesome.icon('fa.id-card-o', color='gray'))

    # 将文件夹中图片的数据读取至内存数组中
    def load_faces(self):
        face_dir = "images/faces"
        # 获得人脸目录下的所有图片名称
        faces = os.listdir(face_dir)
        for face in faces:
            # 获得图片路径
            face_path = os.path.join(face_dir, face)
            # 获取文件名并至数组中（文件名即人脸名）
            self.face_names.append(face.split(".")[0])
            # 获取图片
            img = face_recognition.load_image_file(face_path)
            # 获取图片特征值并保存至数组中
            img_characters = face_recognition.face_encodings(img)[0]
            self.face_characters.append(img_characters)

    # 通用函数：选择图片
    def choose_pic(self):
        file_name = QFileDialog.getOpenFileName(self, "选择图片", "./", "image (*.jpg *.png *.jpeg)")
        # 获取文件路径
        image_path = file_name[0]
        if image_path == "":
            QMessageBox.information(self, "提示", "没有选择图片文件！")
            return 0
        return image_path

    '''
       人脸识别界面函数
    '''
    # 上传人脸照片
    def upload_face_pic(self):
        image_path = self.choose_pic()
        if image_path == 0:
            return
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
        h, w = img.shape[:2]
        ratio1, ratio2 = 500 / h, 800 / w
        img = PicUtils.change_size(img, min(ratio1, ratio2))
        cv2.imwrite("images/tmp/face.jpg", img)
        self.face_upload_label.setPixmap(QPixmap("images/tmp/face.jpg"))
        self.face_upload_label.setFixedSize(img.shape[1], img.shape[0])
        # 判断图像是否有且仅有一个人脸
        load_img = face_recognition.load_image_file("images/tmp/face.jpg")
        img_characters = face_recognition.face_encodings(load_img)
        face_num = len(img_characters)  # 获取人脸得数量
        if face_num == 0:  # 没有人脸
            QMessageBox.information(self, "提示", "当前图片没有发现人脸")
        elif face_num > 1:  # 多个人脸
            QMessageBox.information(self, "提示", "当前图片发现多个人脸")
        else:
            face_name, ok = QInputDialog.getText(self, "提示", "输入对应人名(英文输入):")
            if ok and face_name:
                # 保存人脸数据
                face_characters = img_characters[0]
                img = cv2.imread("images/tmp/face.jpg")
                cv2.imwrite("images/faces/" + face_name + ".jpg", img)
                self.face_names.append(face_name)
                self.face_characters.append(face_characters)
                QtWidgets.QMessageBox.information(self, "提示", "已成功上传")
            else:
                QtWidgets.QMessageBox.information(self, "提示", "已取消上传")
                return

    # 人脸识别, num=0为本地视频，1为摄像头拍摄
    def face_rec(self, num):
        if num == 0:
            file, typ = QFileDialog.getOpenFileName(self, "选择视频", "./", "video (*.mp4)")
            if file:
                self.video_capture = cv2.VideoCapture(file)
        else:
            self.video_capture = cv2.VideoCapture(0)
        th = threading.Thread(target=self.video_face_rec)
        th.start()

    # 视频型人脸识别
    def video_face_rec(self):
        # 首先把打开按钮关闭
        self.image_btn.setEnabled(False)
        self.video_btn.setEnabled(False)
        self.camera_btn.setEnabled(False)
        self.close_btn.setEnabled(True)
        while True:
            ref, img = self.video_capture.read()  # 读取摄像头
            PicUtils.face_rec(img, self.face_characters, self.face_names)
            self.face_rec_label.setPixmap(QPixmap("images/tmp/face_rec.jpg"))
            if cv2.waitKey(25) & self.stopEvent.is_set() == True:
                self.stopEvent.clear()
                self.face_rec_label.clear()
                self.image_btn.setEnabled(True)
                self.video_btn.setEnabled(True)
                self.camera_btn.setEnabled(True)
                self.close_btn.setEnabled(False)
                self.face_rec_set_down()
                break
        self.image_btn.setEnabled(True)
        self.video_btn.setEnabled(True)
        self.camera_btn.setEnabled(True)
        self.close_btn.setEnabled(False)
        self.face_rec_set_down()

    # 图片型人脸识别
    def img_face_rec(self):
        image_path = self.choose_pic()
        if image_path == 0:
            return
        # 展示图片
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
        num = PicUtils.face_rec(img, self.face_characters, self.face_names)
        self.face_rec_label.setPixmap(QPixmap("images/tmp/face_rec.jpg"))
        if num == 0:
            QMessageBox.information(self, "提示", self.tr("图片中识别不到人脸"))

    # 退出人脸识别进程
    def face_rec_close(self):
        self.stopEvent.set()
        self.face_rec_set_down()

    # 初始化人脸识别界面
    def face_rec_set_down(self):
        self.video_capture.release()
        cv2.destroyAllWindows()
        self.face_rec_label.setText("人脸识别")

    '''
       口罩识别界面函数
    '''
    # 口罩识别线程
    def mask_rec_by_camera(self):
        self.video_capture = cv2.VideoCapture(0)
        th = threading.Thread(target=self.mask_rec)
        th.start()

    # 摄像头口罩识别
    def mask_rec(self):
        self.mask_rec_op_btn.setEnabled(False)
        self.mask_rec_cl_btn.setEnabled(True)
        # image = cv2.imread("images/background.jpg")  # 读取背景照片
        # cv2.imshow('skin', image)  # 展示
        # cv2.createTrackbar("h_min", "skin", 0, 90, Detector.back)  # 创建bar
        # cv2.createTrackbar("h_max", "skin", 15, 90, Detector.back)
        while True:
            ref, img = self.video_capture.read()
            PicUtils.mask_rec(img)
            self.mask_rec_label.setPixmap(QPixmap("images/tmp/mask_rec.jpg"))
            # cv2.imshow("skin", thresh)  # 显示肤色图
            # cv2.imshow("img", img_eye)  # 显示肤色图
            # cv2.imwrite('005_result.jpg',img_eye)     保存图片
            if cv2.waitKey(25) & self.stopEvent.is_set() == True:
                self.stopEvent.clear()
                self.mask_rec_label.clear()
                self.mask_rec_op_btn.setEnabled(True)
                self.mask_rec_cl_btn.setEnabled(False)
                self.mask_rec_set_down()
                break
        self.mask_rec_op_btn.setEnabled(True)
        self.mask_rec_cl_btn.setEnabled(False)
        self.mask_rec_set_down()

    # 退出口罩识别进程
    def mask_rec_close(self):
        self.stopEvent.set()
        self.mask_rec_set_down()

    # 初始化口罩识别界面
    def mask_rec_set_down(self):
        self.video_capture.release()
        cv2.destroyAllWindows()
        self.mask_rec_label.setText("人脸识别")

    '''
       佩戴口罩界面函数
    '''
    # 上传待戴口罩图片
    def show_pic_no_mask(self):
        image_path = self.choose_pic()
        if image_path == 0:
            return
        self.img_path1 = image_path
        self.mask_wear_widget1.setPixmap(QPixmap(image_path))
        self.mask_wear_widget2.setText("佩戴效果")

    # 佩戴口罩
    def wear_mask(self, num):
        mask = ["images/mask1.png", "images/mask2.png", "images/mask3.png", "images/mask4.png"]
        self.mask = Image.open(mask[num])
        self.show_pic("_mask")

    # 展示佩戴口罩后的图片
    def show_pic(self, kind):
        if self.img_path1 == '':
            QMessageBox.information(self, "提示", self.tr("请先选择图片"))
            return
        img = cv2.imdecode(np.fromfile(self.img_path1, dtype=np.uint8), -1)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        predictor = dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat')
        detector = dlib.get_frontal_face_detector()
        faces = detector(img_gray, 0)
        if len(faces) == 0:
            QMessageBox.information(self, "提示", self.tr("图片中识别不到人脸"))
            return
        x_min, x_max, y_min, y_max, size = Detector.get_mouth(faces, predictor, img_gray)
        mask = self.mask.resize(size)
        img = Image.fromarray(img[:, :, ::-1])  # 切换RGB格式
        # 在合适位置添加口罩图片
        img.paste(mask, (int(x_min), int(y_min)), mask)
        orig_name = self.img_path1.split('/')[-1]
        image2_path = "result/" + orig_name.split('.')[0] + kind + "." + orig_name.split('.')[1]
        img.save(image2_path)
        self.mask_wear_widget2.setPixmap(QPixmap(image2_path))

    '''
       美化图片界面函数
    '''
    # 上传待并显示美化图片
    def polish_pic(self):
        file_name = QFileDialog.getOpenFileName(self, "选择图片", "./", "images (*.jpg *.png *.jpeg)")
        image_path = file_name[0]
        if image_path == "":
            QMessageBox.information(self, "提示", self.tr("没有选择图片文件！"))
            return
        self.img_path2 = image_path
        self.pic_widget1.setPixmap(QPixmap(image_path))
        self.pic_widget2.setText("效果图")

    # 清除按钮格式
    def clear_button(self):
        self.left_button_1.setStyleSheet("QPushButton{border: none; color: white; font-weight: 400;}"
                                         "QPushButton:hover{border-left: 4px solid red; font-weight: 700;}")
        self.left_button_2.setStyleSheet("QPushButton{border: none; color: white; font-weight: 400;}"
                                         "QPushButton:hover{border-left: 4px solid red; font-weight: 700;}")
        self.left_button_3.setStyleSheet("QPushButton{border: none; color: white; font-weight: 400;}"
                                         "QPushButton:hover{border-left: 4px solid red; font-weight: 700;}")
        self.left_button_4.setStyleSheet("QPushButton{border: none; color: white; font-weight: 400;}"
                                         "QPushButton:hover{border-left: 4px solid red; font-weight: 700;}")
        self.left_button_5.setStyleSheet("QPushButton{border: none; color: white; font-weight: 400;}"
                                         "QPushButton:hover{border-left: 4px solid red; font-weight: 700;}")
        self.left_button_6.setStyleSheet("QPushButton{border: none; color: white; font-weight: 400;}"
                                         "QPushButton:hover{border-left: 4px solid red; font-weight: 700;}")
        self.left_button_7.setStyleSheet("QPushButton{border: none; color: white; font-weight: 400;}"
                                         "QPushButton:hover{border-left: 4px solid red; font-weight: 700;}")
        self.left_button_8.setStyleSheet("QPushButton{border: none; color: white; font-weight: 400;}"
                                         "QPushButton:hover{border-left: 4px solid red; font-weight: 700;}")
        self.left_button_9.setStyleSheet("QPushButton{border: none; color: white; font-weight: 400;}"
                                         "QPushButton:hover{border-left: 4px solid red; font-weight: 700;}")
        self.left_button_10.setStyleSheet("QPushButton{border: none; color: white; font-weight: 400;}"
                                          "QPushButton:hover{border-left: 4px solid red; font-weight: 700;}")
        self.left_button_11.setStyleSheet("QPushButton{border: none; color: white; font-weight: 400;}"
                                          "QPushButton:hover{border-left: 4px solid red; font-weight: 700;}")
        self.left_button_12.setStyleSheet("QPushButton{border: none; color: white; font-weight: 400;}"
                                          "QPushButton:hover{border-left: 4px solid red; font-weight: 700;}")

    # 分配对应的操作函数
    def hand_out(self):
        if self.fun_num == 1:
            self.do_gauss()
        elif self.fun_num == 2:
            self.do_salt()
        elif self.fun_num == 3:
            self.do_gamma()
        elif self.fun_num == 4:
            self.do_uniform()
        elif self.fun_num == 5:
            self.do_rayleigh()
        elif self.fun_num == 6:
            self.do_bilateral()
        elif self.fun_num == 7:
            self.do_median()
        elif self.fun_num == 8:
            self.do_average()
        elif self.fun_num == 9:
            self.do_high_pass()
        elif self.fun_num == 10:
            self.do_low_pass()

    # 设置参数条
    def set_bar(self, fun, m_min, m_max, single, interval, value):
        if self.img_path2 == "":
            QMessageBox.information(self, "提示", self.tr("没有选择图片文件！"))
            return 0
        self.fun_num = fun
        self.bar_widget.setMinimum(m_min)
        self.bar_widget.setMaximum(m_max)
        self.bar_widget.setSingleStep(single)
        self.bar_widget.setTickInterval(interval)
        self.bar_widget.setValue(value)
        self.clear_button()
        return 1

    # 保存操作后图片
    def write_operate(self, img):
        orig_name = self.img_path2.split('/')[-1]
        image2_path = "result/" + orig_name.split('.')[0] + "_gauss." + orig_name.split('.')[1]
        cv2.imwrite(image2_path, img)
        return image2_path

    # 高斯噪声界面
    def gauss_noise(self):
        flag = self.set_bar(1, 10, 60, 2, 5, 40)
        if flag == 0:
            return
        self.text_widget.setText("高斯噪声标准差(" + self.bar_widget.value().__str__() + "/60)")
        self.left_button_1.setStyleSheet("border-left: 4px solid red; font-weight: 700;")
        self.do_gauss()

    # 高斯噪声操作
    def do_gauss(self):
        self.text_widget.setText("高斯噪声标准差(" + self.bar_widget.value().__str__() + "/60)")
        img = cv2.imdecode(np.fromfile(self.img_path2, dtype=np.uint8), -1)
        img = NoiseGenerator.add_gauss_noise(img, self.bar_widget.value())
        image2_path = self.write_operate(img)
        self.pic_widget2.setPixmap(QPixmap(image2_path))

    # 椒盐噪声界面
    def salt_noise(self):
        flag = self.set_bar(2, 0, 10, 1, 1, 8)
        if flag == 0:
            return
        self.text_widget.setText("椒盐噪声信噪比(" + (self.bar_widget.value() / 10).__str__() + "/1)")
        self.left_button_2.setStyleSheet("border-left: 4px solid red; font-weight: 700;")
        self.do_salt()

    # 椒盐噪声操作
    def do_salt(self):
        self.text_widget.setText("椒盐噪声信噪比(" + (self.bar_widget.value() / 10).__str__() + "/1)")
        img = cv2.imdecode(np.fromfile(self.img_path2, dtype=np.uint8), -1)
        img = NoiseGenerator.add_salt_noise(img, self.bar_widget.value() / 10)
        image2_path = self.write_operate(img)
        self.pic_widget2.setPixmap(QPixmap(image2_path))

    # 伽马噪声界面
    def gamma_noise(self):
        flag = self.set_bar(3, 0, 10, 1, 1, 1)
        if flag == 0:
            return
        self.text_widget.setText("伽马噪声方差值(" + (self.bar_widget.value() / 10).__str__() + "/1)")
        self.left_button_3.setStyleSheet("border-left: 4px solid red; font-weight: 700;")
        self.do_gamma()

    # 伽马噪声操作
    def do_gamma(self):
        self.text_widget.setText("伽马噪声方差值(" + (self.bar_widget.value() / 10).__str__() + "/1)")
        img = cv2.imdecode(np.fromfile(self.img_path2, dtype=np.uint8), -1)
        img = NoiseGenerator.add_gamma_noise(img, self.bar_widget.value() / 10)
        image2_path = self.write_operate(img)
        self.pic_widget2.setPixmap(QPixmap(image2_path))

    # 均值噪声界面
    def uniform_noise(self):
        flag = self.set_bar(4, 0, 10, 1, 1, 1)
        if flag == 0:
            return
        self.text_widget.setText("均值噪声上界值(" + self.bar_widget.value().__str__() + "/10)")
        self.left_button_4.setStyleSheet("border-left: 4px solid red; font-weight: 700;")
        self.do_uniform()

    # 均值噪声操作
    def do_uniform(self):
        self.text_widget.setText("均值噪声上界值(" + self.bar_widget.value().__str__() + "/10)")
        img = cv2.imdecode(np.fromfile(self.img_path2, dtype=np.uint8), -1)
        img = NoiseGenerator.add_uniform_noise(img, self.bar_widget.value())
        image2_path = self.write_operate(img)
        self.pic_widget2.setPixmap(QPixmap(image2_path))

    # 瑞利噪声界面
    def rayleigh_noise(self):
        flag = self.set_bar(5, 0, 10, 1, 1, 1)
        if flag == 0:
            return
        self.text_widget.setText("瑞利噪声方差值(" + (self.bar_widget.value() / 10).__str__() + "/1)")
        self.left_button_5.setStyleSheet("border-left: 4px solid red; font-weight: 700;")
        self.do_rayleigh()

    # 瑞利噪声操作
    def do_rayleigh(self):
        self.text_widget.setText("瑞利噪声方差值(" + (self.bar_widget.value() / 10).__str__() + "/1)")
        img = cv2.imdecode(np.fromfile(self.img_path2, dtype=np.uint8), -1)
        img = NoiseGenerator.add_rayleigh_noise(img, self.bar_widget.value() / 10)
        image2_path = self.write_operate(img)
        self.pic_widget2.setPixmap(QPixmap(image2_path))

    # 双边滤波界面
    def bilateral_filter(self):
        flag = self.set_bar(6, 0, 10, 1, 1, 3)
        if flag == 0:
            return
        self.text_widget.setText("双边滤波核大小(" + self.bar_widget.value().__str__() + "/10)")
        self.left_button_6.setStyleSheet("border-left: 4px solid red; font-weight: 700;")
        self.do_bilateral()

    # 双边滤波操作
    def do_bilateral(self):
        self.text_widget.setText("双边滤波核大小(" + self.bar_widget.value().__str__() + "/10)")
        img = cv2.imdecode(np.fromfile(self.img_path2, dtype=np.uint8), -1)
        img = cv2.bilateralFilter(img, self.bar_widget.value(), 20, 20)
        image2_path = self.write_operate(img)
        self.pic_widget2.setPixmap(QPixmap(image2_path))

    # 中值滤波界面
    def median_filter(self):
        flag = self.set_bar(7, 0, 10, 1, 1, 3)
        if flag == 0:
            return
        self.text_widget.setText("中值滤波核大小(" + self.bar_widget.value().__str__() + "/10)")
        self.left_button_7.setStyleSheet("border-left: 4px solid red; font-weight: 700;")
        self.do_median()

    # 中值滤波操作
    def do_median(self):
        self.text_widget.setText("中值滤波核大小(" + self.bar_widget.value().__str__() + "/10)")
        img = cv2.imdecode(np.fromfile(self.img_path2, dtype=np.uint8), -1)
        img = Filter.median_filter(img, self.bar_widget.value())
        image2_path = self.write_operate(img)
        self.pic_widget2.setPixmap(QPixmap(image2_path))

    # 均值滤波界面
    def average_filter(self):
        flag = self.set_bar(8, 0, 10, 1, 1, 3)
        if flag == 0:
            return
        self.text_widget.setText("均值滤波核大小(" + self.bar_widget.value().__str__() + "/10)")
        self.left_button_8.setStyleSheet("border-left: 4px solid red; font-weight: 700;")
        self.do_average()

    # 中值滤波操作
    def do_average(self):
        self.text_widget.setText("均值滤波核大小(" + self.bar_widget.value().__str__() + "/10)")
        img = cv2.imdecode(np.fromfile(self.img_path2, dtype=np.uint8), -1)
        img = Filter.average_filter(img, self.bar_widget.value())
        image2_path = self.write_operate(img)
        self.pic_widget2.setPixmap(QPixmap(image2_path))

    # 高通滤波界面
    def high_pass_filter(self):
        flag = self.set_bar(9, 20, 70, 5, 5, 40)
        if flag == 0:
            return
        self.text_widget.setText("高通滤波半径(" + self.bar_widget.value().__str__() + "/70)")
        self.left_button_9.setStyleSheet("border-left: 4px solid red; font-weight: 700;")
        self.do_high_pass()

    # 高通滤波操作
    def do_high_pass(self):
        self.text_widget.setText("高通滤波半径(" + self.bar_widget.value().__str__() + "/70)")
        img = cv2.imdecode(np.fromfile(self.img_path2, dtype=np.uint8), -1)
        img = Filter.high_pass_filter(img, self.bar_widget.value())
        image2_path = self.write_operate(img)
        self.pic_widget2.setPixmap(QPixmap(image2_path))

    # 低通滤波界面
    def low_pass_filter(self):
        flag = self.set_bar(10, 100, 160, 10, 10, 120)
        if flag == 0:
            return
        self.text_widget.setText("低通滤波半径(" + self.bar_widget.value().__str__() + "/160)")
        self.left_button_10.setStyleSheet("border-left: 4px solid red; font-weight: 700;")
        self.do_low_pass()

    # 低通滤波操作
    def do_low_pass(self):
        self.text_widget.setText("低通滤波半径(" + self.bar_widget.value().__str__() + "/160)")
        img = cv2.imdecode(np.fromfile(self.img_path2, dtype=np.uint8), -1)
        img = Filter.low_pass_filter(img, self.bar_widget.value())
        image2_path = self.write_operate(img)
        self.pic_widget2.setPixmap(QPixmap(image2_path))

    # 直方图正规化界面
    def regularization(self):
        flag = self.set_bar(11, 0, 10, 1, 1, 0)
        if flag == 0:
            return
        self.text_widget.setText("直方图正规化")
        self.left_button_11.setStyleSheet("border-left: 4px solid red; font-weight: 700;")
        self.do_regular()

    # 直方图正规化操作
    def do_regular(self):
        img = cv2.imdecode(np.fromfile(self.img_path2, dtype=np.uint8), -1)
        img = PicUtils.regularize(img)
        image2_path = self.write_operate(img)
        self.pic_widget2.setPixmap(QPixmap(image2_path))

    # 直方图均衡化界面
    def equalization(self):
        flag = self.set_bar(12, 0, 10, 1, 1, 0)
        if flag == 0:
            return
        self.text_widget.setText("直方图均衡化")
        self.left_button_12.setStyleSheet("border-left: 4px solid red; font-weight: 700;")
        self.do_equal()

    # 直方图均衡化操作
    def do_equal(self):
        img = cv2.imdecode(np.fromfile(self.img_path2, dtype=np.uint8), -1)
        img = PicUtils.equalize(img)
        image2_path = self.write_operate(img)
        self.pic_widget2.setPixmap(QPixmap(image2_path))

    '''
       证件照界面函数
    '''
    # 生成证件照
    def show_certificate(self, col):
        self.exist = True
        file_name = QFileDialog.getOpenFileName(self, "选择图片", "./", "images (*.jpg *.png *.jpeg)")
        image_path = file_name[0]
        if image_path == "":
            QMessageBox.information(self, "提示", self.tr("没有选择图片文件！"))
            return
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
        PicUtils.paste_pic(img, col)
        self.certificate_label.setPixmap(QPixmap("result/certificate.png"))

    # 保存证件照
    def save_certificate(self):
        if self.exist is False:
            QMessageBox.information(self, "提示", self.tr("请先选择一种颜色生成证件照"))
            return
        file_name, _ = QFileDialog.getSaveFileName(self, "保存图片", "certificate.png", "images (*.png)")
        if file_name == "":
            return
        img = cv2.imread("result/certificate.png")
        cv2.imwrite(file_name, img)

    '''
        退出程序
    '''
    # 退出程序
    def closeEvent(self, event):
        reply = QMessageBox.question(self, "退出", "是否退出程序？", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.face_rec_close()
            event.accept()
        else:
            event.ignore()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    gui = MainWindow()
    gui.show()
    sys.exit(app.exec_())
