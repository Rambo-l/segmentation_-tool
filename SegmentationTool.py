# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'SegmentationTool.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1304, 947)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.exit_button = QtWidgets.QPushButton(self.centralwidget)
        self.exit_button.setGeometry(QtCore.QRect(930, 750, 101, 61))
        self.exit_button.setObjectName("exit_button")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(230, 0, 231, 61))
        self.label_2.setStyleSheet("font: 18pt \"AcadEref\";")
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(210, 210, 261, 61))
        self.label_3.setStyleSheet("font: 18pt \"Times New Roman\";")
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(770, 0, 391, 61))
        self.label_4.setStyleSheet("font: 18pt \"AcadEref\";")
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(880, 350, 221, 61))
        self.label_5.setStyleSheet("font: 18pt \"Times New Roman\";")
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(210, 430, 261, 61))
        self.label_6.setStyleSheet("font: 18pt \"Times New Roman\";")
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(40, 610, 601, 61))
        self.label_7.setStyleSheet("font: 18pt \"Times New Roman\";")
        self.label_7.setObjectName("label_7")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(718, 410, 511, 241))
        self.layoutWidget.setObjectName("layoutWidget")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout_4.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.sel_model_label = QtWidgets.QPushButton(self.layoutWidget)
        self.sel_model_label.setObjectName("sel_model_label")
        self.gridLayout_4.addWidget(self.sel_model_label, 0, 0, 1, 1)
        self.lineEdit_modellabel = QtWidgets.QLineEdit(self.layoutWidget)
        self.lineEdit_modellabel.setText("")
        self.lineEdit_modellabel.setObjectName("lineEdit_modellabel")
        self.gridLayout_4.addWidget(self.lineEdit_modellabel, 0, 1, 1, 1)
        self.sel_handle_label = QtWidgets.QPushButton(self.layoutWidget)
        self.sel_handle_label.setObjectName("sel_handle_label")
        self.gridLayout_4.addWidget(self.sel_handle_label, 1, 0, 1, 1)
        self.lineEdit_handlelabel = QtWidgets.QLineEdit(self.layoutWidget)
        self.lineEdit_handlelabel.setText("")
        self.lineEdit_handlelabel.setObjectName("lineEdit_handlelabel")
        self.gridLayout_4.addWidget(self.lineEdit_handlelabel, 1, 1, 1, 1)
        self.compute_iou = QtWidgets.QPushButton(self.layoutWidget)
        self.compute_iou.setObjectName("compute_iou")
        self.gridLayout_4.addWidget(self.compute_iou, 2, 0, 1, 1)
        self.textEdit_iou = QtWidgets.QTextEdit(self.layoutWidget)
        self.textEdit_iou.setObjectName("textEdit_iou")
        self.gridLayout_4.addWidget(self.textEdit_iou, 2, 1, 1, 1)
        self.layoutWidget1 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget1.setGeometry(QtCore.QRect(70, 280, 511, 141))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.layoutWidget1)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.sel_jsonjpg_file = QtWidgets.QPushButton(self.layoutWidget1)
        self.sel_jsonjpg_file.setObjectName("sel_jsonjpg_file")
        self.gridLayout_2.addWidget(self.sel_jsonjpg_file, 0, 0, 1, 1)
        self.lineEdit_jsonjpg = QtWidgets.QLineEdit(self.layoutWidget1)
        self.lineEdit_jsonjpg.setObjectName("lineEdit_jsonjpg")
        self.gridLayout_2.addWidget(self.lineEdit_jsonjpg, 0, 1, 1, 1)
        self.sel_jsonsave_file = QtWidgets.QPushButton(self.layoutWidget1)
        self.sel_jsonsave_file.setObjectName("sel_jsonsave_file")
        self.gridLayout_2.addWidget(self.sel_jsonsave_file, 1, 0, 1, 1)
        self.lineEdit_json = QtWidgets.QLineEdit(self.layoutWidget1)
        self.lineEdit_json.setObjectName("lineEdit_json")
        self.gridLayout_2.addWidget(self.lineEdit_json, 1, 1, 1, 1)
        self.sel_jpgsave_file = QtWidgets.QPushButton(self.layoutWidget1)
        self.sel_jpgsave_file.setObjectName("sel_jpgsave_file")
        self.gridLayout_2.addWidget(self.sel_jpgsave_file, 2, 0, 1, 1)
        self.lineEdit_jpg = QtWidgets.QLineEdit(self.layoutWidget1)
        self.lineEdit_jpg.setObjectName("lineEdit_jpg")
        self.gridLayout_2.addWidget(self.lineEdit_jpg, 2, 1, 1, 1)
        self.json_jpg_change = QtWidgets.QPushButton(self.layoutWidget1)
        self.json_jpg_change.setObjectName("json_jpg_change")
        self.gridLayout_2.addWidget(self.json_jpg_change, 3, 0, 1, 1)
        self.lineEdit_jsonjpg_change = QtWidgets.QLineEdit(self.layoutWidget1)
        self.lineEdit_jsonjpg_change.setObjectName("lineEdit_jsonjpg_change")
        self.gridLayout_2.addWidget(self.lineEdit_jsonjpg_change, 3, 1, 1, 1)
        self.layoutWidget2 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget2.setGeometry(QtCore.QRect(70, 60, 511, 151))
        self.layoutWidget2.setObjectName("layoutWidget2")
        self.gridLayout = QtWidgets.QGridLayout(self.layoutWidget2)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.sel_video_file = QtWidgets.QPushButton(self.layoutWidget2)
        self.sel_video_file.setObjectName("sel_video_file")
        self.gridLayout.addWidget(self.sel_video_file, 0, 0, 1, 1)
        self.lineEdit_video = QtWidgets.QLineEdit(self.layoutWidget2)
        self.lineEdit_video.setObjectName("lineEdit_video")
        self.gridLayout.addWidget(self.lineEdit_video, 0, 1, 1, 1)
        self.sel_savepic_file = QtWidgets.QPushButton(self.layoutWidget2)
        self.sel_savepic_file.setObjectName("sel_savepic_file")
        self.gridLayout.addWidget(self.sel_savepic_file, 1, 0, 1, 1)
        self.lineEdit_image = QtWidgets.QLineEdit(self.layoutWidget2)
        self.lineEdit_image.setObjectName("lineEdit_image")
        self.gridLayout.addWidget(self.lineEdit_image, 1, 1, 1, 1)
        self.video_pic_change = QtWidgets.QPushButton(self.layoutWidget2)
        self.video_pic_change.setObjectName("video_pic_change")
        self.gridLayout.addWidget(self.video_pic_change, 2, 0, 1, 1)
        self.lineEdit_videopic_change = QtWidgets.QLineEdit(self.layoutWidget2)
        self.lineEdit_videopic_change.setObjectName("lineEdit_videopic_change")
        self.gridLayout.addWidget(self.lineEdit_videopic_change, 2, 1, 1, 1)
        self.layoutWidget3 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget3.setGeometry(QtCore.QRect(720, 60, 511, 232))
        self.layoutWidget3.setObjectName("layoutWidget3")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.layoutWidget3)
        self.gridLayout_3.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.lineEdit_valtrain = QtWidgets.QLineEdit(self.layoutWidget3)
        self.lineEdit_valtrain.setText("")
        self.lineEdit_valtrain.setObjectName("lineEdit_valtrain")
        self.gridLayout_3.addWidget(self.lineEdit_valtrain, 4, 2, 1, 1)
        self.train_val_change = QtWidgets.QPushButton(self.layoutWidget3)
        self.train_val_change.setObjectName("train_val_change")
        self.gridLayout_3.addWidget(self.train_val_change, 5, 0, 1, 1)
        self.lineEdit_trainimage = QtWidgets.QLineEdit(self.layoutWidget3)
        self.lineEdit_trainimage.setObjectName("lineEdit_trainimage")
        self.gridLayout_3.addWidget(self.lineEdit_trainimage, 0, 2, 1, 1)
        self.sel_vallabel_file = QtWidgets.QPushButton(self.layoutWidget3)
        self.sel_vallabel_file.setObjectName("sel_vallabel_file")
        self.gridLayout_3.addWidget(self.sel_vallabel_file, 3, 0, 1, 2)
        self.label = QtWidgets.QLabel(self.layoutWidget3)
        self.label.setObjectName("label")
        self.gridLayout_3.addWidget(self.label, 4, 0, 1, 2)
        self.lineEdit_valimage = QtWidgets.QLineEdit(self.layoutWidget3)
        self.lineEdit_valimage.setObjectName("lineEdit_valimage")
        self.gridLayout_3.addWidget(self.lineEdit_valimage, 2, 2, 1, 1)
        self.sel_trainlabel_file = QtWidgets.QPushButton(self.layoutWidget3)
        self.sel_trainlabel_file.setObjectName("sel_trainlabel_file")
        self.gridLayout_3.addWidget(self.sel_trainlabel_file, 1, 0, 1, 2)
        self.sel_trainimage_file = QtWidgets.QPushButton(self.layoutWidget3)
        self.sel_trainimage_file.setObjectName("sel_trainimage_file")
        self.gridLayout_3.addWidget(self.sel_trainimage_file, 0, 0, 1, 2)
        self.sel_valimage_file = QtWidgets.QPushButton(self.layoutWidget3)
        self.sel_valimage_file.setObjectName("sel_valimage_file")
        self.gridLayout_3.addWidget(self.sel_valimage_file, 2, 0, 1, 2)
        self.lineEdit_trainval_change = QtWidgets.QLineEdit(self.layoutWidget3)
        self.lineEdit_trainval_change.setObjectName("lineEdit_trainval_change")
        self.gridLayout_3.addWidget(self.lineEdit_trainval_change, 5, 1, 1, 2)
        self.lineEdit_trainlabel = QtWidgets.QLineEdit(self.layoutWidget3)
        self.lineEdit_trainlabel.setObjectName("lineEdit_trainlabel")
        self.gridLayout_3.addWidget(self.lineEdit_trainlabel, 1, 2, 1, 1)
        self.lineEdit_vallabel = QtWidgets.QLineEdit(self.layoutWidget3)
        self.lineEdit_vallabel.setObjectName("lineEdit_vallabel")
        self.gridLayout_3.addWidget(self.lineEdit_vallabel, 3, 2, 1, 1)
        self.layoutWidget4 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget4.setGeometry(QtCore.QRect(70, 500, 511, 101))
        self.layoutWidget4.setObjectName("layoutWidget4")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.layoutWidget4)
        self.gridLayout_5.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.sel_Jsonjpg = QtWidgets.QPushButton(self.layoutWidget4)
        self.sel_Jsonjpg.setObjectName("sel_Jsonjpg")
        self.gridLayout_5.addWidget(self.sel_Jsonjpg, 0, 0, 1, 1)
        self.lineEdit_Jsonjpg = QtWidgets.QLineEdit(self.layoutWidget4)
        self.lineEdit_Jsonjpg.setObjectName("lineEdit_Jsonjpg")
        self.gridLayout_5.addWidget(self.lineEdit_Jsonjpg, 0, 1, 1, 1)
        self.sel_output = QtWidgets.QPushButton(self.layoutWidget4)
        self.sel_output.setObjectName("sel_output")
        self.gridLayout_5.addWidget(self.sel_output, 1, 0, 1, 1)
        self.lineEdit_output = QtWidgets.QLineEdit(self.layoutWidget4)
        self.lineEdit_output.setText("")
        self.lineEdit_output.setObjectName("lineEdit_output")
        self.gridLayout_5.addWidget(self.lineEdit_output, 1, 1, 1, 1)
        self.Do_jsontodataset = QtWidgets.QPushButton(self.layoutWidget4)
        self.Do_jsontodataset.setObjectName("Do_jsontodataset")
        self.gridLayout_5.addWidget(self.Do_jsontodataset, 2, 0, 1, 1)
        self.lineEdit_do_jsontodataset = QtWidgets.QLineEdit(self.layoutWidget4)
        self.lineEdit_do_jsontodataset.setObjectName("lineEdit_do_jsontodataset")
        self.gridLayout_5.addWidget(self.lineEdit_do_jsontodataset, 2, 1, 1, 1)
        self.layoutWidget5 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget5.setGeometry(QtCore.QRect(70, 670, 521, 205))
        self.layoutWidget5.setObjectName("layoutWidget5")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.layoutWidget5)
        self.gridLayout_6.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.sel_Jsonjpg_2 = QtWidgets.QPushButton(self.layoutWidget5)
        self.sel_Jsonjpg_2.setObjectName("sel_Jsonjpg_2")
        self.gridLayout_6.addWidget(self.sel_Jsonjpg_2, 0, 0, 1, 2)
        self.lineEdit_Jsonjpg_2 = QtWidgets.QLineEdit(self.layoutWidget5)
        self.lineEdit_Jsonjpg_2.setObjectName("lineEdit_Jsonjpg_2")
        self.gridLayout_6.addWidget(self.lineEdit_Jsonjpg_2, 0, 2, 1, 1)
        self.sel_Output_file = QtWidgets.QPushButton(self.layoutWidget5)
        self.sel_Output_file.setObjectName("sel_Output_file")
        self.gridLayout_6.addWidget(self.sel_Output_file, 1, 0, 1, 1)
        self.lineEdit_Jsonjpg_7 = QtWidgets.QLineEdit(self.layoutWidget5)
        self.lineEdit_Jsonjpg_7.setText("")
        self.lineEdit_Jsonjpg_7.setObjectName("lineEdit_Jsonjpg_7")
        self.gridLayout_6.addWidget(self.lineEdit_Jsonjpg_7, 1, 1, 1, 2)
        self.sel_Classname_file = QtWidgets.QPushButton(self.layoutWidget5)
        self.sel_Classname_file.setObjectName("sel_Classname_file")
        self.gridLayout_6.addWidget(self.sel_Classname_file, 2, 0, 1, 1)
        self.lineEdit_Jsonjpg_3 = QtWidgets.QLineEdit(self.layoutWidget5)
        self.lineEdit_Jsonjpg_3.setObjectName("lineEdit_Jsonjpg_3")
        self.gridLayout_6.addWidget(self.lineEdit_Jsonjpg_3, 2, 2, 1, 1)
        self.sel_JPG_file = QtWidgets.QPushButton(self.layoutWidget5)
        self.sel_JPG_file.setObjectName("sel_JPG_file")
        self.gridLayout_6.addWidget(self.sel_JPG_file, 3, 0, 1, 1)
        self.lineEdit_Jsonjpg_4 = QtWidgets.QLineEdit(self.layoutWidget5)
        self.lineEdit_Jsonjpg_4.setObjectName("lineEdit_Jsonjpg_4")
        self.gridLayout_6.addWidget(self.lineEdit_Jsonjpg_4, 3, 2, 1, 1)
        self.sel_PNG_file = QtWidgets.QPushButton(self.layoutWidget5)
        self.sel_PNG_file.setObjectName("sel_PNG_file")
        self.gridLayout_6.addWidget(self.sel_PNG_file, 4, 0, 1, 1)
        self.lineEdit_Jsonjpg_5 = QtWidgets.QLineEdit(self.layoutWidget5)
        self.lineEdit_Jsonjpg_5.setObjectName("lineEdit_Jsonjpg_5")
        self.gridLayout_6.addWidget(self.lineEdit_Jsonjpg_5, 4, 2, 1, 1)
        self.do_get_jpgpng = QtWidgets.QPushButton(self.layoutWidget5)
        self.do_get_jpgpng.setObjectName("do_get_jpgpng")
        self.gridLayout_6.addWidget(self.do_get_jpgpng, 5, 0, 1, 1)
        self.lineEdit_Jsonjpg_6 = QtWidgets.QLineEdit(self.layoutWidget5)
        self.lineEdit_Jsonjpg_6.setObjectName("lineEdit_Jsonjpg_6")
        self.gridLayout_6.addWidget(self.lineEdit_Jsonjpg_6, 5, 2, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1304, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.exit_button.setText(_translate("MainWindow", "退出"))
        self.label_2.setText(_translate("MainWindow", "视频转图像功能"))
        self.label_3.setText(_translate("MainWindow", "JSON/JPG文件分离"))
        self.label_4.setText(_translate("MainWindow", "从训练集随机选择比例测试集"))
        self.label_5.setText(_translate("MainWindow", "MIOU/MPA计算"))
        self.label_6.setText(_translate("MainWindow", "JSON_TO_Dataset"))
        self.label_7.setText(_translate("MainWindow", "Get_JPG_PNG(从Dtaset提取训练图像和标签)"))
        self.sel_model_label.setText(_translate("MainWindow", "点击选择模型输出标签图像文件夹"))
        self.sel_handle_label.setText(_translate("MainWindow", "点击选择人工标签图像文件夹"))
        self.compute_iou.setText(_translate("MainWindow", "点击选择IOU等参数"))
        self.sel_jsonjpg_file.setText(_translate("MainWindow", "选择JSON-JPG素材文件夹"))
        self.sel_jsonsave_file.setText(_translate("MainWindow", "点击选择JSON存储文件夹"))
        self.sel_jpgsave_file.setText(_translate("MainWindow", "点击选择JPG存储文件夹"))
        self.json_jpg_change.setText(_translate("MainWindow", "开始转移"))
        self.sel_video_file.setText(_translate("MainWindow", "点击选择视频文件"))
        self.sel_savepic_file.setText(_translate("MainWindow", "点击选择存储图像文件夹"))
        self.video_pic_change.setText(_translate("MainWindow", "开始转换"))
        self.train_val_change.setText(_translate("MainWindow", "开始随机转移"))
        self.sel_vallabel_file.setText(_translate("MainWindow", "点击选择测试集标签文件夹"))
        self.label.setText(_translate("MainWindow", "输入验证集/训练集的比例："))
        self.sel_trainlabel_file.setText(_translate("MainWindow", "点击选择训练集标签文件夹"))
        self.sel_trainimage_file.setText(_translate("MainWindow", "点击选择训练集图像文件夹"))
        self.sel_valimage_file.setText(_translate("MainWindow", "点击选择测试集图像文件夹"))
        self.sel_Jsonjpg.setText(_translate("MainWindow", "点击选择JSON&JPG文件夹"))
        self.sel_output.setText(_translate("MainWindow", "点击选择OUTPUT文件夹"))
        self.Do_jsontodataset.setText(_translate("MainWindow", "执行JsonToDataset"))
        self.sel_Jsonjpg_2.setText(_translate("MainWindow", "点击选择JSON&JPG文件夹"))
        self.sel_Output_file.setText(_translate("MainWindow", "点击选择OUTPUT文件夹"))
        self.sel_Classname_file.setText(_translate("MainWindow", "点击选择ClassName文件"))
        self.sel_JPG_file.setText(_translate("MainWindow", "点击选择输出JPG文件夹"))
        self.sel_PNG_file.setText(_translate("MainWindow", "点击选择输出PNG文件夹"))
        self.do_get_jpgpng.setText(_translate("MainWindow", "执行Get_JPG_PNG"))
