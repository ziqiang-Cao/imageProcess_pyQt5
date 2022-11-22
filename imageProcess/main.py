# -*- coding: utf-8 -*-
# @Project: imageProcess
# @File    : main.py
# @Author  : colby
# @Time    : 2022-10-29 17:25:13
# @Desc    :
import os
import sys

from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import Qt, QRect, QObject, QEvent
from PyQt5.QtGui import QEnterEvent
from PyQt5.QtWidgets import (QApplication, QMainWindow, QAction, QFileDialog,
                             QMessageBox)

from thread_msg import ThreadMsg
from ui.window import Ui_MainWindow
from thread_func import MyThread


class MainWindow(QMainWindow, Ui_MainWindow):
    msg_signal = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.my_thread = None
        self.msg_thread = ThreadMsg(self)
        self.cwd = os.getcwd()
        self.setupUi(self)
        # 无边框
        self.setWindowFlag(Qt.FramelessWindowHint)
        # 背景透明
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.m_flag = None
        self.m_position = None
        self.MARGIN = 5
        self.MOVE_SENSITIVITY = 3
        self.isPressed = False
        self.press_button = None
        self.last_geometry = None
        self._area = None
        self._move_count = 0
        self.src_img1 = None
        self.src_img2 = None
        self.bfmatch_flag = False
        self.dst_img1 = None
        self.img_transform = None
        self.frame.setMouseTracking(True)
        self.frame.installEventFilter(self)
        self.frame_2.installEventFilter(self)
        self.frame_3.installEventFilter(self)
        # 检测出得关键点和描述子，用于子线程进行特征匹配
        self.key_point1 = None
        self.key_point2 = None
        self.describe2 = None
        self.describe2 = None
        self.matches = None
        # flann orb特征匹配
        self.flag_flann_orb = False
        # 判断两张图片都已加载
        self.img_flag = 0
        # 主窗口居中
        self.move(int((QApplication.primaryScreen().availableGeometry().width() - self.width())/2),
                  int((QApplication.primaryScreen().availableGeometry().height() - self.height())/2))
        # 给特征检测按钮添加下拉菜单
        self.menu_featExtraction = QtWidgets.QMenu(self.pushButton_featExtraction)
        self.menu_featExtraction.setObjectName("featExtraction")
        self.act_harris = QAction("Harris", self)
        self.act_harris.setObjectName("Harris")
        self.act_harris.triggered.connect(lambda: self.action_triggered("harris"))
        self.act_shitomasi = QtWidgets.QAction("Shi-Tomasi", self)
        self.act_shitomasi.setObjectName("Shi-Tomasi")
        self.act_shitomasi.triggered.connect(lambda: self.action_triggered("shitomasi"))
        self.act_sift = QtWidgets.QAction("Sift", self)
        self.act_sift.setObjectName("Sift")
        self.act_sift.triggered.connect(lambda: self.action_triggered("sift"))
        self.act_surf = QtWidgets.QAction("Surf", self)
        self.act_surf.setObjectName("Surf")
        self.act_surf.triggered.connect(lambda: self.action_triggered("surf"))
        self.act_orb = QtWidgets.QAction("ORB", self)
        self.act_orb.setObjectName("ORB")
        self.act_orb.triggered.connect(lambda: self.action_triggered("orb"))
        self.menu_featExtraction.addAction(self.act_harris)
        self.menu_featExtraction.addAction(self.act_shitomasi)
        self.menu_featExtraction.addAction(self.act_sift)
        self.menu_featExtraction.addAction(self.act_surf)
        self.menu_featExtraction.addAction(self.act_orb)
        # self.menu_featExtraction.setFixedSize(self.pushButton_featExtraction.minimumSize())
        self.pushButton_featExtraction.setMenu(self.menu_featExtraction)
        self.menu_featExtraction.setStyleSheet("QMenu{\n"
                                               "    background-color: rgb(95, 96, 100);\n"
                                               "    font: 25 10pt \"Microsoft YaHei\";\n"
                                               "    font-weight: bold;\n"
                                               "    color: rgb(218, 218, 218);\n"
                                               "    border-radius: 10px;  \n"
                                               "    border: 2px groove gray;\n"
                                               "    border-style: outset;\n"
                                               "}"
                                               "QMenu::item:selected{background-color:#818181;}")
        # 给特征匹配按钮添加下拉菜单
        self.menu_featMatching = QtWidgets.QMenu(self.pushButton_featMatching)
        self.menu_featMatching.setObjectName("featMatching")
        self.act_forceMatching = QtWidgets.QAction("ForceMatching", self)
        self.act_forceMatching.setObjectName("ForceMatching")
        self.act_forceMatching.triggered.connect(lambda: self.action_triggered("forcematching"))
        self.act_flann = QtWidgets.QAction("FLANN", self)
        self.act_flann.setObjectName("FLANN")
        self.act_flann.triggered.connect(lambda: self.action_triggered("flannmatching"))
        self.menu_featMatching.addAction(self.act_forceMatching)
        self.menu_featMatching.addAction(self.act_flann)
        # self.menu_featExtraction.setFixedSize(self.pushButton_featExtraction.minimumSize())
        self.pushButton_featMatching.setMenu(self.menu_featMatching)
        self.menu_featMatching.setStyleSheet("QMenu{\n"
                                             "    background-color: rgb(95, 96, 100);\n"
                                             "    font: 25 10pt \"Microsoft YaHei\";\n"
                                             "    font-weight: bold;\n"
                                             "    color: rgb(218, 218, 218);\n"
                                             "    border-radius: 10px;  \n"
                                             "    border: 2px groove gray;\n"
                                             "    border-style: outset;\n"
                                             "}"
                                             "QMenu::item:selected{background-color:#818181;}")
        # 给图像拼接按钮添加下拉菜单
        self.pushButton_imgStitching.clicked.connect(lambda: self.action_triggered("stitching"))
        # 给拼接处理按钮添加下拉菜单
        self.pushButton_edgeProcess.clicked.connect(lambda: self.action_triggered("edgeprocess"))
        # 给打开文件按钮绑定槽函数
        self.pushButton_openFile1.clicked.connect(lambda: self.open_file(self.img1))
        self.pushButton_openFile2.clicked.connect(lambda: self.open_file(self.img2))
        # 绑定自定义信号和槽函数
        self.msg_signal.connect(self.print_msg)

    def msg_critical(self, str_info):
        print("msgCritical")
        dlg = QMessageBox(self)
        dlg.setIcon(QMessageBox.Critical)
        dlg.setText(str_info)
        dlg.show()

    def print_msg(self, msg):
        # print(msg)
        self.textBrowser.append(msg)
        self.textBrowser.moveCursor(self.textBrowser.textCursor().End)

    def open_file(self, my_label):
        img_name, img_type = QFileDialog.getOpenFileName(self, "", self.cwd,
                                                         "*.jpg, *.png, *.jpg, *.jpeg, ALL Files(*)")
        if img_name:
            try:
                # print(img_name, my_label.width(), my_label.height())
                img_src = QtGui.QPixmap(img_name).scaled(my_label.width(), my_label.height())
                my_label.setPixmap(img_src)
                self.stackedWidget.setCurrentIndex(0)
                if my_label.objectName() == "img1":
                    self.img_flag = self.img_flag + 1
                    self.src_img1 = QtGui.QPixmap(img_name)
                elif my_label.objectName() == "img2":
                    self.img_flag = self.img_flag + 1
                    self.src_img2 = QtGui.QPixmap(img_name)
            except Exception as e:
                self.msg_thread.msg = e
                self.msg_thread.start()
        else:
            pass

    def action_triggered(self, way):
        if self.img_flag >= 2:
            # 这种正常
            self.my_thread = MyThread(way, self)
            self.my_thread.start()
        else:
            # 不能在局部直接新建线程对象，否则函数调用完此线程即被回收，导致程序崩溃
            self.msg_thread.msg = "请先载入两张图"
            self.msg_thread.start()

    def _resize(self, event):
        """实现拖动调整窗口大小的函数

        以新旧坐标差计算偏移量，使用 QRect 实例附带位置坐标；
        核心算法做了三重校验，以确保任意情况下窗口都能以正确的方式调整大小：
            一: 横纵坐标与最值校验，确保在最值范围内调整大小；
            二: 横纵坐标与左右区块校验，确保鼠标在窗口边缘时才调整大小；
            三: 横纵坐标与极值偏移量校验，确保在改变坐标的情况下，窗口不会发生漂移
        """
        # 鼠标在窗口中的区域
        area = self._area
        # 鼠标偏移量
        offsetPos = event.globalPos() - self._posLast
        # 鼠标在窗口中的坐标
        winPos = event.pos()

        # 矩形实例，被赋予窗口的几何属性（x, y, width, height）
        # 利用其改变左上角坐标，但右下角坐标不变的特性，实现窗口移动效果
        rect = QRect(self.geometry())

        x = rect.x()
        y = rect.y()
        width = rect.width()
        height = rect.height()

        minWidth = self.minimumWidth()
        minHeight = self.minimumHeight()
        maxWidth = self.maximumWidth()
        maxHeight = self.maximumHeight()

        # 根据不同区域选择不同操作
        if area == 11:
            # 左上
            pos = rect.topLeft()

            if offsetPos.x() < 0 and width < maxWidth or offsetPos.x() > 0 and width > minWidth:
                if offsetPos.x() < 0 and winPos.x() <= 0 or offsetPos.x() > 0 and winPos.x() >= 0:
                    if (maxWidth - width) >= -offsetPos.x() and (width - minWidth) >= offsetPos.x():
                        pos.setX(pos.x() + offsetPos.x())

            if offsetPos.y() < 0 and height < maxHeight or offsetPos.y() > 0 and height > minHeight:
                if offsetPos.y() < 0 and winPos.y() <= 0 or offsetPos.y() > 0 and winPos.y() >= 0:
                    if (maxHeight - height) >= -offsetPos.y() and (height - minHeight) >= offsetPos.y():
                        pos.setY(pos.y() + offsetPos.y())

            rect.setTopLeft(pos)

        elif area == 13:
            # 右上
            pos = rect.topRight()

            if offsetPos.x() < 0 and width > minWidth or offsetPos.x() > 0 and width < maxWidth:
                if offsetPos.x() < 0 and winPos.x() <= width or offsetPos.x() > 0 and winPos.x() >= width:
                    pos.setX(pos.x() + offsetPos.x())

            if offsetPos.y() < 0 and height < maxHeight or offsetPos.y() > 0 and height > minHeight:
                if offsetPos.y() < 0 and winPos.y() <= 0 or offsetPos.y() > 0 and winPos.y() >= 0:
                    if (maxHeight - height) >= -offsetPos.y() and (height - minHeight) >= offsetPos.y():
                        pos.setY(pos.y() + offsetPos.y())

            rect.setTopRight(pos)

        elif area == 31:
            # 左下
            pos = rect.bottomLeft()

            if offsetPos.x() < 0 and width < maxWidth or offsetPos.x() > 0 and width > minWidth:
                if offsetPos.x() < 0 and winPos.x() <= 0 or offsetPos.x() > 0 and winPos.x() >= 0:
                    if (maxWidth - width) >= -offsetPos.x() and (width - minWidth) >= offsetPos.x():
                        pos.setX(pos.x() + offsetPos.x())

            if offsetPos.y() < 0 and height > minHeight or offsetPos.y() > 0 and height < maxHeight:
                if offsetPos.y() < 0 and winPos.y() <= height or offsetPos.y() > 0 and winPos.y() >= height:
                    pos.setY(pos.y() + offsetPos.y())

            rect.setBottomLeft(pos)

        elif area == 33:
            # 右下
            pos = rect.bottomRight()

            if offsetPos.x() < 0 and width > minWidth or offsetPos.x() > 0 and width < maxWidth:
                if offsetPos.x() < 0 and winPos.x() <= width or offsetPos.x() > 0 and winPos.x() >= width:
                    pos.setX(pos.x() + offsetPos.x())

            if offsetPos.y() < 0 and height > minHeight or offsetPos.y() > 0 and height < maxHeight:
                if offsetPos.y() < 0 and winPos.y() <= height or offsetPos.y() > 0 and winPos.y() >= height:
                    pos.setY(pos.y() + offsetPos.y())

            rect.setBottomRight(pos)

        elif area == 12:
            # 中上
            if offsetPos.y() < 0 and height < maxHeight or offsetPos.y() > 0 and height > minHeight:
                if offsetPos.y() < 0 and winPos.y() <= 0 or offsetPos.y() > 0 and winPos.y() >= 0:
                    if (maxHeight - height) >= -offsetPos.y() and (height - minHeight) >= offsetPos.y():
                        rect.setTop(rect.top() + offsetPos.y())

        elif area == 21:
            # 中左
            if offsetPos.x() < 0 and width < maxWidth or offsetPos.x() > 0 and width > minWidth:
                if offsetPos.x() < 0 and winPos.x() <= 0 or offsetPos.x() > 0 and winPos.x() >= 0:
                    if (maxWidth - width) >= -offsetPos.x() and (width - minWidth) >= offsetPos.x():
                        rect.setLeft(rect.left() + offsetPos.x())

        elif area == 23:
            # 中右
            if offsetPos.x() < 0 and width > minWidth or offsetPos.x() > 0 and width < maxWidth:
                if offsetPos.x() < 0 and winPos.x() <= width or offsetPos.x() > 0 and winPos.x() >= width:
                    rect.setRight(rect.right() + offsetPos.x())

        elif area == 32:
            # 中下
            if offsetPos.y() < 0 and height > minHeight or offsetPos.y() > 0 and height < maxHeight:
                if offsetPos.y() < 0 and winPos.y() <= height or offsetPos.y() > 0 and winPos.y() >= height:
                    rect.setBottom(rect.bottom() + offsetPos.y())

        # 设置窗口几何属性（坐标，宽高）
        self.setGeometry(rect)

    def _change_cursor_icon(self, area):
        """改变光标在窗口边缘时的图片"""

        # 宽度固定时不应改变宽度
        if self.maximumWidth() == self.minimumWidth() and (area == 21 or area == 23):
            return None
        # 高度固定时不应改变高度
        if self.maximumHeight() == self.minimumHeight() and (area == 12 or area == 32):
            return None

        if area == 11 or area == 33:
            self.setCursor(Qt.SizeFDiagCursor)  # 倾斜光标
        elif area == 12 or area == 32:
            self.setCursor(Qt.SizeVerCursor)  # 垂直大小光标
        elif area == 13 or area == 31:
            self.setCursor(Qt.SizeBDiagCursor)  # 反倾斜光标
        elif area == 21 or area == 23:
            self.setCursor(Qt.SizeHorCursor)  # 水平大小光标
        else:
            self.setCursor(Qt.ArrowCursor)  # 默认光标

    def _compute_area(self, pos):
        """计算鼠标在窗口中的区域

        Args:
            pos: 鼠标相对于窗口的位置
        """
        margin = self.MARGIN  # 以此值为外边框宽度，划为九宫格区域

        # 定位列坐标
        if pos.x() < margin:
            line = 1
        elif pos.x() > self.width() - margin:
            line = 3
        else:
            line = 2
        # 定位行坐标并结合列坐标
        if pos.y() < margin:
            return 10 + line
        elif pos.y() > self.height() - margin:
            return 30 + line
        else:
            return 20 + line

    def _pos_percent(self, pos):
        """返回鼠标相对窗口的纵横百分比"""

        if pos.x() <= 0:
            x = 0
        else:
            x = round(pos.x() / self.width(), 3)

        if pos.y() <= 0:
            y = 0
        else:
            y = round(pos.y() / self.height(), 3)

        return x, y

    def _move(self, event):
        """实现窗口移动"""

        self._move_count += 1

        # 判断移动次数，减少灵敏度
        if self._move_count < self.MOVE_SENSITIVITY:
            return None

        # 最大化时需恢复普通大小，并按相对坐标移动到鼠标位置
        # 普通状态的宽高在窗口最大化之前获取，需借助标题栏的最大化按钮槽函数
        if self.isMaximized():
            relative = self._pos_percent(event.pos())
            self.titlebar.max_button_click()
            gpos = event.globalPos()
            width = self.last_geometry.width()  # 普通大小的几何属性
            height = self.last_geometry.height()
            x = gpos.x() - round(width * relative[0])
            y = gpos.y() - round(height * relative[1])

            self.setGeometry(x, y, width, height)
        else:
            # 鼠标移动偏移量
            offsetPos = event.globalPos() - self._posLast
            # ~ print(self.pos(), '->', self.pos() + offsetPos)
            self.move(self.pos() + offsetPos)

    # 重写事件
    def eventFilter(self, obj, event):
        """事件过滤器,用于解决鼠标进入其它控件后还原为标准鼠标样式"""
        if obj == self.frame and event.type() == QEvent.MouseMove:
            # print("mouse event")
            self.mouseMoveEvent(event)
        if isinstance(event, QEnterEvent) or obj != self.frame:
            self.setCursor(Qt.ArrowCursor)
        return QObject.eventFilter(self, obj, event)  # 交由其他控件处理

    def mousePressEvent(self, event):
        """重写继承的鼠标按住事件"""

        self.isPressed = True  # 判断是否按下
        self.press_button = event.button()  # 按下的鼠标按键
        self._area = self._compute_area(event.pos())  # 计算鼠标所在区域
        self._move_count = 0  # 鼠标移动计数，用于降低灵敏度
        self._posLast = event.globalPos()  # 当前坐标

        return QMainWindow.mousePressEvent(self, event)  # 交由原事件函数处理

    def mouseReleaseEvent(self, event):
        """重写继承的鼠标释放事件"""

        self.isPressed = False  # 重置按下状态
        self.press_button = None  # 清空按下的鼠标按键
        self._area = None  # 清空鼠标区域
        self._move_count = 0  # 清空移动计数
        self.setCursor(Qt.ArrowCursor)  # 还原光标图标

        return QMainWindow.mouseReleaseEvent(self, event)

    # def paintEvent(self, event):
    #     """由于是全透明背景窗口,重绘事件中绘制透明度为1的难以发现的边框,用于调整窗口大小"""
    #     super(FramelessWindow, self).paintEvent(event)
    #     painter = QPainter(self)
    #     painter.setPen(QPen(QColor(255, 255, 255, 1), 2 * self.Margins))
    #     painter.drawRect(self.rect())

    def mouseMoveEvent(self, event):
        """重写继承的鼠标移动事件，实现窗口移动及拖动改变窗口大小"""
        # print("mouse move")
        area = self._compute_area(event.globalPos() - self.pos())  # 计算鼠标区域

        # 调整窗口大小及移动
        if self.isPressed and self.press_button == Qt.LeftButton:
            if self._area == 22:
                if self.isMaximized():
                    self.showNormal()
                self._move(event)  # 调用移动窗口的函数
            elif not self.isMaximized():
                self._resize(event)
                # QLabel图片自适应大小 默认False
                # self.img3.setScaledContents(True)
            # 更新鼠标全局坐标
            self._posLast = event.globalPos()
            return None
        if not self.isPressed and not self.isMaximized():
            # 调整鼠标图标，按下鼠标后锁定状态
            self._change_cursor_icon(area)
            # print("触发了鼠标事件")

        return QMainWindow.mouseMoveEvent(self, event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
