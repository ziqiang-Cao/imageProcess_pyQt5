# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'window.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1263, 582)
        font = QtGui.QFont()
        font.setFamily("Consolas")
        MainWindow.setFont(font)
        MainWindow.setMouseTracking(False)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setMouseTracking(False)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setMouseTracking(True)
        self.frame.setStyleSheet("#frame{\n"
"    background-color: rgba(86, 88, 93, 1);\n"
"    border-radius: 30px;\n"
"}")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.frame)
        self.verticalLayout.setContentsMargins(5, 5, 5, 5)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.frame_2 = QtWidgets.QFrame(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.frame_2.sizePolicy().hasHeightForWidth())
        self.frame_2.setSizePolicy(sizePolicy)
        self.frame_2.setMouseTracking(True)
        self.frame_2.setStyleSheet("#frame_2{\n"
"    background-color: rgb(95, 96, 100);\n"
"    border-top-left-radius: 30px;\n"
"    border-top-right-radius: 30px;\n"
"}")
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.gridLayout = QtWidgets.QGridLayout(self.frame_2)
        self.gridLayout.setContentsMargins(8, 0, 8, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.frame_4 = QtWidgets.QFrame(self.frame_2)
        self.frame_4.setMouseTracking(False)
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.frame_4)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_2 = QtWidgets.QLabel(self.frame_4)
        self.label_2.setMinimumSize(QtCore.QSize(171, 51))
        self.label_2.setStyleSheet("image: url(:/icon/icon/logo.png);")
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_4.addWidget(self.label_2)
        self.widget_5 = QtWidgets.QWidget(self.frame_4)
        self.widget_5.setObjectName("widget_5")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.widget_5)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.pushButton_featExtraction = QtWidgets.QPushButton(self.widget_5)
        self.pushButton_featExtraction.setMinimumSize(QtCore.QSize(120, 28))
        self.pushButton_featExtraction.setStyleSheet("QPushButton{\n"
"    font: 25 10pt \"Microsoft YaHei\";\n"
"    font-weight: bold;\n"
"    color: rgb(218, 218, 218);\n"
"    border-radius: 10px;  \n"
"    border: 2px groove gray;\n"
"    border-style: outset;\n"
"}\n"
"QPushButton::hover{\n"
"    padding-bottom: 5px;\n"
"}")
        self.pushButton_featExtraction.setObjectName("pushButton_featExtraction")
        self.horizontalLayout_2.addWidget(self.pushButton_featExtraction)
        self.pushButton_featMatching = QtWidgets.QPushButton(self.widget_5)
        self.pushButton_featMatching.setMinimumSize(QtCore.QSize(120, 28))
        self.pushButton_featMatching.setStyleSheet("QPushButton{\n"
"    font: 25 10pt \"Microsoft YaHei\";\n"
"    font-weight: bold;\n"
"    color: rgb(218, 218, 218);\n"
"    border-radius: 10px;  \n"
"    border: 2px groove gray;\n"
"    border-style: outset;\n"
"}\n"
"QPushButton:hover{\n"
"    padding-bottom: 5px;\n"
"}")
        self.pushButton_featMatching.setObjectName("pushButton_featMatching")
        self.horizontalLayout_2.addWidget(self.pushButton_featMatching)
        self.pushButton_imgStitching = QtWidgets.QPushButton(self.widget_5)
        self.pushButton_imgStitching.setMinimumSize(QtCore.QSize(120, 28))
        self.pushButton_imgStitching.setStyleSheet("QPushButton{\n"
"    font: 25 10pt \"Microsoft YaHei\";\n"
"    font-weight: bold;\n"
"    color: rgb(218, 218, 218);\n"
"    border-radius: 10px;  \n"
"    border: 2px groove gray;\n"
"    border-style: outset;\n"
"}\n"
"QPushButton:hover{\n"
"    padding-bottom: 5px;\n"
"}")
        self.pushButton_imgStitching.setObjectName("pushButton_imgStitching")
        self.horizontalLayout_2.addWidget(self.pushButton_imgStitching)
        self.pushButton_edgeProcess = QtWidgets.QPushButton(self.widget_5)
        self.pushButton_edgeProcess.setMinimumSize(QtCore.QSize(120, 28))
        self.pushButton_edgeProcess.setStyleSheet("QPushButton{\n"
"    font: 25 10pt \"Microsoft YaHei\";\n"
"    font-weight: bold;\n"
"    color: rgb(218, 218, 218);\n"
"    border-radius: 10px;  \n"
"    border: 2px groove gray;\n"
"    border-style: outset;\n"
"}\n"
"QPushButton:hover{\n"
"    padding-bottom: 5px;\n"
"}")
        self.pushButton_edgeProcess.setObjectName("pushButton_edgeProcess")
        self.horizontalLayout_2.addWidget(self.pushButton_edgeProcess)
        self.horizontalLayout_4.addWidget(self.widget_5)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem)
        self.pushButton_minimize = QtWidgets.QPushButton(self.frame_4)
        self.pushButton_minimize.setStyleSheet("QPushButton{\n"
"    border:none;\n"
"}\n"
"QPushButton:hover{\n"
"    padding-bottom: 5px;\n"
"}")
        self.pushButton_minimize.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icon/icon/最小化.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_minimize.setIcon(icon)
        self.pushButton_minimize.setObjectName("pushButton_minimize")
        self.horizontalLayout_4.addWidget(self.pushButton_minimize)
        self.pushButton_maxmize = QtWidgets.QPushButton(self.frame_4)
        self.pushButton_maxmize.setStyleSheet("QPushButton{\n"
"    border:none;\n"
"}\n"
"QPushButton:hover{\n"
"    padding-bottom: 5px;\n"
"}")
        self.pushButton_maxmize.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/icon/icon/正方形.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_maxmize.setIcon(icon1)
        self.pushButton_maxmize.setObjectName("pushButton_maxmize")
        self.horizontalLayout_4.addWidget(self.pushButton_maxmize)
        self.pushButton_close = QtWidgets.QPushButton(self.frame_4)
        self.pushButton_close.setStyleSheet("QPushButton{\n"
"    border:none;\n"
"}\n"
"QPushButton:hover{\n"
"    padding-bottom: 5px;\n"
"}")
        self.pushButton_close.setText("")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/icon/icon/关闭.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_close.setIcon(icon2)
        self.pushButton_close.setObjectName("pushButton_close")
        self.horizontalLayout_4.addWidget(self.pushButton_close)
        self.gridLayout.addWidget(self.frame_4, 0, 0, 1, 1)
        self.verticalLayout.addWidget(self.frame_2)
        self.frame_3 = QtWidgets.QFrame(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(17)
        sizePolicy.setHeightForWidth(self.frame_3.sizePolicy().hasHeightForWidth())
        self.frame_3.setSizePolicy(sizePolicy)
        self.frame_3.setMouseTracking(True)
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.frame_3)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setSpacing(0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.widget_3 = QtWidgets.QWidget(self.frame_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_3.sizePolicy().hasHeightForWidth())
        self.widget_3.setSizePolicy(sizePolicy)
        self.widget_3.setStyleSheet("QWidget{\n"
"    background-color: rgba(95, 95, 95, 200);\n"
"    border: 0px solid #42adff;\n"
"    border-left: 0px solid rgba(200, 200, 200,100);\n"
"    border-right: 0px solid rgba(29, 83, 185, 255);\n"
"    border-bottom-left-radius: 30px;\n"
"    border-bottom-right-radius: 30px;\n"
"}")
        self.widget_3.setObjectName("widget_3")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.widget_3)
        self.gridLayout_5.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.widget_4 = QtWidgets.QWidget(self.widget_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.widget_4.sizePolicy().hasHeightForWidth())
        self.widget_4.setSizePolicy(sizePolicy)
        self.widget_4.setStyleSheet("QWidget{\n"
"    border-radius: 0px;\n"
"}")
        self.widget_4.setObjectName("widget_4")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.widget_4)
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        spacerItem1 = QtWidgets.QSpacerItem(225, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem1)
        self.pushButton_openFile1 = QtWidgets.QPushButton(self.widget_4)
        self.pushButton_openFile1.setStyleSheet("QPushButton{\n"
"    font: 25 10pt \"Microsoft YaHei\";\n"
"    font-weight: bold;\n"
"    color: rgb(218, 218, 218);\n"
"    border-radius: 00px;  \n"
"    border: none;\n"
"    border-style: outset;\n"
"}\n"
"QPushButton:hover{\n"
"    padding-bottom: 5px;\n"
"}")
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/icon/icon/打开.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_openFile1.setIcon(icon3)
        self.pushButton_openFile1.setObjectName("pushButton_openFile1")
        self.horizontalLayout_5.addWidget(self.pushButton_openFile1)
        spacerItem2 = QtWidgets.QSpacerItem(226, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem2)
        spacerItem3 = QtWidgets.QSpacerItem(226, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem3)
        self.pushButton_openFile2 = QtWidgets.QPushButton(self.widget_4)
        self.pushButton_openFile2.setStyleSheet("QPushButton{\n"
"    font: 25 10pt \"Microsoft YaHei\";\n"
"    font-weight: bold;\n"
"    color: rgb(218, 218, 218);\n"
"    border-radius: 00px;  \n"
"    border: none;\n"
"    border-style: outset;\n"
"}\n"
"QPushButton:hover{\n"
"    padding-bottom: 5px;\n"
"}")
        self.pushButton_openFile2.setIcon(icon3)
        self.pushButton_openFile2.setObjectName("pushButton_openFile2")
        self.horizontalLayout_5.addWidget(self.pushButton_openFile2)
        spacerItem4 = QtWidgets.QSpacerItem(225, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem4)
        self.gridLayout_5.addWidget(self.widget_4, 0, 0, 1, 1)
        self.stackedWidget = QtWidgets.QStackedWidget(self.widget_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(15)
        sizePolicy.setHeightForWidth(self.stackedWidget.sizePolicy().hasHeightForWidth())
        self.stackedWidget.setSizePolicy(sizePolicy)
        self.stackedWidget.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.stackedWidget.setObjectName("stackedWidget")
        self.page = QtWidgets.QWidget()
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.page.sizePolicy().hasHeightForWidth())
        self.page.setSizePolicy(sizePolicy)
        self.page.setObjectName("page")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.page)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.splitter = QtWidgets.QSplitter(self.page)
        self.splitter.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.splitter.sizePolicy().hasHeightForWidth())
        self.splitter.setSizePolicy(sizePolicy)
        self.splitter.setStyleSheet("#splitter::handle{\n"
"    background-color: 1px solid  rgba(200, 200, 200,100);\n"
"}")
        self.splitter.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.splitter.setLineWidth(10)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setHandleWidth(1)
        self.splitter.setObjectName("splitter")
        self.img1 = Label_click_Mouse(self.splitter)
        self.img1.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.img1.sizePolicy().hasHeightForWidth())
        self.img1.setSizePolicy(sizePolicy)
        self.img1.setMinimumSize(QtCore.QSize(200, 0))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(36)
        self.img1.setFont(font)
        self.img1.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.img1.setMouseTracking(True)
        self.img1.setStyleSheet("QLabel{\n"
"    border-radius: 0px;\n"
"}\n"
"\n"
"")
        self.img1.setText("")
        self.img1.setScaledContents(True)
        self.img1.setAlignment(QtCore.Qt.AlignCenter)
        self.img1.setObjectName("img1")
        self.img2 = Label_click_Mouse(self.splitter)
        self.img2.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.img2.sizePolicy().hasHeightForWidth())
        self.img2.setSizePolicy(sizePolicy)
        self.img2.setMinimumSize(QtCore.QSize(200, 0))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(36)
        self.img2.setFont(font)
        self.img2.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.img2.setMouseTracking(True)
        self.img2.setStyleSheet("QLabel{\n"
"    border-radius: 0px;\n"
"}\n"
"\n"
"")
        self.img2.setText("")
        self.img2.setScaledContents(True)
        self.img2.setAlignment(QtCore.Qt.AlignCenter)
        self.img2.setObjectName("img2")
        self.gridLayout_2.addWidget(self.splitter, 0, 0, 1, 1)
        self.stackedWidget.addWidget(self.page)
        self.page_2 = QtWidgets.QWidget()
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.page_2.sizePolicy().hasHeightForWidth())
        self.page_2.setSizePolicy(sizePolicy)
        self.page_2.setObjectName("page_2")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.page_2)
        self.gridLayout_3.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.img3 = QtWidgets.QLabel(self.page_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.img3.sizePolicy().hasHeightForWidth())
        self.img3.setSizePolicy(sizePolicy)
        self.img3.setText("")
        self.img3.setScaledContents(True)
        self.img3.setObjectName("img3")
        self.gridLayout_3.addWidget(self.img3, 0, 0, 1, 1)
        self.stackedWidget.addWidget(self.page_2)
        self.gridLayout_5.addWidget(self.stackedWidget, 1, 0, 1, 1)
        self.widget_7 = QtWidgets.QWidget(self.widget_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(4)
        sizePolicy.setHeightForWidth(self.widget_7.sizePolicy().hasHeightForWidth())
        self.widget_7.setSizePolicy(sizePolicy)
        self.widget_7.setStyleSheet("QWidget{    \n"
"    border-bottom-left-radius: 30px;\n"
"    border-bottom-right-radius: 30px;\n"
"}")
        self.widget_7.setObjectName("widget_7")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.widget_7)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label = QtWidgets.QLabel(self.widget_7)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setMinimumSize(QtCore.QSize(144, 20))
        self.label.setStyleSheet("QLabel\n"
"{\n"
"    border:none;\n"
"    font: 25 10pt \"Microsoft YaHei\";\n"
"    font-weight: bold;\n"
"         border-radius:0px;\n"
"        background:rgba(66, 195, 255, 0);\n"
"color: rgb(218, 218, 218);\n"
"}\n"
"")
        self.label.setObjectName("label")
        self.verticalLayout_2.addWidget(self.label)
        self.scrollArea = QtWidgets.QScrollArea(self.widget_7)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(5)
        sizePolicy.setHeightForWidth(self.scrollArea.sizePolicy().hasHeightForWidth())
        self.scrollArea.setSizePolicy(sizePolicy)
        self.scrollArea.setStyleSheet("QScrollArea{\n"
"    border:none;\n"
"    background-color: rgba(100, 100, 100, 0);\n"
"    border-radius: 300px;\n"
"}")
        self.scrollArea.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.scrollArea.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 1251, 85))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.scrollAreaWidgetContents.sizePolicy().hasHeightForWidth())
        self.scrollAreaWidgetContents.setSizePolicy(sizePolicy)
        self.scrollAreaWidgetContents.setStyleSheet("QWidget{\n"
"    border:none;\n"
"    background-color: rgba(100, 100, 100, 0%);\n"
"    border-radius: 30px;\n"
"}")
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.scrollAreaWidgetContents)
        self.gridLayout_4.setContentsMargins(0, 0, 5, 0)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.textBrowser = QtWidgets.QTextBrowser(self.scrollAreaWidgetContents)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.textBrowser.sizePolicy().hasHeightForWidth())
        self.textBrowser.setSizePolicy(sizePolicy)
        self.textBrowser.setStyleSheet("QTextBrowser{\n"
"    border:none;\n"
"    background-color: rgb(100, 100, 100);\n"
"    border-radius: 30px;  \n"
"}\n"
"QScrollBar:vertical\n"
"{\n"
"    width:15px;\n"
"    background-color: rgb(100, 100, 100);\n"
"    margin-bottom:15px;\n"
"    margin-top:15px;\n"
"}\n"
"QScrollBar::add-line:vertical {\n"
"                  height: 15px;\n"
"                  subcontrol-origin:margin;\n"
"              }\n"
"\n"
"QScrollBar::up-arrow:vertical {\n"
"                 subcontrol-origin: margin;\n"
"                 height: 10px;\n"
"                 border:10 10 10 10;\n"
"                 \n"
"              }\n"
"QScrollBar::sub-line:vertical {\n"
"                  height: 15px;\n"
"                  subcontrol-origin:margin;\n"
"              }\n"
"\n"
" QScrollBar::down-arrow:vertical {\n"
"                 subcontrol-origin: margin;\n"
"                 height: 10px;\n"
"                 border:10 10 10 10;\n"
"              }")
        self.textBrowser.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.textBrowser.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textBrowser.setObjectName("textBrowser")
        self.gridLayout_4.addWidget(self.textBrowser, 0, 0, 1, 1)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.verticalLayout_2.addWidget(self.scrollArea)
        self.gridLayout_5.addWidget(self.widget_7, 2, 0, 1, 1)
        self.horizontalLayout_3.addWidget(self.widget_3)
        self.verticalLayout.addWidget(self.frame_3)
        self.horizontalLayout.addWidget(self.frame)
        MainWindow.setCentralWidget(self.centralwidget)
        self.actionopen = QtWidgets.QAction(MainWindow)
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(":/img/icon/file.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionopen.setIcon(icon4)
        self.actionopen.setObjectName("actionopen")
        self.actionexport = QtWidgets.QAction(MainWindow)
        font = QtGui.QFont()
        font.setFamily("Consolas")
        self.actionexport.setFont(font)
        self.actionexport.setObjectName("actionexport")

        self.retranslateUi(MainWindow)
        self.stackedWidget.setCurrentIndex(0)
        self.pushButton_close.clicked.connect(MainWindow.close)
        self.pushButton_maxmize.clicked.connect(MainWindow.showMaximized)
        self.pushButton_minimize.clicked.connect(MainWindow.showMinimized)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton_featExtraction.setText(_translate("MainWindow", "特征检测"))
        self.pushButton_featMatching.setText(_translate("MainWindow", "特征匹配"))
        self.pushButton_imgStitching.setText(_translate("MainWindow", "图像拼接"))
        self.pushButton_edgeProcess.setText(_translate("MainWindow", "拼接处理"))
        self.pushButton_openFile1.setText(_translate("MainWindow", "图片1"))
        self.pushButton_openFile2.setText(_translate("MainWindow", "图片2"))
        self.label.setText(_translate("MainWindow", "数据统计："))
        self.actionopen.setText(_translate("MainWindow", "open"))
        self.actionexport.setText(_translate("MainWindow", "导出"))
from MouseLabel import Label_click_Mouse
import srcImg_rc