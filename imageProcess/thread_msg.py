import time

from PyQt5.QtCore import QThread, QDateTime


class ThreadMsg(QThread):
    def __init__(self, main_window):
        super().__init__()
        self.msg = None
        self.main_window = main_window
        self.time = None

    def run(self):
        self.time = QDateTime.currentDateTime().toString("[yyyy.MM.dd hh:mm:ss ddd] ")
        self.main_window.msg_signal.emit("<font color=\"#f5f5f5\">" + self.time + self.msg + "</font> ")
