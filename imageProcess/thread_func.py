import cv2
import numpy as np
from PyQt5 import QtGui
from PyQt5.QtCore import QThread


class MyThread(QThread):
    def __init__(self, way, main_window):
        super().__init__()
        self.way = way
        self.main_window = main_window
        self.l_mat = None
        self.r_mat = None
        self.name_harris1 = "img/harris1.jpg"
        self.name_harris2 = "img/harris2.jpg"
        self.name_shitomasi1 = "img/shitomasi1.jpg"
        self.name_shitomasi2 = "img/shitomasi2.jpg"
        self.name_sift1 = "img/sift1.jpg"
        self.name_sift2 = "img/sift2.jpg"
        self.name_surf1 = "img/surf1.jpg"
        self.name_surf2 = "img/surf2.jpg"
        self.name_orb1 = "img/orb1.jpg"
        self.name_orb2 = "img/orb2.jpg"
        self.name_forcematching = "img/forcematching.jpg"
        self.name_flannmatching = "img/flannmatching.jpg"
        self.name_stitching = "img/stitching.jpg"
        self.name_edgeprocess = "img/edgeprocess.jpg"
        self.func = None
        self.dst_img = None
        # 设置匹配点距离
        self.good_points_limited = 0.99

    def run(self):
        """
        将way映射为字符串
        :param way:
        """
        season_name = "way_" + str(self.way)
        fun = getattr(self, season_name, self.default)
        if self.get_image():
            fun()
        else:
            self.default()

    def get_image(self):
        # print(map1, map2)
        # 获取图片，并将原始图片转换成mat
        if self.main_window.src_img1 is not None and self.main_window.src_img2 is not None:
            l_image = self.main_window.src_img1.toImage()
            r_image = self.main_window.src_img2.toImage()
            l_ptr = l_image.constBits()
            r_ptr = r_image.constBits()
            l_ptr.setsize(l_image.byteCount())
            r_ptr.setsize(r_image.byteCount())
            mat1 = np.array(l_ptr).reshape(l_image.height(), l_image.width(), 4)
            mat2 = np.array(r_ptr).reshape(r_image.height(), r_image.width(), 4)
            self.l_mat = cv2.cvtColor(mat1, cv2.COLOR_BGR2RGB)
            self.r_mat = cv2.cvtColor(mat2, cv2.COLOR_BGR2RGB)
        else:
            # threading.Thread(target=self.main_window.msgCritical, args="Invalid file").start()
            return False
        return True

    def convert_mat2img(self):
        if self.func == 1:
            # 显示第一个窗口
            self.main_window.stackedWidget.setCurrentIndex(0)
            map_img1 = QtGui.QImage(self.l_mat.tobytes(),
                                    self.l_mat.shape[1], self.l_mat.shape[0],
                                    self.l_mat.shape[1] * 3, QtGui.QImage.Format_RGB888)
            map_img2 = QtGui.QImage(self.r_mat.tobytes(),
                                    self.r_mat.shape[1], self.r_mat.shape[0],
                                    self.r_mat.shape[1] * 3, QtGui.QImage.Format_RGB888)

            self.main_window.img1.clear()
            self.main_window.img1.setPixmap(QtGui.QPixmap.fromImage(map_img1)
                                            .scaled(self.main_window.img1.width(),
                                                    self.main_window.img1.height()))
            self.main_window.img2.clear()
            self.main_window.img2.setPixmap(QtGui.QPixmap.fromImage(map_img2)
                                            .scaled(self.main_window.img2.width(),
                                                    self.main_window.img2.height()))
            # cv2.imshow("test", self.l_mat)
            # cv2.waitKey(0)
            # self.main_window.img2.clear()
            # self.main_window.img2.setPixmap(QtGui.QPixmap(map_img2))
            # print(self.l_mat, "size:", sys.getsizeof(self.l_mat.data))
            # print(l_img.data)
            # cv2.imshow("harris", self.l_mat)
            # cv2.waitKey(0)
            # 转换成BGR以便存储到本地
            self.l_mat = cv2.cvtColor(self.l_mat, cv2.COLOR_RGB2BGR)
            self.r_mat = cv2.cvtColor(self.r_mat, cv2.COLOR_RGB2BGR)
        elif self.func == 2 and self.dst_img is not None:
            # 显示第二个窗口
            self.main_window.stackedWidget.setCurrentIndex(1)
            map_img3 = QtGui.QImage(self.dst_img.tobytes(),
                                    self.dst_img.shape[1], self.dst_img.shape[0],
                                    self.dst_img.shape[1] * 3, QtGui.QImage.Format_RGB888)

            self.main_window.img3.clear()
            self.main_window.img3.setPixmap(QtGui.QPixmap.fromImage(map_img3)
                                            .scaled(self.main_window.img3.width(),
                                                    self.main_window.img3.height()))
            # 转换成BGR以便存储到本地
            self.dst_img = cv2.cvtColor(self.dst_img, cv2.COLOR_RGB2BGR)
        elif self.func == 3:
            pass
        elif self.func == 4:
            pass
        else:
            self.main_window.msg_thread.msg = "错误操作！"
            self.main_window.msg_thread.start()

    @staticmethod
    def remove_black_edge(image):
        # median filter, to remove the noise interference that may be contained in the black edge
        img = cv2.medianBlur(image, 5)
        # adjust crop effect
        b = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY)
        binary_image = b[1]
        binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)

        x = binary_image.shape[0]
        y = binary_image.shape[1]
        edges_x = []
        edges_y = []
        for i in range(x):
            for j in range(y):
                if binary_image[i][j] == 255:
                    edges_x.append(i)
                    edges_y.append(j)

        left = min(edges_x)
        right = max(edges_x)
        width = right - left
        bottom = min(edges_y)
        top = max(edges_y)
        height = top - bottom

        pre1_picture = image[left:left + width, bottom:bottom + height]
        return pre1_picture

    def way_harris(self):
        self.main_window.msg_thread.msg = "正在进行Harris特征检测..."
        self.main_window.msg_thread.start()

        self.func = 1
        self.main_window.flag_flann_orb = False

        gray1 = cv2.cvtColor(self.l_mat, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(self.r_mat, cv2.COLOR_RGB2GRAY)
        # 返回的东西叫做角点响应. 每一个像素点都能计算出一个角点响应来.
        dst1 = cv2.cornerHarris(gray1, blockSize=2, ksize=3, k=0.04)  # harris角点检测
        dst2 = cv2.cornerHarris(gray2, blockSize=2, ksize=3, k=0.04)
        # print(dst)
        # print(dst.shape)

        self.l_mat[dst1 > 0.01 * dst1.max()] = [0, 0, 255]  # 显示角点 我们认为角点响应大于0.01倍的dst.max()就可以认为是角点了.
        self.r_mat[dst2 > 0.01 * dst2.max()] = [0, 0, 255]
        # 将mat转换回去并显示
        self.convert_mat2img()

        cv2.imwrite(self.name_harris1, self.l_mat)
        cv2.imwrite(self.name_harris2, self.r_mat)

        self.main_window.msg_thread.msg = "Harris特征检测已完成"
        self.main_window.msg_thread.start()

    def way_shitomasi(self):
        self.main_window.msg_thread.msg = "正在进行Shi-Tomasi特征检测..."
        self.main_window.msg_thread.start()

        self.func = 1
        self.main_window.flag_flann_orb = False

        max_corners = 1000
        ql = 0.01
        min_distance = 10

        gray1 = cv2.cvtColor(self.l_mat, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(self.r_mat, cv2.COLOR_RGB2GRAY)

        corners1 = cv2.goodFeaturesToTrack(gray1, max_corners, ql, min_distance)
        corners1 = np.int0(corners1)
        corners2 = cv2.goodFeaturesToTrack(gray2, max_corners, ql, min_distance)
        corners2 = np.int0(corners2)

        for i in corners1:  # Shi-Tomasi绘制角点
            x, y = i.ravel()
            cv2.circle(self.l_mat, (x, y), 3, (255, 0, 0), -1)
        for i in corners2:  # Shi-Tomasi绘制角点
            x, y = i.ravel()
            cv2.circle(self.r_mat, (x, y), 3, (255, 0, 0), -1)

        # 将mat转换回去并显示
        self.convert_mat2img()

        cv2.imwrite(self.name_shitomasi1, self.l_mat)
        cv2.imwrite(self.name_shitomasi2, self.r_mat)

        self.main_window.msg_thread.msg = "Shi-Tomasi特征检测已完成"
        self.main_window.msg_thread.start()

    def way_sift(self):
        self.main_window.msg_thread.msg = "正在进行SIFT特征检测..."
        self.main_window.msg_thread.start()

        self.func = 1
        self.main_window.flag_flann_orb = False
        # 创建sift对象
        sift = cv2.xfeatures2d.SIFT_create()

        # 把关键点和描述子一起检测出来,kp表示关键点 des表示描述子
        self.main_window.key_point1, self.main_window.describe1 = sift.detectAndCompute(self.l_mat, None)
        self.main_window.key_point2, self.main_window.describe2 = sift.detectAndCompute(self.r_mat, None)
        # print(kp)
        # print(des)
        # print(des.shape)
        # 绘制关键点
        self.l_mat = cv2.drawKeypoints(self.l_mat, self.main_window.key_point1, self.l_mat, [255, 0, 0])
        self.r_mat = cv2.drawKeypoints(self.r_mat, self.main_window.key_point2, self.r_mat, [255, 0, 0])

        # 将mat转换回去并显示
        self.convert_mat2img()

        cv2.imwrite(self.name_sift1, self.l_mat)
        cv2.imwrite(self.name_sift2, self.r_mat)

        self.main_window.msg_thread.msg = "SIFT特征检测已完成"
        self.main_window.msg_thread.start()

    def way_surf(self):
        self.main_window.msg_thread.msg = "正在进行SURF特征检测..."
        self.main_window.msg_thread.start()

        self.func = 1
        self.main_window.flag_flann_orb = False
        # 创建SURF对象
        surf = cv2.xfeatures2d.SURF_create()

        # 把关键点和描述子一起检测出来,kp表示关键点 des表示描述子
        self.main_window.key_point1, self.main_window.describe1 = surf.detectAndCompute(self.l_mat, None)
        self.main_window.key_point2, self.main_window.describe2 = surf.detectAndCompute(self.r_mat, None)
        # print(kp)
        # print(des)
        # print(des.shape)

        # 绘制关键点
        self.l_mat = cv2.drawKeypoints(self.l_mat, self.main_window.key_point1, self.l_mat, [255, 0, 0])
        self.r_mat = cv2.drawKeypoints(self.r_mat, self.main_window.key_point2, self.r_mat, [255, 0, 0])

        # 将mat转换回去并显示
        self.convert_mat2img()

        cv2.imwrite(self.name_surf1, self.l_mat)
        cv2.imwrite(self.name_surf2, self.r_mat)

        self.main_window.msg_thread.msg = "SURF特征检测已完成"
        self.main_window.msg_thread.start()

    def way_orb(self):
        self.main_window.msg_thread.msg = "正在进行ORB特征检测..."
        self.main_window.msg_thread.start()

        self.func = 1
        self.main_window.flag_flann_orb = True
        # 创建ORB对象
        orb = cv2.ORB_create()

        # 把关键点和描述子一起检测出来,kp表示关键点 des表示描述子
        self.main_window.key_point1, self.main_window.describe1 = orb.detectAndCompute(self.l_mat, None)
        self.main_window.key_point2, self.main_window.describe2 = orb.detectAndCompute(self.r_mat, None)
        # print(kp)
        # print(des)
        # print(des.shape)

        # 绘制关键点
        self.l_mat = cv2.drawKeypoints(self.l_mat, self.main_window.key_point1, self.l_mat, [255, 0, 0])
        self.r_mat = cv2.drawKeypoints(self.r_mat, self.main_window.key_point2, self.r_mat, [255, 0, 0])

        # 将mat转换回去并显示
        self.convert_mat2img()

        cv2.imwrite(self.name_orb1, self.l_mat)
        cv2.imwrite(self.name_orb2, self.r_mat)

        self.main_window.msg_thread.msg = "ORB特征检测已完成"
        self.main_window.msg_thread.start()

    def way_forcematching(self):
        self.func = 2
        if self.main_window.key_point1 is not None and self.main_window.describe1 is not None \
                and self.main_window.key_point2 is not None and self.main_window.describe2 is not None:
            self.main_window.msg_thread.msg = "暴力特征匹配正在进行中..."
            self.main_window.msg_thread.start()
            # 暴力特征匹配
            # NORM_L1, L1距离, 即绝对值, SIFT和SURF使用. NORM_L2, L2距离,
            # 默认值. 即平方. SIFT和SURF使用
            bf = cv2.BFMatcher()
            self.main_window.matches = bf.match(self.main_window.describe1, self.main_window.describe2)

            # for match in self.main_window.matches:
            #     print("forcematch", match)

            # 绘制匹配特征
            self.dst_img = cv2.drawMatches(self.l_mat, self.main_window.key_point1,
                                             self.r_mat, self.main_window.key_point2,
                                             self.main_window.matches, None)

            # 将mat转换回去并显示
            self.convert_mat2img()
            cv2.imwrite(self.name_forcematching, self.dst_img)
            self.main_window.msg_thread.msg = "暴力特征匹配已完成"
            self.main_window.msg_thread.start()
            self.main_window.bfmatch_flag = True
        else:
            self.main_window.msg_thread.msg = "必须先进行特征检测（sift，surf，orb）"
            self.main_window.msg_thread.start()
            self.main_window.bfmatch_flag = False

    def way_flannmatching(self):
        self.func = 2
        self.main_window.bfmatch_flag = False

        if self.main_window.key_point1 is not None and self.main_window.describe1 is not None \
                and self.main_window.key_point2 is not None and self.main_window.describe2 is not None:
            self.main_window.msg_thread.msg = "FLANN特征匹配正在进行..."
            self.main_window.msg_thread.start()
            # FLANN特征匹配
            if self.main_window.flag_flann_orb:
                print("orb flann")
                # 创建orb匹配器
                index_params = dict(algorithm=6, trees=5)  # FLANN_INDEX_LSH算法 用 6 表示
            else:
                print("sift surf flann")
                # 创建sift,surf匹配器
                index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE算法 用 1 表示

            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)

            # 对描述子进行匹配计算
            # 返回的是第一张图和第二张图的匹配点.
            self.main_window.matches = flann.knnMatch(self.main_window.describe1, self.main_window.describe2, k=2)
            # for match in self.main_window.matches:
            #     print("flannmatch", match)

            good = []
            for i, pair in enumerate(self.main_window.matches):
                try:
                    m, n = pair
                    if m.distance < 0.7 * n.distance:
                        good.append(m)

                except ValueError:
                    pass

            self.dst_img = cv2.drawMatchesKnn(self.l_mat, self.main_window.key_point1, self.r_mat, self.main_window.key_point2, [good], None)

            # 将mat转换回去并显示
            self.convert_mat2img()
            cv2.imwrite(self.name_flannmatching, self.dst_img)

            self.main_window.msg_thread.msg = "FLANN特征匹配已完成"
            self.main_window.msg_thread.start()
            # self.main_window.matches = None
        else:
            self.main_window.msg_thread.msg = "必须先进行特征检测（sift，surf，orb）"
            self.main_window.msg_thread.start()

    def way_stitching(self):
        self.func = 2

        if self.main_window.matches is not None and self.main_window.bfmatch_flag:
            # sorted by distance
            self.main_window.matches = sorted(self.main_window.matches, key=lambda x: x.distance)

            # pick up the good keypoints
            good_points = []
            for j in range(len(self.main_window.matches) - 1):
                if self.main_window.matches[j].distance < self.good_points_limited \
                        * self.main_window.matches[j + 1].distance:
                    good_points.append(self.main_window.matches[j])

            # the index of the keypoints and descriptors
            src_pts = np.float32([self.main_window.key_point1[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
            dst_pts = np.float32([self.main_window.key_point2[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)

            # generate transformation matrix
            M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RHO)

            # get the height and width of the original images
            h1, w1, p1 = self.l_mat.shape
            h2, w2, p2 = self.r_mat.shape
            h = np.maximum(h1, h2)

            move_dis = int(np.maximum(dst_pts[0][0][0], src_pts[0][0][0]))
            self.main_window.img_transform = cv2.warpPerspective(self.r_mat, M, (w1 + w2 - move_dis, h))

            M1 = np.float32([[1, 0, 0], [0, 1, 0]])
            self.main_window.dst_img1 = cv2.warpAffine(self.l_mat, M1, (w1 + w2 - move_dis, h))

            self.dst_img = cv2.add(self.main_window.dst_img1, self.main_window.img_transform)

            # 将mat转换回去并显示
            self.convert_mat2img()
            cv2.imwrite(self.name_stitching, self.dst_img)

            self.main_window.msg_thread.msg = "图像拼接已完成"
            self.main_window.msg_thread.start()
        else:
            self.main_window.msg_thread.msg = "必须先进行特征匹配（ForceMatching）"
            self.main_window.msg_thread.start()

    def way_edgeprocess(self):
        self.func = 2
        if self.main_window.dst_img1 is not None and self.main_window.img_transform is not None:
            self.main_window.msg_thread.msg = "拼接缝正在处理..."
            self.main_window.msg_thread.start()
            self.dst_img = np.maximum(self.main_window.dst_img1, self.main_window.img_transform)
            # self.dst_img = self.remove_black_edge(self.dst_img)

            # 将mat转换回去并显示
            self.convert_mat2img()
            cv2.imwrite(self.name_edgeprocess, self.dst_img)

            self.main_window.msg_thread.msg = "拼接缝处理已完成"
            self.main_window.msg_thread.start()
        else:
            self.main_window.msg_thread.msg = "必须先进行图像拼接"
            self.main_window.msg_thread.start()

    def default(self):
        self.main_window.msg_thread.msg = "No such way!"
        self.main_window.msg_thread.start()
