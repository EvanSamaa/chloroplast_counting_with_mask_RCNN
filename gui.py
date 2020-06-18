import wx
import cv2
import numpy as np
from matplotlib import pyplot as plt
# from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton
from PyQt5.QtWidgets import *
from PyQt5 import QtGui, QtCore
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
import sys
from qimage2ndarray import array2qimage
from PIL import Image
class Interactive_widget(QWidget):
    def __init__(self, img=np.zeros((500, 500, 3)), parent=None):
        # class variables
        super(Interactive_widget, self).__init__(parent=parent)
        self.windowSize = 500
        self.scale = 0
        self.img = None
        self.mask = None
        self.qImg = None
        self.qImg2 = None
        self.contours = []
        # drawing variables


        if len(img.shape) == 2:
            img = img.reshape((img.shape[0], img.shape[1], 1))
        height, width, channel = img.shape
        self.scale = (self.windowSize / max(height, width))
        self.img = img # np image has dimension (height, width, channels)
        self.qImg = array2qimage(self.img)
        self.initUI()
    def initUI(self):
        name = "name"
        self.setWindowTitle(name)
        self.resize(self.windowSize + 100, self.windowSize)
        self.img_field = QLabel(self)
        self.img_field.move(0, 0)
        self.img_field2 = QLabel(self)
        self.img_field2.move(self.windowSize + 10, 0)
        self.display_image1()
    def display_image1(self, event=None):
        if self.qImg:
            a_width = int(self.qImg.width()*self.scale)
            a_height = int(self.qImg.height()*self.scale)
            result = self.qImg.scaled(a_width, a_height, QtCore.Qt.KeepAspectRatioByExpanding, QtCore.Qt.SmoothTransformation)
            self.img_field.setPixmap(QPixmap(result))
    def display_image2(self, event=None):
        if self.qImg2:
            a_width = self.qImg2.width()*self.scale
            a_height = self.qImg2.height()*self.scale
            result = self.qImg2.scaled(a_width, a_height, QtCore.Qt.KeepAspectRatioByExpanding, QtCore.Qt.SmoothTransformation)
            self.img_field2.setPixmap(QPixmap(result))
    def mouseReleaseEvent(self, QMouseEvent):
        print(QMouseEvent.pos())
    def mousePressEvent(self, event):  # click
        print(event.pos())
    def mouseMoveEvent(self, event):  # move
        print(event.pos())

class Interactive_opencv(QMainWindow):
    def __init__(self, img=np.zeros((500, 500, 3))):
        super().__init__()
        self.image_frame = QLabel()
        self.img = img
        self.qimage = None
        # for drawing
        self.drawing = False  # true if mouse is pressed
        self.mode = True  # if True, draw rectangle. Press 'm' to toggle to curve
        self.former_x = -1
        self.former_y = -1
        self.current_former_x = 0
        self.current_former_y = 0
        self.starting_pixel = (-1, -1)
        self.setCentralWidget(self.image_frame)
        self.show()
        self.start_draw()
    def display(self):
        img = np.transpose(self.img, (1, 0, 2)).copy()
        self.qimage = QtGui.QImage(img, img.shape[1], img.shape[0],
                                  QtGui.QImage.Format_RGB888).rgbSwapped()
        self.image_frame.setPixmap(QtGui.QPixmap.fromImage(self.qimage))
    def freehand_draw(self, event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.starting_pixel = (x, y)
            self.drawing = True
            self.current_former_x = x
            self.current_former_y = y
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing == True:
                self.former_x = self.current_former_x
                self.former_y = self.current_former_y
                self.current_former_x = x
                self.current_former_y = y
                cv2.line(self.img, (self.current_former_x, self.current_former_y), (self.former_x, self.former_y), (0, 0, 255), 1)
        elif event == cv2.EVENT_LBUTTONUP:
            cv2.line(self.img, self.starting_pixel, (x, y), (0, 0, 255), 1)
            self.drawing = False
    def start_draw(self):
        self.setMouseTracking(True)
        cv2.setMouseCallback('image', self.freehand_draw)
        while (1):
            self.display()
            key = cv2.waitKey(20)
            if key == ord("b"):
                break
class Main_window(QMainWindow):
    def __init__(self):
        self.app = QApplication(sys.argv)
        super(Main_window, self).__init__()
        self.open_windows = []

        self.setGeometry(50, 50, 300, 120)
        self.setWindowTitle("Chloroplast Quantifier")
        self.setWindowIcon(QtGui.QIcon('pythonlogo.png'))

        self.menu()
        self.home()
        sys.exit(self.app.exec_())
    def menu(self):
        menu_open_file = QAction("Open File", self)
        menu_open_file.setShortcut("Ctrl+O")
        menu_open_file.setStatusTip('Open an image file')
        menu_open_file.triggered.connect(self.open_file)

        menu_open_next_file = QAction("Open Next", self)
        menu_open_next_file.setShortcut("Ctrl+Shift+O")
        menu_open_next_file.setStatusTip('Open the next image file')
        menu_open_next_file.triggered.connect(self.open_next)

        menu_save_file = QAction("Save File", self)
        menu_save_file.setShortcut("Ctrl+S")
        menu_save_file.setStatusTip('Save the File')
        menu_save_file.triggered.connect(self.save_file)

        menu_close_file = QAction("Close File", self)
        menu_close_file.setShortcut("Ctrl+W")
        menu_close_file.setStatusTip('Close the File')
        menu_close_file.triggered.connect(self.close_file)

        menu_undo = QAction("Undo", self)
        menu_undo.setShortcut("Ctrl+Z")
        menu_undo.setStatusTip('Undo something')
        menu_undo.triggered.connect(self.undo_action)

        menu_clear = QAction("Clear", self)
        menu_clear.setStatusTip('Clear All Drawings')
        menu_clear.triggered.connect(self.clear_drawings)

        menu_global_scale = QAction("Set Global Scale", self)
        menu_global_scale.setStatusTip('Set scale for all images opened this session')
        menu_global_scale.triggered.connect(self.set_global_scale)

        menu_measure = QAction("Measure", self)
        menu_measure.setShortcut("Ctrl+M")
        menu_undo.setStatusTip('measure enclosed area inside image')
        menu_measure.triggered.connect(self.measure)
        self.statusBar()
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('&File')
        editMenu = mainMenu.addMenu('&Edit')
        analyzeMenu = mainMenu.addMenu('&Analyze')
        fileMenu.addAction(menu_open_file)
        fileMenu.addAction(menu_open_next_file)
        fileMenu.addAction(menu_save_file)
        fileMenu.addAction(menu_close_file)
        editMenu.addAction(menu_undo)
        analyzeMenu.addAction(menu_global_scale)
    def home(self):
        self.tool_bar_layout = QVBoxLayout()
        win = QWidget()
        self.fh_btn = QPushButton("Freehand \nDrawing", self)
        self.fh_btn.clicked.connect(self.drawing_mode_free_hand)
        self.fh_btn.resize(self.fh_btn.minimumSizeHint())

        self.rc_btn = QPushButton("Draw \nRectangle", self)
        self.rc_btn.clicked.connect(self.drawing_mode_rectangle)
        self.rc_btn.resize(self.rc_btn.minimumSizeHint())

        self.ml_btn = QPushButton("Mask \nR-CNN", self)
        self.rc_btn.clicked.connect(self.mask_rcnn_eval)
        self.rc_btn.resize(self.rc_btn.minimumSizeHint())

        self.tool_bar_layout.addWidget(self.fh_btn)
        self.tool_bar_layout.addWidget(self.rc_btn)
        self.tool_bar_layout.addWidget(self.ml_btn)
        self.setCentralWidget(win)
        win.setLayout(self.tool_bar_layout)
        self.show()

    def open_file(self):
        name = QFileDialog.getOpenFileName(self, 'Open File')
        if name[0] == "":
            return 0
        file = open(name, 'r')
    def open_next(self):
        print("open next")
    def save_file(self):
        print("saved")
    def close_file(self):
        print("close")
    def undo_action(self):
        print("undo")
    def clear_drawings(self):
        print("clear")
    def set_global_scale(self):
        print("set scale")
    def measure(self):
        print("measure")
    def drawing_mode_rectangle(self):
        print("rectangle")
    def drawing_mode_free_hand(self):
        print("freehand")
    def mask_rcnn_eval(self):
        print("analyze with mask RCNN")
class Image_GUI():
    def __init__(self):
        self.drawing = False  # true if mouse is pressed
        self.mode = True  # if True, draw rectangle. Press 'm' to toggle to curve
        self.former_x = -1
        self.former_y = -1
        self.current_former_x = 0
        self.current_former_y = 0
        self.starting_pixel = (-1, -1)
        self.img = np.zeros((512, 512, 3), np.uint8)
    def freehand_draw(self, event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.starting_pixel = (x, y)
            self.drawing = True
            self.current_former_x = x
            self.current_former_y = y
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing == True:
                self.former_x = self.current_former_x
                self.former_y = self.current_former_y
                self.current_former_x = x
                self.current_former_y = y
                cv2.line(self.img, (self.current_former_x, self.current_former_y), (self.former_x, self.former_y), (0, 0, 255), 1)
        elif event == cv2.EVENT_LBUTTONUP:
            cv2.line(self.img, self.starting_pixel, (x, y), (0, 0, 255), 1)
            self.drawing = False
    def demo(self):
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.freehand_draw)
        while (1):
            cv2.imshow('image', self.img)
            key = cv2.waitKey(20)
            if key == ord("b"):
                break
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # main = Main_window()
    app = QApplication(sys.argv)
    img = np.asarray(Image.open("./Smol dataset/raw/FlinYu_Plant1_BS1_011.TIF"))
    interact = Interactive_widget(img)
    interact.show()
    sys.exit(app.exec_())
