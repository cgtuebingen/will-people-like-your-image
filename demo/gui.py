#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Authors: Katharina Schwarz
#          Patrick Wieschollek <mail@patwie.com>

"""
Will People Like Your Image?
"""

from PyQt5 import QtWidgets
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QStatusBar

from main import Ui_MainWindow  # here you need to correct the names

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

import numpy as np
import h5py
import random
import cv2
import sys
import os
import math
import subprocess

# app.setStyleSheet("QToolTip { color: #ffffff; background-color: #2a82da; border: 1px solid white; }")

white = QtGui.QColor(255, 255, 255)
red = QtGui.QColor(255, 0, 0)
black = QtGui.QColor(0, 0, 0)

def dark():
    dark_palette = QtGui.QPalette()
    dark_palette.setColor(QtGui.QPalette.Window, QtGui.QColor(53, 53, 53))
    dark_palette.setColor(QtGui.QPalette.WindowText, white)
    dark_palette.setColor(QtGui.QPalette.Base, QtGui.QColor(25, 25, 25))
    dark_palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(53, 53, 53))
    dark_palette.setColor(QtGui.QPalette.ToolTipBase, white)
    dark_palette.setColor(QtGui.QPalette.ToolTipText, white)
    dark_palette.setColor(QtGui.QPalette.Text, white)
    dark_palette.setColor(QtGui.QPalette.Button, QtGui.QColor(53, 53, 53))
    dark_palette.setColor(QtGui.QPalette.ButtonText, white)
    dark_palette.setColor(QtGui.QPalette.BrightText, red)
    dark_palette.setColor(QtGui.QPalette.Link, QtGui.QColor(42, 130, 218))
    dark_palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(42, 130, 218))
    dark_palette.setColor(QtGui.QPalette.HighlightedText, black)
    return dark_palette


# /graphics/projects/opt_Ubuntu16.04/QT/share/pygt5/pyuic5 main.ui -o main.py
app = QApplication(sys.argv)


def kalman(tour, R=1e-5, Q=1e-5): # noqa
    print tour.shape
    assert len(tour.shape) == 1
    m = tour.shape[0]

    z = np.copy(tour)

    xhat = np.zeros((m,))
    P = np.zeros((m,)) # noqa
    xhatminus = np.zeros((m,))
    Pminus = np.zeros((m,)) # noqa
    K = np.zeros((m,)) # noqa

    xhat[0] = tour[0]
    P[0] = tour[0]

    for k in range(1, m):
        # time update
        xhatminus[k] = xhat[k - 1]
        Pminus[k] = P[k - 1] + Q

        # measurement update
        K[k] = Pminus[k] / (Pminus[k] + R)
        xhat[k] = xhatminus[k] + K[k] * (z[k] - xhatminus[k])
        P[k] = (1 - K[k]) * Pminus[k]

    return xhat

class MplCanvas(FigureCanvas):
    """Class to represent the FigureCanvas widget"""
    def __init__(self, trigger):
        self.fig = plt.figure()        
        self.ax = self.fig.add_subplot(111)
        self.trigger = trigger

        FigureCanvas.__init__(self, self.fig)
        FigureCanvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        self.fig.canvas.mpl_connect('button_press_event',self._on_press)

    def _on_press(self,event):
        if self.trigger:
            if event.xdata is not None:
                self.trigger(event.xdata)


class MplWidget(QtWidgets.QWidget):
    """Widget defined in Qt Designer"""
    def __init__(self, trigger=None, parent = None):
        QtWidgets.QWidget.__init__(self, parent)

        self.canvas = MplCanvas(trigger)

        self.canvas.setFocusPolicy( QtCore.Qt.ClickFocus )
        self.canvas.setFocus()

        self.vbl = QtWidgets.QVBoxLayout()
        self.vbl.addWidget(self.canvas)

        self.setLayout(self.vbl)


class Overlay(QWidget):

    def __init__(self, parent = None, gui=None):
    
        QWidget.__init__(self, parent)
        palette = QtGui.QPalette(self.palette())
        palette.setColor(palette.Background, QtCore.Qt.transparent)
        self.setPalette(palette)
        self.gui = gui
    
    def paintEvent(self, event):
    
        painter = QtGui.QPainter()
        painter.begin(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.fillRect(event.rect(), QtGui.QBrush(QtGui.QColor(255, 255, 255, 127)))
        painter.setPen(QtGui.QPen(QtCore.Qt.NoPen))
        
        for i in range(6):
            if (self.counter / 5) % 6 == i:
                painter.setBrush(QtGui.QBrush(QtGui.QColor(127 + (self.counter % 5)*32, 127, 127)))
            else:
                painter.setBrush(QtGui.QBrush(QtGui.QColor(127, 127, 127)))
            painter.drawEllipse(
                self.width()/2 + 30 * math.cos(2 * math.pi * i / 6.0) - 10,
                self.height()/2 + 30 * math.sin(2 * math.pi * i / 6.0) - 10,
                20, 20)
        
        painter.end()
    
    def showEvent(self, event):
        self.timer = self.startTimer(50)
        self.counter = 0
    
    def timerEvent(self, event):
        self.counter += 1
        self.update()
        if self.gui.continue_gui:
            self.killTimer(self.timer)
            self.hide()
        self.counter = self.counter % 60
        # if self.counter == 60:
        #     self.killTimer(self.timer)
        #     self.hide()


class ExampleApp(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self)

        self.btnLoad.clicked.connect(self.loadVideo)

        self.mp4_filename = None
        self.continue_gui = True
        self.cap = None
        self.cap_len = None
        self.frame = None
        self.frame_id = None
        self.data = None

        self.plotWidget = MplWidget(self.showFrame)
        self.plotWidget.setVisible(False)

        self.verticalLayout.addWidget(self.plotWidget)

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)


        self.info_label = QtWidgets.QLabel()
        self.frame_label = QtWidgets.QLabel()
        self.statusBar.addWidget(self.info_label)
        self.statusBar.addWidget(self.frame_label)

        self.overlay = Overlay(parent=self.centralWidget(), gui=self)
        self.overlay.hide()

    def resizeEvent(self, event):
        self.overlay.resize(event.size())
        event.accept()

    def loadVideo(self):
        fn = QtWidgets.QFileDialog.getOpenFileName(self, 'Video File', '', filter='*.mp4')
        if fn:
            self.continue_gui = False
            self.overlay.show()
            print fn[0]
            self.mp4_filename = "%s" % fn[0]

            if '-small' in self.mp4_filename:
                self.mp4_filename = self.mp4_filename.replace('-small', '')

            self.info_label.setText(self.mp4_filename)
            self.cap = cv2.VideoCapture(self.mp4_filename)
            self.cap_len = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.score_name = self.mp4_filename + '.score'
            self.get_score(self.score_name)
            self.showFrame(1)
            self.update_plot()
            self.plotWidget.setVisible(True)
            self.continue_gui = True

    def displayImg(self, frame):
        w_width = self.frameGeometry().width()
        w_height = self.frameGeometry().height()

        height, width = frame.shape[:2]

        if height > w_height*0.8:
            scale = w_height*0.8 / float(height)
            frame = cv2.resize(frame, (0,0), fx=scale, fy=scale)
            height, width = frame.shape[:2]

        data = np.array(frame[:, :, ::-1], dtype=np.uint8) # np.zeros((height, width, 3), dtype=np.uint8)
        q_img = QtGui.QImage(data, width, height, 3 * width, QtGui.QImage.Format_RGB888)
        pixmap01 = QtGui.QPixmap.fromImage(q_img)
        self.image_preview.setPixmap(pixmap01)
        self.image_preview.setFixedWidth(width)
        self.image_preview.setFixedHeight(height)

    def showFrame(self, frame_id):
        self.frame_id = int(frame_id)
        if self.data is not None and self.frame_id is not None:
            self.frame_label.setText("frame: %i, score: %f" % (self.frame_id, self.data[self.frame_id]))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        success, self.frame = self.cap.read()
        self.displayImg(self.frame)
        self.update_plot()

    def get_score(self, score):
        print score
        assert os.path.isfile(score)
        hf = h5py.File(score)
        data = hf['group1']['dataset1']
        data = np.array(data)
        data -= data.min()
        data /= data.max()
        print len(data)

        self.data = np.log(1 + kalman(data, R=1e-1, Q=1e-4))
        self.data -= self.data.min()
        self.data /= self.data.max()

    def update_plot(self):
        if self.data is not None:
            self.plotWidget.canvas.fig.clear()
            self.plotWidget.canvas.fig.patch.set_facecolor('#353535')
            ax = self.plotWidget.canvas.fig.add_subplot(111, axisbg='#353535')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.plot(self.data, '#2a82da' ) #'#0ef42d')
            ax.axis('off')
            ax.set_xlim([0,len(self.data)])
            ax.set_ylim([0,1])
            if self.frame_id:
                ax.axvline(x=self.frame_id, color='#f57c00')
            self.plotWidget.canvas.draw()


app.setStyle("fusion")
app.setPalette(dark())

form = ExampleApp()
form.show()
sys.exit(app.exec_())
