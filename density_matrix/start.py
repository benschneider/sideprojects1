from PyQt4.uic import loadUiType
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import threading
import logging
import numpy as np
import sys
import os
import functions

import random
from matplotlib.backends import qt_compat
use_pyside = qt_compat.QT_API == qt_compat.QT_API_PYSIDE
if use_pyside:
    from PySide import QtGui, QtCore
else:
    from PyQt4 import QtGui, QtCore

Ui_MainWindow, QMainWindow = loadUiType('density_matrix.ui')

logging.basicConfig(level=logging.DEBUG, format='[%(threadName)-10s] %(message)s',)


class dApp(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(dApp, self).__init__(parent)
        self.setupUi(self)
        self.init_defValues()
        self.init_UI()  # connect buttons etc to programms
        # self.testfigure()

    def init_defValues(self):
        self.dispData = {}
        self.dicData = {}
        self.dispData['Frequency 1'] = 4.8e9
        self.dispData['Frequency 2'] = 4.1e9
        self.dispData['Gain 1'] = 602.0e7
        self.dispData['Gain 2'] = 685.35e7
        self.dispData['Bandwidth'] = 1e5
        self.dispData['mapdim'] = [80, 80]
        self.dispData['lags'] = 20

    def init_UI(self):
        self.open_hdf5_on.triggered.connect(self.browse_hdf5_on)
        self.open_hdf5_off.triggered.connect(self.browse_hdf5_off)
        self.selectBtn.clicked.connect(self.select_number)
        self.ExitButton.clicked.connect(QtGui.qApp.quit)
        self.action_Quit.triggered.connect(QtGui.qApp.quit)
        self.save_mtx.triggered.connect(self.browse_saveMtx)
        self.makeHistrogram.clicked.connect()

    def show_figure(self, fig):
        ''' Adds figure as a Widget to the Right Vertical Layout part'''
        logging.debug('Plot Figure')
        self.canvas = FigureCanvas(fig)
        self.RightVertLayout.addWidget(self.canvas)
        self.canvas.draw()
        self.toolbar = NavigationToolbar(self.canvas, self.dWindow1, coordinates=True)
        self.RightVertLayout.addWidget(self.toolbar)
        # self.toolbar = NavigationToolbar(self.canvas, self, coordinates=True)
        # self.addToolBar(self.toolbar)

    def remove_figure(self):
        self.RightVertLayout.removeWidget(self.canvas)
        self.canvas.close()
        self.RightVertLayout.removeWidget(self.toolbar)
        self.toolbar.close()

    def testfigure(self):
        fig1 = Figure()
        y = np.random.rand(10)
        x = range(10)
        ax1 = fig1.add_subplot(2, 2, 1)
        ax2 = fig1.add_subplot(2, 2, 2, sharex=ax1, sharey=ax1)
        ax3 = fig1.add_subplot(2, 2, 3, sharex=ax1, sharey=ax1)
        ax4 = fig1.add_subplot(2, 2, 4, sharex=ax1, sharey=ax1)
        ax1.axis('auto')
        ax1.tick_params(labelbottom='off')
        ax2.tick_params(labelbottom='off', labelleft='off')
        ax4.tick_params(labelleft='off')
        # fig1.set_title('Sharing x per column, y per row')
        ax1.plot(x, y)
        ax2.scatter(x, y)
        ax3.scatter(x, 2 * y ** 2 - 1, color='r')
        ax4.plot(x, 2 * y ** 2 - 1, color='r')
        self.show_figure(fig1)  # send figure to the show_figure Terminal

    def plot_data(self):
        on = self.dicData['hdf5_on']  # this one contains all the histogram data
        # off = self.dicData['hdf5_off']
        fig1 = Figure()
        ax1 = fig1.add_subplot(2, 2, 1)
        ax2 = fig1.add_subplot(2, 2, 2, sharex=ax1, sharey=ax1)
        ax3 = fig1.add_subplot(2, 2, 3, sharex=ax1, sharey=ax1)
        ax4 = fig1.add_subplot(2, 2, 4, sharex=ax1, sharey=ax1)
        ax1.axis('auto')
        ax1.tick_params(labelbottom='off')
        ax2.tick_params(labelbottom='off', labelleft='off')
        ax4.tick_params(labelleft='off')
        # fig1.set_title('Sharing x per column, y per row')

        ax1.title('II')
        ax1.imshow(on.IIdmap, interpolation='nearest', origin='low',
                   extent=[on.xII[0], on.xII[-1], on.yII[0], on.yII[-1]])

        ax2.title('QQ')
        ax2.imshow(on.QQdmap, interpolation='nearest', origin='low',
                   extent=[on.xQQ[0], on.xQQ[-1], on.yQQ[0], on.yQQ[-1]])

        ax3.title('I1Q2')
        ax3.imshow(on.IQdmap, interpolation='nearest', origin='low',
                   extent=[on.xIQ[0], on.xIQ[-1], on.yIQ[0], on.yIQ[-1]])

        ax4.title('Q1I2')
        ax4.imshow(on.QIdmap, interpolation='nearest', origin='low',
                   extent=[on.xQI[0], on.xQI[-1], on.yQI[0], on.yQI[-1]])

        self.show_figure(fig1)  # send figure to the show_figure Terminal

    def browse_hdf5_on(self):
        self.browse_hdf5('hdf5_on')

    def browse_hdf5_off(self):
        self.browse_hdf5('hdf5_off')

    def browse_hdf5(self, ext):
        dialog_txt = 'Pick a file for :' + str(ext)
        openname = QtGui.QFileDialog.getOpenFileName(self, dialog_txt)
        self.dispData[str(ext)] = openname
        logging.debug(str(ext)+':'+str(openname))
        functions.load_dataset(self.dispData, self.dicData, ext)
        self.update_data_disp()

    def browse_saveMtx(self):
        savename = QtGui.QFileDialog.getSaveFileName(self, "file")
        self.dispData['mtx '] = savename
        logging.debug('Save .mtx:'+str(savename))
        self.update_data_disp()

    def select_number(self):
        number = self.selectNum.text()
        self.dispData['select '] = number
        # do calculations here
        self.update_data_disp()

    def update_data_disp(self):
        QtGui
        self.selectDat.clear()
        for key in self.dispData:
            newItem = key + ': ' + str(self.dispData[key])
            self.selectDat.addItem(newItem)


def main():
    app = QtGui.QApplication(sys.argv)
    form = dApp()
    form.show()
    # app.exec_()
    sys.exit(app.exec_())  # it there actually a difference?

if __name__ == '__main__':
    main()  # activate Gui Interface
