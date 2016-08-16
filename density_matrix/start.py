from PyQt4.QtGui import QApplication, QTableWidgetItem, qApp, QFileDialog
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt4.uic import loadUiType
# import threading
import logging
import numpy as np
import sys
import functions
import random

Ui_MainWindow, QMainWindow = loadUiType('density_matrix.ui')
logging.basicConfig(level=logging.DEBUG, format='[%(threadName)-10s] %(message)s',)


class dApp(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        super(dApp, self).__init__(parent)
        self.setupUi(self)
        self.init_defValues()
        self.update_table()
        self.update_data_disp()
        self.init_UI()  # connect buttons etc to programs
        self.update_data_disp()
        self.widgetStyle = ("QWidget {background-color: #ffffff}," +
                            "QWidget::item {background: transparent," +
                            "QWidget::item:selected {background: #ffffff}}")

    def init_defValues(self):
        self.canvas_1 = False
        self.canvas_2 = False
        self.dispData = {}
        self.dicData = {}
        self.dispData['f1'] = 4.8e9
        self.dispData['f2'] = 4.1e9
        self.dispData['g1'] = 602.0e7
        self.dispData['g2'] = 685.35e7
        self.dispData['B'] = 1e5
        self.dispData['select'] = 1
        self.dispData['mapdim'] = [80, 80]
        self.dispData['lags'] = 20

    def init_UI(self):
        self.open_hdf5_on.triggered.connect(self.browse_hdf5_on)
        self.open_hdf5_off.triggered.connect(self.browse_hdf5_off)
        # self.selectBtn.clicked.connect(self.select_number)
        # self.ExitButton.clicked.connect(QtGui.qApp.quit)
        self.action_Quit.triggered.connect(qApp.quit)
        self.save_mtx.triggered.connect(self.browse_saveMtx)
        self.makeHistogram.clicked.connect(self.make_Histogram)
        self.tableWidget.itemChanged.connect(self.read_table)

    def update_page_1(self, fig):
        self.clear_page_1()
        logging.debug('Update Page 1 Figures')
        self.canvas_1 = FigureCanvas(fig)
        self.HistLayout.addWidget(self.canvas_1)
        self.canvas_1.draw()
        self.toolbar_1 = NavigationToolbar(self.canvas_1, self.page_0, coordinates=True)
        self.HistLayout.addWidget(self.toolbar_1)

    def clear_page_1(self):
        if self.canvas_1:
            logging.debug('Clear Page 1 Figures')
            self.HistLayout.removeWidget(self.canvas_1)
            self.canvas_1.close()
            self.HistLayout.removeWidget(self.toolbar_1)
            self.toolbar_1.close()

    def make_Histogram(self):
        functions.process(self.dispData, self.dicData)
        on = self.dicData['hdf5_on']  # this one contains all the histogram data
        fig1 = Figure()
        ax1 = fig1.add_subplot(2, 2, 1)
        ax2 = fig1.add_subplot(2, 2, 2, sharex=ax1, sharey=ax1)
        ax3 = fig1.add_subplot(2, 2, 3, sharex=ax1, sharey=ax1)
        ax4 = fig1.add_subplot(2, 2, 4, sharex=ax1, sharey=ax1)
        ax1.axis('tight')
        ax1.imshow(on.IIdmap, interpolation='nearest', origin='low',
                   extent=[on.xII[0], on.xII[-1], on.yII[0], on.yII[-1]])
        ax2.imshow(on.QQdmap, interpolation='nearest', origin='low',
                   extent=[on.xQQ[0], on.xQQ[-1], on.yQQ[0], on.yQQ[-1]])
        ax3.imshow(on.IQdmap, interpolation='nearest', origin='low',
                   extent=[on.xIQ[0], on.xIQ[-1], on.yIQ[0], on.yIQ[-1]])
        ax4.imshow(on.QIdmap, interpolation='nearest', origin='low',
                   extent=[on.xQI[0], on.xQI[-1], on.yQI[0], on.yQI[-1]])
        ax1.tick_params(labelbottom='off')
        ax2.tick_params(labelbottom='off', labelleft='off')
        ax4.tick_params(labelleft='off')
        ax1.set_title('II')
        ax2.set_title('QQ')
        ax3.set_title('I1Q2')
        ax4.set_title('Q1I2')
        self.update_page_1(fig1)  # send figure to the show_figure Terminal

    def browse_hdf5_on(self):
        self.browse_hdf5('hdf5_on')

    def browse_hdf5_off(self):
        self.browse_hdf5('hdf5_off')

    def browse_hdf5(self, ext):
        dialog_txt = 'Pick a file for :' + str(ext)
        openname = QFileDialog.getOpenFileName(self, dialog_txt)
        self.dispData[str(ext)] = openname
        logging.debug(str(ext)+':'+str(openname))
        functions.load_dataset(self.dispData, self.dicData, ext)
        self.update_data_disp()

    def browse_saveMtx(self):
        savename = QFileDialog.getSaveFileName(self, "file")
        self.dispData['mtx '] = savename
        logging.debug('Save .mtx:'+str(savename))
        self.update_data_disp()

    def update_data_disp(self):
        self.selectDat.clear()
        for key in self.dispData:
            newItem = key + ': ' + str(self.dispData[key])
            self.selectDat.addItem(newItem)

    def read_table(self):
        table = self.tableWidget
        logging.debug('Read Table Widget')
        self.dispData['f1'] = np.float(table.item(0, 0).text())
        self.dispData['f2'] = np.float(table.item(1, 0).text())
        self.dispData['g1'] = np.float(table.item(2, 0).text())
        self.dispData['g2'] = np.float(table.item(3, 0).text())
        self.dispData['B'] = np.float(table.item(4, 0).text())
        self.dispData['select'] = int(eval(str(table.item(5, 0).text())))
        self.dispData['lags'] = int(eval(str(table.item(6, 0).text())))
        self.dispData['mapdim'][0] = np.float(table.item(7, 0).text())
        self.dispData['mapdim'][1] = np.float(table.item(8, 0).text())
        self.update_data_disp()

    def update_table(self):
        logging.debug('Update Table Widget')
        table = self.tableWidget
        d = self.dispData
        table.setItem(0, 0, QTableWidgetItem(str(d['f1'])))
        table.setItem(1, 0, QTableWidgetItem(str(d['f2'])))
        table.setItem(2, 0, QTableWidgetItem(str(d['g1'])))
        table.setItem(3, 0, QTableWidgetItem(str(d['g2'])))
        table.setItem(4, 0, QTableWidgetItem(str(d['B'])))
        table.setItem(5, 0, QTableWidgetItem(str(d['select'])))
        table.setItem(6, 0, QTableWidgetItem(str(d['lags'])))
        table.setItem(7, 0, QTableWidgetItem(str(d['mapdim'][0])))
        table.setItem(8, 0, QTableWidgetItem(str(d['mapdim'][1])))
        table.resizeColumnsToContents()
        table.resizeRowsToContents()
        table.show()

    def tab2array(self, table):
        nT = np.zeros([table.rowCount(), table.columnCount()])
        for i in range(table.rowCount()):
            for j in range(table.columnCount()):
                val = table.item(i, j)
                if val:
                    nT[i, j] = np.float(val.text())
        return nT


class stuff():

    def __init__(self):
        self.temp = []


def main():
    app = QApplication(sys.argv)
    form = dApp()
    form.show()
    # app.exec_()
    sys.exit(app.exec_())  # it there actually a difference?

if __name__ == '__main__':
    app = QApplication(sys.argv)
    form = dApp()
    form.show()
    # app.exec_()
    sys.exit(app.exec_())  # it there actually a difference?
