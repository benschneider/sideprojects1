#!/usr/local/bin/python

from PyQt4.QtGui import QApplication, QTableWidgetItem, qApp, QFileDialog
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt4.uic import loadUiType
from parsers import savemtx
# import threading
import logging
import numpy as np
import sys
import functions
import cPickle
import subprocess
import threading

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
        self.canvas_3 = False
        self.dic_canvas = {}
        self.dispData = {}
        self.dicData = {}
        self.dispData['f1'] = 4.8e9
        self.dispData['f2'] = 4.1e9
        self.dispData['g1'] = 602.0e7
        self.dispData['g2'] = 685.35e7
        self.dispData['B'] = 5e5
        self.dispData['select'] = 1
        self.dispData['mapdim'] = [20, 20]
        self.dispData['lags'] = 20
        self.dispData['Phase correction'] = False
        self.dispData['Trigger correction'] = False
        self.dispData['FFT-Filter'] = False
        self.dispData['Section Data'] = True
        self.dispData['Settings file'] = 'density_matrix.set'

    def init_UI(self):
        self.open_hdf5_on.triggered.connect(self.browse_hdf5_on)
        self.open_hdf5_off.triggered.connect(self.browse_hdf5_off)
        self.action_Quit.triggered.connect(qApp.quit)
        self.save_mtx_as.triggered.connect(self.browse_saveMtx)
        self.save_mtx.triggered.connect(self.saveMtx)
        self.makeHistogram.clicked.connect(self.make_Histogram)
        self.tableWidget.itemChanged.connect(self.read_table)
        self.actionLoadPrev.triggered.connect(self.load_settings)
        self.actionSavePrev.triggered.connect(self.save_settings)
        self.actionMtx_files.triggered.connect(self.open_mtx_spyview)

    def save_settings(self):
        with open(self.dispData['Settings file'], "wb") as myFile:
            cPickle.dump(self.dispData, myFile)
        logging.debug('settings and files saved')

    def load_settings(self):
        with open(self.dispData['Settings file'], "rb") as myFile:
            self.dispData = cPickle.load(myFile)
        functions.load_dataset(self.dispData, self.dicData, 'hdf5_on')
        functions.load_dataset(self.dispData, self.dicData, 'hdf5_off')
        self.update_table()
        logging.debug('settings and files loaded')

    def browse_hdf5_on(self):
        self.browse_hdf5('hdf5_on')

    def browse_hdf5_off(self):
        self.browse_hdf5('hdf5_off')

    def browse_hdf5(self, ext):
        dialog_txt = 'Pick a file for :' + str(ext)
        openname = QFileDialog.getOpenFileName(self, dialog_txt)
        if openname:
            self.dispData[str(ext)] = openname
            logging.debug(str(ext) + ':' + str(openname))
            functions.load_dataset(self.dispData, self.dicData, ext)
            self.update_data_disp()

    def browse_saveMtx(self):
        savename = QFileDialog.getSaveFileName(self, "Select for 'base-name'+cII.mtx .. files")
        if savename:
            self.dispData['mtx '] = savename
            logging.debug('Save .mtx:' + str(savename))
            self.saveMtx()

    def saveMtx(self):
        if self.dispData['mtx ']:
            savename = self.dispData['mtx ']
            logging.debug('Save .mtx:' + str(savename))
            on = self.dicData['hdf5_on']
            savemtx(savename + 'cII.mtx', np.expand_dims(on.IIdmap, axis=0), on.headerII)
            savemtx(savename + 'cQQ.mtx', np.expand_dims(on.QQdmap, axis=0), on.headerQQ)
            savemtx(savename + 'cIQ.mtx', np.expand_dims(on.IQdmap, axis=0), on.headerIQ)
            savemtx(savename + 'cQI.mtx', np.expand_dims(on.QIdmap, axis=0), on.headerQI)
            self.update_data_disp()

    def open_mtx_spyview(self):
        if self.dispData['mtx ']:
            d = threading.Thread(name='spyview', target=self._spyview)
            d.setDaemon(True)
            d.start()

    def _spyview(self):
        logging.debug('Spyview started')
        basen = self.dispData['mtx ']
        subprocess.call(['spyview', basen + 'cII.mtx', basen + 'cQQ.mtx',
                         basen + 'cIQ.mtx', basen + 'cQI.mtx'])
        logging.debug('Spyview closed')

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
        self.dispData['mapdim'][0] = int(table.item(7, 0).text())
        self.dispData['mapdim'][1] = int(table.item(8, 0).text())
        self.dispData['Phase correction'] = bool(eval(str(table.item(9, 0).text())))
        self.dispData['Trigger correction'] = bool(eval(str(table.item(10, 0).text())))
        self.dispData['FFT-Filter'] = bool(eval(str(table.item(11, 0).text())))
        self.dispData['Section Data'] = bool(eval(str(table.item(12, 0).text())))
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
        table.setItem(9, 0, QTableWidgetItem(str(d['Phase correction'])))
        table.setItem(10, 0, QTableWidgetItem(str(d['Trigger correction'])))
        table.setItem(11, 0, QTableWidgetItem(str(d['FFT-Filter'])))
        table.setItem(12, 0, QTableWidgetItem(str(d['Section Data'])))
        table.resizeColumnsToContents()
        table.resizeRowsToContents()
        table.show()

    def update_data_disp(self):
        xr = (np.array([-self.dispData['lags'], self.dispData['lags']]) / self.dispData['B'])
        self.dispData['minmax lags (s)'] = xr
        self.dicData['xaxis'] = np.linspace(xr[0], xr[1], self.dispData['lags'] * 2 + 1)
        self.selectDat.clear()
        for key in self.dispData:
            newItem = key + ': ' + str(self.dispData[key])
            self.selectDat.addItem(newItem)

    def tab2array(self, table):
        nT = np.zeros([table.rowCount(), table.columnCount()])
        for i in range(table.rowCount()):
            for j in range(table.columnCount()):
                val = table.item(i, j)
                if val:
                    nT[i, j] = np.float(val.text())
        return nT

    def update_page_1(self, fig):
        self.clear_page_1()
        logging.debug('Update Histogram Figures')
        self.canvas_1 = FigureCanvas(fig)
        self.HistLayout.addWidget(self.canvas_1)
        self.canvas_1.draw()
        self.toolbar_1 = NavigationToolbar(self.canvas_1, self.page_1, coordinates=True)
        self.HistLayout.addWidget(self.toolbar_1)

    def clear_page_1(self):
        if self.canvas_1:
            logging.debug('Clear Histogram Figures')
            self.HistLayout.removeWidget(self.canvas_1)
            self.canvas_1.close()
            self.HistLayout.removeWidget(self.toolbar_1)
            self.toolbar_1.close()

    def update_page_2(self, fig):
        self.clear_page_2()
        logging.debug('Update Correlation Figures')
        self.canvas_2 = FigureCanvas(fig)
        self.CorrLayout.addWidget(self.canvas_2)
        self.canvas_2.draw()
        self.toolbar_2 = NavigationToolbar(self.canvas_2, self.page_2, coordinates=True)
        self.CorrLayout.addWidget(self.toolbar_2)

    def clear_page_2(self):
        if self.canvas_2:
            logging.debug('Clear Correlation Figures')
            self.CorrLayout.removeWidget(self.canvas_2)
            self.canvas_2.close()
            self.CorrLayout.removeWidget(self.toolbar_2)
            self.toolbar_2.close()

    def update_page_3(self, fig):
        self.clear_page_3()
        logging.debug('Update TMS Figures')
        self.canvas_3 = FigureCanvas(fig)
        self.TMSLayout.addWidget(self.canvas_3)
        self.canvas_3.draw()
        self.toolbar_3 = NavigationToolbar(self.canvas_3, self.page_3, coordinates=True)
        self.TMSLayout.addWidget(self.toolbar_3)

    def clear_page_3(self):
        if self.canvas_3:
            logging.debug('Clear TMS Figures')
            self.TMSLayout.removeWidget(self.canvas_3)
            self.canvas_3.close()
            self.TMSLayout.removeWidget(self.toolbar_3)
            self.toolbar_3.close()

    def make_Histogram(self):
        functions.process(self.dispData, self.dicData)
        self.make_CorrFigs()
        self.make_TMSFig()
        on = self.dicData['hdf5_on']  # this one contains all the histogram data
        fig1 = Figure(facecolor='white', edgecolor='white')
        ax1 = fig1.add_subplot(2, 2, 1)
        ax2 = fig1.add_subplot(2, 2, 2)  # , sharex=ax1, sharey=ax1)
        ax3 = fig1.add_subplot(2, 2, 3)  # , sharex=ax1, sharey=ax1)
        ax4 = fig1.add_subplot(2, 2, 4)  # , sharex=ax1, sharey=ax1)
        ax1.imshow(on.IIdmap, interpolation='nearest', origin='low',
                   extent=[on.xII[0], on.xII[-1], on.yII[0], on.yII[-1]], aspect='auto')
        ax2.imshow(on.QQdmap, interpolation='nearest', origin='low',
                   extent=[on.xQQ[0], on.xQQ[-1], on.yQQ[0], on.yQQ[-1]], aspect='auto')
        ax3.imshow(on.IQdmap, interpolation='nearest', origin='low',
                   extent=[on.xIQ[0], on.xIQ[-1], on.yIQ[0], on.yIQ[-1]], aspect='auto')
        ax4.imshow(on.QIdmap, interpolation='nearest', origin='low',
                   extent=[on.xQI[0], on.xQI[-1], on.yQI[0], on.yQI[-1]], aspect='auto')
        fig1.tight_layout()
        # ax1.tick_params(labelbottom='off')
        # ax2.tick_params(labelbottom='off', labelleft='off')
        # ax4.tick_params(labelleft='off')
        ax1.set_title('IIc')
        ax2.set_title('QQc')
        ax3.set_title('IQc')
        ax4.set_title('QIc')
        self.update_page_1(fig1)  # send figure to the show_figure terminal
        self.update_table()

    def make_CorrFigs(self):
        fig2 = Figure(facecolor='white', edgecolor='black')
        xCorr1 = fig2.add_subplot(2, 2, 1)
        xCorr2 = fig2.add_subplot(2, 2, 2)
        xCorr3 = fig2.add_subplot(2, 2, 3)
        xCorr4 = fig2.add_subplot(2, 2, 4)
        xCorr1.set_title('<IIc>')
        xCorr2.set_title('<QQc>')
        xCorr3.set_title('<IQc>')
        xCorr4.set_title('<QIc>')
        xCorr1.plot(self.dicData['xaxis'], self.dicData['hdf5_on'].cII)
        xCorr2.plot(self.dicData['xaxis'], self.dicData['hdf5_on'].cQQ)
        xCorr3.plot(self.dicData['xaxis'], self.dicData['hdf5_on'].cIQ)
        xCorr4.plot(self.dicData['xaxis'], self.dicData['hdf5_on'].cQI)
        fig2.tight_layout()
        self.update_page_2(fig2)

    def make_TMSFig(self):
        fig3 = Figure(facecolor='white', edgecolor='black')
        xTMS1 = fig3.add_subplot(1, 2, 1)
        xTMS2 = fig3.add_subplot(1, 2, 2)
        xTMS1.set_title('Magnitue')
        xTMS2.set_title('Phase')
        xTMS1.plot(self.dicData['xaxis'], self.dicData['hdf5_on'].PSI_mag)
        xTMS2.plot(self.dicData['xaxis'], self.dicData['hdf5_on'].PSI_phs)
        fig3.tight_layout()
        self.update_page_3(fig3)


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
