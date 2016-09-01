#!/usr/local/bin/python

from PyQt4.QtGui import QApplication, QTableWidgetItem, qApp, QFileDialog
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt4.uic import loadUiType
from parsers import savemtx
import logging
import numpy as np
import sys
import functions
import cPickle
import subprocess
import threading
import PyGnuplot as gp

Ui_MainWindow, QMainWindow = loadUiType('density_matrix.ui')
logging.basicConfig(level=logging.DEBUG, format='[%(threadName)-10s] %(message)s',)


class dApp(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        super(dApp, self).__init__(parent)
        self.setupUi(self)
        self.init_defValues()
        self.update_table()
        self.update_data_disp()
        self.init_UI()
        self.update_data_disp()
        self.widgetStyle = ("QWidget {background-color: #ffffff}," +
                            "QWidget::item {background: transparent," +
                            "QWidget::item:selected {background: #ffffff}}")

    def init_defValues(self):
        self.canvas_1 = False
        self.canvas_2 = False
        self.canvas_3 = False
        self.canvas_4 = False
        self.canvas_5 = False
        self.canvas_6 = False
        self.dic_canvas = {}
        self.dispData = {}
        self.dicData = {}
        self.dicData['res'] = empty_class()  # class to store results
        dD = self.dispData
        dD['f1'] = 4.8e9
        dD['f2'] = 4.1e9
        dD['g1'] = 6.4757e9  # <- fixed R # 602.0e7  # dR
        dD['g2'] = 7.5135e9  # <- fixed R # 685.35e7  # dR
        dD['Cross_correlation_Gain_f1f1'] = 1.4240e7
        dD['Cross_correlation_Gain_f2f2'] = 1.7251e7
        dD['B'] = 5e5
        dD['select'] = 0
        dD['mapdim'] = [20, 20]
        dD['lags'] = 1000
        dD['Phase correction'] = True
        dD['Trigger correction'] = True
        dD['FFT-Filter'] = False
        dD['Segment Size'] = 0
        dD['Averages'] = 1
        dD['Low Pass'] = 0
        dD['dim1 pt'] = 201
        dD['dim1 start'] = 2.03
        dD['dim1 stop'] = 0.03
        dD['dim1 name'] = 'RF power'
        dD['dim2 pt'] = 100
        dD['dim2 start'] = 0
        dD['dim2 stop'] = 1
        dD['dim2 name'] = 'Magnet'
        dD['dim3 pt'] = 100
        dD['dim3 start'] = 0
        dD['dim3 stop'] = 1
        dD['dim3 name'] = 'Nothing'
        dD['Process Num'] = 'Nothing'
        dD['Settings file'] = 'density_matrix.set'
        dD['dim1 lin'] = np.linspace(dD['dim1 start'], dD['dim1 stop'], dD['dim1 pt'])
        dD['dim2 lin'] = np.linspace(dD['dim2 start'], dD['dim2 stop'], dD['dim2 pt'])
        dD['dim3 lin'] = np.linspace(dD['dim3 start'], dD['dim3 stop'], dD['dim3 pt'])

    def init_UI(self):
        ''' connect buttons to programs '''
        self.open_hdf5_on.triggered.connect(lambda: self.add_hdf5data('hdf5_on'))
        self.open_hdf5_off.triggered.connect(lambda: self.add_hdf5data('hdf5_off'))
        self.action_Quit.triggered.connect(qApp.quit)
        self.save_mtx_as.triggered.connect(self.browse_saveMtx)
        self.save_mtx.triggered.connect(self.saveMtx)
        self.makeHistogram.clicked.connect(self.make_Histogram)
        self.Process_all.clicked.connect(self.process_all)
        self.tableWidget.itemChanged.connect(self.read_table)
        self.actionLoadPrev.triggered.connect(self.load_settings)
        self.actionSavePrev.triggered.connect(self.save_settings)
        self.actionMtx_files.triggered.connect(self.open_mtx_spyview)
        self.action11.triggered.connect(lambda: self.add_hdf5data('on11'))
        self.action22.triggered.connect(lambda: self.add_hdf5data('on22'))
        self.action12.triggered.connect(lambda: self.add_hdf5data('on12'))
        self.action21.triggered.connect(lambda: self.add_hdf5data('on21'))
        self.action12_OFF.triggered.connect(lambda: self.add_hdf5data('off12'))
        self.action21_OFF.triggered.connect(lambda: self.add_hdf5data('off21'))

    def save_settings(self):
        with open(self.dispData['Settings file'], "wb+") as myFile:
            cPickle.dump(self.dispData, myFile)
        logging.debug('settings and files saved')

    def load_settings(self):
        with open(self.dispData['Settings file'], "rb") as myFile:
            self.dispData = cPickle.load(myFile)
        functions.load_dataset(self.dispData, self.dicData, 'hdf5_on')
        functions.load_dataset(self.dispData, self.dicData, 'hdf5_off')
        self.update_table()
        logging.debug('settings and files loaded')

    def add_hdf5data(self, frequency_configuration):
        dialog_txt = 'Pick a file for :' + str(frequency_configuration)
        openname = QFileDialog.getOpenFileName(self, dialog_txt)
        if openname:
            logging.debug(str(frequency_configuration) + ':' + str(openname))
            self.dispData[str(frequency_configuration)] = openname
            functions.load_dataset(self.dispData, self.dicData, frequency_configuration)
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
            res = self.dicData['res']  # this contains the calculation results
            logging.debug('Save .mtx:' + str(savename))
            on = self.dicData['hdf5_on']
            savemtx(savename + 'cII.mtx', np.expand_dims(res.IQmapM_avg[0], axis=0), on.headerII)
            savemtx(savename + 'cQQ.mtx', np.expand_dims(res.IQmapM_avg[1], axis=0), on.headerQQ)
            savemtx(savename + 'cIQ.mtx', np.expand_dims(res.IQmapM_avg[2], axis=0), on.headerIQ)
            savemtx(savename + 'cQI.mtx', np.expand_dims(res.IQmapM_avg[3], axis=0), on.headerQI)
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
        # logging.debug('Read Table Widget')
        dD = self.dispData
        dD['f1'] = np.float(table.item(0, 0).text())
        dD['f2'] = np.float(table.item(1, 0).text())
        dD['g1'] = np.float(table.item(2, 0).text())
        dD['g2'] = np.float(table.item(3, 0).text())
        dD['B'] = np.float(table.item(4, 0).text())
        dD['select'] = int(eval(str(table.item(5, 0).text())))
        dD['lags'] = int(eval(str(table.item(6, 0).text())))
        dD['mapdim'][0] = int(table.item(7, 0).text())
        dD['mapdim'][1] = int(table.item(8, 0).text())
        dD['Phase correction'] = bool(eval(str(table.item(9, 0).text())))
        dD['Trigger correction'] = bool(eval(str(table.item(10, 0).text())))
        dD['FFT-Filter'] = bool(eval(str(table.item(11, 0).text())))
        dD['Segment Size'] = int(eval(str(table.item(12, 0).text())))
        dD['Low Pass'] = np.float(table.item(13, 0).text())
        dD['Averages'] = np.int(table.item(14, 0).text())
        dD['dim1 pt'] = np.int(table.item(15, 0).text())
        dD['dim1 start'] = np.float(table.item(16, 0).text())
        dD['dim1 stop'] = np.float(table.item(17, 0).text())
        dD['dim1 lin'] = np.linspace(dD['dim1 start'], dD['dim1 stop'], 201)
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
        table.setItem(12, 0, QTableWidgetItem(str(d['Segment Size'])))
        table.setItem(13, 0, QTableWidgetItem(str(d['Low Pass'])))
        table.setItem(14, 0, QTableWidgetItem(str(d['Averages'])))
        table.setItem(15, 0, QTableWidgetItem(str(d['dim1 pt'])))
        table.setItem(16, 0, QTableWidgetItem(str(d['dim1 start'])))
        table.setItem(17, 0, QTableWidgetItem(str(d['dim1 stop'])))
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
        self.toolbar_1 = NavigationToolbar(self.canvas_1, self.tab_2, coordinates=True)
        self.HistLayout.addWidget(self.toolbar_1)

    def update_page_2(self, fig):
        self.clear_page_2()
        logging.debug('Update Correlation Figures')
        self.canvas_2 = FigureCanvas(fig)
        self.CorrLayout.addWidget(self.canvas_2)
        self.canvas_2.draw()
        self.toolbar_2 = NavigationToolbar(self.canvas_2, self.cc_page, coordinates=True)
        self.CorrLayout.addWidget(self.toolbar_2)

    def update_page_3(self, fig):
        self.clear_page_3()
        logging.debug('Update TMS Figures')
        self.canvas_3 = FigureCanvas(fig)
        self.TMSLayout.addWidget(self.canvas_3)
        self.canvas_3.draw()
        self.toolbar_3 = NavigationToolbar(self.canvas_3, self.TMS_page, coordinates=True)
        self.TMSLayout.addWidget(self.toolbar_3)

    def update_page_5(self, fig):
        self.clear_page_5()
        logging.debug('Update page 5: phn1')
        self.canvas_5 = FigureCanvas(fig)
        self.phn1.addWidget(self.canvas_5)
        self.canvas_5.draw()
        self.toolbar_5 = NavigationToolbar(self.canvas_5, self.phn1_page, coordinates=True)
        self.phn1.addWidget(self.toolbar_5)

    def update_page_6(self, fig):
        self.clear_page_6()
        logging.debug('Update page 6: phn2')
        self.canvas_6 = FigureCanvas(fig)
        self.phn2.addWidget(self.canvas_6)
        self.canvas_6.draw()
        self.toolbar_6 = NavigationToolbar(self.canvas_6, self.phn2_page, coordinates=True)
        self.phn2.addWidget(self.toolbar_6)

    def clear_page_1(self):
        if self.canvas_1:
            logging.debug('Clear Histogram Figures')
            self.HistLayout.removeWidget(self.canvas_1)
            self.canvas_1.close()
            self.HistLayout.removeWidget(self.toolbar_1)
            self.toolbar_1.close()

    def clear_page_2(self):
        if self.canvas_2:
            logging.debug('Clear Correlation Figures')
            self.CorrLayout.removeWidget(self.canvas_2)
            self.canvas_2.close()
            self.CorrLayout.removeWidget(self.toolbar_2)
            self.toolbar_2.close()

    def clear_page_3(self):
        if self.canvas_3:
            logging.debug('Clear TMS Figures')
            self.TMSLayout.removeWidget(self.canvas_3)
            self.canvas_3.close()
            self.TMSLayout.removeWidget(self.toolbar_3)
            self.toolbar_3.close()

    def clear_page_5(self):
        if self.canvas_5:
            logging.debug('Clear page 5: phn1')
            self.phn1.removeWidget(self.canvas_5)
            self.canvas_5.close()
            self.phn1.removeWidget(self.toolbar_5)
            self.toolbar_5.close()

    def clear_page_6(self):
        if self.canvas_6:
            logging.debug('Clear page 5: phn2')
            self.phn2.removeWidget(self.canvas_6)
            self.canvas_6.close()
            self.phn2.removeWidget(self.toolbar_6)
            self.toolbar_6.close()

    def process_all(self):
        logging.debug('start processing')
        functions.process_all_points(self.dispData, self.dicData)
        res = self.dicData['res']

        fig1 = Figure(facecolor='white', edgecolor='white')
        pl1 = fig1.add_subplot(1, 1, 1)
        pl1.plot(res.ns[:, 0], label='f1')
        pl1.plot(res.ns[:, 1], label='f2')
        pl1.set_title('Photon numbers')

        fig2 = Figure(facecolor='white', edgecolor='white')
        pl3 = fig2.add_subplot(2, 1, 1)
        pl4 = fig2.add_subplot(2, 1, 2)
        pl3.plot(res.sqs, label='Sq Mag')
        pl3.plot(res.ineqs, label='Ineq_req')
        pl3.set_title('Squeezing Mag')
        pl4.plot(res.sqphs)
        pl4.set_title('Squeezing Phase')

        self.update_page_5(fig1)
        self.update_page_6(fig2)
        self.update_table()
        self.save_processed()

    def save_processed(self):
        if self.dispData['mtx ']:
            savename = self.dispData['mtx ']
            logging.debug('Save data as mtx files: ' + str(savename))
            on = self.dicData['hdf5_on']
            res = self.dicData['res']  # this contains the calculation results
            savemtx(savename + 'IImaps.mtx', res.IQmapMs_avg[:, 0, :, :], on.headerII)
            savemtx(savename + 'QQmaps.mtx', res.IQmapMs_avg[:, 1, :, :], on.headerQQ)
            savemtx(savename + 'IQmaps.mtx', res.IQmapMs_avg[:, 2, :, :], on.headerIQ)
            savemtx(savename + 'QImaps.mtx', res.IQmapMs_avg[:, 3, :, :], on.headerQI)
            savemtx(savename + 'cs_avg_QI.mtx', res.cs_avg, on.headerQI)
            savemtx(savename + 'cs_avg_QI_off.mtx', res.cs_avg_off, on.headerQI)
            filename = 'n1n2sqIneqRawSQNoise.dat'
            gp.s([res.ns[:, 0], res.ns[:, 1], res.sqs, res.ineqs, res.noise], filename=filename)
            gp.c('plot "' + filename + '" u 3 w lp t "Squeezing"')
            gp.c('replot "' + filename + '" u 4 w lp t "Ineq"')
            self.update_data_disp()

    def make_Histogram(self):
        functions.process(self.dispData, self.dicData)
        self.make_CorrFigs()
        self.make_TMSFig()
        on = self.dicData['hdf5_on']  # this one contains all the histogram axis
        res = self.dicData['res']  # this contains the calculation results
        fig1 = Figure(facecolor='white', edgecolor='white')
        ax1 = fig1.add_subplot(2, 2, 1)
        ax2 = fig1.add_subplot(2, 2, 2)
        ax3 = fig1.add_subplot(2, 2, 3)
        ax4 = fig1.add_subplot(2, 2, 4)
        ax1.imshow(res.IQmapM_avg[0], interpolation='nearest', origin='low',
                   extent=[on.xII[0], on.xII[-1], on.yII[0], on.yII[-1]], aspect='auto')
        ax2.imshow(res.IQmapM_avg[1], interpolation='nearest', origin='low',
                   extent=[on.xQQ[0], on.xQQ[-1], on.yQQ[0], on.yQQ[-1]], aspect='auto')
        ax3.imshow(res.IQmapM_avg[2], interpolation='nearest', origin='low',
                   extent=[on.xIQ[0], on.xIQ[-1], on.yIQ[0], on.yIQ[-1]], aspect='auto')
        ax4.imshow(res.IQmapM_avg[3], interpolation='nearest', origin='low',
                   extent=[on.xQI[0], on.xQI[-1], on.yQI[0], on.yQI[-1]], aspect='auto')
        fig1.tight_layout()
        ax1.set_title('IIc')
        ax2.set_title('QQc')
        ax3.set_title('IQc')
        ax4.set_title('QIc')
        self.update_page_1(fig1)  # send figure to the show_figure terminal
        self.update_table()

    def make_CorrFigs(self):
        fig2 = Figure(facecolor='white', edgecolor='black')
        res = self.dicData['res']
        xCorr1 = fig2.add_subplot(2, 2, 1)
        xCorr2 = fig2.add_subplot(2, 2, 2)
        xCorr3 = fig2.add_subplot(2, 2, 3)
        xCorr4 = fig2.add_subplot(2, 2, 4)
        xCorr1.set_title('<IIc>')
        xCorr2.set_title('<QQc>')
        xCorr3.set_title('<IQc>')
        xCorr4.set_title('<QIc>')
        xCorr1.plot(self.dicData['xaxis'], res.c_avg[0])
        xCorr2.plot(self.dicData['xaxis'], res.c_avg[1])
        xCorr3.plot(self.dicData['xaxis'], res.c_avg[2])
        xCorr4.plot(self.dicData['xaxis'], res.c_avg[3])
        fig2.tight_layout()
        self.update_page_2(fig2)

    def make_TMSFig(self):
        fig3 = Figure(facecolor='white', edgecolor='black')
        res = self.dicData['res']
        xTMS1 = fig3.add_subplot(1, 2, 1)
        xTMS2 = fig3.add_subplot(1, 2, 2)
        xTMS1.set_title('Magnitude')
        xTMS2.set_title('Phase')
        xTMS1.plot(self.dicData['xaxis'], np.abs(res.psi_avg[0]))
        xTMS2.plot(self.dicData['xaxis'], np.angle(res.psi_avg[0]))
        fig3.tight_layout()
        self.update_page_3(fig3)


class empty_class():

    def __init__(self):
        pass


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
    sys.exit(app.exec_())  # it there actually a difference?
