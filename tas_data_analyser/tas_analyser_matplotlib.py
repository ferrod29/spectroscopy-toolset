import glob, os, sys

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import pyqtgraph as pg
import pyqtgraph.exporters
from pyqtgraph import mkPen
from PyQt5.QtWidgets import QMainWindow, QAction, QApplication, QFileDialog, QWidget, QGridLayout, QSplitter
from PyQt5.QtGui import QDoubleValidator, QFont, QColor, QPalette
from PyQt5.QtCore import Qt, QRectF
from PyQt5 import uic

# Now use a palette to switch to dark colors:
palette = QPalette()
palette.setColor(QPalette.Window, QColor(53, 53, 53))
palette.setColor(QPalette.WindowText, Qt.white)
palette.setColor(QPalette.Base, QColor(25, 25, 25))
palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
palette.setColor(QPalette.ToolTipBase, Qt.white)
palette.setColor(QPalette.ToolTipText, Qt.white)
palette.setColor(QPalette.Text, Qt.white)
palette.setColor(QPalette.Button, QColor(53, 53, 53))
palette.setColor(QPalette.ButtonText, Qt.white)
palette.setColor(QPalette.BrightText, Qt.red)
palette.setColor(QPalette.Link, QColor(42, 130, 218))
palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
palette.setColor(QPalette.HighlightedText, Qt.black)

pg.setConfigOptions(imageAxisOrder='row-major', antialias=True, crashWarning=True)

import pyqtgraph.widgets.MatplotlibWidget as mplWidegt

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import ticker, cm
plt.rcParams['savefig.facecolor'] = "0.95"
plt.rcParams['figure.figsize'] = 5.0, 5.0
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['font.size'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['legend.title_fontsize'] = 8
plt.rcParams['figure.titlesize'] = 8

class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, vcenter=None, clip=False):
        """ centering data on 0.0 from normaliztion """
        self.vcenter = vcenter
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1]

        return np.ma.masked_array(np.interp(value, x, y))


class PlotsWidget(QMainWindow):
    def __init__(self):
        """
        Main window with the 2D map of intensities and traces preview.
        Control panel for data manipulation, set T0, average on lambdas.
        """
        super(QMainWindow, self).__init__()
        self._main = QWidget()
        self.setCentralWidget(self._main)
        layout = QGridLayout(self._main)

        self.setWindowTitle('TAS Analyser')
        self.setGeometry(20,20,1800,1200)

        intmap_canvas = mplWidegt.MatplotlibWidget(size=(5.0, 5.0), dpi=125)
        layout.addWidget(intmap_canvas,0,0)
        self.intmap_curve = intmap_canvas.getFigure().add_subplot()

        wlentrace_canvas = mplWidegt.MatplotlibWidget(size=(5.0, 5.0), dpi=125)
        layout.addWidget(wlentrace_canvas,0,1)
        self.wlentrace_curve = wlentrace_canvas.getFigure().add_subplot()

        timetrace_canvas = mplWidegt.MatplotlibWidget(size=(5.0, 5.0), dpi=125)
        layout.addWidget(timetrace_canvas,1,1)
        self.timetrace_curve = timetrace_canvas.getFigure().add_subplot()

        self.controls_widget = uic.loadUi("controls.ui")
        self.controls_widget.W0_value.setValidator(QDoubleValidator(-999.999,999.999,3))
        self.controls_widget.wlen_min.setValidator(QDoubleValidator(-999.999,999.999,3))
        self.controls_widget.wlen_max.setValidator(QDoubleValidator(-999.999,999.999,3))
        self.controls_widget.T0_value.setValidator(QDoubleValidator(-999.999,999.999,3))
        self.controls_widget.time_min.setValidator(QDoubleValidator(-999.999,999.999,3))
        self.controls_widget.time_max.setValidator(QDoubleValidator(-999.999,999.999,3))
        self.controls_widget.inty_min.setValidator(QDoubleValidator(-9999.999,9999.999,3))
        self.controls_widget.inty_max.setValidator(QDoubleValidator(-9999.999,9999.999,3))

        layout.addWidget(self.controls_widget,1,0)
        self._main.adjustSize()

    def closeEvent(self, event):
        TAS_Analiser.traceswidget.close()
        event.accept()

class TracesWidget(QMainWindow):
    def __init__(self):
        """ Windows to compare spectral and kinetics traces  """
        super(TracesWidget, self).__init__()
        self._main = QWidget()
        self.setCentralWidget(self._main)
        layout = QGridLayout(self._main)

        timetraces_canvas = mplWidegt.MatplotlibWidget(size=(2.0, 2.0), dpi=100)
        self.timetraces_plot = timetraces_canvas.getFigure().add_subplot()

        wlentraces_canvas = mplWidegt.MatplotlibWidget(size=(2.0, 2.0), dpi=100)
        self.wlentraces_plot = wlentraces_canvas.getFigure().add_subplot()

        layout.addWidget(timetraces_canvas,2,0)
        layout.addWidget(wlentraces_canvas,2,1)

        self._main.adjustSize()


class TAS_Analiser(object):
    def __init__(self):
        self.traceswidget = TracesWidget()
        self.plotswidget = PlotsWidget()
        self.plotswidget.showMaximized()
        self.connectUI()

    def connectUI(self):
        self.plotswidget.controls_widget.loadData_button.clicked.connect(self.loadData)
        self.plotswidget.controls_widget.correctData_button.clicked.connect(self.correctData)
        self.plotswidget.controls_widget.plotData_button.clicked.connect(self.plotData)
        self.plotswidget.controls_widget.exportplot_button.clicked.connect(self.exportPlot)
        self.plotswidget.controls_widget.gettraces_button.clicked.connect(self.getTraces)
        self.plotswidget.controls_widget.plottraces_button.clicked.connect(self.plotTraces)
        self.plotswidget.controls_widget.cleartraces_button.clicked.connect(self.clearTracesPlots)

        # self.mouse_clicked = pg.SignalProxy(self.plotswidget.intymap_plot.scene().sigMouseClicked, rateLimit=60, slot=self.mouseClicked)
        self.plotswidget.intmap_curve.figure.canvas.mpl_connect('motion_notify_event', self.cursorMove)
        self.plotswidget.intmap_curve.figure.canvas.mpl_connect('button_press_event', self.cursorClick)

    def writeUI(self, delta_wl, W0, wlen_min, wlen_max, T0, time_min, time_max, inty_min, inty_max):

        self.plotswidget.controls_widget.W0_value.setProperty("text",str(round(W0,3)))
        self.plotswidget.controls_widget.delta_wl.setProperty("text",str(round(delta_wl,3)))
        self.plotswidget.controls_widget.wlen_min.setProperty("text",str(round(wlen_min,3)))
        self.plotswidget.controls_widget.wlen_max.setProperty("text",str(round(wlen_max,3)))

        self.plotswidget.controls_widget.T0_value.setProperty("text",str(round(T0,3)))
        self.plotswidget.controls_widget.time_min.setProperty("text",str(round(time_min,3)))
        self.plotswidget.controls_widget.time_max.setProperty("text",str(round(time_max,3)))

        self.plotswidget.controls_widget.inty_min.setProperty("text",str(round(inty_min,3)))
        self.plotswidget.controls_widget.inty_max.setProperty("text",str(round(inty_max,3)))

        self.plotswidget.controls_widget.I0_value.setProperty("text",str(round(self.snr_t0,3)))

    def readUI(self):
        delta_wl = float(self.plotswidget.controls_widget.delta_wl.text())
        W0 = float(self.plotswidget.controls_widget.W0_value.text())
        wlen_min = float(self.plotswidget.controls_widget.wlen_min.text())
        wlen_max = float(self.plotswidget.controls_widget.wlen_max.text())

        T0 = float(self.plotswidget.controls_widget.T0_value.text())
        time_min = float(self.plotswidget.controls_widget.time_min.text())
        time_max = float(self.plotswidget.controls_widget.time_max.text())

        inty_min = float(self.plotswidget.controls_widget.inty_min.text())
        inty_max = float(self.plotswidget.controls_widget.inty_max.text())

        self.W0 = W0

        return delta_wl, W0, wlen_min, wlen_max, T0, time_min, time_max, inty_min, inty_max

    def loadData(self):
        file_dialog = QFileDialog()
        self.files_list, _ = file_dialog.getOpenFileNames()
        self.file_extension = os.path.splitext(os.path.basename(self.files_list[0]))[1]

        if self.file_extension=='.scan':
            try:
                self.W0
            except:
                self.W0 = 380.0
            else:
                pass
            finally:
                wl, tl, il, std_il = self.readJsonData(self.files_list)
                self.delta_wl = 1.0
                self.n = 4
                self.T0 = 0.5
        else:
            try:
                self.W0
            except:
                self.W0 = 435.0
            else:
                pass
            finally:
                wl, tl, il, std_il = self.readData(self.files_list)
                self.delta_wl = 1.0
                self.n = 6
                self.T0 = 0.5

        time_min, time_max = tl[0], tl[-1]
        wlen_min, wlen_max = wl[0], wl[-1]
        # inty_min, inty_max = il.min(), il.max()
        inty_min, inty_max = np.floor(np.amin(il)), np.ceil(abs(np.amin(il)))

        self.avg_wl = wl
        self.tl_t0 = tl
        self.avg_il = il
        self.avg_std_il = std_il

        self.std_t0 = np.std(self.avg_il[:,self.tl_t0<self.T0])
        self.var_t0 = np.var(self.avg_il[:,self.tl_t0<self.T0])
        self.mean_t0 = np.mean(self.avg_il[:,self.tl_t0<self.T0])
        self.cov_t0 = abs(self.std_t0/self.mean_t0)
        self.snr_t0 = abs(self.mean_t0/self.std_t0)

        self.writeUI(self.delta_wl, self.W0, wlen_min, wlen_max, self.T0, time_min, time_max, inty_min, inty_max)
        self.plotswidget.controls_widget.buttons_box.setEnabled(True)
        self.plotswidget.controls_widget.plotOpt_box.setEnabled(True)
        self.plotswidget.controls_widget.dataOpt_box.setEnabled(True)
        self.plotswidget.controls_widget.loadData_button.setProperty("text","New Data")

    def correctData(self):
        if (self.plotswidget.controls_widget.w0_checkbox.isChecked() or self.plotswidget.controls_widget.w0_checkbox.isChecked()):
            self.delta_wl, self.W0, wlen_min, wlen_max, self.T0, time_min, time_max, inty_min, inty_max = self.readUI()
            self.reloadData()
            if self.plotswidget.controls_widget.dlambda_checkbox.isChecked():
                self.smoothLambda()
                if self.plotswidget.controls_widget.chirp_checkbox.isChecked():
                    self.correctChirp()
            elif self.plotswidget.controls_widget.chirp_checkbox.isChecked():
                self.correctChirp()
            else:
                pass
        elif self.plotswidget.controls_widget.dlambda_checkbox.isChecked():
            self.reloadData()
            self.smoothLambda()
            if self.plotswidget.controls_widget.chirp_checkbox.isChecked():
                self.correctChirp()
        elif self.plotswidget.controls_widget.chirp_checkbox.isChecked():
            self.correctChirp()
        else:
            self.reloadData()

        time_min, time_max = self.tl_t0[0], self.tl_t0[-1]
        wlen_min, wlen_max = self.avg_wl[0], self.avg_wl[-1]
        inty_min, inty_max = np.floor(np.amin(self.avg_il)), np.ceil(abs(np.amin(self.avg_il)))
        self.writeUI(self.delta_wl, self.W0, wlen_min, wlen_max, self.T0, time_min, time_max, inty_min, inty_max)
        self.plotData()

    def reloadData(self):
        if self.file_extension=='.scan':
            wl, tl, il, std_il = self.readJsonData(self.files_list)
        else:
            wl, tl, il, std_il = self.readData(self.files_list)

        time_min, time_max = tl[0], tl[-1]
        wlen_min, wlen_max = wl[0], wl[-1]
        # inty_min, inty_max = np.amin(il), np.amax(il)
        inty_min, inty_max = np.floor(np.amin(il)), np.ceil(abs(np.amin(il)))

        self.avg_wl = wl
        self.tl_t0 = tl
        self.avg_il = il
        self.avg_std_il = std_il

        self.std_t0 = np.std(self.avg_il[:,self.tl_t0<=self.T0])
        self.var_t0 = np.var(self.avg_il[:,self.tl_t0<=self.T0])
        self.mean_t0 = np.mean(self.avg_il[:,self.tl_t0<=self.T0])
        self.cov_t0 = abs(self.std_t0/self.mean_t0)
        self.snr_t0 = abs(self.mean_t0/self.std_t0)**2

        self.writeUI(self.delta_wl, self.W0, wlen_min, wlen_max, self.T0, time_min, time_max, inty_min, inty_max)

    def readJsonData(self, files_list):
        for file in files_list:
            try:
                data
            except:
                data = pd.read_json(file, typ='series', convert_axes=np.float64)
                wl = np.asarray(data[0][0], dtype=np.float64)
                tl = np.asarray(data[1][0], dtype=np.float64)
                il = np.nan_to_num(np.asarray(data[2], dtype=np.float64).T)
                bkg = np.asarray(data[3][0], dtype=np.float64)
                il_i = il
            else:
                data = pd.read_json(file, typ='series', convert_axes=np.float64)
                wl = np.add(wl, np.asarray(data[0][0], dtype=np.float64))
                tl = np.add(tl, np.asarray(data[1][0], dtype=np.float64))
                il = np.add(il, np.nan_to_num(np.asarray(data[2], dtype=np.float64).T))
                bkg = np.add(bkg, np.asarray(data[3][0], dtype=np.float64))
                il_i = np.append(il_i, il)
            finally:
                self.dirname = os.path.dirname(file)
                splitteddir = os.path.split(self.dirname)
                self.sample_name = splitteddir[-2].split('/')[-1]+' -- '+splitteddir[-1]

        NFiles = len(files_list)

        il_i = np.reshape(np.asarray(il_i), (np.size(wl), np.size(tl), NFiles))

        ### Averaging and Std. Err. of measurements (data files)
        wl = (1.0/NFiles)*wl
        tl = 1e-3*(1.0/NFiles)*tl
        il = 1e3*(1.0/NFiles)*il

        # std =0.0
        # std_il = np.empty(0)
        # for i in range(np.shape(il_i)[0]):
        #     for j in range(np.shape(il_i)[1]):
        #         std_il = np.append(std_il, np.std(il_i[i,j,:]))
        # std_il = 1e3*(1.0/NFiles)*np.reshape(np.asarray(std_il),(np.size(wl), np.size(tl)))
        std_il = np.zeros((np.size(wl), np.size(tl)))
        ###

        # W0 correction
        old_len = np.size(wl)
        wl = np.delete(wl, np.where(wl < self.W0))
        new_len = np.size(wl)
        W0_index = old_len - new_len
        il = np.delete(np.delete(il, slice(0,0), 1), slice(0,W0_index), 0)
        ###

        ### Time data displacement to positive values
        tl = tl - tl[0]
        ###

        return wl, tl, il, std_il

    def readData(self, files_list):
        for file in files_list:
            try:
                data
            except:
                data = pd.read_csv(file, header=None, delim_whitespace=True, dtype=np.float64)
                wl = data.iloc[1:,0].to_numpy()
                tl = data.iloc[0,1:].to_numpy()
                il = data.iloc[1:,1:].to_numpy()
                il_i = il
            else:
                data = pd.read_csv(file, header=None, delim_whitespace=True, dtype=np.float64)
                wl = np.add(wl, data.iloc[1:,0].to_numpy())
                tl = np.add(tl, data.iloc[0,1:].to_numpy())
                il = np.add(il, data.iloc[1:,1:].to_numpy())
                il_i = np.append(il_i, il)
            finally:
                self.dirname = os.path.dirname(file)
                splitteddir = os.path.split(self.dirname)
                self.sample_name = splitteddir[-2].split('/')[-1]+' -- '+splitteddir[-1]

        NFiles = len(files_list)

        il_i = np.reshape(np.asarray(il_i), (np.size(wl), np.size(tl), NFiles))

        # Averaging and Std. Err. of measurements (data files)
        wl = (1.0/NFiles)*wl
        tl = (1.0/NFiles)*tl
        il = 1e3*(1.0/NFiles)*il

        # std =0.0
        # std_il = np.empty(0)
        # for i in range(np.shape(il_i)[0]):
        #     for j in range(np.shape(il_i)[1]):
        #         std_il = np.append(std_il, np.std(il_i[i,j,:]))
        # std_il = 1e3*(1.0/NFiles)*np.reshape(np.asarray(std_il),(np.size(wl), np.size(tl)))
        std_il = np.zeros((np.size(wl), np.size(tl)))
        ###
        # W0 correction
        old_len = np.size(wl)
        wl = np.delete(wl, np.where(wl < self.W0))
        new_len = np.size(wl)
        W0_index = old_len - new_len
        il = np.delete(np.delete(il, slice(0,0), 1), slice(0,W0_index), 0)
        ###

        ### Time data displacement to positive values
        tl = tl - tl[0]
        ###

        return wl, tl, il, std_il

    def smoothLambda(self):
        n_dwl = int(self.n*self.delta_wl)

        # averaging over delta wlen
        avg_wl = np.empty(0)
        avg_il = np.empty(0)
        std_il = np.empty(0)
        snr_il = np.empty(0)
        avg_std_il = np.empty(0)

        for j in range(0,np.size(self.avg_wl)-n_dwl,n_dwl):
            avg_wl = np.append(avg_wl, np.mean(self.avg_wl[j:j+n_dwl]))

            avg_il = np.append(avg_il, [np.mean(self.avg_il[j:j+n_dwl,i]) for i in range(np.size(self.tl_t0))])
            std_il = np.append(std_il, [np.std(self.avg_il[j:j+n_dwl,i]) for i in range(np.size(self.tl_t0))])

            avg_std_il = np.append(avg_std_il, [np.mean(self.avg_std_il[j:j+n_dwl,i]) for i in range(np.size(self.tl_t0))])

        avg_il = np.reshape(avg_il, (np.size(avg_wl), np.size(self.tl_t0)))
        std_il = np.reshape(std_il, (np.size(avg_wl), np.size(self.tl_t0)))
        avg_std_il = np.reshape(avg_std_il, (np.size(avg_wl), np.size(self.tl_t0)))

        snr_il = np.power(avg_il/std_il,2)

        time_min, time_max = self.tl_t0[0], self.tl_t0[-1]
        wlen_min, wlen_max = avg_wl[0], avg_wl[-1]
        inty_min, inty_max = np.amin(avg_il), np.amax(avg_il)

        self.avg_wl = avg_wl
        self.avg_il = avg_il
        self.std_il = std_il
        self.snr_il = snr_il
        self.avg_std_il = avg_std_il

        self.std_t0 = np.std(self.avg_il[:,self.tl_t0<=self.T0])
        self.var_t0 = np.var(self.avg_il[:,self.tl_t0<=self.T0])
        self.mean_t0 = np.mean(self.avg_il[:,self.tl_t0<=self.T0])
        self.cov_t0 = abs(self.std_t0/self.mean_t0)
        self.snr_t0 = abs(self.mean_t0/self.std_t0)**2

        self.writeUI(self.delta_wl, self.W0, wlen_min, wlen_max, self.T0, time_min, time_max, inty_min, inty_max)

    def correctChirp(self):
        new_avg_il = np.empty(np.shape(self.avg_il))
        new_avg_std_il = np.empty(np.shape(self.avg_std_il))

        wl_indexes = list(range(len(self.avg_wl)))
        T0_idx = len(self.tl_t0[self.tl_t0<=self.T0])
        tl_indexes = list(range(len(self.tl_t0)))

        unch_il = self.avg_il
        unch_std_il = self.avg_std_il
        unch_il[:,self.tl_t0<self.T0]=0.0


        ##########**********matplolib**********#################################
        # # [:3*T0_idx]
        # for wlidx in wl_indexes:
        #     for tlidx in tl_indexes[:2*T0_idx]:
        #         if abs(self.avg_il[wlidx,tlidx]-self.std_t0)>=self.mean_t0:
        #         # if self.snr_il[wlidx,tlidx]<=self.snr_t0:
        #             unch_il[wlidx][tlidx]=0.0
        #
        #
        # for wlidx in wl_indexes:
        #     # unch_il[wlidx][:][(unch_il[wlidx][:] - self.std_t0 < self.mean_t0) & (unch_il[wlidx][:] + self.std_t0 > self.mean_t0)]=0.0
        #     new_avg_il_row = np.delete(unch_il[wlidx],np.where(unch_il[wlidx] == 0.0))
        #     new_avg_il_row = np.append(new_avg_il_row, np.zeros(len(self.tl_t0)-len(new_avg_il_row)))
        #     new_avg_il[wlidx] = new_avg_il_row
        #     new_avg_std_il_row = np.delete(unch_il[wlidx],np.where(unch_il[wlidx] == 0.0))
        #     new_avg_std_il_row = np.append(new_avg_std_il_row, np.zeros(len(self.tl_t0)-len(new_avg_std_il_row)))
        #     new_avg_std_il[wlidx] = new_avg_std_il_row
        ########################################################################


        ##########**********pyqtgraph**********#################################
        new_avg_il = np.empty(np.shape(self.avg_il))
        new_avg_std_il = np.empty(np.shape(self.avg_std_il))

        T0_idx = len(self.tl_t0[self.tl_t0<=self.T0])

        unch_il = self.avg_il
        unch_std_il = self.avg_std_il

        std_t0 = np.std(self.avg_il[:,self.tl_t0<=self.T0])
        var_t0 = np.var(self.avg_il[:,self.tl_t0<=self.T0])
        mean_t0 = np.mean(self.avg_il[:,self.tl_t0<=self.T0])

        cov_t0 = abs(std_t0/mean_t0)
        snr_t0 = abs(mean_t0/std_t0)**2.0

        for wlidx in range(len(self.avg_wl)):
            unch_il[wlidx][:3*T0_idx][(unch_il[wlidx][:3*T0_idx]-std_t0 <= mean_t0) & (unch_il[wlidx][:3*T0_idx]+std_t0 >= mean_t0)] = 0.0
            new_avg_il_row = np.delete(unch_il[wlidx],np.where(unch_il[wlidx] == 0.0))
            new_avg_il_row = np.append(new_avg_il_row, np.zeros(len(self.tl_t0)-len(new_avg_il_row)))
            new_avg_il[wlidx] = new_avg_il_row
            new_avg_std_il_row = np.delete(unch_il[wlidx],np.where(unch_il[wlidx] == 0.0))
            new_avg_std_il_row = np.append(new_avg_std_il_row, np.zeros(len(self.tl_t0)-len(new_avg_std_il_row)))
            new_avg_std_il[wlidx] = new_avg_std_il_row
            # break
        ########################################################################

        ##########**********Fitting**********###################################
        # for wlidx in wl_indexes:
        #     for tlidx in range(3*T0_idx):
        #         std_t = np.std(self.avg_il[wlidx,:T0_idx+tlidx])
        #         var_t = np.var(self.avg_il[wlidx,:T0_idx+tlidx])
        #         mean_t = np.mean(self.avg_il[wlidx,:T0_idx+tlidx])
        #         cov_t = abs(std_t/mean_t)
        #         snr_t = abs(mean_t/std_t)**2
        #         # if snr_t<=self.snr_t0:
        #         if self.avg_il[wlidx][tlidx] - self.std_t0 < self.mean_t0 and self.avg_il[wlidx][tlidx] + self.std_t0 > self.mean_t0:
        #             unch_il[wlidx][:T0_idx+tlidx] = 0.0
        #             break
        #
        # t0idx=np.zeros(len(self.avg_wl))
        # for wlidx in wl_indexes:
        #     for tlidx in range(3*T0_idx):
        #         if unch_il[wlidx][tlidx]!=0.0:
        #             t0idx[wlidx]=tlidx
        #             break
        # time0 = [self.tl_t0[idx] for idx in t0idx]
        #
        # popt, pcov = curve_fit(self.fit_indexes, self.avg_wl, time0)
        # # a, b, c = round(popt[0],2), round(popt[1],2), round(popt[2],2)
        # a, b, c ,d = round(popt[0],2), round(popt[1],2), round(popt[2],2), round(popt[2],2)
        # new_time0 = self.fit_indexes(self.avg_wl, *popt)
        # # [print(m,n) for m,n in zip(t0idx,new_t0idx)new_time0]
        # for wlidx in wl_indexes:
        #     new_avg_il_row=unch_il[wlidx, self.tl_t0>=new_time0[wlidx]]
        #     new_avg_il_row = np.append(new_avg_il_row, np.zeros(len(self.tl_t0)-len(new_avg_il_row)))
        #     new_avg_il[wlidx] = new_avg_il_row
        ########################################################################

        self.avg_il = new_avg_il
        self.avg_std_il = new_avg_il

        self.plotData()

    def plotData(self):
        self.clearPlots()
        self.delta_wl, self.W0, wlen_min, wlen_max, self.T0, time_min, time_max, inty_min, inty_max = self.readUI()
        levels = np.linspace(inty_min, inty_max, 400, endpoint=True)
        midnorm = MidpointNormalize(vmin=inty_min, vcenter=0.0, vmax=inty_max)

        self.intmap = self.plotswidget.intmap_curve.contourf(self.tl_t0, self.avg_wl, self.avg_il, norm=midnorm, cmap='seismic', levels=levels, vmin=inty_min, vmax=inty_max, extend='both')
        self.intmap.ax.set_title(self.sample_name)
        self.intmap.autoscale()

        try:
            self.cbar
        except:
            self.cbar = self.plotswidget.intmap_curve.figure.colorbar(self.intmap, label='Intensity (mOD)', norm=midnorm, cmap='seismic', extend='both')
            self.cbar.ax.set_autoscale_on(True)
            self.cbar.draw_all()
        else:
            self.intmap.set_clim(vmin=inty_min, vmax=inty_max)
            self.intmap.set_norm(midnorm)
            self.cbar.draw_all()

        self.setPlotsAxes()
        self.plotswidget.intmap_curve.figure.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        self.image_name = self.dirname+'/intymap_{}-{}ps.png'.format(int(time_min), int(time_max))

    def exportPlot(self):
        self.plotswidget.intmap_curve.figure.savefig(self.image_name)

    def clearPlots(self):
        self.plotswidget.intmap_curve.clear()
        self.plotswidget.intmap_curve.figure.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        self.plotswidget.timetrace_curve.clear()
        self.plotswidget.wlentrace_curve.clear()
        self.traceswidget.timetraces_plot.clear()
        self.traceswidget.wlentraces_plot.clear()
    def setPlotsAxes(self):
        delta_wl, W0, wlen_min, wlen_max, T0, time_min, time_max, inty_min, inty_max = self.readUI()

        self.plotswidget.intmap_curve.set_xlim(time_min,time_max)
        self.plotswidget.intmap_curve.set_ylim(wlen_min,wlen_max)
        self.plotswidget.intmap_curve.set_xlabel('Time (ps)')
        self.plotswidget.intmap_curve.set_ylabel('Wavelength (nm)')

        self.plotswidget.wlentrace_curve.set_xlim(wlen_min,wlen_max)
        self.plotswidget.timetrace_curve.set_xlim(time_min,time_max)

        self.traceswidget.wlentraces_plot.set_xlim(wlen_min,wlen_max)
        self.traceswidget.wlentraces_plot.set_xlabel('Wavelength (nm)')
        self.traceswidget.wlentraces_plot.set_ylabel('Intensity')

        self.traceswidget.timetraces_plot.set_xlim(time_min,time_max)
        self.traceswidget.timetraces_plot.set_xlabel('Time (ps)')
        self.traceswidget.timetraces_plot.set_ylabel('Intensity')

        self.plotswidget.intmap_curve.figure.canvas.draw()
        self.plotswidget.wlentrace_curve.figure.canvas.draw()
        self.plotswidget.timetrace_curve.figure.canvas.draw()
        self.traceswidget.wlentraces_plot.figure.canvas.draw()
        self.traceswidget.timetraces_plot.figure.canvas.draw()

    def cursorMove(self, event):
        if not event.inaxes: return
        self.plotswidget.intmap_curve.figure.canvas.draw()

    def cursorClick(self, event):
        if not event.inaxes: return
        self.x, self.y = event.xdata, event.ydata

        delta_wl, W0, wlen_min, wlen_max, T0, time_min, time_max, inty_min, inty_max = self.readUI()

        self.wlen_idx = min(np.searchsorted(self.avg_wl, self.y), len(self.avg_wl) - 1)
        self.wmin_idx = min(np.searchsorted(self.avg_wl, wlen_min), len(self.avg_wl) - 1)
        self.wmax_idx = min(np.searchsorted(self.avg_wl, wlen_max), len(self.avg_wl) - 1)

        self.time_idx = min(np.searchsorted(self.tl_t0, self.x), len(self.tl_t0) - 1)
        self.tmin_idx = min(np.searchsorted(self.tl_t0, time_min), len(self.tl_t0) - 1)
        self.tmax_idx = min(np.searchsorted(self.tl_t0, time_max), len(self.tl_t0) - 1)

        self.wlen_trace = self.avg_il[self.wlen_idx, self.tmin_idx:self.tmax_idx]
        self.time_trace = self.avg_il[self.wmin_idx:self.wmax_idx, self.time_idx]

        self.wlen_std_trace = self.avg_std_il[self.wlen_idx, self.tmin_idx:self.tmax_idx]
        self.time_std_trace = self.avg_std_il[self.wmin_idx:self.wmax_idx, self.time_idx]

        self.plotswidget.wlentrace_curve.clear()
        self.plotswidget.wlentrace_curve.set_xlabel('Wavelength (nm)')
        self.plotswidget.wlentrace_curve.set_ylabel('Intensity (mOD)')

        self.plotswidget.timetrace_curve.clear()
        self.plotswidget.timetrace_curve.set_xlabel('Time (ps)')
        self.plotswidget.timetrace_curve.set_ylabel('Intensity (mOD)')

        if self.plotswidget.controls_widget.std_checkbox.isChecked():
            self.plotswidget.wlentrace_curve.errorbar(self.avg_wl[self.wmin_idx:self.wmax_idx], self.time_trace, yerr=self.time_std_trace, fmt='-', ecolor='k', elinewidth=0.5, mew=4, label=r'$\Delta$t={} ps'.format(round(self.x,2)))
            self.plotswidget.timetrace_curve.errorbar(self.tl_t0[self.tmin_idx:self.tmax_idx], self.wlen_trace, yerr=self.wlen_std_trace, fmt='-', ecolor='k', elinewidth=0.5, mew=4, label=r'$\lambda$={} nm'.format(round(self.y,2)))
            if self.plotswidget.controls_widget.fit_checkbox.isChecked():
                popt, pcov = curve_fit(self.fitfunc, self.tl[self.tmin_idx:self.tmax_idx], self.wlen_trace)
                a,b,c = round(popt[0],2),round(popt[1],2),round(popt[2],2)
                self.plotswidget.timetrace_curve.plot(self.tl_t0[self.tmin_idx:self.tmax_idx], self.fitfunc(self.tl_t0[self.tmin_idx:self.tmax_idx], *popt), 'r.', label=r'$\lambda$={} nm, $\tau$={} ps'.format(round(self.y,2),b))
        elif self.plotswidget.controls_widget.fft_checkbox.isChecked():
            self.plotswidget.wlentrace_curve.plot(self.avg_wl[self.wmin_idx:self.wmax_idx], self.time_trace, linewidth=1, label=r'$\Delta$t={} ps'.format(round(self.x,2)))
            self.plotswidget.timetrace_curve.set_xlabel(r'Wavenumber ($cm^{-1}$)')
            self.plotswidget.timetrace_curve.set_ylabel('Amplitude')

            hamm = 1.0#np.hanning(self.wlen_trace.size)

            self.time_trace_fft = np.fft.rfft(hamm*self.wlen_trace)
            self.time_trace_wnum = (33.35641)*np.fft.rfftfreq(self.wlen_trace.size, 0.03)

            data_fft = 2.0*self.time_trace_fft[:self.wlen_trace.size]/self.wlen_trace.size
            self.plotswidget.timetrace_curve.plot(self.time_trace_wnum[1:self.wlen_trace.size], data_fft[1:self.wlen_trace.size], linewidth=1, label=r'$\lambda$={} nm'.format(round(self.y,2)))
        else:
            self.plotswidget.wlentrace_curve.plot(self.avg_wl[self.wmin_idx:self.wmax_idx], self.time_trace, linewidth=1, label=r'$\Delta$t={} ps'.format(round(self.x,2)))
            self.plotswidget.timetrace_curve.plot(self.tl_t0[self.tmin_idx:self.tmax_idx], self.wlen_trace, linewidth=1, label=r'$\lambda$={} nm'.format(round(self.y,2)))
            if self.plotswidget.controls_widget.fit_checkbox.isChecked():
                popt, pcov = curve_fit(self.fitfunc, self.tl_t0[self.tmin_idx:self.tmax_idx], self.wlen_trace)
                a,b,c = round(popt[0],2),round(popt[1],2),round(popt[2],2)
                self.plotswidget.timetrace_curve.plot(self.tl_t0[self.tmin_idx:self.tmax_idx], self.fitfunc(self.tl_t0[self.tmin_idx:self.tmax_idx], *popt), 'r.', label=r'$\lambda$={} nm, $\tau$={} ps'.format(round(self.y,2),b))

        self.plotswidget.wlentrace_curve.legend()
        self.plotswidget.wlentrace_curve.figure.canvas.draw()
        self.plotswidget.wlentrace_curve.figure.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

        self.plotswidget.timetrace_curve.legend()
        self.plotswidget.timetrace_curve.figure.canvas.draw()
        self.plotswidget.timetrace_curve.figure.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    def getTraces(self):
        self.traceswidget.wlentraces_plot.plot(self.avg_wl[self.wmin_idx:self.wmax_idx], self.time_trace, linewidth=1,label=r'$\Delta$t={} ps'.format(round(self.x,2)))
        self.traceswidget.wlentraces_plot.legend()

        self.traceswidget.timetraces_plot.plot(self.tl_t0[self.tmin_idx:self.tmax_idx], self.wlen_trace, linewidth=1,label=r'$\lambda$={} nm'.format(round(self.y,2)))
        self.traceswidget.timetraces_plot.legend()

    def plotTraces(self):
        self.traceswidget.wlentraces_plot.set_xlabel('Wavelenght (nm)')
        self.traceswidget.wlentraces_plot.set_ylabel('Intensity (mOD)')
        self.traceswidget.wlentraces_plot.figure.canvas.draw()
        self.traceswidget.wlentraces_plot.figure.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

        self.traceswidget.timetraces_plot.set_xlabel('Time (ps)')
        self.traceswidget.timetraces_plot.set_ylabel('Intensity (mOD)')
        self.traceswidget.timetraces_plot.figure.canvas.draw()
        self.traceswidget.timetraces_plot.figure.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

        self.traceswidget.show()

    def clearTracesPlots(self):
        self.traceswidget.wlentraces_plot.clear()
        self.traceswidget.wlentraces_plot.figure.canvas.draw()

        self.traceswidget.timetraces_plot.clear()
        self.traceswidget.timetraces_plot.figure.canvas.draw()

    def fitfunc(self, x, a, b, c):
        return a*np.exp(-b*x)+c
    def fit_indexes(self, x, a, b, c,d):
        return a*x**2+b*x+c+x**d

if __name__=='__main__':
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setPalette(palette)
    TAS_Analiser.plotswidget = PlotsWidget()
    TAS_Analiser.traceswidget = TracesWidget()
    tas_analiser = TAS_Analiser()
    tas_analiser.plotswidget.showMaximized()
    sys.exit(app.exec_())

















#################################################
### ------------ fitting tools -------------- ###

# from scipy.optimize import curve_fit
# from scipy.signal import savgol_filter
# from lmfit import Minimizer, Parameters

    # def func(self, pars, x, data=None):
    #     a, b, c = pars['a'], pars['b'], pars['c']
    #     model = a * np.exp(-b*x) + c
    #     if data is None:
    #         return model
    #     return model - data
    #
    #
    # def dfunc(self, pars, x, data=None):
    #     a, b = pars['a'], pars['b']
    #     v = np.exp(-b*x)
    #     return np.array([v, -a*x*v, np.ones(len(x))])
    #
    #
    # def fitting(self, var, x):
    #     params = Parameters()
    #     params.add('a', value=10)
    #     params.add('b', value=10)
    #     params.add('c', value=10)
    #
    #     min2 = Minimizer(self.func, params, fcn_args=(self.tl,), fcn_kws={'data': wlen_trace})
    #     out2 = min2.leastsq(Dfun=self.dfunc, col_deriv=1)
    #     fit2 = func(out2.params, self.tl)
    #     a=out2.params['a']
    #     b=out2.params['b']
    #     c=out2.params['c']

#
# params = Parameters()
# params.add('a', value=10)
# params.add('b', value=10)
# params.add('c', value=10)
#
# a, b, c = 2.5, 1.3, 0.8
# x = np.linspace(0, 4, 50)
# y = f([a, b, c], x)
# data = y + 0.15*np.random.normal(size=x.size)
#
# # fit without analytic derivative
# min1 = Minimizer(func, params, fcn_args=(x,), fcn_kws={'data': data})
# out1 = min1.leastsq()
# fit1 = func(out1.params, x)
#
# # fit with analytic derivative
# min2 = Minimizer(func, params, fcn_args=(x,), fcn_kws={'data': data})
# out2 = min2.leastsq(Dfun=dfunc, col_deriv=1)
# fit2 = func(out2.params, x)
