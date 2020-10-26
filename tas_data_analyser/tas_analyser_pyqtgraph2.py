import os
import sys
import glob

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy import interpolate

from scipy.signal.windows import hann, hamming # FFT window functions
import nfft # Non-uniform FFT

import matplotlib.pyplot as plt

from gui_config import *

def exp_decay(x, p):
    '''
    Exponential decay of population.
    It is a bi-exponential decay function
    '''
    return p[2]*np.exp(-x/p[0]) + p[3]*np.exp(-x/p[1]) + p[4]

def exp_decay_fit(p, x, y):
    '''
    Exponential decay of population.
    This function is called from the nonlinear fitting engine
    '''
    return exp_decay(x, p) - y

def vib_decay(x, p):
    '''
    Vibrational coherence function.
    It is aimed to model the residuals form the bi-exponential decay
    '''
    return p[0]*np.multiply(np.exp(-x/p[1]), np.sin(p[2]*x/5309 + p[3])) + p[4]

def vib_decay_fit(p ,x, y):
    '''
    Vibrational coherence function.
    This function is called from the nonlinear fitting engine
    '''
    return vib_decay(x, p) - y


class MainWindow(QMainWindow):
    '''
    Fron this windows multiple instances of the TAS analyser can be opened and
    it can display a comparison frame for curves of different set of data (samples)
    '''
    def __init__(self):
        super(MainWindow, self).__init__()
        self._main = QWidget()
        self.setCentralWidget(self._main)
        layout = QGridLayout(self._main)

        # List of TAS analyser instances, to be manipulated (open, close)
        self.list_of_windows = list()

        self.setWindowTitle('T.A.S. Analyser Suite')
        self.setGeometry(20,20,1800,1200)

        self.exitAct = QAction('Exit', self)
        self.exitAct.setShortcut('Ctrl+Q')
        self.exitAct.setStatusTip('Exit application')
        self.exitAct.triggered.connect(self.close)

        self.newGraphAct = QAction('New DataSet', self)
        self.newGraphAct.setShortcut('Ctrl+N')
        self.newGraphAct.setStatusTip("Open a new TAS analyser window")
        self.newGraphAct.triggered.connect(self.graphWindow)

        self.menubar = self.menuBar()
        self.fileMenu = self.menubar.addMenu('&Menu')
        self.fileMenu.addAction(self.exitAct)
        self.fileMenu.addAction(self.newGraphAct)

        self._main.adjustSize()

        self.exportedTracesWidget = TracesWidget()
        self.list_of_windows.append(self.exportedTracesWidget)

    def closeEvent(self, event):
        self.deleteLater()
        for window in self.list_of_windows:
            window.close()
        event.accept()
            # close = QMessageBox.question(self,"Warning","Confirm to close the application, every child window will be closed and not saved data will be lost", QMessageBox.Yes | QMessageBox.No)
            # if close == QMessageBox.Yes:
            #     for window in self.list_of_windows:
            #         window.close()
            #     event.accept()
            # else:
            #     event.ignore()

    def graphWindow(self):
        graph_window = TAS_Analiser(self)
        # graph_window.showMaximized()
        self.list_of_windows.append(graph_window)

class ClassName(object):
    """docstring for ."""

    def __init__(self, arg):
        super(, self).__init__()
        self.arg = arg

def IntensityMap2D(self):
    self.intymap_plotWidget = pg.PlotWidget()
    self.intymap_plotItem = self.intymap_plotWidget.getPlotItem()
    self.intymap_plotItem.enableAutoRange(enable=True)
    self.intymap_plotItem.setDownsampling(auto=True)
    self.intymap_plotItem.setLabel('left',  'Wavelength', units='nm', **labelstyle)
    self.intymap_plotItem.setLabel('bottom', 'Time', units='ps', **labelstyle)

    self.intymap_imageItem = pg.ImageItem()
    self.intymap_imageItem.setLookupTable(BuRd_lut, update=True)

    self.intymap_imageViewWidget = pg.ImageView(parent=self.intymap_plotWidget, view=self.intymap_plotItem, imageItem=self.intymap_imageItem)
    self.intymap_imageViewWidget.getView().invertY(False)
    self.intymap_imageViewWidget.setColorMap(BuRd_map)
    self.intymap_imageViewWidget.autoLevels()
    self.intymap_imageViewWidget.ui.menuBtn.hide()
    self.intymap_imageViewWidget.ui.roiBtn.hide()

    self.intymap_histogramWidget = self.intymap_imageViewWidget.getHistogramWidget()
    self.intymap_histogramWidget.setImageItem(self.intymap_imageViewWidget.getImageItem())
    self.intymap_histogramWidget.setBackground(None)
    # self.intymap_histogramItem = pg.HistogramLUTItem(self.intymap_imageItem, fillHistogram=True)
    # self.intymap_histogramItem.autoHistogramRange()
    # self.intymap_histogramWidget.addItem(self.intymap_histogramItem)
    # self.intymap_histogramWidget.setColorMap(BuRd_map)

    # self.intymap_plotItem.vb.addItem(self.intymap_imageItem)
    # self.intymap_plotWidget.addItem(self.intymap_histogramWidget)

    #cross hair
    self.intymap_vLine = pg.InfiniteLine(angle=90, pen='b', movable=False)
    self.intymap_hLine = pg.InfiniteLine(angle=0, pen='b', movable=False)

    self.intymap_plotItem.addItem(self.intymap_imageItem)
    self.intymap_plotItem.addItem(self.intymap_vLine, ignoreBounds=True)
    self.intymap_plotItem.addItem(self.intymap_hLine, ignoreBounds=True)

    self.plot_export = pg.exporters.ImageExporter(self.intymap_plotItem)
    self.plot_export.parameters()['height'] = 2000

    # self.hist_export = pg.exporters.ImageExporter(self.intymap_histogram.scene())
    # self.hist_export.parameters()['height'] = 2000


class TASAnalyserGUI(QMainWindow):
    '''
    This windows contains the graphical widgets for TAS analyser
    - 2D map of the absorption spectrum
    - Kinetics and spectral plots wits their respective residual and FFT analysis
    - Control box
    '''
    def __init__(self):
        super(QMainWindow, self).__init__()
        self._main = QWidget()
        self.setCentralWidget(self._main)
        layout = QGridLayout(self._main)

        self.setWindowTitle('TAS Analyser')
        self.setGeometry(20,20,1800,1200)

        self.widget_vsplit = QSplitter(Qt.Vertical)
        self.widget_hsplit = QSplitter(Qt.Horizontal)
        self.traces_widget = pg.GraphicsLayoutWidget()

        self.controlsWidget()
        self.absorption_map2d()
        self.plotTracesWidget()

        self.widget_vsplit.addWidget(self.intymap_imageViewWidget)
        self.widget_vsplit.addWidget(self.controls_widget)
        self.widget_hsplit.addWidget(self.widget_vsplit)
        self.widget_hsplit.addWidget(self.traces_widget)
        layout.addWidget(self.widget_hsplit,0,0)

        self._main.adjustSize()

        self.mouse_moved = pg.SignalProxy(self.intymap_plotItem.scene().sigMouseMoved, rateLimit=60, slot=self.mouseMoved)

    def closeEvent(self, event):
            self.deleteLater()
            # close = QMessageBox.question(self,"Warning","Confirm to close the application, not saved data will be lost", QMessageBox.Yes | QMessageBox.No)
            # if close == QMessageBox.Yes:
            #     event.accept()
            # else:
            #     event.ignore()

            event.accept()

    def controlsWidget(self):
        self.controls_widget = uic.loadUi("controls.ui")
        self.controls_widget.W0_value.setValidator(QDoubleValidator(-999.999,999.999,3))
        self.controls_widget.wlen_min.setValidator(QDoubleValidator(-999.999,999.999,3))
        self.controls_widget.wlen_max.setValidator(QDoubleValidator(-999.999,999.999,3))
        self.controls_widget.T0_value.setValidator(QDoubleValidator(-999.999,999.999,3))
        self.controls_widget.time_min.setValidator(QDoubleValidator(-999.999,999.999,3))
        self.controls_widget.time_max.setValidator(QDoubleValidator(-999.999,999.999,3))
        self.controls_widget.inty_min.setValidator(QDoubleValidator(-9999.999,9999.999,3))
        self.controls_widget.inty_max.setValidator(QDoubleValidator(-9999.999,9999.999,3))

    def plotTracesWidget(self):
        self.kinetics_plot = self.traces_widget.addPlot(0,0)
        self.kinetics_plot.addLegend()
        self.kinetics_plot.showGrid(x=True, y=True)

        self.kinetics_curve = self.kinetics_plot.plot(pen='b')
        self.kinetics_curve_fit = self.kinetics_plot.plot(pen='r')
        # self.kinetics_curve_fft = self.kinetics_plot.plot(pen='w')

        self.spectral_plot = self.traces_widget.addPlot(1,0)
        self.spectral_plot.setLabel('left', '<math>&Delta;A', units='OD', unitPrefix='m', **labelstyle)
        self.spectral_plot.setLabel('bottom', 'Wavelength', units='nm', **labelstyle)
        self.spectral_plot.addLegend()
        self.spectral_plot.showGrid(x=True, y=True)
        self.spectral_curve = self.spectral_plot.plot(pen='b')

    def absorption_map2d(self):
        self.intymap_plotWidget = pg.PlotWidget()
        self.intymap_plotItem = self.intymap_plotWidget.getPlotItem()
        self.intymap_plotItem.enableAutoRange(enable=True)
        self.intymap_plotItem.setDownsampling(auto=True)
        self.intymap_plotItem.setLabel('left',  'Wavelength', units='nm', **labelstyle)
        self.intymap_plotItem.setLabel('bottom', 'Time', units='ps', **labelstyle)

        self.intymap_imageItem = pg.ImageItem()
        self.intymap_imageItem.setLookupTable(BuRd_lut, update=True)

        self.intymap_imageViewWidget = pg.ImageView(parent=self.intymap_plotWidget, view=self.intymap_plotItem, imageItem=self.intymap_imageItem)
        self.intymap_imageViewWidget.getView().invertY(False)
        self.intymap_imageViewWidget.setColorMap(BuRd_map)
        self.intymap_imageViewWidget.autoLevels()
        self.intymap_imageViewWidget.ui.menuBtn.hide()
        self.intymap_imageViewWidget.ui.roiBtn.hide()

        self.intymap_histogramWidget = self.intymap_imageViewWidget.getHistogramWidget()
        self.intymap_histogramWidget.setImageItem(self.intymap_imageViewWidget.getImageItem())
        self.intymap_histogramWidget.setBackground(None)
        # self.intymap_histogramItem = pg.HistogramLUTItem(self.intymap_imageItem, fillHistogram=True)
        # self.intymap_histogramItem.autoHistogramRange()
        # self.intymap_histogramWidget.addItem(self.intymap_histogramItem)
        # self.intymap_histogramWidget.setColorMap(BuRd_map)

        # self.intymap_plotItem.vb.addItem(self.intymap_imageItem)
        # self.intymap_plotWidget.addItem(self.intymap_histogramWidget)

        #cross hair
        self.intymap_vLine = pg.InfiniteLine(angle=90, pen='b', movable=False)
        self.intymap_hLine = pg.InfiniteLine(angle=0, pen='b', movable=False)

        self.intymap_plotItem.addItem(self.intymap_imageItem)
        self.intymap_plotItem.addItem(self.intymap_vLine, ignoreBounds=True)
        self.intymap_plotItem.addItem(self.intymap_hLine, ignoreBounds=True)

        self.plot_export = pg.exporters.ImageExporter(self.intymap_plotItem)
        self.plot_export.parameters()['height'] = 2000

        # self.hist_export = pg.exporters.ImageExporter(self.intymap_histogram.scene())
        # self.hist_export.parameters()['height'] = 2000

    def mouseMoved(self, event):
        pos = event[0]  ## using signal proxy turns original arguments into a tuple
        if self.intymap_plotItem.sceneBoundingRect().contains(pos):
            self.mousePoint = self.intymap_plotItem.vb.mapSceneToView(pos)
            lx = self.mousePoint.x()
            ly = self.mousePoint.y()
            # self.intymap_vLine.setPos(lx)
            # self.intymap_hLine.setPos(ly)


class TracesWidget(QMainWindow):
    def __init__(self):
        """ Windows to compare spectral and kinetics traces  """
        super(QMainWindow, self).__init__()
        layout = pg.GraphicsLayoutWidget()

        self.kinetics_traces = layout.addPlot(0,0)
        self.kinetics_traces.setLabel('left', '<math>&Delta;A', units='OD', unitPrefix='m', **labelstyle)
        self.kinetics_traces.setLabel('bottom', 'Time', units='ps', **labelstyle)
        self.kinetics_traces.addLegend(offset=(-50, -50))

        self.spectral_traces = layout.addPlot(0,1)
        self.spectral_traces.setLabel('left', '<math>&Delta;A', units='OD', unitPrefix='m', **labelstyle)
        self.spectral_traces.setLabel('bottom', 'Wavelength', units='nm', **labelstyle)
        self.spectral_traces.addLegend(offset=(-50, -50))

        self.setCentralWidget(layout)
        self.adjustSize()


class TAS_Analiser(QMainWindow):
    def __init__(self, parent=None):
        super(QMainWindow, self).__init__(parent)
        self.parent = parent
        self.gotTracesWidget = TracesWidget()
        self.parent.list_of_windows.append(self.gotTracesWidget)
        self.plotswidget = PlotsWidget()
        self.plotswidget.showMaximized()
        self.connectUI()

    def closeEvent(self, event):
        self.deleteLater()
        self.plotswidget.close()

    def connectUI(self):
        self.plotswidget.controls_widget.loadData_button.clicked.connect(self.loadData)
        self.plotswidget.controls_widget.correctData_button.clicked.connect(self.correctData)
        self.plotswidget.controls_widget.plotData_button.clicked.connect(self.plotData)
        self.plotswidget.controls_widget.exportplot_button.clicked.connect(self.exportPlot)
        self.plotswidget.controls_widget.gettraces_button.clicked.connect(self.getTraces)
        self.plotswidget.controls_widget.plottraces_button.clicked.connect(self.plotTraces)
        self.plotswidget.controls_widget.exporttraces_button.clicked.connect(self.exportTraces)
        self.plotswidget.controls_widget.cleartraces_button.clicked.connect(self.clearTracesPlots)

        self.mouse_clicked = pg.SignalProxy(self.plotswidget.intymap_plotItem.scene().sigMouseClicked, rateLimit=60, slot=self.mouseClicked)

    def writeUI(self, delta_wl, W0, wlen_min, wlen_max, T0, time_min, time_max, inty_min, inty_max):

        self.plotswidget.controls_widget.W0_value.setText(str(round(W0,3)))
        self.plotswidget.controls_widget.delta_wl.setText(str(round(delta_wl,3)))
        self.plotswidget.controls_widget.wlen_min.setText(str(round(wlen_min,3)))
        self.plotswidget.controls_widget.wlen_max.setText(str(round(wlen_max,3)))

        self.plotswidget.controls_widget.T0_value.setText(str(round(T0,3)))
        self.plotswidget.controls_widget.time_min.setText(str(round(time_min,3)))
        self.plotswidget.controls_widget.time_max.setText(str(round(time_max,3)))

        self.plotswidget.controls_widget.inty_min.setText(str(round(inty_min,3)))
        self.plotswidget.controls_widget.inty_max.setText(str(round(inty_max,3)))

        self.plotswidget.controls_widget.I0_value.setText(str(round(self.snr_t0,3)))

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
        self.file_dirname = self.dirname = os.path.dirname(self.files_list[0])
        self.file_basename = os.path.splitext(os.path.basename(self.files_list[0]))[0]
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
        self.std_il = std_il

        self.std_t0 = np.std(self.avg_il[:,self.tl_t0<self.T0])
        self.var_t0 = np.var(self.avg_il[:,self.tl_t0<self.T0])
        self.mean_t0 = np.mean(self.avg_il[:,self.tl_t0<self.T0])
        self.cov_t0 = abs(self.std_t0/self.mean_t0)
        self.snr_t0 = abs(self.mean_t0/self.std_t0)

        self.writeUI(self.delta_wl, self.W0, wlen_min, wlen_max, self.T0, time_min, time_max, inty_min, inty_max)
        self.plotswidget.controls_widget.buttons_box.setEnabled(True)
        self.plotswidget.controls_widget.plotOpt_box.setEnabled(True)
        self.plotswidget.controls_widget.dataOpt_box.setEnabled(True)
        self.plotswidget.controls_widget.loadData_button.setText("New Data")
        self.plotswidget.controls_widget.filename_loaded.setText(self.files_list[0])
        self.plotswidget.controls_widget.dataname_input.setText(self.file_basename)

    def correctData(self):
        if (self.plotswidget.controls_widget.w0_checkbox.isChecked() or self.plotswidget.controls_widget.t0_checkbox.isChecked()):
            self.delta_wl, self.W0, wlen_min, wlen_max, self.T0, time_min, time_max, inty_min, inty_max = self.readUI()
            self.reloadData()
            # self.avg_il[self.avg_wl>=self.W0, :]
            # self.avg_il[:, self.tl_t0>=self.T0]
            # self.avg_wl = self.avg_wl[self.avg_wl>=self.W0]
            # self.tl_t0 = self.tl_t0[self.tl_t0>=self.T0]

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
        self.std_il = std_il

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
        il = (1.0/NFiles)*il

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
        il = (1.0/NFiles)*il
        # considering dat files are coming from the pharos TA system thus the only file is already averaged data
        std_il = il

        # std =0.0
        # std_il = np.empty(0)
        # for i in range(np.shape(il_i)[0]):
        #     for j in range(np.shape(il_i)[1]):
        #         std_il = np.append(std_il, np.std(il_i[i,j,:]))
        # std_il = 1e3*(1.0/NFiles)*np.reshape(np.asarray(std_il),(np.size(wl), np.size(tl)))
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

        # averaging over delta wlen, now std_il is coming from wavelenght averaged data
        avg_wl = np.empty(0)
        avg_il = np.empty(0)
        std_il = np.empty(0)
        snr_il = np.empty(0)
        avg_std_il = np.empty(0)

        for j in range(0,np.size(self.avg_wl)-n_dwl,n_dwl):
            avg_wl = np.append(avg_wl, np.mean(self.avg_wl[j:j+n_dwl]))

            avg_il = np.append(avg_il, [np.mean(self.avg_il[j:j+n_dwl,i]) for i in range(np.size(self.tl_t0))])
            std_il = np.append(std_il, [np.std(self.avg_il[j:j+n_dwl,i]) for i in range(np.size(self.tl_t0))])

        avg_il = np.reshape(avg_il, (np.size(avg_wl), np.size(self.tl_t0)))
        std_il = np.reshape(std_il, (np.size(avg_wl), np.size(self.tl_t0)))

        snr_il = np.power(avg_il/std_il,2)

        time_min, time_max = self.tl_t0[0], self.tl_t0[-1]
        wlen_min, wlen_max = avg_wl[0], avg_wl[-1]
        inty_min, inty_max = np.amin(avg_il), np.amax(avg_il)

        self.avg_wl = avg_wl
        self.avg_il = avg_il
        self.std_il = std_il
        self.snr_il = snr_il

        self.std_t0 = np.std(self.avg_il[:,self.tl_t0<=self.T0])
        self.var_t0 = np.var(self.avg_il[:,self.tl_t0<=self.T0])
        self.mean_t0 = np.mean(self.avg_il[:,self.tl_t0<=self.T0])
        self.cov_t0 = abs(self.std_t0/self.mean_t0)
        self.snr_t0 = abs(self.mean_t0/self.std_t0)**2

        self.writeUI(self.delta_wl, self.W0, wlen_min, wlen_max, self.T0, time_min, time_max, inty_min, inty_max)

    def correctChirp(self):
        new_avg_il = np.empty(np.shape(self.avg_il))
        new_std_il = np.empty(np.shape(self.std_il))

        wl_indexes = list(range(len(self.avg_wl)))
        T0_idx = len(self.tl_t0[self.tl_t0<=self.T0])
        tl_indexes = list(range(len(self.tl_t0)))

        unch_il = self.avg_il
        unch_std_il = self.std_il
        unch_il[:,self.tl_t0<self.T0]=0.0

        il_max = np.empty((0))
        new_t0 = np.empty((0))
        for wlidx in wl_indexes:
            wls = np.abs(unch_il[wlidx][self.tl_t0<=3*self.T0])
            il_max = np.append(il_max, np.where(wls==wls.max()))
            new_t0 = np.append(new_t0, self.tl_t0[np.where(wls==wls.max())])

        plt.plot(self.avg_wl, new_t0)

        popt, pcov = curve_fit(self.fit_indexes, self.avg_wl, new_t0)
        new_il_max = self.fit_indexes(self.avg_wl, *popt)

        plt.plot(self.avg_wl, new_il_max, 'r')
        plt.show()

        # for wlidx in wl_indexes:
        #     new_avg_il_row=unch_il[wlidx, self.tl_t0>=new_time0[wlidx]]
        #     new_avg_il_row = np.append(new_avg_il_row, np.zeros(len(self.tl_t0)-len(new_avg_il_row)))
        #     new_avg_il[wlidx] = new_avg_il_row

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
        new_std_il = np.empty(np.shape(self.std_il))

        T0_idx = len(self.tl_t0[self.tl_t0<=self.T0])

        unch_il = self.avg_il
        unch_std_il = self.std_il

        std_t0 = np.std(self.avg_il[:,self.tl_t0<=self.T0])
        var_t0 = np.var(self.avg_il[:,self.tl_t0<=self.T0])
        mean_t0 = np.mean(self.avg_il[:,self.tl_t0<=self.T0])

        cov_t0 = abs(std_t0/mean_t0)
        snr_t0 = abs(mean_t0/std_t0)**2.0

        for wlidx in range(len(self.avg_wl)):
            unch_il[wlidx][:3*T0_idx][(unch_il[wlidx][:3*T0_idx]-std_t0 <= mean_t0) & (unch_il[wlidx][:3*T0_idx]+std_t0 >= mean_t0)] = 0.0
            new_avg_il_row = np.delete(unch_il[wlidx],np.where(unch_il[wlidx] == 0.0))
            new_avg_il_row = np.append(new_avg_il_row, new_avg_il_row[-1]*np.ones(len(self.tl_t0)-len(new_avg_il_row)))
            new_avg_il[wlidx] = new_avg_il_row

            unch_std_il[wlidx][:3*T0_idx][(unch_std_il[wlidx][:3*T0_idx]-std_t0 <= mean_t0) & (unch_std_il[wlidx][:3*T0_idx]+std_t0 >= mean_t0)] = 0.0
            new_std_il_row = np.delete(unch_std_il[wlidx],np.where(unch_std_il[wlidx] == 0.0))
            new_std_il_row = np.append(new_std_il_row, new_std_il_row[-1]*np.ones(len(self.tl_t0)-len(new_std_il_row)))
            new_std_il[wlidx] = new_std_il_row
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
        self.std_il = new_std_il

        self.plotData()

    def updateHistogram(self):
        hist_arr = self.plotswidget.intymap_imageItem.getHistogram()
        self.plotswidget.intymap_histogramWidget.setHistogramRange(hist_arr[0].min(), hist_arr[0].max(), padding=None)
        self.plotswidget.intymap_histogramWidget.setLevels(hist_arr[0].min(), hist_arr[0].max())

        hist_indx0 = np.where(hist_arr[0]==hist_arr[0].min())[0][0]
        hist_indx1 = np.where(hist_arr[1]==hist_arr[1].max())[0][0]
        hist_indx2 = np.where(hist_arr[0]==hist_arr[0].max())[0][0]

        hist_pos_values = hist_arr[0][hist_arr[0]>=0.0]
        hist_neg_values = hist_arr[0][hist_arr[0]<0.0]
        grad_pos_linsp = np.linspace(0.5, 1.0, num=len(hist_pos_values), endpoint=True)
        grad_neg_linsp = np.linspace(0.0, 0.5, num=len(hist_neg_values), endpoint=False)
        grad_pos_interp = np.interp(grad_pos_linsp, [hist_pos_values.min(), hist_pos_values.max()], [grad_pos_linsp.min(), grad_pos_linsp.max()])
        grad_neg_interp =  np.interp(grad_neg_linsp, [hist_neg_values.min(), hist_neg_values.max()], [grad_neg_linsp.min(), grad_neg_linsp.max()])

        hist_new = hist_arr[0]/hist_arr[0][0]+1.0
        grad_linsp = np.linspace(0.0, 1.0, num=len(hist_arr[0]), endpoint=True)
        grad_interp = np.interp(grad_linsp, hist_new, grad_linsp)

        hist_gradient = self.plotswidget.intymap_histogramWidget.gradient.getGradient()

        grad_neg_interp = np.interp(hist_pos_values, [hist_pos_values.min(), 0.0], [0.0, 0.5])
        grad_pos_interp =  np.interp(hist_neg_values, [0.0, hist_neg_values.max()], [0.5, 1.0])

        self.plotswidget.intymap_histogramWidget.gradient.setTickValue(0, 0.0)
        self.plotswidget.intymap_histogramWidget.gradient.setTickValue(1, 0.5)
        self.plotswidget.intymap_histogramWidget.gradient.setTickValue(2, 1.0)

    def plotData(self):
        self.delta_wl, self.W0, wlen_min, wlen_max, self.T0, time_min, time_max, inty_min, inty_max = self.readUI()
        time_min_idx = np.where(np.around(self.tl_t0,decimals=3)==time_min)[0][0]
        time_max_idx = np.where(np.around(self.tl_t0,decimals=3)==time_max)[0][0]
        wlen_min_idx = np.where(np.around(self.avg_wl,decimals=3)==wlen_min)[0][0]
        wlen_max_idx = np.where(np.around(self.avg_wl,decimals=3)==wlen_max)[0][0]
        self.image = self.avg_il[wlen_min_idx:wlen_max_idx, time_min_idx:time_max_idx]
        # self.plotswidget.intymap_imageItem.setImage(self.image)

        self.plotswidget.intymap_imageViewWidget.setImage(self.image, autoRange=True, autoLevels=True, autoHistogramRange=True)

        self.updateHistogram()

        self.plotswidget.intymap_imageItem.translate(0, self.W0)
        self.plotswidget.intymap_imageItem.scale((time_max-time_min)/np.size(self.image[0,:]), (wlen_max-wlen_min)/np.size(self.image[:,0]))
        self.plotswidget.intymap_plotItem.setLimits(xMin=time_min, xMax=1.01*time_max, yMin=wlen_min, yMax=1.01*wlen_max)

        self.plotswidget.kinetics_plot.setLimits(xMin=time_min, xMax=1.01*time_max)
        self.plotswidget.spectral_plot.setLimits(xMin=wlen_min, xMax=1.01*wlen_max)

        self.image_name = self.dirname+'/intymap_{}-{}ps.png'.format(int(time_min), int(time_max))
        # self.histogram_name = self.dirname+'/histogram_{}-{}ps_{}.png'.format(int(time_min), int(time_max), self.chirpStat)

        # viewrect = QRectF(0.0, W0, (time_max-time_min), wlen_max-wlen_min)
        # self.plotswidget.intymap_plot.vb.setRange(rect=viewrect, update=True)
        # self.plotswidget.intymap_imag.setRect(viewrect)

        # self.plotswidget.intymap_plot.vb.setXRange(0, len(self.tl_t0), update=True)
        # self.plotswidget.intymap_plot.vb.setYRange(0, len(self.avg_wl), update=True)
        # self.plotswidget.intymap_plot.vb.setLimits(xMin=time_min, xMax=time_max, yMin=wlen_min, yMax=wlen_max)
        #
        # piximag = self.plotswidget.intymap_imag.pixelSize()
        # pixview = self.plotswidget.intymap_plot.vb.viewPixelSize()
        # rangview = self.plotswidget.intymap_plot.vb.viewRange()
        # rectview = self.plotswidget.intymap_plot.vb.viewRect()
        # trectview = self.plotswidget.intymap_plot.vb.targetRect()
        # geomview = self.plotswidget.intymap_plot.viewGeometry()
        # print('\n', width, height, piximag, pixview, rangview, rectview, trectview, geomview)

    def updatePlot(self):
        self.delta_wl, self.W0, wlen_min, wlen_max, self.T0, time_min, time_max, inty_min, inty_max = self.readUI()
        self.plotswidget.intymap_plotItem.setLimits(xMin=time_min, xMax=1.01*time_max, yMin=wlen_min, yMax=1.01*wlen_max)
        time_min_idx = np.where(np.around(self.tl_t0,decimals=3)==time_min)[0][0]
        time_max_idx = np.where(np.around(self.tl_t0,decimals=3)==time_max)[0][0]
        wlen_min_idx = np.where(np.around(self.avg_wl,decimals=3)==wlen_min)[0][0]
        wlen_max_idx = np.where(np.around(self.avg_wl,decimals=3)==wlen_max)[0][0]
        self.image = self.avg_il[wlen_min_idx:wlen_max_idx,time_min_idx:time_max_idx]

        self.plotswidget.kinetics_plot.setLimits(xMin=time_min, xMax=1.01*time_max)
        self.plotswidget.spectral_plot.setLimits(xMin=wlen_min, xMax=1.01*wlen_max)

        # self.plotswidget.intymap_imageItem.setImage(self.image)

        self.plotswidget.intymap_imageViewWidget.setImage(self.image, autoRange=True, autoLevels=True, autoHistogramRange=True)
        self.plotswidget.intymap_imageItem.setRect(self.plotswidget.intymap_plotItem.vb.viewRect())

        # self.updateHistogram()

        self.image_name = self.dirname+'/intymap_{}-{}ps_{}.png'.format(int(time_min), int(time_max), self.chirpStat)
        # self.histogram_name = self.dirname+'/histogram_{}-{}ps_{}.png'.format(int(time_min), int(time_max), self.chirpStat)

    def exportPlot(self):
        self.plotswidget.plot_export.export(self.image_name)
        # self.plotswidget.hist_export.export(self.histogram_name)

    def clearAllPlots(self):
        self.clearTracesPlots()
        self.kinetics_plot.clear()
        self.spectral_plot.clear()
        self.plotswidget.intymap_plotItem.clear()

    def mouseClicked(self, event):
        self.x, self.y = self.plotswidget.mousePoint.x(), self.plotswidget.mousePoint.y()

        self.delta_wl, self.W0, wlen_min, wlen_max, self.T0, time_min, time_max, inty_min, inty_max = self.readUI()

        self.wlen_idx = min(np.searchsorted(self.avg_wl, self.y), len(self.avg_wl) - 1)
        self.wmin_idx = min(np.searchsorted(self.avg_wl, wlen_min), len(self.avg_wl) - 1)
        self.wmax_idx = min(np.searchsorted(self.avg_wl, wlen_max), len(self.avg_wl) - 1)

        self.time_idx = min(np.searchsorted(self.tl_t0, self.x), len(self.tl_t0) - 1)
        self.tmin_idx = min(np.searchsorted(self.tl_t0, time_min), len(self.tl_t0) - 1)
        self.tmax_idx = min(np.searchsorted(self.tl_t0, time_max), len(self.tl_t0) - 1)

        self.kinetics_trace = (1.0/np.amax(self.avg_il[self.wlen_idx, self.tmin_idx:self.tmax_idx]))*self.avg_il[self.wlen_idx, self.tmin_idx:self.tmax_idx]
        self.spectrum_trace = self.avg_il[self.wmin_idx:self.wmax_idx, self.time_idx]

        self.kinetics_std_trace = self.std_il[self.wlen_idx, self.tmin_idx:self.tmax_idx]
        self.spectrum_std_trace = self.std_il[self.wmin_idx:self.wmax_idx, self.time_idx]

        self.plotswidget.kinetics_plot.legend.clear()
        self.plotswidget.kinetics_plot.clear()

        self.plotswidget.spectral_plot.legend.clear()
        self.plotswidget.spectral_plot.clear()

        if self.plotswidget.controls_widget.std_checkbox.isChecked():
            self.plotswidget.kinetics_plot.setLabel('left', '<math>&Delta;A', units='mOD', unitPrefix='m', **labelstyle)
            self.plotswidget.kinetics_plot.setLabel('bottom', 'Time', units='ps', **labelstyle)
            #self.plotswidget.wlentrace_curve.errorbar(self.avg_wl[self.wmin_idx:self.wmax_idx], self.time_trace, yerr=self.time_std_trace, fmt='-', ecolor='k', elinewidth=0.5, mew=4, label=r'$\Delta$t={} ps'.format(round(self.x,2)))
            #self.plotswidget.timetrace_curve.errorbar(self.t0_tl[self.tmin_idx:self.tmax_idx], self.wlen_trace, yerr=self.wlen_std_trace, fmt='-', ecolor='k', elinewidth=0.5, mew=4, label=r'$\lambda$={} nm'.format(round(self.y,2)))
            #self.kinetics_curve_err.setData(self.t0_tl[self.tmin_idx:self.tmax_idx], self.fitfunc(self.t0_tl[self.tmin_idx:self.tmax_idx], *popt))

            if self.plotswidget.controls_widget.fit_checkbox.isChecked():
                pen = pg.mkPen(color='r', width=2, style=Qt.DashLine)
                popt, pcov = curve_fit(self.fitfunc, self.tl_t0[self.tmin_idx:self.tmax_idx], self.kinetics_trace, sigma=self.kinetics_std_trace)
                # exp_result = least_squares(exp_decay, x0=[200, 800, 2000], args=(rho_data['time'], rho_data[state]), loss='soft_l1',  method='trf', jac='3-point', max_nfev=2000)
                exp_result = least_squares(exp_decay, x0=[200, 800, 1000], args=(self.tl_t0[self.tmin_idx:self.tmax_idx], self.kinetics_trace), loss='soft_l1', f_scale=0.1)
                popt = np.copy(exp_result.x)

                # a,b,c = round(popt[0],2),round(popt[1],2),round(popt[2],2)
                #self.plotswidget.timetrace_curve.plot(self.t0_tl[self.tmin_idx:self.tmax_idx], self.fitfunc(self.t0_tl[self.tmin_idx:self.tmax_idx], *popt), 'r.', label=r'$\lambda$={} nm, $\tau$={} ps'.format(round(self.y,2),b))
                # self.plotswidget.kinetics_curve_fit.setData(self.tl_t0[self.tmin_idx:self.tmax_idx], self.fitfunc(self.tl_t0[self.tmin_idx:self.tmax_idx], *popt))
                self.plotswidget.kinetics_plot.plot(self.tl_t0[self.tmin_idx:self.tmax_idx], func_exp_decay(self.tl_t0[self.tmin_idx:self.tmax_idx], popt))

        elif self.plotswidget.controls_widget.fft_checkbox.isChecked():
            # self.plotswidget.kinetics_plot.setLabel('left', 'Amplitude', **labelstyle)
            # self.plotswidget.kinetics_plot.setLabel('bottom', 'Wavenumber', units='<math>cm<sup>-1</sup></math>', **labelstyle)
            self.plotswidget.kinetics_plot.setLabel('left', '<math>&Delta;A', units='OD', unitPrefix='m', **labelstyle)
            self.plotswidget.kinetics_plot.setLabel('bottom', 'Time', units='ps', **labelstyle)

            a0=0.5*np.amax(self.kinetics_trace)

            exp_result = least_squares(exp_decay_fit, x0=[0.2, 1.0, a0, a0, 0.005], args=(self.tl_t0[self.tmin_idx:self.tmax_idx], self.kinetics_trace), loss='soft_l1', f_scale=0.1)

            resi = exp_result.fun

            residuals_amp = resi[0]
            vib_coherence_fit = least_squares(vib_decay_fit,
                                                x0=[residuals_amp , 0.200, 1000, 0.1, 0.005],
                                                args=(self.tl_t0[self.tmin_idx:self.tmax_idx], resi),
                                                loss='soft_l1',
                                                f_scale=0.1)

            popt = vib_coherence_fit.x

            self.plotswidget.kinetics_plot.plot(self.tl_t0[self.tmin_idx:self.tmax_idx], resi, pen='b', name='<math>&lambda; = {} nm</math'.format(round(self.y,2)))

            pen = pg.mkPen(color='r', width=2, style=Qt.DashLine)
            self.plotswidget.kinetics_plot.plot(self.tl_t0[self.tmin_idx:self.tmax_idx], vib_decay(self.tl_t0[self.tmin_idx:self.tmax_idx], popt), pen=pen, name='<math>&tau; = {} ps<math>\t<math>&omega; = {} <math>cm<sup>-1</sup></math>'.format(round(popt[1],2),round(popt[2],2)))



            # hann_window = hann(resi.size-1, sym=True)
            # # self.kinetics_trace_fft = np.fft.rfft(hamm*resi, norm='ortho')
            # self.kinetics_trace_fft = nfft.ndft(self.tl_t0[self.tmin_idx+1:self.tmax_idx], hann_window*resi[1:])
            # self.kinetics_trace_wnum = (33.35641)*np.fft.rfftfreq(2*self.kinetics_trace_fft.size, 0.03)
            # # self.kinetics_trace_wnum = nfft.ndft_adjoint(self.tl_t0[self.tmin_idx:self.tmax_idx], self.kinetics_trace)
            # data_fft = np.real(2.0*self.kinetics_trace_fft)
            # self.plotswidget.kinetics_plot.plot(self.kinetics_trace_wnum[1:], data_fft, pen='b', name='<math>&lambda; = {} nm</math'.format(round(self.y,2)))
            # # self.plotswidget.kinetics_plot.plot(self.kinetics_trace_wnum, self.kinetics_trace_fft, pen='b', name='<math>&lambda; = {} nm</math'.format(round(self.y,2)))
            self.plotswidget.spectral_plot.plot(self.avg_wl[self.wmin_idx:self.wmax_idx], self.spectrum_trace, pen='b', name='<math>&Delta;t = {} ps<math>'.format(round(self.x,2)))




        else:
            self.plotswidget.kinetics_plot.setLabel('left', '<math>&Delta;A', units='OD', unitPrefix='m', **labelstyle)
            self.plotswidget.kinetics_plot.setLabel('bottom', 'Time', units='ps', **labelstyle)
            self.plotswidget.kinetics_plot.plot(self.tl_t0[self.tmin_idx:self.tmax_idx], self.kinetics_trace, pen='b', name='<math>&lambda; = {} nm</math>'.format(round(self.y,2)))
            self.plotswidget.spectral_plot.plot(self.avg_wl[self.wmin_idx:self.wmax_idx], self.spectrum_trace, pen='b', name='<math>&Delta;t = {} ps<math>'.format(round(self.x,2)))

            if self.plotswidget.controls_widget.fit_checkbox.isChecked():
                pen = pg.mkPen(color='r', width=2, style=Qt.DashLine)
                # popt, pcov = curve_fit(self.fitfunc, self.tl_t0[self.tmin_idx:self.tmax_idx], self.kinetics_trace, sigma=self.kinetics_std_trace)
                # exp_result = least_squares(exp_decay, x0=[200, 800, 2000], args=(rho_data['time'], rho_data[state]), loss='soft_l1',  method='trf', jac='3-point', max_nfev=2000)
                a0=0.5*np.amax(self.kinetics_trace)
                exp_result = least_squares(exp_decay_fit, x0=[0.2, 1.0, 5.0, a0, a0, a0, 0.005], args=(self.tl_t0[self.tmin_idx:self.tmax_idx], self.kinetics_trace), loss='soft_l1', f_scale=0.1)
                popt = np.copy(exp_result.x)
                print(popt)
                # self.plotswidget.kinetics_plot.plot(self.tl_t0[self.tmin_idx:self.tmax_idx], self.fitfunc(self.tl_t0[self.tmin_idx:self.tmax_idx], *popt), pen=pen, name='<math>&tau; = {} ps<math>\n<math>&tau; = {} ps<math>'.format(round(popt[1],2),round(popt[3],2)))
                self.plotswidget.kinetics_plot.plot(self.tl_t0[self.tmin_idx:self.tmax_idx], exp_decay(self.tl_t0[self.tmin_idx:self.tmax_idx], popt), pen=pen, name='<math>&tau<sub>1</sub>; = {} ps<math>\t<math>&tau<sub>2</sub>; = {} ps<math>\t<math>&tau<sub>3</sub>; = {} ps<math>'.format(round(popt[0],2),round(popt[1],2),round(popt[2],2)))

    def getTraces(self):
        try:
            self.gotTracesWidget.ccount
        except:
            self.gotTracesWidget.ccount = 0
        finally:
            if self.gotTracesWidget.ccount<10:
                pen = pg.mkPen(color=colors[self.gotTracesWidget.ccount], width=2)
                self.gotTracesWidget.ccount += 1
            else:
                self.gotTracesWidget.ccount = 0
                pen = pg.mkPen(color=colors[self.gotTracesWidget.ccount], width=2)

            dataname = self.plotswidget.controls_widget.dataname_input.text()
            self.gotTracesWidget.kinetics_traces.setTitle(dataname)
            self.gotTracesWidget.spectral_traces.setTitle(dataname)
            self.gotTracesWidget.kinetics_traces.plot(self.tl_t0[self.tmin_idx:self.tmax_idx], self.kinetics_trace, pen=pen, name='<math>&lambda; = {} nm</math>'.format(round(self.y,2)))
            self.gotTracesWidget.spectral_traces.plot(self.avg_wl[self.wmin_idx:self.wmax_idx], self.spectrum_trace, pen=pen, name='<math>&Delta;t = {} ps<math>'.format(round(self.x,2)))

    def exportTraces(self):
        try:
            self.parent.exportedTracesWidget.ccount
        except:
            self.parent.exportedTracesWidget.ccount = 0
        finally:
            if self.parent.exportedTracesWidget.ccount>9:
                self.parent.exportedTracesWidget.ccount = 0

            pen = pg.mkPen(color=colors[self.parent.exportedTracesWidget.ccount], width=2)
            self.parent.exportedTracesWidget.ccount += 1

            dataname = self.plotswidget.controls_widget.dataname_input.text()
            self.parent.exportedTracesWidget.kinetics_traces.setLabel('left', '<math>&Delta;A', units='normalized', **labelstyle)
            self.parent.exportedTracesWidget.spectral_traces.setLabel('left', '<math>&Delta;A', units='normalized', **labelstyle)
            self.parent.exportedTracesWidget.kinetics_traces.setTitle('<math>&lambda; = {} nm</math>'.format(round(self.y,2)))
            self.parent.exportedTracesWidget.spectral_traces.setTitle('<math>&Delta;t = {} ps<math>'.format(round(self.x,2)))

            self.parent.exportedTracesWidget.kinetics_traces.plot(self.tl_t0[self.tmin_idx:self.tmax_idx], self.kinetics_trace/self.kinetics_trace.max(), pen=pen, name=dataname)
            self.parent.exportedTracesWidget.spectral_traces.plot(self.avg_wl[self.wmin_idx:self.wmax_idx], self.spectrum_trace/self.spectrum_trace.max(), pen=pen, name=dataname)

    def plotTraces(self):
        self.gotTracesWidget.showMaximized()
        self.parent.exportedTracesWidget.showMaximized()

    def clearTracesPlots(self):
        self.gotTracesWidget.kinetics_traces.legend.clear()
        self.gotTracesWidget.spectral_traces.clear()

        self.gotTracesWidget.spectral_traces.legend.clear()
        self.gotTracesWidget.kinetics_traces.clear()

        self.parent.exportedTracesWidget.kinetics_traces.legend.clear()
        self.parent.exportedTracesWidget.spectral_traces.clear()

        self.parent.exportedTracesWidget.spectral_traces.legend.clear()
        self.parent.exportedTracesWidget.kinetics_traces.clear()

    def fitfunc(self, x, a, b, c, d, e):
        return a*np.exp(-b*x) + c*np.exp(-d*x) + e

    def fit_indexes(self, x, a, b, c):
        return a + b*x + c*pow(x,2) #+ p[3]*pow(x,3)


if __name__=='__main__':
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setPalette(palette)
    screen_resolution = app.desktop().screenGeometry()
    width, height = screen_resolution.width(), screen_resolution.height()
    mainwindow = MainWindow()
    mainwindow.showMaximized()
    # tas_analiser = TAS_Analiser()
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
#
