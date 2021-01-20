import os
import sys
from gui_config import *

import random
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


class MainWindow(QMainWindow):
    '''
    Main application window, from this different instances of UV-Vis Abs.
    Spectra Plotter can be open.
    It maight later hold a TabWidget to analyse or compare data from different
    Plotter instances
    '''
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle('UV-Vis Plots Manager')
        self.setGeometry(20,20,1800,1200)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QGridLayout(self.central_widget)
        self.central_widget.adjustSize()

        self.exitAct = QAction('Exit', self)
        self.exitAct.setShortcut('Ctrl+Q')
        self.exitAct.setStatusTip('Exit application')
        self.exitAct.triggered.connect(self.close)

        self.newGraphAct = QAction('New Plotter', self)
        self.newGraphAct.setShortcut('Ctrl+N')
        self.newGraphAct.setStatusTip("Creates a new plotter window")
        self.newGraphAct.triggered.connect(self.plotterWindow)

        self.menubar = self.menuBar()
        self.fileMenu = self.menubar.addMenu('&Menu')
        self.fileMenu.addAction(self.exitAct)
        self.fileMenu.addAction(self.newGraphAct)

        self.list_of_windows = list()

    def closeEvent(self, event):
        '''
        This function will call close of every plotter instance so data could be
        saved before closing
        '''
        for window in self.list_of_windows:
            window.close()
        event.accept()

    def plotterWindow(self):
        '''
        Making a list of initiated plotter windows
        '''
        plotter_window = PlotterWindow(self)
        plotter_window.showMaximized()
        self.list_of_windows.append(plotter_window)


class PlotterWindow(QMainWindow):
    '''
    This is the main graphical area plus the formatting controls ui.
    '''
    def __init__(self, parent=None):
        super(PlotterWindow, self).__init__(parent)
        self.parent = parent
        self.setWindowTitle('UV-Vis Abs. Spectra Plotter')
        self.setGeometry(20,20,1800,1200)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QGridLayout(self.central_widget)
        self.central_widget.adjustSize()

        self.statusbar = self.statusBar()

        self.exitAct = QAction('Exit', self)
        self.exitAct.setShortcut('Ctrl+Q')
        self.exitAct.setStatusTip('Exit application')
        self.exitAct.triggered.connect(self.close)

        self.loadDataAct = QAction('Load Data', self)
        self.loadDataAct.setShortcut('Ctrl+D')
        self.loadDataAct.setStatusTip('Load data file')
        self.loadDataAct.triggered.connect(self.loadData)

        self.clearGraphAct = QAction('Clear', self)
        self.clearGraphAct.setShortcut('Ctrl+K')
        self.clearGraphAct.setStatusTip('Clear Plot')
        self.clearGraphAct.triggered.connect(self.clearGraph)

        self.saveGraphAct = QAction('Save', self)
        self.saveGraphAct.setShortcut('Ctrl+S')
        self.saveGraphAct.setStatusTip('Save Plot')
        self.saveGraphAct.triggered.connect(self.saveGraph)

        self.plotAvgAct = QAction('Avg. Data', self)
        self.plotAvgAct.setStatusTip('Plot Avg. Data')
        self.plotAvgAct.triggered.connect(self.plotAvgData)

        self.toolbar = self.addToolBar('toolbar')
        self.toolbar.addAction(self.loadDataAct)
        self.toolbar.addAction(self.clearGraphAct)
        self.toolbar.addAction(self.saveGraphAct)
        self.toolbar.addAction(self.plotAvgAct)

        self.GraphWidget = pg.PlotWidget()
        self.GraphArea = self.GraphWidget.getPlotItem()
        self.GraphArea.getAxis("left").tickFont = QFont("Helvetica [Cronyx]", 14)
        self.GraphArea.getAxis("bottom").tickFont = QFont("Helvetica [Cronyx]", 14)
        self.GraphArea.setLabel('left', 'Absorbance', **labelstyle)
        self.GraphArea.setLabel('bottom', 'Wavelength (nm)', **labelstyle)
        self.GraphArea.addLegend(size=(50,50), offset=(-50, 50))
        self.GraphArea.showGrid(x=True, y=True, alpha=0.5)

        self.GrapExporter = pg.exporters.ImageExporter(self.GraphArea)
        self.GrapExporter.parameters()['height'] = 2000

        self.setupUI()
        self.connectUI()

        layout.addWidget(self.GraphWidget,0,0)
        layout.addWidget(self.FormatWidget,0,1)

    def setupUI(self):
        self.FormatWidget = uic.loadUi("formatting.ui")
        self.FormatWidget.xaxis_min.setValidator(QDoubleValidator(-999.999,999.999,3))
        self.FormatWidget.xaxis_max.setValidator(QDoubleValidator(-999.999,999.999,3))
        self.FormatWidget.yaxis_min.setValidator(QDoubleValidator(-999.999,999.999,3))
        self.FormatWidget.yaxis_max.setValidator(QDoubleValidator(-999.999,999.999,3))

        self.FormatWidget.xaxis_title.setText("Wavelength (nm)")
        self.FormatWidget.yaxis_title.setText("Absorbance")
        self.FormatWidget.curve_marker.addItems(list(symbols.keys()))

    def connectUI(self):
        self.FormatWidget.graphopts_button.clicked.connect(self.applyGraphOpts)
        self.FormatWidget.curveopts_button.clicked.connect(self.applyCurveOpts)
        self.FormatWidget.recolor_button.clicked.connect(self.reColor)
        self.FormatWidget.legend_list.currentIndexChanged.connect(self.on_current_index_changed)
        self.FormatWidget.curve_title.editingFinished.connect(self.on_editing_finished)

    def on_current_index_changed(self):
        self.FormatWidget.curve_title.setText(self.FormatWidget.legend_list.currentText())

    def on_editing_finished(self):
        self.GraphArea.legend.removeItem(self.FormatWidget.legend_list.currentText())
        idx = self.FormatWidget.legend_list.currentIndex()
        name = self.FormatWidget.curve_title.text()
        self.list_of_datanames[idx] = name
        self.FormatWidget.legend_list.setItemText(idx, name)
        self.GraphArea.legend.addItem(self.list_of_curves[idx], name=self.list_of_datanames[idx])

    def closeEvent(self, event):
            close = QMessageBox.question(self,"Warning","Confirm to close the application, not saved data will be lost", QMessageBox.Yes | QMessageBox.No)
            if close == QMessageBox.Yes:
                self.parent.list_of_windows.remove(self)
                event.accept()
            else:
                event.ignore()

    def loadData(self):
        file_dialog = QFileDialog()
        files_list, _ = file_dialog.getOpenFileNames()
        if len(files_list)>0:
            self.dirname = os.path.dirname(files_list[0])
            splitteddir = os.path.split(self.dirname)
            self.graph_area_title = splitteddir[-2].split('/')[-1]+' -- '+splitteddir[-1]

            try:
                self.list_of_datanames
            except:
                self.list_of_datanames = list()
                self.list_of_xdata = list()
                self.list_of_ydata = list()
                self.list_of_curves = list()

            for file in files_list:
                try:
                    data = pd.read_csv(file, header=None, delim_whitespace=True, dtype=np.float64, error_bad_lines=False)
                except:
                    data = pd.read_csv(file, header=None, delim_whitespace=True, decimal=',', skiprows=2, dtype=np.float64, error_bad_lines=False)

                xdata = data.iloc[:,0].to_numpy()
                ydata = data.iloc[:,1].to_numpy()

                splitteddir = os.path.split(self.dirname)
                dataname = os.path.splitext(os.path.basename(file))[0]

                idx = len(self.list_of_datanames)
                width = 2.0
                pen=pg.mkPen(color=colors[idx],width=width)

                curve = self.GraphArea.plot(xdata, ydata, pen=pen, name=dataname)

                self.list_of_datanames.append(dataname)
                self.list_of_xdata.append(xdata)
                self.list_of_ydata.append(ydata)
                self.list_of_curves.append(curve)

                # try:
                #     self.sum_ydata
                # except:
                #     # sum_xdata = np.zeros(len(xdata))
                #     self.sum_ydata = np.zeros(len(ydata))
                # finally:
                #     # sum_xdata = np.add(sum_xdata, xdata, dtype=np.float64)
                #     self.sum_ydata = np.add(self.sum_ydata, ydata, dtype=np.float64)

                self.FormatWidget.legend_list.addItem(dataname)

            # self.mean_xdata = 1.0/len(self.list_of_curves)*sum_xdata

            xmin = round(np.amin(xdata),3)
            xmax = round(np.amax(xdata),3)
            ymin = round(np.amin(ydata),3)
            ymax = round(np.amax(ydata),3)

            self.FormatWidget.xaxis_min.setProperty("text",str(xmin))
            self.FormatWidget.xaxis_max.setProperty("text",str(xmax))
            self.FormatWidget.yaxis_min.setProperty("text",str(ymin))
            self.FormatWidget.yaxis_max.setProperty("text",str(ymax))
            self.FormatWidget.graph_title.setText(self.graph_area_title)

            self.statusbar.showMessage('Succeeded to load datafiles')

        else:
            self.statusbar.showMessage('Failed to load any datafile')
            pass

    def plotAvgData(self):
        maxlen = min([len(data) for data in self.list_of_ydata])
        sum_ydata = np.zeros(maxlen)
        for data in self.list_of_ydata:
            sum_ydata = [a+b for a, b in zip(sum_ydata, data)]
        ydata = [(1.0/len(self.list_of_ydata))*yval for yval in sum_ydata]
        xdata_o = self.list_of_xdata[0][0]
        xdata_f = self.list_of_xdata[0][-1]
        xdata = np.linspace(xdata_o, xdata_f, maxlen, endpoint=True)
        self.GraphArea.plot(xdata, ydata, pen=pg.mkPen(color=colors['gray'],width=2.0), name='avg. data')

    def clearGraph(self):
        # self.GraphArea.vb.removeItem(self.GraphArea.legend)

        [self.GraphArea.legend.removeItem(self.FormatWidget.legend_list.itemText(idx)) for idx in range(len(self.list_of_curves))]
        self.FormatWidget.legend_list.clear()
        self.GraphArea.clear()
        self.list_of_datanames = list()
        self.list_of_xdata = list()
        self.list_of_ydata = list()
        self.list_of_curves = list()

    def saveGraph(self):
        try:
            self.graph_area_title
        except:
            msgbox = QMessageBox()
            msgbox.setIcon(QMessageBox.Warning)
            msgbox.setWindowTitle('Metadata Missing')
            msgbox.setText("Image filename is not defined.")
            msgbox.setInformativeText("The image's filename is defined by the title of the graph.")
            msgbox.show()
            msgbox.exec_()
        else:
            self.GrapExporter.export(self.dirname+'/'+self.graph_area_title+'.png')

    def readGraphOpts(self):
        self.graph_area_title = self.FormatWidget.graph_title.text()
        self.xaxis_title = self.FormatWidget.xaxis_title.text()
        self.yaxis_title = self.FormatWidget.yaxis_title.text()

        self.xaxis_min = float(self.FormatWidget.xaxis_min.text())
        self.xaxis_max = float(self.FormatWidget.xaxis_max.text())
        self.yaxis_min = float(self.FormatWidget.yaxis_min.text())
        self.yaxis_max = float(self.FormatWidget.yaxis_max.text())

    def applyGraphOpts(self):
        self.readGraphOpts()
        self.plotCurves()
        self.GraphArea.setTitle(self.graph_area_title)
        self.GraphArea.setLabel('bottom', self.xaxis_title, **labelstyle)
        self.GraphArea.setLabel('left', self.yaxis_title, **labelstyle)
        self.GraphArea.setLimits(xMin=self.xaxis_min, xMax=self.xaxis_max, yMin=self.yaxis_min, yMax=self.yaxis_max)

    def applyCurveOpts(self):
        width = self.FormatWidget.curve_thickness.value()
        idx = self.FormatWidget.legend_list.currentIndex()
        pen = pg.mkPen(color=colors[idx], width=width)
        marker = symbols[self.FormatWidget.curve_marker.currentText()]
        self.list_of_curves[idx].setPen(pen)
        self.list_of_curves[idx].setSymbol(marker)

        xdata = self.list_of_xdata[idx]
        if self.FormatWidget.normalize_checkbox.isChecked():
            ydata = self.list_of_ydata[idx]*(1.0/np.amax(self.list_of_ydata[idx]))
        else:
            ydata = self.list_of_ydata[idx]

        self.list_of_curves[idx].setData(xdata, ydata, name=self.list_of_datanames[idx])

    def reColor(self):
        width = self.FormatWidget.curve_thickness.value()
        idx = self.FormatWidget.legend_list.currentIndex()
        new_color = random.choice(list(colors.values()))
        #colors[np.random.randint(len(self.list_of_curves)+1)]
        pen = pg.mkPen(color=new_color, width=width)
        self.list_of_curves[idx].setPen(pen)

        xdata = self.list_of_xdata[idx]
        if self.FormatWidget.normalize_checkbox.isChecked():
            ydata = self.list_of_ydata[idx]*(1.0/np.amax(self.list_of_ydata[idx]))
        else:
            ydata = self.list_of_ydata[idx]

        self.list_of_curves[idx].setData(xdata, ydata, name=self.list_of_datanames[idx])

    def plotCurves(self):
        for curve in self.list_of_curves:
            idx = self.list_of_curves.index(curve)

            xdata = self.list_of_xdata[idx]
            if self.FormatWidget.normalize_checkbox.isChecked():
                ydata = self.list_of_ydata[idx]*(1.0/np.amax(self.list_of_ydata[idx]))
                self.GraphArea.setLabel('left', self.yaxis_title, units='normalized', **labelstyle)
            else:
                ydata = self.list_of_ydata[idx]

            self.list_of_curves[idx].setData(xdata, ydata, name=self.list_of_datanames[idx])

if __name__=='__main__':
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setPalette(palette)
    mainwindow = MainWindow()
    mainwindow.showMaximized()
    sys.exit(app.exec_())
