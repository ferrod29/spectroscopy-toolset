from PyQt5 import uic
from PyQt5.Qt import QObject, Qt, QRectF, \
                    QDoubleValidator, QFont, QColor, QImage, QPalette, \
                    QMainWindow, QAction, QApplication, \
                    QFileDialog, QWidget, QGridLayout, \
                    QSplitter, QMessageBox, QSizePolicy

import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter
import sys
import os
import numpy as np
import pandas as pd
from snippets import *


DarkPalette = QPalette()
DarkPalette.setColor(QPalette.Window, QColor(53, 53, 53))
DarkPalette.setColor(QPalette.WindowText, Qt.white)
DarkPalette.setColor(QPalette.Base, QColor(25, 25, 25))
DarkPalette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
DarkPalette.setColor(QPalette.ToolTipBase, Qt.white)
DarkPalette.setColor(QPalette.ToolTipText, Qt.white)
DarkPalette.setColor(QPalette.Text, Qt.white)
DarkPalette.setColor(QPalette.Button, QColor(53, 53, 53))
DarkPalette.setColor(QPalette.ButtonText, Qt.white)
DarkPalette.setColor(QPalette.BrightText, Qt.red)
DarkPalette.setColor(QPalette.Link, QColor(42, 130, 218))
DarkPalette.setColor(QPalette.Highlight, QColor(42, 130, 218))
DarkPalette.setColor(QPalette.HighlightedText, Qt.black)

pg.setConfigOptions(imageAxisOrder='row-major', antialias=True, crashWarning=True)
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

BuRd_map = pg.ColorMap((1.0,0.5,0.0), ((255,0,0,255), (255,255,255,255), (0,0,255,255)))
BuRd_lut = BuRd_map.getLookupTable()

colors = {
            'black':(0, 0 ,0),
            'red':(255, 0, 0),
            'blue':(0, 0, 255),
            'green':(0, 255, 0),
            'magenta':(255, 0, 255),
            'skyblue':(127,191,255),
            'cyan':(0, 255, 255),
            'orange':(255, 127, 64),
            'violet':(200, 20, 120),
            'turquoise':(127,255,191),
            'yellow':(255, 255, 127),
            'pink':(255,127,127),
            'gray':(127, 127, 127)
        }

colors = list(colors.values())

symbols = {"None":None, u"\u2022":"o", u"\u002B":"+", u"\u2266":"d", u"\u25B2":"t", u"\u25A0":"s", u"\u2B1F":"p"}
labelstyle = {'color': 'k', 'font-size': '12pt'}



class DataAnalysis(QObject):
    def __init__(self, parent=None):
        super(DataAnalysis, self).__init__(parent)
        self.parent = parent
        self.setupUI()
        self.connectUI()
        
    def setupUI(self):
        self.ui = QMainWindow()
        self.ui.setWindowTitle('Data Plotter Manager')

        central_widget = QWidget()
        layout = QGridLayout(central_widget)
        central_widget.adjustSize()
        self.ui.setCentralWidget(central_widget)

        self.ui.exit_action = QAction('Exit', self)
        self.ui.exit_action.setShortcut('Ctrl+Q')
        self.ui.exit_action.setStatusTip('Exit application')

        self.ui.new_action = QAction('New Plotter', self)
        self.ui.new_action.setShortcut('Ctrl+N')
        self.ui.new_action.setStatusTip("Creates a new plotter window")

        self.ui.menu_bar = self.ui.menuBar()
        self.ui.file_menu = self.ui.menu_bar.addMenu('&Menu')
        self.ui.file_menu.addAction(self.ui.exit_action)
        self.ui.file_menu.addAction(self.ui.new_action)

        self.list_of_windows = list()
                
        
    def connectUI(self):
        self.ui.exit_action.triggered.connect(self.ui.close)
        self.ui.new_action.triggered.connect(self.newPlotter)
        
        
    def closeEvent(self, event):
        '''
        This function will call close of every plotter instance so data could be
        saved before closing
        '''
        for window in self.list_of_windows:
            window.close()
        event.accept()


    def newPlotter(self):
        '''
        Making a list of initiated plotter windows
        '''
        plotter_window = DataPlotter(self)
        plotter_window.ui.showMaximized()
        self.list_of_windows.append(plotter_window)
        
        
    
class DataPlotter(QObject):
    '''
    This is the main graphical area plus the formatting controls ui.
    '''
    def __init__(self, parent=None):
        super(DataPlotter, self).__init__(parent)
        self.parent = parent
        self.setupUI()
        self.connectUI()
        
    def setupUI(self):
        self.ui = QMainWindow()
        self.ui.setWindowTitle('Data Plotter')

        central_widget = QWidget()
        layout = QGridLayout(central_widget)
        central_widget.adjustSize()
        self.ui.setCentralWidget(central_widget)

        self.ui.status_bar = self.ui.statusBar()

        self.ui.exit_action = QAction('Exit', self)
        self.ui.exit_action.setShortcut('Ctrl+Q')
        self.ui.exit_action.setStatusTip('Exit application')

        self.ui.load_action = QAction('Load Data', self)
        self.ui.load_action.setShortcut('Ctrl+D')
        self.ui.load_action.setStatusTip('Load data file')

        self.ui.clear_action = QAction('Clear', self)
        self.ui.clear_action.setShortcut('Ctrl+K')
        self.ui.clear_action.setStatusTip('Clear Plot')

        self.ui.save_action = QAction('Save', self)
        self.ui.save_action.setShortcut('Ctrl+S')
        self.ui.save_action.setStatusTip('Save Plot')

        self.ui.plotAvg_action = QAction('Avg. Data', self)
        self.ui.plotAvg_action.setStatusTip('Plot Avg. Data')

        self.ui.toolbar = self.ui.addToolBar('toolbar')
        self.ui.toolbar.addAction(self.ui.load_action)
        self.ui.toolbar.addAction(self.ui.clear_action)
        self.ui.toolbar.addAction(self.ui.save_action)
        self.ui.toolbar.addAction(self.ui.plotAvg_action)

        self.ui.GraphWidget = pg.PlotWidget()
        self.ui.GraphArea = self.ui.GraphWidget.getPlotItem()
        self.ui.GraphArea.getAxis("left").tickFont = QFont("Helvetica [Cronyx]", 14)
        self.ui.GraphArea.getAxis("bottom").tickFont = QFont("Helvetica [Cronyx]", 14)
        self.ui.GraphArea.setLabel('left', 'Absorbance', **labelstyle)
        self.ui.GraphArea.setLabel('bottom', 'Wavelength (nm)', **labelstyle)
        self.ui.GraphArea.addLegend(size=(50,50), offset=(-50, 50))
        self.ui.GraphArea.showGrid(x=True, y=True, alpha=0.5)
        self.ui.GraphWidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ui.GraphArea.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.ui.control_box = uic.loadUi("formatting.ui")
        self.ui.control_box.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.MinimumExpanding)
        self.ui.control_box.xaxis_min.setValidator(QDoubleValidator(-999.999,999.999,3))
        self.ui.control_box.xaxis_max.setValidator(QDoubleValidator(-999.999,999.999,3))
        self.ui.control_box.yaxis_min.setValidator(QDoubleValidator(-999.999,999.999,3))
        self.ui.control_box.yaxis_max.setValidator(QDoubleValidator(-999.999,999.999,3))

        # self.ui.control_box.xaxis_title.setText("Wavelength (nm)")
        # self.ui.control_box.yaxis_title.setText("Absorbance")
        self.ui.control_box.curve_marker.addItems(list(symbols.keys()))
        
        layout.addWidget(self.ui.GraphWidget,0,0)
        layout.addWidget(self.ui.control_box,0,1)
        
        self.GrapExporter = ImageExporter(self.ui.GraphArea)
        self.GrapExporter.parameters()['height'] = 2000


    def connectUI(self):
        self.ui.exit_action.triggered.connect(self.ui.close)
        self.ui.load_action.triggered.connect(self.loadData)
        self.ui.save_action.triggered.connect(self.saveGraph)
        self.ui.clear_action.triggered.connect(self.clearGraph)
        self.ui.plotAvg_action.triggered.connect(self.plotAvgData)
        
        self.ui.control_box.applyOpts_button.clicked.connect(self.applyGraphOpts)
        self.ui.control_box.curveOpts_button.clicked.connect(self.applyCurveOpts)
        self.ui.control_box.recolor_button.clicked.connect(self.reColor)
        self.ui.control_box.legend_list.currentIndexChanged.connect(self.on_current_index_changed)
        self.ui.control_box.curve_title.editingFinished.connect(self.on_editing_finished)
        
        self.ui.control_box.calcAnalysis_button.clicked.connect(self.dataAnalyse)
        self.ui.control_box.clearAnalysis_button.clicked.connect(self.dataReset)


    def on_current_index_changed(self):
        self.ui.control_box.curve_title.setText(self.ui.control_box.legend_list.currentText())


    def on_editing_finished(self):
        self.ui.GraphArea.legend.removeItem(self.ui.control_box.legend_list.currentText())
        idx = self.ui.control_box.legend_list.currentIndex()
        name = self.ui.control_box.curve_title.text()
        self.list_of_datanames[idx] = name
        self.ui.control_box.legend_list.setItemText(idx, name)
        self.ui.GraphArea.legend.addItem(self.list_of_curves[idx], name=self.list_of_datanames[idx])


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
                basename = os.path.basename(file).split('.')
                ext = basename[1]
                if ext =='csv':
                    data = pd.read_csv(file, header=None, dtype=np.float64,  skiprows=1, error_bad_lines=False)
                else:
                    try:
                        data = pd.read_csv(file, header=None, delim_whitespace=True, dtype=np.float64, error_bad_lines=False)
                    except:
                        data = pd.read_csv(file, header=None, delim_whitespace=True, decimal=',', skiprows=1, dtype=np.float64, error_bad_lines=False)

                xdata = data.iloc[:,0].to_numpy()
                ydata = data.iloc[:,1].to_numpy()

                splitteddir = os.path.split(self.dirname)
                dataname = os.path.splitext(os.path.basename(file))[0]

                idx = len(self.list_of_datanames)
                width = 2.0
                pen=pg.mkPen(color=colors[idx],width=width)

                curve = self.ui.GraphArea.plot(xdata, ydata, pen=pen, name=dataname)

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

                self.ui.control_box.legend_list.addItem(dataname)

            # self.mean_xdata = 1.0/len(self.list_of_curves)*sum_xdata

            xmin = round(np.amin(xdata),3)
            xmax = round(np.amax(xdata),3)
            ymin = round(np.amin(ydata),3)
            ymax = round(np.amax(ydata),3)

            self.ui.control_box.xaxis_min.setProperty("text",str(xmin))
            self.ui.control_box.xaxis_max.setProperty("text",str(xmax))
            self.ui.control_box.yaxis_min.setProperty("text",str(ymin))
            self.ui.control_box.yaxis_max.setProperty("text",str(ymax))
            self.ui.control_box.graph_title.setText(self.graph_area_title)

            self.status_bar.showMessage('Succeeded to load datafiles')

        else:
            self.status_bar.showMessage('Failed to load any datafile')
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
        avg_plot = self.ui.GraphArea.plot(xdata, ydata, pen=pg.mkPen(color=colors['gray'],width=2.0), name='avg. data')
        self.list_of_curves.append(avg_plot)


    def clearGraph(self):
        # self.ui.GraphArea.vb.removeItem(self.ui.GraphArea.legend)

        [self.ui.GraphArea.legend.removeItem(self.ui.control_box.legend_list.itemText(idx)) for idx in range(len(self.list_of_curves))]
        self.ui.control_box.legend_list.clear()
        self.ui.GraphArea.clear()
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
        self.graph_area_title = self.ui.control_box.graph_title.text()
        self.xaxis_title = self.ui.control_box.xaxis_title.text()
        self.yaxis_title = self.ui.control_box.yaxis_title.text()

        self.xaxis_min = float(self.ui.control_box.xaxis_min.text())
        self.xaxis_max = float(self.ui.control_box.xaxis_max.text())
        self.yaxis_min = float(self.ui.control_box.yaxis_min.text())
        self.yaxis_max = float(self.ui.control_box.yaxis_max.text())


    def applyGraphOpts(self):
        self.readGraphOpts()
        self.plotCurves()
        self.ui.GraphArea.setTitle(self.graph_area_title)
        self.ui.GraphArea.setLabel('bottom', self.xaxis_title, **labelstyle)
        self.ui.GraphArea.setLabel('left', self.yaxis_title, **labelstyle)
        self.ui.GraphArea.setLimits(xMin=self.xaxis_min, xMax=self.xaxis_max, yMin=self.yaxis_min, yMax=self.yaxis_max)


    def applyCurveOpts(self):
        width = self.ui.control_box.curve_thickness.value()
        idx = self.ui.control_box.legend_list.currentIndex()
        pen = pg.mkPen(color=colors[idx], width=width)
        marker = symbols[self.ui.control_box.curve_marker.currentText()]
        self.list_of_curves[idx].setPen(pen)
        self.list_of_curves[idx].setSymbol(marker)

        xdata = self.list_of_xdata[idx]
        if self.ui.control_box.normalize_checkbox.isChecked():
            ydata = self.list_of_ydata[idx]*(1.0/np.amax(self.list_of_ydata[idx]))
        else:
            ydata = self.list_of_ydata[idx]

        self.list_of_curves[idx].setData(xdata, ydata, name=self.list_of_datanames[idx])


    def reColor(self):
        width = self.ui.control_box.curve_thickness.value()
        idx = self.ui.control_box.legend_list.currentIndex()
        new_color = random.choice(list(colors.values()))
        #colors[np.random.randint(len(self.list_of_curves)+1)]
        pen = pg.mkPen(color=new_color, width=width)
        self.list_of_curves[idx].setPen(pen)

        xdata = self.list_of_xdata[idx]
        if self.ui.control_box.normalize_checkbox.isChecked():
            ydata = self.list_of_ydata[idx]*(1.0/np.amax(self.list_of_ydata[idx]))
        else:
            ydata = self.list_of_ydata[idx]

        self.list_of_curves[idx].setData(xdata, ydata, name=self.list_of_datanames[idx])


    def plotCurves(self):
        for curve in self.list_of_curves:
            idx = self.list_of_curves.index(curve)

            xdata = self.list_of_xdata[idx]
            if self.ui.control_box.normalize_checkbox.isChecked():
                ydata = self.list_of_ydata[idx]*(1.0/np.amax(self.list_of_ydata[idx]))
                self.ui.GraphArea.setLabel('left', self.yaxis_title, units='normalized', **labelstyle)
            else:
                ydata = self.list_of_ydata[idx]

            self.list_of_curves[idx].setData(xdata, ydata, name=self.list_of_datanames[idx])
            
            
    def findPeaks(self, data):
        peaks_plots = list(np.empty((len(self.list_of_ydata))))
        for indx in enumerate(self.list_of_ydata)[0]:
            xdata = self.list_of_xdata[indx]
            ydata = self.list_of_ydata[indx]
            intensities = np.asarray(ydata)
            indexes = peakutils_indexes(intensities, thres=1.0/max(intensities), min_dist=10.0)
            peaks_val = [intensities[i] for i in indexes]
            peaks_pos = [xdata[i] for i in indexes]
            widths = peak_widths(intensities, indexes)
            peaks_fwhm = peak_widths(intensities, indexes, rel_height=0.5)
            #norm_intensitites = [a*b/np.amax(intensities) for (a,b) in zip(peaks,widths[0])]
            
            marker = list(symbols.values())[indx+1]
            peaks_plots[indx] = self.ui.GraphArea.plot(xdata, ydata, symbol=marker)
            self.list_of_curves.append(peaks_plots[indx])
    
    
    def fitData(self):            
        for curve in self.list_of_curves:
            idx = self.list_of_curves.index(curve)

            xdata = self.list_of_xdata[idx]
            if self.ui.control_box.normalize_checkbox.isChecked():
                ydata = self.list_of_ydata[idx]*(1.0/np.amax(self.list_of_ydata[idx]))
                              
                self.ui.GraphArea.setLabel('left', self.yaxis_title, units='norm.', **labelstyle)
                
            else:
                ydata = self.list_of_ydata[idx]
                
            if self.ui.control_box.exp_button.isChecked():
                results = least_squares(exp_func,
                                   x0=[1.0, 10, 0.1],
                                   bounds=([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf]),
                                   args=(xdata, ydata),
                                   method='trf',
                                   tr_solver='lsmr',
                                   loss='soft_l1',
                                   jac='3-point',
                                   x_scale='jac',
                                   max_nfev=10000)
                params = results.x
                fitted_data = exp_func(params, xdata)
            
            
            elif self.ui.control_box.biexp_button.isChecked():
                results = least_squares(biexp_func,
                                   x0=[1.0,10, 0.1, 100],
                                   bounds=([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf]),
                                   args=(xdata, ydata),
                                   method='trf',
                                   tr_solver='lsmr',
                                   loss='soft_l1',
                                   jac='3-point',
                                   x_scale='jac',
                                   max_nfev=10000)
                params = results.x
                fitted_data = biexp_func(params, xdata)
            
            
            elif self.ui.control_box.triexp_button.isChecked():
                results = least_squares(triexp_func,
                                   x0=[1.0, 10, 0.1, 100, 0.01, 1000],
                                   bounds=([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf]),
                                   args=(xdata, ydata),
                                   method='trf',
                                   tr_solver='lsmr',
                                   loss='soft_l1',
                                   jac='3-point',
                                   x_scale='jac',
                                   max_nfev=10000)
                params = results.x
                fitted_data = triexp_func(params, xdata)
            
            
            elif self.ui.control_box.power_button.isChecked():
                results = least_squares(power_func,
                                   x0=[1.0, 0.5, 0.1],
                                   bounds=([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf]),
                                   args=(xdata, ydata),
                                   method='trf',
                                   tr_solver='lsmr',
                                   loss='soft_l1',
                                   jac='3-point',
                                   x_scale='jac',
                                   max_nfev=10000)
                params = results.x
                fitted_data = power_func(params, xdata)
            
            
            elif self.ui.control_box.gaussian_button.isChecked():
                results = least_squares(gaussian_func,
                                   x0=[1.0, 0.5, 0.0],
                                   bounds=([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf]),
                                   args=(xdata, ydata),
                                   method='trf',
                                   tr_solver='lsmr',
                                   loss='soft_l1',
                                   jac='3-point',
                                   x_scale='jac',
                                   max_nfev=10000)
                params = results.x
                fitted_data = gaussian_func(params, xdata)
            
            elif self.ui.control_box.cauchy_button.isChecked():
                results = least_squares(cauchy_func,
                                   x0=[1.0, 0.5, 0.0],
                                   bounds=([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf]),
                                   args=(xdata, ydata),
                                   method='trf',
                                   tr_solver='lsmr',
                                   loss='soft_l1',
                                   jac='3-point',
                                   x_scale='jac',
                                   max_nfev=10000)
                params = results.x
                fitted_data = cauchy_func(params, xdata)
            
            else:
                pass
                
                
        
    
    def dataReset(self):
        pass
        
            
    def dataAnalyse(self):
        if self.ui.control_box.findpeaks_checkBox.isChecked():
            self.findPeaks(self)
            
        if self.ui.control_box.fitting_checkBox.isChecked():
            self.fitData(self)
        

        

if __name__=='__main__':

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setPalette(DarkPalette)
    screen_resolution = app.desktop().screenGeometry()
    width, height = screen_resolution.width(), screen_resolution.height()

    mainwindow = DataAnalysis()
    mainwindow.ui.showMaximized()

    sys.exit(app.exec_())
