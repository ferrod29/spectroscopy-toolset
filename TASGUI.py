from PyQt5 import uic
from PyQt5.QtCore import QObject, Qt, QRectF
from PyQt5.QtGui import QDoubleValidator, QFont, QColor, QImage, QPalette
from PyQt5.QtWidgets import QMainWindow, QAction, QApplication, \
                            QFileDialog, QWidget, QGridLayout, \
                            QSplitter, QMessageBox

from pyqtgraph.exporters import ImageExporter

from DataHandler import *
import pyqtgraph as pg
from matplotlib.pyplot import axis


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

colors_dict = {
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

colors_list = list(colors_dict.values())

symbols = {"None":None, u"\u2022":"o", u"\u002B":"+", u"\u2266":"d", u"\u25B2":"t", u"\u25A0":"s", u"\u2B1F":"p"}
labelstyle = {'color': 'k', 'font-size': '12pt'}



class MainWindow(QMainWindow):
    '''
    Main application window, from this different instances of TAS Analyser
    Plotter can be open.
    It holds a Widget to analyse or compare data from different Analyser instances
    '''
    def __init__(self):
        super(MainWindow, self).__init__()

        self.setGeometry(20,20,1800,1200)
        self.setWindowTitle('T.A.S. Analyser Suite')

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QGridLayout(self.central_widget)
        self.central_widget.adjustSize()

        open_action = QAction('Open', self)
        open_action.setShortcut('Ctrl+N')
        open_action.setStatusTip("Open a New Window")
        open_action.triggered.connect(self.openWindow)

        close_action = QAction('Close', self)
        close_action.setShortcut('Ctrl+Q')
        close_action.setStatusTip('Close Application')
        close_action.triggered.connect(self.close)

        load_action = QAction('Load', self)
        load_action.setShortcut('Ctrl+L')
        load_action.setStatusTip("Load Central Widget")
        load_action.triggered.connect(self.loadWidget)

        self.menubar = self.menuBar()
        self.menu = self.menubar.addMenu('&Menu')
        self.menu.addAction(open_action)
        self.menu.addAction(close_action)

        self.toolbar = self.addToolBar('toolbar')
        self.toolbar.addAction(load_action)

        self.statusbar = self.statusBar()

        self.windows_list = []
        
        self.traces_widget = TracesWidget(self)
        self.layout.addWidget(self.traces_widget)
        self.traces_widget.hide()
        self.windows_list.append(self.traces_widget)


    def closeEvent(self, event):
        '''
        Check for every opened window and close them
        '''
        [window.close() for window in self.windows_list]
        event.accept()

    def openWindow(self):
        '''
        Open a new window and list it
        '''
        new_window = PlotterWindow(self)
        new_window.showMaximized()
        self.windows_list.append(new_window)

    def loadWidget(self):
        '''
        Load central widget and list it
        '''
        if self.traces_widget.isVisible():
            self.traces_widget.hide()
        else:
            self.traces_widget.show()



class PlotterWindow(QMainWindow):
    def __init__(self, parent=None):
        super(PlotterWindow, self).__init__(parent)
        self.parent = parent

        self.setGeometry(20,20,1800,1200)
        self.setWindowTitle('TAS Inspector')

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QGridLayout(self.central_widget)
        self.central_widget.adjustSize()

        self.statusbar = self.statusBar()
        self.toolbar = self.addToolBar('toolbar')

        connect_action = QAction('connect', self)
        connect_action.setStatusTip("Connect to data handler class")
        connect_action.triggered.connect(self.connectHandler)
        self.toolbar.addAction(connect_action)

        self.windows_list = []
        self.file_dialog = QFileDialog()

        self.surfplot_widget = SurfPlotWidget(self)
        self.traces_widget = TracesWidget(self)
        self.controls_widget = ControlPanelWidget(self)

        self.vsplit = QSplitter(Qt.Vertical)
        self.hsplit = QSplitter(Qt.Horizontal)

        self.vsplit.addWidget(self.surfplot_widget)
        self.vsplit.addWidget(self.controls_widget)
        self.hsplit.addWidget(self.vsplit)
        self.hsplit.addWidget(self.traces_widget)
        layout.addWidget(self.hsplit,0,0)
                
        self.compare_traces_widget = TracesWidget(axis='r')
        self.compare_traces_widget.hide()
        self.windows_list.append(self.compare_traces_widget)

        self.surfplot_widget.plotItem.scene().sigMouseMoved.connect(self.onMouseMoved)


    def closeEvent(self, event):
            close = QMessageBox.question(self,"Warning","Confirm to close the application, not saved data will be lost", QMessageBox.Yes | QMessageBox.No)
            if close == QMessageBox.Yes:
                for window in self.windows_list:
                    self.windows_list.remove(window)
                    window.close()
                self.parent.windows_list.remove(self)
                event.accept()
            else:
                event.ignore()


    def onMouseMoved(self, point):
        if self.surfplot_widget.plotItem.sceneBoundingRect().contains(point):
            self.mouse_point = self.surfplot_widget.plotItem.vb.mapSceneToView(point)
            self.surfplot_widget.vLine.setPos(self.mouse_point.x())
            self.surfplot_widget.hLine.setPos(self.mouse_point.y())

    def connectHandler(self):

        data_handler = DataControlWidget(self)


class ControlPanelWidget(QWidget):

    def __init__(self, parent=None):
        super(ControlPanelWidget, self).__init__(parent)
        uic.loadUi('ControlPanel.ui', self)
        self.adjustSize()

        self.W0_value.setValidator(QDoubleValidator(-999.999,999.999,3))
        self.wlen_min.setValidator(QDoubleValidator(-999.999,999.999,3))
        self.wlen_max.setValidator(QDoubleValidator(-999.999,999.999,3))
        self.T0_value.setValidator(QDoubleValidator(-999.999,999.999,3))
        self.time_min.setValidator(QDoubleValidator(-999.999,999.999,3))
        self.time_max.setValidator(QDoubleValidator(-999.999,999.999,3))
        self.inty_min.setValidator(QDoubleValidator(-9999.999,9999.999,3))
        self.inty_max.setValidator(QDoubleValidator(-9999.999,9999.999,3))
        
        self.savenotes_button.clicked.connect(self.saveNotes)
        
    def saveNotes(self):
        dataname = self.dataname_input.text()
        with open(dataname+'_notes.txt', 'w') as notes_file:
            notes_file.write(self.annotations.toPlainText())


    def writeUI(self, delta_wl, W0, wlen_min, wlen_max, T0, time_min, time_max, inty_min, inty_max):

        self.W0_value.setText(str(round(W0,3)))
        self.delta_wl.setText(str(round(delta_wl,3)))
        self.wlen_min.setText(str(round(wlen_min,3)))
        self.wlen_max.setText(str(round(wlen_max,3)))

        self.T0_value.setText(str(round(T0,3)))
        self.time_min.setText(str(round(time_min,3)))
        self.time_max.setText(str(round(time_max,3)))

        self.inty_min.setText(str(round(inty_min,3)))
        self.inty_max.setText(str(round(inty_max,3)))


    def readUI(self):
        delta_wl = float(self.delta_wl.text())
        W0 = float(self.W0_value.text())
        wlen_min = float(self.wlen_min.text())
        wlen_max = float(self.wlen_max.text())

        T0 = float(self.T0_value.text())
        time_min = float(self.time_min.text())
        time_max = float(self.time_max.text())

        inty_min = float(self.inty_min.text())
        inty_max = float(self.inty_max.text())

        self.W0 = W0

        return delta_wl, W0, wlen_min, wlen_max, T0, time_min, time_max, inty_min, inty_max

class SurfPlotWidget(QWidget):

    def __init__(self, parent=None):
        super(SurfPlotWidget, self).__init__(parent)
        self.parent = parent

        self.plotWidget = pg.PlotWidget()
        self.plotItem = self.plotWidget.getPlotItem()
        self.plotItem.enableAutoRange(enable=True)
        self.plotItem.setDownsampling(auto=True)
        self.plotItem.setLabel('left',  'Wavelength', units='nm', **labelstyle)
        self.plotItem.setLabel('bottom', 'Time', units='ps', **labelstyle)

        self.imageItem = pg.ImageItem()
        self.imageItem.setLookupTable(BuRd_lut, update=True)

        self.imageViewWidget = pg.ImageView(parent=self.plotWidget, view=self.plotItem, imageItem=self.imageItem)
        self.imageViewWidget.getView().invertY(False)
        self.imageViewWidget.setColorMap(BuRd_map)
        self.imageViewWidget.autoLevels()
        self.imageViewWidget.ui.menuBtn.hide()
        self.imageViewWidget.ui.roiBtn.hide()

        self.histogramWidget = self.imageViewWidget.getHistogramWidget()
        self.histogramWidget.setImageItem(self.imageViewWidget.getImageItem())
        self.histogramWidget.setBackground(None)

        #cross hair
        dashedblue = pg.mkPen(color='b', width=1, style=Qt.DashLine)
        self.vLine = pg.InfiniteLine(angle=90, pen=dashedblue, movable=False)
        self.hLine = pg.InfiniteLine(angle=0, pen=dashedblue, movable=False)

        self.plotItem.addItem(self.imageItem)
        self.plotItem.addItem(self.vLine, ignoreBounds=True)
        self.plotItem.addItem(self.hLine, ignoreBounds=True)

        self.exporter = ImageExporter(self.plotItem)
        self.exporter.parameters()['height'] = 2400

        layout = QGridLayout(self)
        layout.addWidget(self.imageViewWidget)
        self.adjustSize()
        self.parent.windows_list.append(self)



class TracesWidget(pg.GraphicsLayoutWidget):

    def __init__(self, parent=None, axis='c'):
        '''
        Windows to compare spectral and kinetics traces
        '''
        super(TracesWidget, self).__init__(parent, axis)

        if axis=='r':
            r, c = 0, 1
        else:
            r, c = 1, 0

        self.kinetics = self.addPlot(0,0)
        self.kinetics.setLabel('left', '<math>&Delta;A', units='OD', unitPrefix='m', **labelstyle)
        self.kinetics.setLabel('bottom', 'Time', units='ps', **labelstyle)
        self.kinetics.showGrid(x=True, y=True)
        self.kinetics.addLegend(offset=(-50, -50))
        self.kinetics_curve = self.kinetics.plot(pen='b')

        self.spectral = self.addPlot(r,c)
        self.spectral.setLabel('left', '<math>&Delta;A', units='OD', unitPrefix='m', **labelstyle)
        self.spectral.setLabel('bottom', 'Wavelength', units='nm', **labelstyle)
        self.spectral.showGrid(x=True, y=True)
        self.spectral.addLegend(offset=(-50, -50))
        self.spectral_curve = self.spectral.plot(pen='b')

#         self.setCentralWidget(layout)
        self.adjustSize()


if __name__=='__main__':
    import sys

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setPalette(DarkPalette)
    screen_resolution = app.desktop().screenGeometry()
    width, height = screen_resolution.width(), screen_resolution.height()

    main_window = MainWindow()
    main_window.showMaximized()
    sys.exit(app.exec_())
