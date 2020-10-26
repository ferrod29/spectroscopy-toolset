from PyQt5 import uic

from PyQt5.QtCore import Qt, QRectF

from PyQt5.QtGui import QDoubleValidator, QFont, QColor, QPalette

from PyQt5.QtWidgets import QMainWindow, QAction, QApplication, \
                            QFileDialog, QWidget, QGridLayout, \
                            QSplitter, QMessageBox, QLineEdit, QPushButton

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

import pyqtgraph as pg
import pyqtgraph.exporters

pg.setConfigOptions(imageAxisOrder='row-major', antialias=True, crashWarning=True)
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

BuRd_map = pg.ColorMap((1.0,0.5,0.0),
                        ((255,0,0,255), (255,255,255,255), (0,0,255,255)))
BuRd_lut = BuRd_map.getLookupTable()
colors = {
            'black':(0, 0 ,0),
            'red':(255, 0, 0),
            'blue':(0, 0, 255),
            'green':(0, 255, 0),
            'magenta':(255, 0, 255),
            'cyan':(0, 255, 255),
            'orange':(255, 127, 64),
            'violet':(200, 20, 120),
            'yellow':(255, 255, 0),
            'gray':(127, 127, 127)
        }
symbols = {"None":None, u"\u2022":"o", u"\u002B":"+", u"\u2266":"d",
            u"\u25B2":"t", u"\u25A0":"s", u"\u2B1F":"p"}
labelstyle={'color': 'k', 'font-size': '12pt'}
