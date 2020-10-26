# transform dim NxM TAS data to dim Lx3 / L=NxM

import os, sys
import numpy as np
import pandas as pd

import PyQt5
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QFileDialog, QPushButton

import pyqtgraph as pg
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

BuRd_map = pg.ColorMap((1.0,0.5,0.0), ((255,0,0,255), (255,255,255,255), (0,0,255,255)))
BuRd_lut = BuRd_map.getLookupTable()
cmap = plt.get_cmap('RdBu_r')


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, vcenter=None, clip=False):
        """ centering data on 0.0 from normaliztion """
        self.vcenter = vcenter
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1]

        return np.ma.masked_array(np.interp(value, x, y))


class SurfacePlotWidget(mplWidegt.MatplotlibWidget):
    def __init__(self, parent=None, size=(5.0, 5.0), dpi=125):
        super(SurfacePlotWidget, self).__init__(parent)
        # self.opts['distance'] = 2000
        layout = QGridLayout(self)
        self.splotItem = self.getFigure().add_subplot()
        self.show()


class MainWidget(QWidget):
    def __init__(self):
        super(MainWidget, self).__init__()
        layout =QGridLayout(self)
        self.load_button = QPushButton('Load data')
        self.load_button.clicked.connect(self.loadData)
        layout.addWidget(self.load_button)
        self.splotWidget = SurfacePlotWidget()

    def loadData(self):
        file_dialog = QFileDialog()
        datafile, _ = file_dialog.getOpenFileName()
        dataname = os.path.basename(datafile)

        data = pd.read_csv(datafile, header=None, delim_whitespace=True, dtype=np.float64)
        wl = data.iloc[1:,0].to_numpy()
        tl = data.iloc[0,1:].to_numpy()
        il = data.iloc[1:,1:].to_numpy()

        W0 = 520.0
        old_len = np.size(wl)
        wl = np.delete(wl, np.where(wl < W0))
        new_len = np.size(wl)
        W0_index = old_len - new_len

        il = np.delete(np.delete(il, slice(0,0), 1), slice(0,W0_index), 0)
        tl = tl - tl[0]
        tl_t0 = tl
        T0 = 0.5
        n_dwl = 6

        avg_wl = np.empty((0), dtype=np.float64)
        avg_il = np.empty((0), dtype=np.float64)
        std_il = np.empty((0), dtype=np.float64)
        snr_il = np.empty((0), dtype=np.float64)
        avg_std_il = np.empty((0), dtype=np.float64)

        for j in range(0,np.size(wl)-n_dwl,n_dwl):
            avg_wl = np.append(avg_wl, np.mean(wl[j:j+n_dwl]))

            avg_il = np.append(avg_il, [np.mean(il[j:j+n_dwl,i]) for i in range(np.size(tl_t0))])
            std_il = np.append(std_il, [np.std(il[j:j+n_dwl,i]) for i in range(np.size(tl_t0))])

        avg_il = np.reshape(avg_il, (np.size(avg_wl), np.size(tl_t0)))
        std_il = np.reshape(std_il, (np.size(avg_wl), np.size(tl_t0)))

        snr_il = np.power(avg_il/std_il,2)

        time_min, time_max = tl_t0[0], tl_t0[-1]
        wlen_min, wlen_max = avg_wl[0], avg_wl[-1]
        inty_min, inty_max = np.amin(avg_il), np.amax(avg_il)

        std_t0 = np.std(avg_il[:,tl_t0<=T0])
        var_t0 = np.var(avg_il[:,tl_t0<=T0])
        mean_t0 = np.mean(avg_il[:,tl_t0<=T0])
        cov_t0 = abs(std_t0/mean_t0)
        snr_t0 = abs(mean_t0/std_t0)**2

        xdata = np.empty((0), dtype=np.float64)
        ydata = np.empty((0), dtype=np.float64)
        zdata = np.empty((0), dtype=np.float64)

        data3d = open('data3d.dat', 'w')
        for i in range(len(avg_wl)):
            for j in range(len(tl_t0)):
                data3d.write('{}\t{}\t{}\n'.format(avg_wl[i], tl_t0[j], 1e3*avg_il[i][j]))
                xdata = np.append(xdata, tl_t0[j])
                ydata = np.append(ydata, avg_wl[i])
                zdata = np.append(zdata, 1e3*avg_il[i][j])
            data3d.write('\n')
        data3d.close()

        X, Y = np.meshgrid(xdata, ydata)
        Z = np.asarray(zdata, dtype=np.float64)
        Z = np.ma.array(Z, mask=np.isnan(Z))
        levels = MaxNLocator(nbins=999).tick_values(np.amin(Z), np.amax(Z))
        # norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        rgba_img = cmap((1e3*zdata-minZ)/(maxZ -minZ))

        midnorm = MidpointNormalize(vmin=minZ, vcenter=0.0, vmax=maxZ)

        self.intmap = self.splotWidget.splotItem.contourf(X, Y, Z, norm=midnorm, cmap='seismic', vmin=minZ, vmax=maxZ, extend='both')
        # self.splotWidget.splotItem.setData(tl_t0, avg_wl, 1e3*avg_il.T, colors=rgba_img)
        # self.splotWidget.splotItem.scale(3,1,1)


if __name__=='__main__':
    app = QApplication(sys.argv)
    window = MainWidget()
    window.show()

    sys.exit(app.exec_())
