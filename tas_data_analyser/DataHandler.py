import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit, least_squares
from scipy.signal import welch, lombscargle
from scipy.signal.windows import hann, hamming

from PyQt5.QtCore import QObject, Qt

from snippets import *


def fit_chirp(p, x, y):
    return func_residuals(func_exponent, p, x, y)


def fit_trace(p, x, y):
    return func_residuals(biexp_decay, p, x, y)

labelstyle = {'color': 'k', 'font-size': '12pt'}
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

class DataControlWidget(QObject):

    def __init__(self, parent=None):
        super(DataControlWidget, self).__init__(parent)
        self.parent = parent
        self.ui = self.parent.controls_widget

        self.connectUI()


    def connectUI(self):
        self.ui.loadData_button.clicked.connect(self.loadData)
        self.ui.correctData_button.clicked.connect(self.correctData)
        self.ui.plotData_button.clicked.connect(self.plotData)
        self.ui.exportplot_button.clicked.connect(self.exportPlot)
        self.ui.gettraces_button.clicked.connect(self.getTraces)
        self.ui.plottraces_button.clicked.connect(self.plotTraces)
        self.ui.exporttraces_button.clicked.connect(self.exportTraces)
        self.ui.cleartraces_button.clicked.connect(self.clearTracesPlots)

        self.parent.surfplot_widget.plotItem.scene().sigMouseClicked.connect(self.onMouseClicked)


    def loadData(self):

        self.files_list, _ = self.parent.file_dialog.getOpenFileNames()

        if len(self.files_list) > 0:
            self.file_dirname = os.path.dirname(self.files_list[0])
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
            self.snr_t0 = abs(self.var_t0/self.std_t0)

            self.ui.writeUI(self.delta_wl, self.W0, wlen_min, wlen_max, self.T0, time_min, time_max, inty_min, inty_max)
            self.ui.buttons_box.setEnabled(True)
            self.ui.plotOpt_box.setEnabled(True)
            self.ui.dataOpt_box.setEnabled(True)
            self.ui.loadData_button.setText("New Data")
            self.ui.filename_loaded.setText(self.files_list[0])
            self.ui.dataname_input.setText(self.file_basename)

            self.parent.statusbar.showMessage('Succeeded to load datafiles')

        else:
            self.parent.statusbar.showMessage('Failed to load any datafile')


    def correctData(self):
        if (self.ui.w0_checkbox.isChecked() or self.ui.t0_checkbox.isChecked()):

            self.delta_wl, self.W0, wlen_min, wlen_max, self.T0, time_min, time_max, inty_min, inty_max = self.ui.readUI()
            self.reloadData()

            chirp='No'
            smooth='No'

            if self.ui.dlambda_checkbox.isChecked():
                self.smoothLambda()
                smooth='yes'
                if self.ui.chirp_checkbox.isChecked():
                    self.correctChirp()
                    chirp='yes'
            elif self.ui.chirp_checkbox.isChecked():
                self.correctChirp()
                chirp='yes'
            else:
                chirp='No'
                smooth='No'
                pass

        elif self.ui.dlambda_checkbox.isChecked():
            self.reloadData()
            self.smoothLambda()
            smooth='yes'
            if self.ui.chirp_checkbox.isChecked():
                self.correctChirp()
                chirp='yes'

        elif self.ui.chirp_checkbox.isChecked():
            self.correctChirp()
            chirp='yes'

        else:
            self.reloadData()
            chirp='No'
            smooth='No'

        time_min, time_max = self.tl_t0[0], self.tl_t0[-1]
        wlen_min, wlen_max = self.avg_wl[0], self.avg_wl[-1]
        inty_min, inty_max = np.floor(np.amin(self.avg_il)), np.ceil(abs(np.amin(self.avg_il)))

        self.ui.writeUI(self.delta_wl, self.W0, wlen_min, wlen_max, self.T0, time_min, time_max, inty_min, inty_max)

        self.parent.statusbar.showMessage('Data have been corrected: smooth={}, chirp={}'.format(smooth, chirp))

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
        self.snr_t0 = abs(self.var_t0/self.std_t0)

        self.ui.writeUI(self.delta_wl, self.W0, wlen_min, wlen_max, self.T0, time_min, time_max, inty_min, inty_max)


    def readJsonData(self, files_list):
        for file in files_list:
            if 'data' not in locals():
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


        self.dirname = os.path.dirname(files_list[0])
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
            if 'data' not in locals():
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

        self.dirname = os.path.dirname(files_list[0])
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

        avg_wl = mean_dim_reduction(self.avg_wl, n_dwl)
        avg_il = mean_dim_reduction(self.avg_il, n_dwl)

        self.avg_wl = avg_wl
        self.avg_il = avg_il

        self.std_t0 = np.std(self.avg_il[:,self.tl_t0<=self.T0])
        self.var_t0 = np.var(self.avg_il[:,self.tl_t0<=self.T0])
        self.mean_t0 = np.mean(self.avg_il[:,self.tl_t0<=self.T0])
        self.cov_t0 = abs(self.std_t0/self.mean_t0)
        self.snr_t0 = abs(self.var_t0/self.std_t0)**2

        time_min, time_max = self.tl_t0[0], self.tl_t0[-1]
        wlen_min, wlen_max = avg_wl[0], avg_wl[-1]
        inty_min, inty_max = np.amin(avg_il), np.amax(avg_il)

        self.ui.writeUI(self.delta_wl, self.W0, wlen_min, wlen_max, self.T0, time_min, time_max, inty_min, inty_max)


    def correctChirp(self):
        new_avg_il = np.empty(np.shape(self.avg_il))

        wl_indexes = list(range(len(self.avg_wl)))
        T0_idx = len(self.tl_t0[self.tl_t0<=self.T0])
        tl_indexes = list(range(len(self.tl_t0)))

        unch_il = self.avg_il
        unch_il[:,self.tl_t0<self.T0]=0.0

        il_max = np.empty((0))
        new_t0 = np.empty((0))
        for wlidx in wl_indexes:
            wls = np.abs(unch_il[wlidx][self.tl_t0<=3*self.T0])
            il_max = np.append(il_max, np.where(wls==wls.max()))
            new_t0 = np.append(new_t0, self.tl_t0[np.where(wls==wls.max())])

        plt.plot(self.avg_wl, new_t0)

        exp_result = least_squares(fit_chirp,
                                    bounds=([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf]),
                                    args=(self.avg_wl, new_t0),
                                    method='trf',
                                    tr_solver='lsmr',
                                    loss='soft_l1',
                                    jac='3-point',
                                    x_scale='jac',
                                    max_nfev=10000)

        params = exp_result.x

        new_il_max = func_exponent(params, self.avg_wl)

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

        T0_idx = len(self.tl_t0[self.tl_t0<=self.T0])

        unch_il = self.avg_il

        std_t0 = np.std(self.avg_il[:,self.tl_t0<=self.T0])
        var_t0 = np.var(self.avg_il[:,self.tl_t0<=self.T0])
        mean_t0 = np.mean(self.avg_il[:,self.tl_t0<=self.T0])

        cov_t0 = abs(std_t0/mean_t0)
        snr_t0 = abs(var_t0/std_t0)

        for wlidx in range(len(self.avg_wl)):
            unch_il[wlidx][:3*T0_idx][(unch_il[wlidx][:3*T0_idx]-std_t0 <= mean_t0) & (unch_il[wlidx][:3*T0_idx]+std_t0 >= mean_t0)] = 0.0
            new_avg_il_row = np.delete(unch_il[wlidx],np.where(unch_il[wlidx] == 0.0))
            new_avg_il_row = np.append(new_avg_il_row, new_avg_il_row[-1]*np.ones(len(self.tl_t0)-len(new_avg_il_row)))
            new_avg_il[wlidx] = new_avg_il_row

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

        self.plotData()


    def updateHistogram(self):
        hist_arr = self.parent.surfplot_widget.imageItem.getHistogram()
        self.parent.surfplot_widget.histogramWidget.setHistogramRange(hist_arr[0].min(), hist_arr[0].max(), padding=None)
        self.parent.surfplot_widget.histogramWidget.setLevels(hist_arr[0].min(), hist_arr[0].max())

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

        hist_gradient = self.parent.surfplot_widget.histogramWidget.gradient.getGradient()

        grad_neg_interp = np.interp(hist_pos_values, [hist_pos_values.min(), 0.0], [0.0, 0.5])
        grad_pos_interp =  np.interp(hist_neg_values, [0.0, hist_neg_values.max()], [0.5, 1.0])

        self.parent.surfplot_widget.histogramWidget.gradient.setTickValue(0, 0.0)
        self.parent.surfplot_widget.histogramWidget.gradient.setTickValue(1, 0.5)
        self.parent.surfplot_widget.histogramWidget.gradient.setTickValue(2, 1.0)


    def plotData(self):
        self.delta_wl, self.W0, wlen_min, wlen_max, self.T0, time_min, time_max, inty_min, inty_max = self.ui.readUI()

        time_min_idx = np.where(np.around(self.tl_t0,decimals=3)==time_min)[0][0]
        time_max_idx = np.where(np.around(self.tl_t0,decimals=3)==time_max)[0][0]
        wlen_min_idx = np.where(np.around(self.avg_wl,decimals=3)==wlen_min)[0][0]
        wlen_max_idx = np.where(np.around(self.avg_wl,decimals=3)==wlen_max)[0][0]

        self.image = self.avg_il[wlen_min_idx:wlen_max_idx, time_min_idx:time_max_idx]

        self.parent.surfplot_widget.imageViewWidget.setImage(self.image, autoRange=True, autoLevels=True, autoHistogramRange=True)

        self.updateHistogram()

        self.parent.surfplot_widget.imageItem.translate(0, self.W0)
        self.parent.surfplot_widget.imageItem.scale((time_max-time_min)/np.size(self.image[0,:]), (wlen_max-wlen_min)/np.size(self.image[:,0]))
        self.parent.surfplot_widget.plotItem.setLimits(xMin=time_min, xMax=1.01*time_max, yMin=wlen_min, yMax=1.01*wlen_max)

        self.image_name = self.dirname+'/surfplot_{}-{}ps.png'.format(int(time_min), int(time_max))


    def updatePlot(self):
        self.delta_wl, self.W0, wlen_min, wlen_max, self.T0, time_min, time_max, inty_min, inty_max = self.ui.readUI()

        self.parent.surfplot_widget.plotItem.setLimits(xMin=time_min, xMax=1.01*time_max, yMin=wlen_min, yMax=1.01*wlen_max)
        time_min_idx = np.where(np.around(self.tl_t0,decimals=3)==time_min)[0][0]
        time_max_idx = np.where(np.around(self.tl_t0,decimals=3)==time_max)[0][0]
        wlen_min_idx = np.where(np.around(self.avg_wl,decimals=3)==wlen_min)[0][0]
        wlen_max_idx = np.where(np.around(self.avg_wl,decimals=3)==wlen_max)[0][0]

        self.image = self.avg_il[wlen_min_idx:wlen_max_idx,time_min_idx:time_max_idx]

        self.parent.surfplot_widget.imageViewWidget.setImage(self.image, autoRange=True, autoLevels=True, autoHistogramRange=True)
        self.parent.surfplot_widget.imageItem.setRect(self.parent.surfplot_widget.plotItem.vb.viewRect())

        self.image_name = self.dirname+'/intymap_{}-{}ps_{}.png'.format(int(time_min), int(time_max), self.chirpStat)


    def exportPlot(self):
        self.parent.surfplot_widget.exporter.export(self.image_name)


    def clearAllPlots(self):
        self.clearTracesPlots()
        self.parent.traces_widget.kinetics.clear()
        self.parent.traces_widget.spectral.clear()
        self.parent.surfplot_widget.plotItem.clear()


    def onMouseClicked(self, point):
        self.x, self.y = self.parent.mouse_point.x(), self.parent.mouse_point.y()

        self.delta_wl, self.W0, wmin, wmax, self.T0, tmin, tmax, imin, imax = self.ui.readUI()

        wlen_idx = min(np.searchsorted(self.avg_wl, self.y), len(self.avg_wl) - 1)
        wmin_idx = min(np.searchsorted(self.avg_wl, wmin), len(self.avg_wl) - 1)
        wmax_idx = min(np.searchsorted(self.avg_wl, wmax), len(self.avg_wl) - 1)

        time_idx = min(np.searchsorted(self.tl_t0, self.x), len(self.tl_t0) - 1)
        tmin_idx = min(np.searchsorted(self.tl_t0, tmin), len(self.tl_t0) - 1)
        tmax_idx = min(np.searchsorted(self.tl_t0, tmax), len(self.tl_t0) - 1)

        time_range = self.tl_t0[tmin_idx:tmax_idx]
        tmax_idx = tmax_idx if int(time_range.size % 2) == 0 else tmax_idx-1
        time_range = self.tl_t0[tmin_idx:tmax_idx]


        self.kinetics_trace = self.avg_il[wlen_idx, tmin_idx:tmax_idx]
        self.kinetics_trace = np.asarray([i - np.sign(i)*self.snr_t0 for i in self.kinetics_trace])
        self.spectrum_trace = self.avg_il[wmin_idx:wmax_idx, time_idx]

        self.kinetics_std_trace = self.std_il[wlen_idx, tmin_idx:tmax_idx]
        self.spectrum_std_trace = self.std_il[wmin_idx:wmax_idx, time_idx]

        self.parent.traces_widget.kinetics.legend.clear()
        self.parent.traces_widget.kinetics.clear()

        self.parent.traces_widget.spectral.legend.clear()
        self.parent.traces_widget.spectral.clear()

        if self.ui.std_checkbox.isChecked():
            # not yet implemented
            self.parent.traces_widget.kinetics.setLabel('left', '<math>&Delta;A</math>', units='OD', unitPrefix='m', **labelstyle)
            self.parent.traces_widget.kinetics.setLabel('bottom', 'Time', units='ps', **labelstyle)

            self.parent.traces_widget.spectral.plot(self.avg_wl[wmin_idx:wmax_idx], self.spectrum_trace, pen='b', name='<math>&Delta;t = {} ps<math>'.format(round(self.x,2)))

            if self.ui.fit_checkbox.isChecked():

                exp_result = least_squares(fit_trace,
                                           x0=[100, 1000, -0.5, 0.5, 0.05],
                                           bounds=([1e1, 1e2, -np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf]),
                                           args=(time_range, self.kinetics_trace),
                                           method='trf',
                                           tr_solver='lsmr',
                                           loss='soft_l1',
                                           jac='3-point',
                                           x_scale='jac',
                                           # f_scale=0.04,
                                           max_nfev=10000)

                params = exp_result.x
                fit_curve = triexp_decay(params, time_range)

                self.parent.traces_widget.kinetics.plot(time_range, fit_curve, pen={'color':'r', 'width':2, 'style':Qt.DashLine}, name='<math>&tau<sub>1</sub>; = {} ps<math>\t<math>&tau<sub>2</sub>; = {} ps<math>\t<math>&tau<sub>3</sub>; = {} ps<math>'.format(round(params[0],2),round(params[1],2),round(params[2],2)))

        elif self.ui.fft_checkbox.isChecked():
            self.parent.traces_widget.kinetics.setLabel('left', '<math>&Delta;A</math>', units='OD', unitPrefix='m', **labelstyle)
            self.parent.traces_widget.kinetics.setLabel('bottom', 'Wavenumber', units='cm<sup>-1</sup>', unitPrefix='c', **labelstyle)

            exp_result = least_squares(fit_trace,
                                       x0=[100, 1000, -0.5, 0.5, 0.05],
                                       bounds=([1e1, 1e2, -np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf]),
                                       args=(time_range, self.kinetics_trace),
                                       method='trf',
                                       tr_solver='lsmr',
                                       loss='soft_l1',
                                       jac='3-point',
                                       x_scale='jac',
                                       # f_scale=0.04,
                                       max_nfev=10000)

            params = exp_result.x
            residuals = exp_result.fun

            #N = residuals.size #if int(len(resi)%2)==0 else len(resi)-1

            w = hann(residuals.size, sym=False)
            wavenumbers = np.linspace(0.01,10,100)/0.02997924536
            intensities = lombscargle(time_range, w*residuals, wavenumbers, normalize=False, precenter=False)
            # intensities = nfft.ndft(time_range, hann_window*residuals)
            # intensities = nfft.ndft_adjoint(time_range, hann_window*residuals, N).real
            # wavenumbers = (33.35641)*np.fft.fftfreq(intensities.size, 5.54)
            self.kinetics_trace_fft = np.sqrt(intensities)#intensities#
            self.kinetics_trace_wnum = wavenumbers
            # self.kinetics_trace_fft = nfft.ndft(time_range, hann_window*residualsu)
            # self.kinetics_trace_wnum = (333.5641)*np.fft.fftfreq(self.kinetics_trace_fft.size, 0.554)
            # # self.kinetics_trace_wnum = nfft.ndft_adjoint(time_range, self.kinetics_trace, N).real
            # fft_real = self.kinetics_trace_fft.real#np.real(2.0*self.kinetics_trace_fft)
            # fft_imag = self.kinetics_trace_fft.imag
            self.parent.traces_widget.kinetics.plot(self.kinetics_trace_wnum, self.kinetics_trace_fft, pen='b', name='<math>&lambda; = {} nm</math>'.format(round(self.y,2)))
            # self.parent.traces_widget.kinetics.plot(self.kinetics_trace_wnum, fft_imag, pen='r', name='<math>&lambda; = {} nm</math>'.format(round(self.y,2)))
            self.parent.traces_widget.spectral.plot(self.avg_wl[wmin_idx:wmax_idx], self.spectrum_trace, pen='b', name='<math>&Delta;t = {} ps<math>'.format(round(self.x,2)))

        else:
            self.parent.traces_widget.kinetics.setLabel('left', '<math>&Delta;A</math>', units='OD', unitPrefix='m', **labelstyle)
            self.parent.traces_widget.kinetics.setLabel('bottom', 'Time', units='ps', **labelstyle)
            self.parent.traces_widget.kinetics.plot(time_range, self.kinetics_trace, pen='b', name='<math>&lambda; = {} nm</math>'.format(round(self.y,2)))
            self.parent.traces_widget.spectral.plot(self.avg_wl[wmin_idx:wmax_idx], self.spectrum_trace, pen='b', name='<math>&Delta;t = {} ps<math>'.format(round(self.x,2)))

            if self.ui.fit_checkbox.isChecked():

                exp_result = least_squares(fit_trace,
                                           x0=[100, 1000, -0.5, 0.5, 0.05],
                                           bounds=([1e1, 1e2, -np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf]),
                                           args=(time_range, self.kinetics_trace),
                                           method='trf',
                                           tr_solver='lsmr',
                                           loss='soft_l1',
                                           jac='3-point',
                                           x_scale='jac',
                                           # f_scale=0.04,
                                           max_nfev=10000)

                params = exp_result.x
                fit_curve = triexp_decay(params, time_range)

                self.parent.traces_widget.kinetics.plot(time_range, fit_curve, pen={'color':'r', 'width':2, 'style':Qt.DashLine}, name='<math>&tau<sub>1</sub>; = {} ps<math>\t<math>&tau<sub>2</sub>; = {} ps<math>\t<math>&tau<sub>3</sub>; = {} ps<math>'.format(round(params[0],2),round(params[1],2),round(params[2],2)))


    def getTraces(self):
        try:
            self.compare_traces_widget.ccount
        except:
            self.compare_traces_widget.ccount = 0
        finally:
            if self.compare_traces_widget.ccount < 10:
                pen = pg.mkPen(color=colors_list[self.compare_traces_widget.ccount], width=2)
                self.compare_traces_widget.ccount += 1
            else:
                self.compare_traces_widget.ccount = 0
                pen = pg.mkPen(color=colors_list[self.compare_traces_widget.ccount], width=2)

            dataname = self.plotswidget.controls_widget.dataname_input.text()
            self.compare_traces_widget.kinetics.setTitle(dataname)
            self.compare_traces_widget.spectral.setTitle(dataname)
            self.compare_traces_widget.kinetics.plot(self.tl_t0[self.tmin_idx:self.tmax_idx], self.kinetics_trace, pen=pen, name='<math>&lambda; = {} nm</math>'.format(round(self.y,2)))
            self.compare_traces_widget.spectral.plot(self.avg_wl[self.wmin_idx:self.wmax_idx], self.spectrum_trace, pen=pen, name='<math>&Delta;t = {} ps<math>'.format(round(self.x,2)))


    def exportTraces(self):
        try:
            self.parent.parent.traces_widget.ccount
        except:
            self.parent.parent.traces_widget.ccount = 0
        finally:
            if self.parent.parent.traces_widget.ccount > 9:
                self.parent.parent.races_widget.ccount = 0

            pen = pg.mkPen(color=colors_list[self.parent.parent.traces_widget.ccount], width=2)
            self.parent.parent.traces_widget.ccount += 1

            dataname = self.dataname_input.text()
            self.parent.parent.traces_widget.kinetics.setLabel('left', '<math>&Delta;A', units='normalized', **labelstyle)
            self.parent.parent.traces_widget.spectral.setLabel('left', '<math>&Delta;A', units='normalized', **labelstyle)
            self.parent.parent.traces_widget.kinetics.setTitle('<math>&lambda; = {} nm</math>'.format(round(self.y,2)))
            self.parent.parent.traces_widget.spectral.setTitle('<math>&Delta;t = {} ps<math>'.format(round(self.x,2)))

            self.parent.parent.traces_widget.kinetics.plot(self.tl_t0[self.tmin_idx:self.tmax_idx], self.kinetics_trace/self.kinetics_trace.max(), pen=pen, name=dataname)
            self.parent.parent.traces_widget.spectral.plot(self.avg_wl[self.wmin_idx:self.wmax_idx], self.spectrum_trace/self.spectrum_trace.max(), pen=pen, name=dataname)


    def plotTraces(self):
        self.compare_traces_widget = TracesWidget(self.parent, axis='r')
        self.compare_traces_widget.showMaximized()
        #self.parent.exportedTracesWidget.showMaximized()


    def clearTracesPlots(self):
        self.compare_traces_widgett.kinetics.legend.clear()
        self.compare_traces_widget.spectral.clear()

        self.compare_traces_widget.spectral.legend.clear()
        self.compare_traces_widget.kinetics.clear()

        self.parent.parent.traces_widget.kinetics.legend.clear()
        self.parent.parent.traces_widget.spectral.clear()

        self.parent.parent.traces_widget.spectral.legend.clear()
        self.parent.parent.traces_widget.kinetics.clear()
