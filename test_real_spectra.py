import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from scipy.special import wofz
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy.signal import find_peaks


class SpecBroadener():
    def __init__(self, freq, inti, spec_lenth, broaden_factor) -> None:

        self.n_mode = len(freq)
        self.freq = freq
        self.inti = inti
        self.spec_length = spec_lenth

        self.fwhm = broaden_factor

    def _Voigt(self, x, y0, amp, pos, fwhm, shape=1):
        tmp = 1 / wofz(np.zeros((len(x))) + 1j * np.sqrt(np.log(2.0)) * shape).real
        return y0 + tmp * amp * wofz(
            2 * np.sqrt(np.log(2.0)) * (x - pos) / fwhm + 1j * np.sqrt(np.log(2.0)) * shape).real

    def _form_data(self):
        x = np.array([i for i in range(self.spec_length)])
        y0 = np.array([0.0 for _ in range(self.spec_length)])
        yf = y0
        for n in range(self.n_mode):
            y = self._Voigt(x, y0, self.inti[n], self.freq[n], self.fwhm)
            yf = yf + y
        return x, yf

def broaden(freq, intens, length):
    xxx = SpecBroadener(freq, intens, length, 10)  # 5000 10
    x_freq, y_intens = xxx._form_data()
    return y_intens

def normlize(data):
    d_max=max(data)
    ratio = 600/d_max
    norm_data=[ratio * i for i in data]
    return norm_data

from framework_1 import Batch_Net
my_net = torch.load(f'm_1.pkl')
my_net.eval()

data = np.loadtxt(f'real_spectra.csv', delimiter=',', dtype=str)
data = data[1:]
data = data.astype(np.float64)
# print(data)
x = data[:, 0]
# print(x)
for i in range(1, 11):
    y = data[:, i + 1]
    new_y = []
    freq = []
    intens = []
    print(i)
    y = np.log10(1 / y)
    max = y.max()
    thre = max * 0.2
    for ii in y:
        if ii > thre:
            new_y.append(ii)
        else:
            new_y.append(0)
    peaks, _ = find_peaks(new_y, height=0)
    for iii in peaks:
        freq.append(x[iii])
        intens.append(new_y[iii])
    intens = normlize(intens)
    # print(freq)
    # print(intens)
    input = broaden(freq, intens, 5000)
    input = input.reshape(1, input.shape[0])
    input = torch.tensor(input, dtype=torch.float32)
    output = my_net(input)
    pred = []
    for iiii in output[0]:
        if iiii > 0:
            pred.append(1)
        else:
            pred.append(0)
    print(pred)  # ['*CHO','*CHOH','*CO','*COH','*COOH','*OCH3','*OCHO']


