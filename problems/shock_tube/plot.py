import sys
sys.path.insert(0, '../../src')
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from globe import *

def line_plot():

    for num in range(0, 11):

        f, ax1 = plt.subplots()

        x1p, x2p, Prho, Pvx1, Pprs = np.loadtxt(f'output/pluto/data.{num:04}.tab', unpack=True)
        V, x1 = np.load(f'output/1D/data.{num:04}.npy')

        ax1.plot(x1, V[rho], 'b')
        ax1.plot(x1, V[prs], 'r')
        ax1.plot(x1, V[vx1], 'g')
        ax1.plot(x1p, Prho, 'b--')
        ax1.plot(x1p, Pprs, 'r--')
        ax1.plot(x1p, Pvx1, 'g--')

        plt.savefig(f'output/plots/line_{num:04}.png')
        plt.close()

line_plot()
