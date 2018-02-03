
import numpy as np
from globe import *


class States:

    def __init__(self):
        None

    def build(self, g, axis):
        if axis == 'i':
            self.flux = np.zeros(shape=g.shape_flux_x1)
            self.FL = np.zeros(shape=g.shape_flux_x1)
            self.FR = np.zeros(shape=g.shape_flux_x1) 
            self.UL = np.zeros(shape=g.shape_flux_x1)
            self.UR = np.zeros(shape=g.shape_flux_x1)
            self.VL = np.zeros(shape=g.shape_flux_x1)
            self.VR = np.zeros(shape=g.shape_flux_x1)
            self.SL = np.zeros(shape=g.shape_flux_x1)
            self.SR = np.zeros(shape=g.shape_flux_x1)
        if axis == 'j':
            self.flux = np.zeros(shape=g.shape_flux_x2)
            self.FL = np.zeros(shape=g.shape_flux_x2)
            self.FR = np.zeros(shape=g.shape_flux_x2)
            self.UL = np.zeros(shape=g.shape_flux_x2)
            self.UR = np.zeros(shape=g.shape_flux_x2)
            self.VL = np.zeros(shape=g.shape_flux_x2)
            self.VR = np.zeros(shape=g.shape_flux_x2)
            self.SL = np.zeros(shape=g.shape_flux_x2)
            self.SR = np.zeros(shape=g.shape_flux_x2)
        if axis == 'k':
            self.flux = np.zeros(shape=g.shape_flux_x3)
            self.FL = np.zeros(shape=g.shape_flux_x3)
            self.FR = np.zeros(shape=g.shape_flux_x3)
            self.UL = np.zeros(shape=g.shape_flux_x3)
            self.UR = np.zeros(shape=g.shape_flux_x3)
            self.VL = np.zeros(shape=g.shape_flux_x3)
            self.VR = np.zeros(shape=g.shape_flux_x3)
            self.SL = np.zeros(shape=g.shape_flux_x3)
            self.SR = np.zeros(shape=g.shape_flux_x3)