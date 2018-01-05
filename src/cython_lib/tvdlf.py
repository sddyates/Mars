import numpy as np

def tvdlf(s):
    Smax = max(np.amax(abs(s.SL)), np.amax(abs(s.SR)))
    return 0.5*(s.FL + s.FR - Smax*(s.UR - s.UL))

