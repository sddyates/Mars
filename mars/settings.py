"""
Synopsis
--------
Define default global variables and indexes for element reference.

Args
----
None.

Attributes
----------
None.

TODO
----
None.
"""

global small_pressure
global rho, prs, eng
global vx1, vx2, vx3
global mvx1, mvx2, mvx3
global u, v, w

rho = 0
prs = 1
vx1 = 2
vx2 = 3
vx3 = 4

eng = prs
u = mvx1 = vx1 
v = mvx2 = vx2
w = mvx3 = vx3

small_pressure = 1.0e-12