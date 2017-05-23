import numpy as np
from grid import Grid,Boundary
from user import parameters, initialise
from evolve import Evolve
from output import Mesh_plot

# Get the parameters from the user.
para = parameters()

# Initialise grid.
g = Grid(para)

# Generate state vector to hold variables.
U = g.state_vector()

# Initialise the state vector.
initialise(U, g)

# Apply boundary conditions.
Boundary(U, g, para)

# Check grid.
Mesh_plot(U, g, 0)

# Integrate in time.
e = Evolve()
e(U, g, para)


