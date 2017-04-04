# hyperamoeba

Stochastic Latin Hypercube or Monte Carlo sampler with amoeba (Nelder-Mead) search for optimization.

## Requirements
* numpy
* 

## Usage
'''python
import hyperamoeba.minimize as ha
import numpy as np

def rosen(x_p,b):
    ''' Rosenbrock function
    '''
    cost = 0.
    for i in range(x_p.shape[0]-1):
        cost += b*(x_p[i+1] - x_p[i]**2)**2 + (x_p[i] - 1.)**2
    return cost

# Inputs
bounds = np.array([[-20.,20.],[-20.,20.]])  # Bounds of search space
n_samp = 10                                 # Number of samples
args = [100.]                               # Rosenbrock args

# Run optimizer
best_pt = ha.minimize(rosen,args,bounds,n_samp)
if ha.rank == 0:
    print 'Best Point:',best_pt
'''
