# This will be the main file so you will do the following:
#
# import hyperamoeba.minimize as ha
#
# ha.minimize(*args,**kwargs)
#
from __future__ import division
import numpy as np 
import sys, os, time
import dhsimp
from mpi4py import MPI

# MPI setup
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
name = MPI.Get_processor_name()

def minimize(cost_fun,bounds,n_samp,n_chains=size):
    '''Minimize a function with the hypercube amoeba search

    Args:
        Function to optimize
        Bounds in each direction (N x 2 array)
        Number of hypercube samples (Will be rounded down to a multiple of the number of cores.)

    Kwargs:
        Number of simplex chains (Default size of communicator. Free cores will pull from a stack of starting points if n_chains > number of cores.)
        Stopping tolerance for the simplex algorithm
        Chain proximity tolerance (Check with stopped chains as well as those currently running.)

    Returns:

    '''
    # Round number of samples to a multiple of the number of cores for load balancing
    n_samp = size*int(np.floor(n_samp/size))

    # Get input space dimension
    n_dim = bounds.shape[0]

    # Sample starting points: monte-carlo or latin hypercube
    if rank == 0:
        all_samps = np.random.uniform(size=n_samp*n_dim).reshape([n_samp,n_dim])
        all_samps = bound_map(all_samps,bounds)
        sample_list = np.array_split(all_samps,size)
    else:
        sample_list = []

    # Run function at each starting point (MPI)
    cost_list = grid_search(cost_fun,sample_list)

    # Process grid search results
    if rank == 0:

        # Recombine error pairs
        cost_arr = np.concatenate(cost_list)
        for i in range(n_samp):
            print all_samps[i,:],cost_arr[i]

        # Get n_chains best sample points
        chain_inds = cost_arr.argsort()[:n_chains]
        print chain_inds
        print all_samps[chain_inds,:]

    # Run simplex at each point (MPI)

    # Return best point

    if rank == 0:
        return 4
    else:
        return None

def bound_map(all_samps,bounds,to_og=True):
    '''Map bounds from [0,1] to original coordinates and back

    Args:

    Kwargs:

    Returns:

    '''
    if to_og:
        for i in range(all_samps.shape[0]):
            for j in range(all_samps.shape[1]):
                all_samps[i,j] = bounds[j,0] + all_samps[i,j]*(bounds[j,1] - bounds[j,0])
    else:
        for i in range(all_samps.shape[0]):
            for j in range(all_samps.shape[1]):
                all_samps[i,j] = (all_samps[i,j] - bounds[j,0])/(bounds[j,1] - bounds[j,0])
    return all_samps

def grid_search(cost_fun,sample_list):
    '''Evaluate cost function at each sample point

    Args:

    Kwargs:

    Returns:

    '''
    # Scatter input arrays
    my_ins = comm.scatter(sample_list,root=0)

    # Do work on personal input set
    my_cost = np.zeros(my_ins.shape[0])
    for i in range(my_ins.shape[0]):
        my_cost[i] = cost_fun(my_ins[i,:])

    # Gather outputs
    cost_list = comm.gather(my_cost,root=0)

    return cost_list



