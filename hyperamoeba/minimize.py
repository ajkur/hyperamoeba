# Usage:
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

def minimize(cost_fun,cost_args,bounds,n_samp,sampler='mc',n_chains=size,verb=0):
    ''' Minimize a function with the hypercube amoeba search

    Args:
        cost_fun (function): function to optimize
        cost_args (list): list of arguments for cost function
        bounds (array): bounds in each direction (number of dimensions x 2 array)
        n_samp (int): number of hypercube samples (rounded down to a multiple of the number of cores)

    Kwargs:
        sampler (str): sampler type (mc: Monte Carlo/lhs: Latin Hypercube Sampler, defalut mc)
        n_chains (int): number of simplex chains (default size of the MPI communicator)
        verb (int): verbosity level (0/1/2)

    To-do:
        Clean up number of print statements in verbose modes
        Make a smoother boundary transition
        Stopping tolerance for the simplex algorithm (both standard dev and minimum error of simplex verticies)
        Chain proximity tolerance (check against stopped chains as well as those currently running)
        Simplex chains pull from a stack of starting points

    Returns:
        best_pt (array): location of best point returned by root processor
    '''
    # Round number of samples to a multiple of the number of cores for load balancing
    n_samp = size*int(np.floor(n_samp/size))

    # Get input space dimension
    n_dim = bounds.shape[0]

    # Sample starting points: monte-carlo or latin hypercube
    if rank == 0:
        if sampler == 'mc':
            all_samps = mc_samp(n_samp,n_dim)
        elif sampler == 'lhs':
            all_samps = lhs(n_samp,n_dim)
        else:
            raise ValueError('\''+sampler+'\' is not a sampler option.')
        all_samps = bound_map(all_samps,bounds)
        sample_list = np.array_split(all_samps,size)
    else:
        sample_list = []

    # Run function at each starting point (MPI)
    cost_list = grid_search(cost_fun,cost_args,sample_list)

    # Process grid search results
    if rank == 0:

        # Recombine error pairs
        cost_arr = np.concatenate(cost_list)

        # Get n_chains best sample points
        chain_inds = cost_arr.argsort()[:n_chains]
        chain_pts = all_samps[chain_inds,:]
        chain_pts_list = np.array_split(chain_pts,n_chains)
    else:
        chain_pts_list = []

    # Run simplex at each point (MPI)
    # Scatter input arrays
    my_ins = comm.scatter(chain_pts_list,root=0)

    # Run simplex on own input point
    my_ins = bound_map(my_ins,bounds,to_og=False)
    x_one,min_err,i_cur,my_path = dhsimp.dhsimp(my_ins,bound_wrap,(cost_fun,cost_args,bounds),s_scale=0.1,verb=verb)

    # Gather best points and errors
    pt_list = comm.gather(x_one,root=0)
    err_list = comm.gather(min_err,root=0)
        
    # Return best point
    if rank == 0:
        best_pt = pt_list[np.argmin(err_list)]
        best_pt = bound_map(best_pt,bounds)
        if verb > 0:
            print 'Best Point:',best_pt
        if verb > 1:
            print 'Best point from each chain:',pt_list
            print 'Error from each chain:',err_list
        return best_pt
    else:
        return None

def bound_map(sample_arr,bounds,to_og=True):
    ''' Map bounds from [0,1] to original coordinates and back

    Args:
        sample_arr (array): array of sample points (can be 1D or 2D)
        bounds (array): bounds in each direction (number of dimensions x 2 array)

    Kwargs:
        to_og (bool): True - transform to original function coordinates, False - transform to [0,1]

    Returns:
        sample_trans (array): transformed array of sample points
    '''
    # Transform array
    sample_trans = np.zeros(sample_arr.shape)

    # Check dimension of sample array and perform transform
    if sample_arr.ndim == 1:
        if to_og:
            for j in range(bounds.shape[0]):
                sample_trans[j] = bounds[j,0] + sample_arr[j]*(bounds[j,1] - bounds[j,0])
        else:
            for j in range(bounds.shape[0]):
                sample_trans[j] = (sample_arr[j] - bounds[j,0])/(bounds[j,1] - bounds[j,0])
    elif sample_arr.ndim == 2:
        if to_og:
            for i in range(sample_arr.shape[0]):
                for j in range(sample_arr.shape[1]):
                    sample_trans[i,j] = bounds[j,0] + sample_arr[i,j]*(bounds[j,1] - bounds[j,0])
        else:
            for i in range(sample_arr.shape[0]):
                for j in range(sample_arr.shape[1]):
                    sample_trans[i,j] = (sample_arr[i,j] - bounds[j,0])/(bounds[j,1] - bounds[j,0])
    return sample_trans

def bound_wrap(x_p,(cost_fun,cost_args,bounds),big_num=1e14):
    ''' Return large error if out of bounds, otherwise return the cost function evaluation.

    Args:
        x_p (array): point 
        cost_fun (function): function to optimize
        cost_args (list): list of arguments for cost function
        bounds (array): bounds in each direction (number of dimensions x 2 array)

    Kwargs:
        big_num (float): large number for when the point is out of bounds

    Returns:
        (float): evaluation of the cost function or a large number
    '''
    bound_flag = False
    for i in range(x_p.shape[0]):

        # Check if point is out of bounds
        if x_p[i] <= 0. or x_p[i] >= 1.:
            bound_flag = True
            break
    if bound_flag:

        # Return large number if out of bounds
        return big_num
    else:

        # Transform to original coordinates and evaluate cost function
        x_p_og = bound_map(x_p,bounds)
        return cost_fun(x_p_og,*cost_args)

def grid_search(cost_fun,cost_args,sample_list):
    ''' Evaluate cost function at each sample point

    Args:
        cost_fun (function): function to optimize
        cost_args (list): list of arguments for cost function
        sample_list (list): list of input arrays in original coordinates

    Returns:
        cost_list (list): list of arrays of the cost functions evaluated by each processor
    '''
    # Scatter input arrays
    my_ins = comm.scatter(sample_list,root=0)

    # Evaluate cost function of own input set
    my_cost = np.zeros(my_ins.shape[0])
    for i in range(my_ins.shape[0]):
        my_cost[i] = cost_fun(my_ins[i,:],*cost_args)

    # Gather outputs
    cost_list = comm.gather(my_cost,root=0)

    return cost_list

def lhs(n_samp,n_dim):
    ''' Latin Hypercube Sampler. Takes n_samp draws from n_dim space such that
        each draw is in its own hyperplane.

    Args:
        n_samp (int): number of samples
        n_dim (int): dimension of sample space

    Returns:
        all_samps (array): n_samp by n_dim array of LHS sample points
    '''
    # Width of hypercubes
    dx = 1./n_samp

    # Sample each dimension
    all_samps = np.zeros([n_samp,n_dim])
    for j in range(n_dim):

        # Shuffle array of indices
        tmp_arr = np.arange(n_samp)
        np.random.shuffle(tmp_arr)

        # Uniform draws in each hyperplane for dimension j
        rand_arr = np.random.uniform(size=n_samp)

        # Transform to [0,1] and store
        all_samps[:,j] = dx*(tmp_arr + rand_arr)

    return all_samps

def mc_samp(n_samp,n_dim):
    ''' Monte Carlo Sampler. Takes n_samp draws from n_dim space.

    Args:
        n_samp (int): number of samples
        n_dim (int): dimension of sample space

    Returns:
        all_samps (array): n_samp by n_dim array of sample points
    '''
    all_samps = np.random.uniform(size=n_samp*n_dim).reshape([n_samp,n_dim])
    return all_samps
