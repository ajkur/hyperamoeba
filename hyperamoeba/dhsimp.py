from __future__ import division
import numpy as np 

def cent_calc(x_vec):
    ''' Calculate centroid

    Args:
        x_vec (array): 2d array of vertices

    Returns:
        x_o (array): coordinates of centroid
    '''
    x_o = np.zeros(x_vec.shape[0])
    for i in range(x_vec.shape[0]):
        for j in range(x_vec.shape[1]):
            x_o[i] = x_o[i] + x_vec[i,j]
    x_o = x_o/x_vec.shape[1]
    return x_o

def dhsimp(x_o,the_fun,fun_args,alpha=1.0,gamma=2.0,rho=-0.5,sigma=0.5,err_tol=1e-8,std_tol=1e-8,s_scale=1,itmax=1000,verb=0):
    ''' Downhill simplex (amoeba) method
    
    Args:
        x_o (array): starting point
        the_fun (function): function to be minimized
        fun_args (tuple): tuple of arguments to pass to the function

    Kwargs:
        alpha (float): reflection coefficient
        gamma (float): expansion coefficient
        rho (float): contraction coefficient
        sigma (float): reduction coefficient
        err_tol (float): cost function error stopping tolerance
        std_tol (float): stopping tolerance for standard deviation of the errors
        s_scale (float): scaling for starting simplex
        itmax (int): maximum number of iterations
        verb (int): verbosity (0,1,2)

    Returns:
        x_one (array): coordinates of best point
        min_err (array): error at best point
        i_cur (array): number of simplex iterations
        my_path (list): list of centroid locations at each iteration
    '''
    if verb > 0:
        print '\nStarting Downhill Simplex'
        print 'Sum of\tStd dev of\tIteration'
        print 'Errors\tErrors'
    # Store path of the centroid as a list
    my_path = []
    my_path.append(x_o)

    # Get dimensions
    x_o = x_o.flatten()
    N = x_o.shape[0]

    # Set vertices of a regular simplex
    verts = np.zeros([N,N+1])
    verts1 = np.zeros([N,N])
    errs = np.arange(N+1,dtype=float)
    errs1 = np.zeros(N)
    verts[0,0] = 1.0
    for i in range(N):
        verts[0,i+1] = -1.0/N
    for k in range(1,N):
        v_temp = 0.0
        for j in range(0,k):
            v_temp = v_temp + verts[j,k]**2
        verts[k,k] = (1.0-v_temp)**0.5
        for i in range(k+1,N+1):
            v_temp = 0.0
            for l in range(0,k):
                v_temp = v_temp + verts[l,k]*verts[l,i]
            verts[k,i] = (-1.0/N - v_temp)/verts[k,k]

    # Scale and shift simplex to specified starting centroid 
    for i in range(N):
        for j in range(N+1):
            verts[i,j] = verts[i,j]*s_scale + x_o[i]

    # Calculate erros at each vertex
    for i in range(N+1):
        errs[i] = the_fun(verts[:,i],fun_args)
    
    # Begin simplex loop
    i_cur = 0
    while np.sum(errs) > err_tol and np.std(errs) > std_tol and i_cur < itmax:
        n1_loc = np.argmax(errs)
        one_loc = np.argmin(errs)
        x_n1 = verts[:,n1_loc]
        x_one = verts[:,one_loc]

        # Print progress
        if i_cur%2 == 0 and verb > 0:
            print str(np.sum(errs))+'\t'+str(np.std(errs))+'\t'+str(i_cur)

        # Cut out max error
        j = 0
        for i in range(N+1):
            if i != n1_loc:
                verts1[:,j] = verts[:,i]
                errs1[j] = errs[i]
                j += 1

        # Get index of second highest error
        sec_loc = np.argmax(errs1)

        # Calculate new centroid
        x_o = cent_calc(verts1)
        my_path.append(x_o)

        # Reflection
        x_r = x_o + alpha*(x_o - x_n1)
        err_r = the_fun(x_r,fun_args)
        if err_r < errs1[sec_loc] and err_r > errs[one_loc]:
            if verb > 1:
                print 'Reflection'
            verts[:,n1_loc] = 1.0*x_r
            errs[n1_loc] = 1.0*err_r

        # Expansion
        elif err_r < errs[one_loc]:
            if verb > 1:
                print 'Expansion'
            x_e = x_o + gamma*(x_o - x_n1)
            err_e = the_fun(x_e,fun_args)
            if err_e < err_r:
                verts[:,n1_loc] = 1.0*x_e
                errs[n1_loc] = 1.0*err_e
            else:
                verts[:,n1_loc] = 1.0*x_r
                errs[n1_loc] = 1.0*err_r

        else:
            # Contraction
            x_c = x_o + rho*(x_o - x_n1)
            err_c = the_fun(x_c,fun_args)
            if err_c < errs[n1_loc]:
                verts[:,n1_loc] = 1.0*x_c
                errs[n1_loc] = 1.0*err_c
                if verb > 1:
                    print 'Contraction'
            else:
                # Reduction
                if verb > 1:
                    print 'Reduction'
                for ll in range(N+1):
                    if ll != one_loc:
                        verts[:,ll] = x_one + sigma*(verts[:,ll] - x_one)
                        errs[ll] = the_fun(verts[:,ll],fun_args)

        # Increment iteration counter
        i_cur += 1

    # Get the best vertex
    one_loc = np.argmin(errs)
    x_one = verts[:,one_loc]
    min_err = np.min(errs)
    if verb > 0:
        print '\nComplete\n# of iterations:',i_cur
        print 'Error at best point:',min_err
        print 'Best Point:',x_one
    return x_one,min_err,i_cur,my_path
    
if __name__ == '__main__':
    ''' Testing simplex method
    '''
    cnt = 0
    def quad(x_o,(a,b)):
        global cnt
        cnt += 1
        return 1. + a*(x_o[0]+5)**2 + b*(x_o[1]-7)**2
    a = 1
    b = 1
    x_o = np.zeros(2)#+500
    x_one,min_err,i_cur,my_path = dhsimp(x_o,quad,(a,b),s_scale=10,verb=1)
    print 'Number of Model Calls:',cnt
