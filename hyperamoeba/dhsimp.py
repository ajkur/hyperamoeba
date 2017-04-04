from __future__ import division
import numpy as np 

# Calculate centroid
def centCalc(xVec):
    xC = np.zeros(xVec.shape[0])
    for i in range(xVec.shape[0]):
        for j in range(xVec.shape[1]):
            xC[i] = xC[i] + xVec[i,j]
    return xC/xVec.shape[1]

def boundWrap(var_arr,(og_wrap,obs_data,b_list),big_num=1e14):
    '''Call boundTrans to translate to user supplied bounds
       Return high error if out of bounds, otherwise return L2 norm
    '''
    var_trans = boundTrans(var_arr,b_list)
    bound_flag = False
    for i in range(var_trans.shape[0]):
        if var_trans[i] < b_list[i][0] or var_trans[i] > b_list[i][1]:
            bound_flag = True
            break
    if bound_flag:
        return big_num
    else:
        return np.sum((og_wrap(var_trans)-obs_data)**2)**0.5

def boundTrans(var_arr,b_list):
    '''Translate point from [0,1] to user supplied bounds
    '''
    var_trans = np.zeros(var_arr.shape[0])
    for i in range(var_arr.shape[0]):
        var_trans[i] = var_arr[i]*(b_list[i][1] - b_list[i][0]) + b_list[i][0]
    return var_trans

# Downhill Simplex
def dhSimp(x0,theFun,funArgs,alpha=1.0,gamma=2.0,rho=-0.5,sigma=0.5,tol=1e-8,sScale=1,prOut=True):
    if prOut == True:
        print '\nStarting Downhill Simplex'
        print 'Sum of\tStd dev of\tIteration'
        print 'Errors\tErrors'
    # Store path as a list
    myPath = []
    myPath.append(x0)

    # Get dimensions
    N = x0.shape[0]

    # Distance of each point form centroid
    dist_arr = np.zeros(N)

    # Set vertices of a regular simplex
    verts = np.zeros([N,N+1])
    verts1 = np.zeros([N,N])
    errs = np.arange(N+1,dtype=float)
    errs1 = np.zeros(N)
    verts[0,0] = 1.0
    for i in range(N):
        verts[0,i+1] = -1.0/N
    for k in range(1,N):
        vTemp = 0.0
        for j in range(0,k):
            vTemp = vTemp + verts[j,k]**2
        verts[k,k] = (1.0-vTemp)**0.5
        for i in range(k+1,N+1):
            vTemp = 0.0
            for l in range(0,k):
                vTemp = vTemp + verts[l,k]*verts[l,i]
            verts[k,i] = (-1.0/N - vTemp)/verts[k,k]

    # Scale and shift simplex to specified starting centroid 
    for i in range(N):
        for j in range(N+1):
            verts[i,j] = verts[i,j]*sScale + x0[i]

    # Calculate erros at each vertex
    for i in range(N+1):
        errs[i] = theFun(verts[:,i],funArgs)

    # print verts,np.exp(verts)
    iCur = 0
    while np.sum(errs) > tol and np.std(errs) > tol and iCur < 1000:
        N1Loc = np.argmax(errs)
        OneLoc = np.argmin(errs)
        xN1 = verts[:,N1Loc]
        xOne = verts[:,OneLoc]
        if iCur%2 == 0 and prOut == True:
            print str(np.sum(errs))+'\t'+str(np.std(errs))+'\t'+str(iCur)
        # Cut out max error
        j = 0
        for i in range(N+1):
            if i != N1Loc:
                verts1[:,j] = verts[:,i]
                errs1[j] = errs[i]
                j += 1

        secLoc = np.argmax(errs1)
        xN = verts1[:,secLoc]

        # Calculate new centroid
        x0 = centCalc(verts1)
        myPath.append(x0)

        # Reflection
        xR = x0 + alpha*(x0 - xN1)
        errR = theFun(xR,funArgs)
        if errR < errs1[secLoc] and errR > errs[OneLoc]:
            # print 'Reflection'
            verts[:,N1Loc] = 1.0*xR
            errs[N1Loc] = 1.0*errR

        # Expansion
        elif errR < errs[OneLoc]:
            # print 'Expansion'
            xE = x0 + gamma*(x0 - xN1)
            errE = theFun(xE,funArgs)
            if errE < errR:
                verts[:,N1Loc] = 1.0*xE
                errs[N1Loc] = 1.0*errE
            else:
                verts[:,N1Loc] = 1.0*xR
                errs[N1Loc] = 1.0*errR

        # Contraction
        else:
            xC = x0 + rho*(x0 - xN1)
            errC = theFun(xC,funArgs)
            if errC < errs[N1Loc]:
                verts[:,N1Loc] = 1.0*xC
                errs[N1Loc] = 1.0*errC
                # print 'Contraction'
            else:
                # Reduction
                # print 'Reduction'
                for ll in range(N+1):
                    if ll != OneLoc:
                        verts[:,ll] = xOne + sigma*(verts[:,ll] - xOne)
                        errs[ll] = theFun(verts[:,ll],funArgs)

        # Calculate centroid and check spread of points
        x0 = centCalc(verts)
        for i in range(N):
            dist_arr[i] = np.sum((x0 - verts[:,i])**2)**0.5
        if np.max(dist_arr) < tol:
            break

        iCur += 1

    # Get the best vertex
    OneLoc = np.argmin(errs)
    xOne = verts[:,OneLoc]
    if prOut == True:
	print verts
        print '\nComplete\n# of iterations:',iCur
        print 'Error at best point:',np.min(errs)
        print 'Best Point:',xOne
    return iCur,myPath,xOne

if __name__ == '__main__':
    cnt = 0
    def quad(x0,(a,b)):
        global cnt
        cnt += 1
        return 1. + a*(x0[0]+5)**2 + b*(x0[1]-7)**2

    a = 1
    b = 1
    x0 = np.zeros(2)#+500
    iCur,myPath,x0 = dhSimp(x0,quad,(a,b),sScale=10,prOut=False)
    print 'Number of Model Calls:',cnt
    print 'Best Point:',x0
