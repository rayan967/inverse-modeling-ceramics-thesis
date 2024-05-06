import numpy as np

from basicfunctions.utils.creategrid import *
from basicfunctions.covariance.cov import *
from basicfunctions.derivative.dGPR import *
from basicfunctions.utils.creategrid import *

from hyperparameter.hyperparameteroptimization import *
from hyperparameter.utils.setstartvalues import *

from optimization.utils.computationalwork import *

from reconstruction.utils.perror import *

from scipy.linalg import block_diag

' Lossfunction and its derivate '
def loss(w0, idx, x,
         Xtextended, Xgradextended, ytextended, ygradextended,
         epsXtextended, epsXgradextended,
         HPm , m, dim,
         epsphys, gflag):

    """ Calcualte eps(W) """
    current_eps = epsofw(w0, dim)

    """ Set new eps-training data """
    epsXtextended[0,idx] = current_eps**2

    """ Calculate dp with new eps at x """
    if gflag == 0:
        matrices = kernelmatrices(x, Xtextended, HPm[0, :], epsXtextended)
        metaerror = matrices[0] - (matrices[1]@np.linalg.inv(matrices[2])@matrices[1].T)

        L = HPm[0, 1:]
        alpha = np.linalg.inv(matrices[2]) @ ytextended
        df = dGPR(x, Xtextended, matrices[1], L)@alpha

    elif gflag == 1:
        covmatrices = kernelmatricesgrad(x, Xtextended, Xgradextended, HPm[0, :], epsXtextended, epsXgradextended)
        Kxx,KXXt,KXXg,K = covmatrices[0] ,covmatrices[1],covmatrices[3],covmatrices[5]
        invK = np.linalg.inv(K)
        KXXtXXg = np.concatenate((KXXt,KXXg),axis = 1)
        metaerror = Kxx - KXXtXXg@invK@KXXtXXg.T

        sigma, L = HPm[0,0], HPm[0, 1:]
        ytilde = np.concatenate((ytextended,ygradextended), axis=0 )
        alpha = invK @ ytilde
        df = dGPRgrad(x,Xtextended,Xgradextended,sigma,L)@alpha

    dp = parametererror(df,metaerror,epsphys,m,dim)

    return np.linalg.norm(dp,2)**2

def losswithoutgflag(w0, idx, x,
         Xtextended, Xgradextended, ytextended, ygradextended,
         epsXtextended, epsXgradextended,
         HPm , m, dim,
         epsphys):

    """ Calcualte eps(W) """
    current_eps = epsofw(w0, dim)

    """ Set new eps-training data """
    epsXtextended[0,idx] = current_eps**2

    """ Calculate dp with new eps at x """
    if Xgradextended is None:
        matrices = kernelmatrices(x, Xtextended, HPm[0, :], epsXtextended)
        metaerror = matrices[0] - (matrices[1]@np.linalg.inv(matrices[2])@matrices[1].T)

        L = HPm[0, 1:]
        alpha = np.linalg.inv(matrices[2]) @ ytextended
        df = dGPR(x, Xtextended, matrices[1], L)@alpha

    else:
        covmatrices = kernelmatricesgrad(x, Xtextended, Xgradextended, HPm[0, :], epsXtextended, epsXgradextended)
        Kxx,KXXt,KXXg,K = covmatrices[0] ,covmatrices[1],covmatrices[3],covmatrices[5]
        invK = np.linalg.inv(K)
        KXXtXXg = np.concatenate((KXXt,KXXg),axis = 1)
        metaerror = Kxx - KXXtXXg@invK@KXXtXXg.T

        sigma, L = HPm[0,0], HPm[0, 1:]
        ytilde = np.concatenate((ytextended,ygradextended), axis=0 )
        alpha = invK @ ytilde
        df = dGPRgrad(x,Xtextended,Xgradextended,sigma,L)@alpha

    dp = parametererror(df,metaerror,epsphys,m,dim)

    return np.linalg.norm(dp,2)**2

def dlosswithoutgflag(w0, idx, x, Xtextended,Xgradextended, ytextended, ygradextended, epsXtextended, epsXgradextended, HPm , m, dim, epsphys):
    ' Key numbers '
    n = Xtextended.shape[0]    

    ' Calcualte eps(W) '
    current_eps = epsofw(w0, dim) # Float

    ' Set new eps-training data '
    epsXtextended[0,idx] = current_eps**2 #Shape [[1,n]]

    ' Calcualte dp with new eps at x '
    if Xgradextended is None:
        
        ' All necessary kernel matrices '
        matrices = kernelmatrices(x, Xtextended, HPm[0, :], epsXtextended)
        Kxx,KxX,KXX = matrices[0], matrices[1], matrices[2]
        invKXX = np.linalg.inv(KXX)

        ' Variance at x '
        varxs = Kxx - (KxX@invKXX@KxX.T) #Shape [[1,1]]
        
        ' df at x '
        L = HPm[0, 1:]
        alpha = invKXX @ ytextended
        dKxX = dGPR(x, Xtextended, KxX, L)
        df = dKxX@alpha #Shape [[dim,1]]
      
        ' Derivative dE / deps'
        dEdepsi = np.zeros((n, n)) #[[n,n]]
        epsj = epsXtextended[0,idx] #Float
        dEdepsi[idx, idx] = 2*np.sqrt(epsj)

        ' Derivative df/deps '
        dfdepsj= -dKxX @ (invKXX @ dEdepsi @ invKXX) @ ytextended  #Shape [[dim,1]]
                           
        ' Derivative dsigma/deps'
        dvardepsj = KxX @ invKXX @ dEdepsi @ invKXX @ KxX.T #ddvardepsj mit Mathematica gecheckt. Ist immer R , Shape [[1,1]]
    
    if Xgradextended is not None:

        ngrad = Xgradextended.shape[0]*dim

        covmatrices = kernelmatricesgrad(x, Xtextended, Xgradextended, HPm[0, :], epsXtextended, epsXgradextended)
        Kxx,KXXt,KXXg,K = covmatrices[0], covmatrices[1], covmatrices[3], covmatrices[5]
        invKXX = np.linalg.inv(K)
        KXXtXXg = np.concatenate((KXXt,KXXg),axis = 1)

        ' Variance at x '
        varxs = Kxx - KXXtXXg@invKXX@KXXtXXg.T

        ' df at x '
        sigma, L = HPm[0, 0], HPm[0, 1:]
        ytilde = np.concatenate((ytextended, ygradextended),axis = 0)
        alpha = np.linalg.solve(K,ytilde)
        dKxX = dGPRgrad(x, Xtextended, Xgradextended, sigma, L)
        df = dKxX@alpha

        ' Derivative dE / deps'
        dEdepsi = np.zeros((n+ngrad, n+ngrad))
        epsj = epsXtextended[0,idx]
        dEdepsi[idx, idx] = 2*np.sqrt(epsj)

        ' Derivative df/deps '
        dfdepsj= -dKxX @ (invKXX @ dEdepsi @ invKXX) @ ytilde  #Shape [[dim,1]]

        ' Derivative dsigma/deps'
        dvardepsj = KXXtXXg @ invKXX @ dEdepsi @ invKXX @ KXXtXXg.T #ddvardepsj[i,:]  mit Mathematica gecheckt

    ' Derivative d(f/<f,f>)/deps'
    dffdeps = ((df*df)*dfdepsj - df*(2*(dfdepsj*df))) / (df*df)**2 #Ist immer R^dim , Shape [[dim,1]]

    ddpdepsj = -dffdeps * varxs - df/(np.dot(df.T,df))*dvardepsj #Ist immer R^dim, Shape [[dim,1]] 
           
    dp = parametererror(df, varxs, 0.0, m, dim) #Ist immer R^dim, Shape [[dim,1]] 
   
    ddpdeps = 2*np.dot((dp).T, ddpdepsj) #Ist immer R, , Shape [[1,1]] 

    ' deps / dw '
    depsdW = deps(w0,dim)

    ' Compose everything'
    dndpdW = ddpdeps * depsdW #Ist immer R, , Shape [[1,1]] 

    DEBUG = False
    if DEBUG:
        print("Opt. var: "+str(varxs))
        print("Opt. df: "+str(df))
        print("Opt. dfdepsj: "+str(dfdepsj))
        print("Opt. dvardepsj: "+str( dvardepsj))
        print("Opt. ddpdepsj: "+ str(ddpdepsj))
        print("Opt. dp: "+str(dp))
        print("Opt. ddpdeps: "+str(ddpdeps))
        print("Optimized version: "+str(dndpdW))
        print("\n")
        
    ' return '
    return dndpdW[0,0]


def lossgrad(w0con, idx, x,
            Xtextended, Xgradextended, ytextended, ygradextended,
            epsXtextended, epsXgradextended,
            HPm, m, dim,
            epsphys, gflag):

    'Split w0 into work of eps and epsgrad'
    w0 = w0con[0]
    w0grad = w0con[1:]

    'Calcualte the accuracy eps for the data points given the work w'
    current_eps = epsofw(w0, dim) #e(W)

    ' Set new eps-training data '
    epsXtextended[0,idx] = current_eps**2

    'Calcualte the accuracy eps for the gradient data points given the work w'
    current_eps_grad = epsofwgrad(w0grad, dim)

    ' Set new eps-gradient data '
    epsXgradextended[0,idx*dim:idx*dim+dim] = current_eps_grad**2

    covmatrices = kernelmatricesgrad(x, Xtextended, Xgradextended, HPm[0, :], epsXtextended, epsXgradextended)
    Kxx,KXXt,KXXg,K = covmatrices[0] ,covmatrices[1],covmatrices[3],covmatrices[5]
    invK = np.linalg.inv(K)
    KXXtXXg = np.concatenate((KXXt,KXXg),axis = 1)
    varianceatx = Kxx - KXXtXXg@invK@KXXtXXg.T

    sigma, L = HPm[0,0], HPm[0, 1:]
    ytilde = np.concatenate((ytextended,ygradextended), axis=0 )
    df = dGPRgrad(x,Xtextended,Xgradextended,sigma,L)@ invK @ ytilde

    dp = parametererror(df,varianceatx,epsphys,m,dim)

    return np.linalg.norm(dp,2)**2

def dlossgrad(w0con, idx, x, Xtextended, Xgradextended,
              ytextended, ygradextended, epsXtextended,
              epsXgradextended, HPm , m, dim, epsphys):

    'Split w0 into work of eps and epsgrad'
    w0 = w0con[0]
    w0grad = w0con[1:]

    'Calcualte the accuracy eps for the data points given the work w'
    current_eps = epsofw(w0, dim) #e(W)

    ' Set new eps-training data '
    epsXtextended[0,idx] = current_eps**2

    'Calcualte the accuracy eps for the gradient data points given the work w'
    current_eps_grad = epsofwgrad(w0grad, dim)

    ' Set new eps-gradient data '
    epsXgradextended[0,idx*dim:idx*dim+dim] = current_eps_grad**2

    ' Calculate all needed covariance matrices'
    covmatrices = kernelmatricesgrad(x, Xtextended, Xgradextended, HPm[0, :], epsXtextended, epsXgradextended)
    Kxx,KXXt,KXXg,K = covmatrices[0],covmatrices[1],covmatrices[3],covmatrices[5]
    invK = np.linalg.inv(K)
    KXXtXXg = np.concatenate((KXXt,KXXg),axis = 1) #Concatenate covariance matrices between x/Xt and x/Xgrad - data points
    ytilde = np.concatenate((ytextended, ygradextended),axis = 0) #concatenate function and gradient values

    ' Calculate the surrogat error with gradient data and adapted accuracies'
    varxs = Kxx - KXXtXXg@invK@KXXtXXg.T

    ' Calcualte dp with new eps and epsgrad at x '
    sigma, L = HPm[0, 0], HPm[0, 1:]
    dkdx = dGPRgrad(x,Xtextended, Xgradextended, sigma, L)
    alpha = invK @ ytilde
    df = dkdx@alpha

    """ Calculate dpdw with new eps at x.
    Since now there are gradient information the derivative becomes a vector of R^(1+dim) """

    """
    Derivative of dp with respect to epsXtextended[idx] , d(dp)/deps with gradient data in the covariance structure
    Due to the strcuture only Epsilon has to be adjusted such that we take a block approach and concatenate along the diagonal
    """

    dfdepsj = np.zeros((m, dim))

    sizeupperblock = Xtextended.shape[0]
    sizelowerblock = Xgradextended.shape[0]*dim

    dEdepsi = np.zeros((sizeupperblock+sizelowerblock,sizeupperblock+sizelowerblock,1+dim))

    """ Use a tensor structure for dEdepsi since the overall structure doesn't change.
    """
    jac = np.zeros((dim+1))

    for ii in range(1+dim):
        dEdepsupperblock = np.zeros((sizeupperblock, sizeupperblock))

        # Derivative with respect to eps
        if ii == 0:
            ' deps / dw '
            w = w0
            depsdW = deps(w,dim)

            dEdepslowerblock = np.zeros((sizelowerblock, sizelowerblock))
            epsatidx = epsXtextended[0,idx]
            dEdepsupperblock[idx,idx] = 2*np.sqrt(epsatidx)
            dEdepsi[:,:,0] = block_diag(dEdepsupperblock,dEdepslowerblock)

        # Derivative with respect to epsgrad_1 , ..., epsgrad_d
        else:
            w = w0grad[ii-1]
            depsdW = depsgrad(w, dim)

            epsatidx = epsXgradextended[0,idx*dim:idx*dim+dim] # Subvector
            eps = epsatidx[ii-1] # Get eps at subvector
            tmp = np.zeros((1,dim)) # Create zero vector
            tmp[0,ii-1] = 2*np.sqrt(eps)

            subdiag = np.zeros((1,sizelowerblock))
            subdiag[0,idx*dim:idx*dim+dim] = tmp
            dEdepslowerblock = np.diagflat(subdiag)
            dEdepsi[:,:,ii] = block_diag(dEdepsupperblock,dEdepslowerblock)

        ' Derivative df/deps '
        dfdepsj= -dkdx @ (invK @ dEdepsi[:,:,ii] @ invK) @ ytilde  #Shape [[dim,1]]

        ' Derivative dsigma/deps'
        dvardepsj = KXXtXXg @ invK @ dEdepsi[:,:,ii] @ invK @ KXXtXXg.T #ddvardepsj[i,:]  mit Mathematica gecheckt

        ' Derivative d(f/<f,f>)/deps'
        dffdeps = ((df*df)*dfdepsj - df*(2*(dfdepsj*df))) / (df*df)**2 #Ist immer R^dim , Shape [[dim,1]]

        ddpdepsj = -dffdeps * varxs - df/(np.dot(df.T,df))*dvardepsj #Ist immer R^dim, Shape [[dim,1]] 
               
        dp = parametererror(df, varxs, 0.0, m, dim) #Ist immer R^dim, Shape [[dim,1]] 
       
        ddpdeps = 2*np.dot((dp).T, ddpdepsj) #Ist immer R, Shape [[1,1]] 

        ' deps / dw '
        depsdW = deps(w0,dim)

        ' Compose everything'
        dpdW = ddpdeps * depsdW #Ist immer R, , Shape [[1,1]] 

        jac[ii] = dpdW


        DEBUG = False
        if DEBUG:
            print("Opt. var: "+str(varxs))
            print("Opt. df: "+str(df))
            print("Opt. dfdepsj: "+str(dfdepsj))
            print("Opt. dvardepsj: "+str( dvardepsj))
            print("Opt. ddpdepsj: "+ str(ddpdepsj))
            print("Opt. dp: "+str(dp))
            print("Opt. ddpdeps: "+str(ddpdeps))
            print("Optimized version: "+str(dpdW))
            print("\n")
    
    return jac





























































# Lossfunction and its derivate '
def lossmultibleexp(w0, idx, x, Xtextended, ytextended, epsXtextended, HPm , m, dim, epsphys):
    """ Calcualte eps(W) """
    current_eps = epsofw(w0, dim)

    """ Set new eps-training data """
    epsXtextended[:,idx] = current_eps**2

    ' Zeros '
    df = np.zeros((m, dim))
    metaerror = np.zeros((m,1))

    """ Calculate dp with new eps at x """
    for i in range(m):

        matrices = kernelmatrices(x, Xtextended, HPm[i, :], epsXtextended[i,:])
        metaerror[i,:] = matrices[0] - (matrices[1]@np.linalg.inv(matrices[2])@matrices[1].T)

        L = HPm[i, 1:]
        alpha = np.linalg.inv(matrices[2]) @ ytextended[:,i]
        df[i,:] = dGPR(x, Xtextended, matrices[1], L)@alpha

    #SigmaInv = np.diagflat(1/metaerror)

    dp = parametererror(df,metaerror,epsphys,m,dim)

# =============================================================================
#     if dim != 1:
#         A = metaerror * 1/(np.dot(df.T,df))
#         B = df.T*SigmaInv*metaerror.T
#         dp = -A*B
#
#         return np.linalg.norm(dp,2)**2
#     else:
#         A = np.linalg.inv(df.T@SigmaInv@df)
#         B = df.T@SigmaInv@metaerror.T
#         dp = -A@B
# =============================================================================

    return np.linalg.norm(dp,2)**2

def gradlossmultibleexp(w0, idx, x, Xtextended, ytextended, epsXtextended, HPm , m, dim, epsphys):

    jac = np.zeros((m))

    """ First step: calculate dp with the given experiments """
    df = np.zeros((m,dim))
    metaerror = np.zeros((m,1))

    """ Calcualte eps(W) """
    current_eps = epsofw(w0, dim)

    """ Set new eps-training data """
    epsXtextended[:,idx] = current_eps**2

    for ii in range(m):

        """ Calcualte dp with new eps at x """
        matrices = kernelmatrices(x, Xtextended, HPm[ii, :], epsXtextended[ii,:])
        metaerror[ii,:] = matrices[0] - (matrices[1]@np.linalg.inv(matrices[2])@matrices[1].T)

        L = HPm[ii, 1:]
        alpha = np.linalg.inv(matrices[2]) @ ytextended[:,ii].T
        df[ii,:] = dGPR(x, Xtextended, matrices[1], L)@alpha

    SigmaInv = np.diagflat(1/(metaerror+epsphys) )

    reg = 1E-6*np.eye((dim))
    A = np.linalg.inv(df.T@SigmaInv@df+reg)
    B = df.T@SigmaInv@metaerror
    dp = -A@B

    """ Second step: calculate dpdw with new eps at x """

    for jj in range(m):

        'Matrix for derivative of f in respect to eps'
        dfdepsj = np.zeros((m, dim))
        'Matrix for derivative of E in respect to eps'
        dEdepsi = np.zeros((Xtextended.shape[0], Xtextended.shape[0]))

        'Epsilon of every data point for surrogat model 1,...,m'
        eps = epsXtextended[jj,:]
        'Epsilon at data point idx'
        epsj = epsXtextended[jj,idx]
        'Derivative of Epsilon in respect to epsilon at given index idx'
        dEdepsi[idx, idx] = 2*np.sqrt(epsj)

        ' Covariance matrices for further usage '
        matricesdf = kernelmatrices(x, Xtextended, HPm[jj, :], eps)
        L = HPm[jj, 1:]
        invK = np.linalg.inv(matricesdf[2])

        ' Derivative of f in respect to eps at x'
        # (invK @ dEdepsi @ invK)mit Mathematica getestet
        #dfdepsj[i,:]  mit Mathematica gecheckt
        dfdepsj[jj, :] = (-dGPR(x, Xtextended, matricesdf[1],L) @ (invK @ dEdepsi @ invK) @ ytextended[:,ii].T).reshape((1,-1))

        ' Derivative of variance in respect to eps'
        dvardepsj = np.zeros((m, 1))
        dvardepsj[jj, :] = matricesdf[1] @ invK @ dEdepsi @ invK @ matricesdf[1].T
        #ddvardepsj[i,:]  mit Mathematica gecheckt

        'Derivative of the inverse of sigma in respect to eps, where sigma is a diagonal matrix with the variance on the diagonal'
        dSigmaInvdepsi = np.zeros((m, m))
        dSigmaInvdepsi[jj, jj] = (matricesdf[1]@invK @ dEdepsi @ invK@matricesdf[1].T)[0, 0]

        if dim != 1:
            dAdepsj = -A@dfdepsj.T@(SigmaInv@df) - df.T@(SigmaInv@dSigmaInvdepsi@SigmaInv@df) + df.T @ (SigmaInv@dfdepsj)@A
            dBdepsj = dfdepsj.T@SigmaInv@(metaerror+epsphys) - df.T@(SigmaInv@dSigmaInvdepsi@SigmaInv@(metaerror+epsphys)) + df.T@SigmaInv@dvardepsj
            ddpdepsj = -((dAdepsj@B) + (A@dBdepsj))
            dndpdeps = 2*np.dot((dp.T), ddpdepsj)

        else:

            dAdepsj = -A@(dfdepsj.T@SigmaInv@df - df.T@SigmaInv @ dSigmaInvdepsi@SigmaInv @ df + df.T @ SigmaInv@dfdepsj)@A
            dBdepsj = dfdepsj.T@SigmaInv@(metaerror+epsphys) - df.T@SigmaInv@dSigmaInvdepsi@SigmaInv@(metaerror+epsphys) + df.T@SigmaInv@dvardepsj
            ddpdepsj = -((dAdepsj@B) + (A@dBdepsj))
            dndpdeps = 2*np.dot((dp).T, ddpdepsj)

        """ Thrid step: calculate depsdw for every m """
        ' deps / dw '
        w = W(epsj, dim)
        depsdW = deps(w,dim)

        ' Compose everything'
        dndpdW = dndpdeps * depsdW

        """ Fourth step: concatenate everything """
        jac[jj] = dndpdW

    return jac


