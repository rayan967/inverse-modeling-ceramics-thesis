import numpy as np
from scipy.linalg import solve_triangular

def logmarginallikelihood(L, Xt, Xgrad, yt, ygrad, eps, epsgrad, gflag):
    """
    Parameters
    ----------
    H : list 1 x (1+d)
        Hyperparameter values used in the minimization algorithm
    Xt : np.array n x d
        Training data
    Xgrad : np.array n x d
        Gradient data.
    yt : np.array 1 x d
        Values at Xt
    ytg : np.array n x d
        Values at Xgrad.
    eps : TYPE
        DESCRIPTION.
    epsgrad : TYPE
        DESCRIPTION.
    gflag : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    Log of marginal likelihood

    log (p(y|X,H)) := 1/2 * y^T K^(-1) y - 1/2 log(|K|) - N/2 log(2 pi)

    where K is the Kovariancematrix with or without gradient information0

    N = N without gradient info
    N = N + N*d  with gradient info

    K = | KXX  KXXG  |
        | KXXG KXGXG |


    H = (sigma,L1, .... , Ld) without eps since eps is known
    """

    dim = Xt.shape[1]

    """ Without gradient information """
    if gflag == False:

        """ Build K """
        N2      = Xt.shape[0]
        epsilon = np.diagflat(eps)

        """ Prescale data """
        Xtscaled = Xt/L
        n2sq     = np.sum(Xtscaled**2,axis=1)
        DYY      = np.transpose(np.outer(np.ones(N2),n2sq)) + np.outer(np.ones(N2),n2sq) - 2*(np.dot(Xtscaled,np.transpose(Xtscaled)))
        DYY[np.abs(DYY) < 1E-5] = 0.0
        
        K        = np.exp(-DYY / (2.0))
        K        = K + epsilon
        
        "Regularisation"
        reg      = np.eye(N2)*1E-4
        K        = K + reg 

        # Numerically more stable implementation as described
        # in http://www.gaussianprocess.org/gpml/chapters/RW2.pdf, Section
        # 2.2, Algorithm 2.1
       
        PD = False
        while PD == False:
            PD = True
            try:
                LU = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                PD = False
                #LU = np.linalg.cholesky(K)
                LU = LU + np.eye(2) * 0.0001

        S1 = solve_triangular(LU  , yt, lower=True)
        S2 = solve_triangular(LU.T, S1, lower=False)

        logp = np.sum(np.log(np.diagonal(LU))) + 0.5 * yt.T@S2 + 0.5*N2*np.log(2*np.pi)

        return logp

    elif gflag == True:

        """ Build K """
        N2 = Xt.shape[0]
        N3 = Xgrad.shape[0]
        D3 = Xgrad.shape[1]

        """ Prescale data for the covariance matrices only """
        Xtscaled    = Xt/L
        Xgradscaled = Xgrad/L

        """ Error matrices """
        epsilon     = np.diagflat(eps)
        epsgradient = np.diagflat(epsgrad)

        n2sq = np.sum(Xtscaled**2,axis=1);
        n3sq = np.sum(Xgradscaled**2,axis=1);

        # Kernel matrix Xt Xt
        DYY = np.transpose(np.outer(np.ones(N2),n2sq)) + np.outer(np.ones(N2),n2sq)-2* (np.dot(Xtscaled,np.transpose(Xtscaled)))
        KYY = np.exp(-DYY / 2.0)
        KYY = KYY + epsilon

        # Kernel matrix Xt Xgrad
        DXtXgrad = np.transpose(np.outer(np.ones(N3),n2sq)) + np.outer(np.ones(N2),n3sq)-2* (np.dot(Xtscaled,np.transpose(Xgradscaled)))
        DXtXgrad[np.abs(DXtXgrad) < 1E-5] = 0.0
        kXtXgrad = np.exp(-DXtXgrad / 2.0)
        KXtXgrad = np.zeros((N2,N3*D3))
        for i in range(N2):
            tmp = (Xt[i,:]-Xgrad)/L**2
            A = kXtXgrad[i,:]
            A = A[:,None] # Cast to coloumn vector
            tmp = np.multiply(tmp,A)
            res = np.reshape(tmp,(1,-1))
            KXtXgrad[i,:] = res

        # Kernel matrix Xgrad Xgrad
        DXgradXgrad = np.transpose(np.outer(np.ones(N3),n3sq)) + np.outer(np.ones(N3),n3sq)-2* (np.dot(Xgradscaled,np.transpose(Xgradscaled)))
        DXgradXgrad[np.abs(DXgradXgrad) < 1E-5] = 0.0
        KXgXg       = np.exp(-DXgradXgrad / 2.0)
        # Second derivative
        #Kfdy   = np.zeros((N3*D3,N3*D3));
        tmprow = np.array([])
        Kfdy = np.array([])
        for i in range(N3):
            xi = Xgrad[i,:]
            for j in range(N3):
                xj = Xgrad[j,:]
                diff = np.outer(((xi-xj)/(L**2)),((xi-xj)/(L**2)))
                tmp = KXgXg[i,j]*( -diff + np.diag(1/L**2))
                if j == 0:
                    tmprow = tmp
                else:
                    tmprow = np.concatenate((tmprow,tmp),axis=1);
            if i == 0:
                Kfdy = tmprow
            else:
                Kfdy = np.concatenate((Kfdy,tmprow),axis=0);

        Kfdy = Kfdy + epsgradient

        # Concatenate matrices
        K = np.concatenate((KYY,KXtXgrad),axis =1)
        K = np.concatenate((K,np.concatenate((np.transpose(KXtXgrad),Kfdy),axis =1)) ,axis =0)

        """ concatenate y and y grad """
        ytilde = np.concatenate((yt.reshape(1,-1),ygrad.reshape(1,-1)),axis = 1)

        """ Return value """
        N = N2+N3*D3

        """ Regularisation """
        gamma   = 1E-4
        reg     = gamma * np.eye(N)

        K = K+reg 

        #alpha = np.linalg.solve(K+reg,ytilde.T)
        #logp = 1/2*ytilde @ alpha + 1/2*np.log(np.linalg.det(K)) + N/2*np.log(2*np.pi)
        
        # Numerically more stable implementation as described
        # in http://www.gaussianprocess.org/gpml/chapters/RW2.pdf, Section
        # 2.2, Algorithm 2.1
       
        ytilde = np.squeeze(ytilde)
        PD = False
        while PD == False:
            PD = True
            try:
                LU = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                PD = False
                LU = LU + np.eye(dim) * 0.0001

        S1 = solve_triangular(LU, ytilde.T, lower=True)
        S2 = solve_triangular(LU.T, S1, lower=False)
        
        logp = np.sum(np.log(np.diagonal(LU))) + 0.5 * ytilde.T@S2 + 0.5*N2*np.log(2*np.pi)

        """ log marginal likelihood """
        return np.squeeze(logp)

def logmarginallikelihood_der(L,Xt,Xgrad,yt,ytg,eps,epsgrad,gflag):

    if gflag == False:
        # Numerically more stable implementation of the derivative
        # Rasmussen Section 5.4.1, Eqation 5.9

        N2 = Xt.shape[0]
        D2 = Xt.shape[1]

        ytilde = yt

        epsilon = np.diagflat(eps)

        """ Prescale data """
        Xtscaled = Xt/L
        n2sq     = np.sum(Xtscaled**2,axis=1)

        """ Covariance matrix K(X,X) """
        DYY = np.transpose(np.outer(np.ones(N2),n2sq)) + np.outer(np.ones(N2),n2sq) - 2 * (np.dot(Xtscaled,np.transpose(Xtscaled)))
        DYY[np.abs(DYY) < 1E-5] = 0.0
        K   =  np.exp(-DYY / 2.0) + epsilon

        """ Calculate alpha, dK/dsigma """
        invK     = np.linalg.inv(K)
        alpha    = invK@yt
        #dKdsigma = 2*sigma * np.exp(-DYY / 2.0)
               
        """ dp / dsigma """
        #dpdsigma = -0.5*np.trace((np.outer(alpha,alpha) - invK)@dKdsigma)

        """ dp / dL, dK/dL """
        dtmpdL  = np.zeros((N2,N2,D2))
        dpdL    = np.zeros((D2))
        # First derivates
        for j in range(0,N2):

            tmp = np.transpose((Xt[j,:]-Xt)**2/L**3)
            for i in range(0,D2):
                dtmpdL[j,:,i] = tmp[i,:]

        for ii in range(0,D2):
            dKdL     = K*dtmpdL[:,:,ii]
            dpdL[ii] = -0.5*np.trace((np.outer(alpha,alpha) - invK)@dKdL)
        """  concatenate """
        jacobi =  dpdL.reshape(D2,-1)

        return np.squeeze(jacobi)

    if gflag == True:

        """ HP from H """
        sigma   = 1

        """ Build K """
        N2 = Xt.shape[0]
        D2 = Xt.shape[1]
        N3 = Xgrad.shape[0]
        D3 = Xgrad.shape[1]
        N = N2+N3*D3

        """ Prescale data for the covaraince matrices only """
        Xtscaled = Xt/L
        Xgradscaled = Xgrad/L

        """ Error matrices """
        epsilon = np.diagflat(eps)
        epsgradient = np.diagflat(epsgrad)

        n2sq = np.sum(Xtscaled**2, axis=1)
        n3sq = np.sum(Xgradscaled**2, axis=1)

        ytilde = np.concatenate((yt.reshape(1, -1), ytg.reshape(1, -1)), axis=1)

        for jj in range(2, 1, -1):

            # Kernel matrix Xt Xt
            DYY = np.transpose(np.outer(np.ones(N2), n2sq)) + np.outer(np.ones(N2),
                                                                       n2sq)-2 * (np.dot(Xtscaled, np.transpose(Xtscaled)))
            KYY = sigma**jj * np.exp(-DYY / 2.0)

            # Kernel matrix Xt Xgrad
            # Checked with Mathematica for 1 and 2D , 21.5.2021
            DXtXgrad = np.transpose(np.outer(np.ones(N3), n2sq)) + np.outer(
                np.ones(N2), n3sq)-2 * (np.dot(Xtscaled, np.transpose(Xgradscaled)))
            kXtXgrad = sigma**jj * np.exp(-DXtXgrad / 2.0)
            KXtXgrad = np.zeros((N2, N3*D3))
            for i in range(0, N2):
                tmp = (Xt[i, :]-Xgrad)/L**2
                A = kXtXgrad[i, :]
                A = A[:, None]  # Cast to column vector
                tmp = np.multiply(tmp, A)
                res = np.reshape(tmp, (1, -1))
                KXtXgrad[i, :] = res

            # Kernel matrix Xgrad Xgrad
            # Checked with Mathematica for 1D  and 2D 21.5.2021
            DXgradXgrad = np.transpose(np.outer(np.ones(N3), n3sq)) + np.outer(
                np.ones(N3), n3sq)-2 * (np.dot(Xgradscaled, np.transpose(Xgradscaled)))
            KXgXg = sigma**jj * np.exp(-DXgradXgrad / 2.0)
            # Second derivative
            #Kfdy   = np.zeros((N3*D3,N3*D3));
            tmprow = np.array([])
            Kfdy = np.array([])
            for i in range(0, N3):
                xi = Xgrad[i, :]
                for j in range(0, N3):
                    xj = Xgrad[j, :]
                    diff = np.outer(((xi-xj)/(L**2)), ((xi-xj)/(L**2)))
                    assert np.diag(1/L**2).shape[0] == Xgrad.shape[1]
                    tmp = KXgXg[i, j]*(-diff + np.diag(1/L**2))
                    if j == 0:
                        tmprow = tmp
                    else:
                        tmprow = np.concatenate((tmprow, tmp), axis=1)
                if i == 0:
                    Kfdy = tmprow
                else:
                    Kfdy = np.concatenate((Kfdy, tmprow), axis=0)
            'dKdL'
            if jj == 2:

                KYY = KYY + epsilon
                Kfdy = Kfdy + epsgradient
                # Concatenate matrices
                K = np.concatenate((KYY, KXtXgrad), axis=1)
                K = np.concatenate(
                    (K, np.concatenate((np.transpose(KXtXgrad), Kfdy), axis=1)), axis=0)
                # Regularization
                K = K+1E-6*np.eye(N)
                """ Calculate alpha, dK/dsigma """
                invK = np.linalg.inv(K)
                alpha = invK@ytilde.T

                """ dKXX/dL """
                #Checked with Mathematica 21.5.2021
                dtmpdL = np.zeros((N2, N2, D2))
                dpdL = np.zeros((D2))

                """ dKXXgrad / dL """
                # dKXXgdL
                # Checked with Mathematica for 1D  and 2D 21.5.2021
                row = []
                col = np.array([])
                dKXXgdL = np.zeros((N2, N3*D2))

                """ dKXgXg/dL """
                dtmpXgXgdL = np.zeros((N3, N3, D3))

                'dKXgradXgrad/dL'
                #Checked with Mathematica 21.5.2021
                tmplist = []
                colXgXg = []
                dXgXgdltmp = np.array([])
                dXgXgdL = np.zeros((N3*D3, N3*D3))

                for j in range(0, N2):
                    tmp = np.transpose((Xt[j, :]-Xt)**2/L**3)
                    #t1 = np.transpose((Xgrad[j, :]-Xgrad)**2/L**3)
                    for i in range(0, D2):
                        dtmpdL[j, :, i] = tmp[i, :]
                        #dtmpXgXgdL[j, :, i] = t1[i, :]

                for j in range(0, N3):
                    #tmp = np.transpose((Xt[j, :]-Xt)**2/L**3)
                    t1 = np.transpose((Xgrad[j, :]-Xgrad)**2/L**3)
                    for i in range(0, D3):
                        #dtmpdL[j, :, i] = tmp[i, :]
                        dtmpXgXgdL[j, :, i] = t1[i, :]

                for dim in range(D3):
                    'dKXX/dL'
                    dKdL = KYY*dtmpdL[:, :, dim]

                    'dKXXgrad/dL'
                    currL = L[dim]
                    tv = np.zeros((D2))
                    for i in range(N2):
                        xi = Xt[i, :]
                        xii = xi[dim]  # komponent
                        #print("xi: {}, xii: {}".format(xi, xii))
                        for j in range(N3):

                            xj = Xgrad[j, :]
                            xjj = xj[dim]
                            #print("xj: {}, xjj: {}".format(xj, xjj))
                            tv[dim] = -2*((xii-xjj)/currL**3)
                            # GEÄNDERT
                            tmp = kXtXgrad[i, j] * (((xii-xjj)**2 / currL**3) * (xi-xj)/L**2 + tv)
                            row.append(tmp)

                        col = np.asarray(row)
                        col = col.reshape(1, -1)
                        row = []
                        dKXXgdL[i, :] = col

                    'dKXgradXgrad/dL'
                    tt = np.zeros((D3, D3))
                    currL = L[dim]
                    #dKXgXgdL = KXgXg*dtmpdL[:, :, dim]
                    dKXgXgdL = KXgXg*dtmpXgXgdL[:, :, dim]

                    for i in range(0, N3):
                        xi = Xgrad[i, :]
                        for j in range(0, N3):
                            xj = Xgrad[j, :]

                            diff = np.outer(((xi-xj)/(L**2)), ((xi-xj)/(L**2)))

                            rhs = dKXgXgdL[i, j] * \
                                (-diff + np.diag(1/L**2))  # Rechte Seite

                            #Geprüft mit Mathematica
                            mat = np.zeros((D3, D3))
                            Lcurr = L[dim]
                            tmp = diff[dim, :]*(1/Lcurr)
                            tmp = -2*tmp
                            tmp[dim] = 2*tmp[dim]
                            mat[dim, :] = tmp.reshape((1, -1))
                            valind = mat[dim, dim]
                            mat = mat+mat.T
                            mat[dim, dim] = mat[dim, dim]-valind

                            tt[dim, dim] = -2/currL**3

                            lhs = KXgXg[i, j] * (mat-tt)

                            res = rhs-lhs
                            'Build the matrix'
                            tmplist.append(res)

                        'Form final matrix'
                        colXgXg.append(np.hstack(tmplist))

                        'Clear up'
                        tmplist = []

                    dXgXgdL = np.vstack(colXgXg)
                    colXgXg = []

                    #Concatenate matrices
                    dKtildedL = np.concatenate((dKdL, dKXXgdL), axis=1)
                    dKtildedL = np.concatenate((dKtildedL, np.concatenate(
                        (np.transpose(dKXXgdL), dXgXgdL), axis=1)), axis=0)

                    dpdL[dim] = -0.5 * np.trace((np.outer(alpha, alpha) - invK)@dKtildedL)

# =============================================================================
#             elif jj == 1:
# 
#                 Kt = np.concatenate((KYY, KXtXgrad), axis=1)
#                 Kt = np.concatenate((Kt, np.concatenate((np.transpose(KXtXgrad), Kfdy), axis=1)), axis=0)
#                 dKdsigma = 2 * Kt
# 
#                 """ Calculate alpha, dK/dsigma """
#                 invK = np.linalg.inv(K)
#                 alpha = invK@ytilde.T
# 
#                 """ dp / dsigma """
#                 dpdsigma = -0.5*np.trace((np.outer(alpha, alpha) - invK)@dKdsigma)
# =============================================================================

        #jacobi = dpdL.reshape(D2, -1)
        #return jacobi
        return dpdL