import numpy as np

from optimization.utils.loss import *
from optimization.utils.computationalwork import *

from basicfunctions.utils.prettyplot import *

def findepsgrad(x,Xtextended,Xgradextended,ytextended,ygradextended,HPm,epsXtextended,epsXgradextended,
		    m,dim,epsphys,gflag,initsize,dptarget,itermax,threshold,Wbudget,eta=1,gamma=1.05,verbose=0):

    """ ----------------------------------------- Find epsilon -----------------------------------------
        ...

    """
    n = Xtextended.shape[0]
    solution,index,computationalcost,accuracy,accuracygrad,pointtype = np.array([]), 0, 0.0, 0.0, 0.0,0
    sumofcost = 1E20

    for idx in range(Xtextended.shape[0]):

        """
        With given index idx it is tested here whether with or
        without gradient information the parameter error can be brought to the desired tolerance.

        Both fuctions return either the optimal epsilon or nothing when there is no solution, such that there
        are four different outcomes

        (i) Both are none
        (ii) Withoutgi has solution, with is none
        (iii) Withoutgi is none, withgi has solution
        (iv) Both have a solution

        In the end the cheapest solution is chosen whether this means that we need to calculate the gradients or just adapt...

        """

        if idx >= initsize:
            'Initial value if the current point is a ghost point'
            w0 = W(1E-1, dim)

            'Initial value for the gradients of the current ghost point'
            w0grad = Wgrad( np.array( 1E-1*np.ones((dim))) ,dim)

            ' Concatenate both for optimization with variable gradient data  '
            w0con = np.concatenate([np.array([w0]),np.asarray(w0grad)])

            print("\n")
            plotiterationinformation(idx,dim,Xtextended,epsXtextended,epsXgradextended,w0,w0grad,"(g)")

            ' Set initial values '
            epsinitial      = epsXtextended[0,idx]
            epsinitialgrad  = epsXgradextended[0,idx*dim:idx*dim+dim] #This slice is mutable so it is changed to be a list ??
            epsinitialgrad  = list(epsinitialgrad)

            dw = 0
            v = 1

        else:
            'Initial value if the current point is a data point'
            w0 = W(np.sqrt(epsXtextended[0,idx]), dim)

            'Initial value for the gradients of the current data point'
            w0grad = Wgrad(np.sqrt(epsXgradextended[0,idx*dim:idx*dim+dim]), dim)

            'Concatenate both for optimization with variable gradient data'
            w0con = np.concatenate([np.array([w0]),np.asarray(w0grad)])

            print("\n")
            plotiterationinformation(idx,dim,Xtextended,epsXtextended,epsXgradextended,w0,w0grad,"(s)")

            ' Set initial values '
            epsinitial      = epsXtextended[0,idx]
            epsinitialgrad  = epsXgradextended[0,idx*dim:idx*dim+dim]
            epsinitialgrad  = list(epsinitialgrad)

            dw = 0
            v = 1

        # Gradient info is present, but the errors are kept constant
        print("--Start optimizing eps")
        epswithoutgi = optimizewithoutgi(w0,v,dw,epsinitial,epsinitialgrad,idx,
                                         x,Xtextended,Xgradextended,ytextended,ygradextended,HPm,epsXtextended,epsXgradextended,
                                         m,dim,epsphys,
                                         initsize,dptarget,itermax,threshold,Wbudget)

        # Gradient info is present, and the errors are optimized
        print("--Start optimizing eps and epsgrad ")
        epswithgi = optimizewithgi(w0con,v,dw,epsinitial,epsinitialgrad,idx,
                                      x,Xtextended,Xgradextended,ytextended,ygradextended,HPm,epsXtextended,epsXgradextended,
                                      m,dim,epsphys,
                                      initsize,dptarget,itermax,threshold,Wbudget)


        ' Sort the reults two times but instead of adding the result to a list we compare the costs and take the cheapest result'

        ' Check where a solution is present '
        print("\n")
        print("--- Optimization results ---")

        if epswithoutgi[0] is None:
            print(" No solution was found for eps")
            print("  Last function value: {}".format(epswithoutgi[1]))

            # No solution was found with eps and epsgrad at x
            if epswithgi[0] is None:
                print(" No solution was found for eps and epsgrad")
                print("  Last function value: {}".format(epswithgi[1]))

            # Solution was found with eps and epsgrad at x
            elif epswithgi[0] is not None:
                print(" Solution was found for eps and epsgrad")
                print("  Last function value: {}".format(epswithgi[4]))


                ccostgrad = epswithgi[2][0]+np.max(epswithgi[2][1:])
                print("   -Computational cost: "  +str(ccostgrad))

                if ccostgrad <= sumofcost:
                    print("Solution is new cheapest solution")
                    #index = epswithgi[0]
                    point = epswithgi[1]
                    acccasetwo = epswithgi[3]

                    print("   -Associanted accuracy: "+str(acccasetwo))
                    #indextoreturn = index
                    resulttoreturn = point
                    costtoreturn = ccostgrad
                    accuraciestoreturn = acccasetwo
                    typetoreturn = 1
                    sumofcost = ccostgrad

        elif epswithoutgi[0] is not None:
            print(" Solution was found for eps")

            ccost = np.max(epswithoutgi[2])
            print("   -Computational cost: "+ str(ccost))

            if epswithgi[0] is None:
                print(" No solution was found for eps and epsgrad")
                print("  Last function value: {}".format(epswithgi[1]))
                'Case 1'
                if ccost <= sumofcost:
                    print("Solution is new cheapest solution")
                    print("  Last function value: {}".format(epswithoutgi[4]))
                    #index = epswithoutgi[0]
                    point = epswithoutgi[1]
                    acccasethree = epswithoutgi[3]
                    print("   -Associanted accuracy: "+str(acccasethree))

                    #indextoreturn = index
                    resulttoreturn = point
                    costtoreturn = ccost
                    accuraciestoreturn = acccasethree
                    typetoreturn = 0

                    sumofcost = ccost

            # Solution was found with eps and epsgrad at x
            elif epswithgi[0] is not None:
                print(" Solution was found for eps and epsgrad")
                print("    Last function value: {}".format(epswithgi[4]))
                ccostgrad = epswithgi[2][0]+np.max(epswithgi[2][1:])  # Sum of eps + highest epsgrad !!!!

                print("   -Computational cost: "+ str(ccostgrad))
                print("Compare both solutions")

                if ccost < ccostgrad:

                    if ccost <= sumofcost:
                        print("Solution is new cheapest solution")
                        print(" --Optimization without gradient is cheaper, no gradient info is added")

                        #index = epswithoutgi[0]
                        point = epswithoutgi[1]
                        accwithout = epswithoutgi[3]
                        print("   -Associanted accuracy: "+str(accwithout))
                        resulttoreturn = point
                        costtoreturn = ccost
                        accuraciestoreturn = accwithout
                        typetoreturn = 0
                        sumofcost = ccost
                else:
                    if ccostgrad <= sumofcost:
                        print(" --Optimization with gradient is cheaper, adding gradient info is added")

                        acccasefour = epswithgi[3]
                        print("   -Associanted accuracy: "+str(acccasefour))
                        point = epswithgi[1]
                        resulttoreturn = point

                        costtoreturn = ccostgrad
                        accuraciestoreturn = acccasefour # eps and epsgrad !
                        typetoreturn = 1
                        sumofcost = ccostgrad

        ' Return the solution (s) for adding / altering'

    return resulttoreturn,costtoreturn,accuraciestoreturn,typetoreturn


def optimizewithgi(w0,v,dw,epsinitial,epsinitialgrad,idx,
                      x,Xtextended,Xgradextended,ytextended,ygradextended,HPm,epsXtextended,epsXgradextended,
                      m,dim,epsphys,
                      initsize,dptarget,itermax,threshold,Wbudget,
                      eta=1,gamma=1.05,verbose=0):

    """  At this point the gradient data is visible in that sense, that theo accuracies are a variable in the optimization"""
    fn = 0
    for i in range(itermax+1):

        #print("Iteration i: {}".format(i))

        f   = lossgrad(w0         , idx, x, Xtextended, Xgradextended, ytextended, ygradextended, epsXtextended, epsXgradextended, HPm, m, dim, epsphys, 1)
        jac = dlossgrad(w0+gamma*v, idx, x, Xtextended, Xgradextended, ytextended, ygradextended, epsXtextended, epsXgradextended, HPm, m, dim, epsphys)

        ' The accuracy is bounded from below since simulations cant be arbitrary accurate'
        if np.abs(epsofw((w0+dw),dim)).any() < threshold:
            if verbose:
                print("Iteration {}, no solution found".format(i))
                print("Accuracy {} is below allowed threshold".format(str(epsofw((w0+dw),dim))))
                print("Last function value {}".format(np.abs(np.sqrt(f))))
                print("\n")
            epsXtextended[0,idx] = epsinitial
            epsXgradextended[0,idx*dim:idx*dim+dim] = epsinitialgrad
            dw = 0
            return None, np.abs(np.sqrt(f))

        elif i == itermax-1:
            if verbose:
                print("Iteration {}, no solution found - maximum number of iterations reached".format(i))
                print("Last function value {}".format(np.abs(np.sqrt(f))))
                print("\n")
            epsXtextended[0,idx] = epsinitial
            epsXgradextended[0,idx*dim:idx*dim+dim] = epsinitialgrad
            dw = 0
            return None, np.abs(np.sqrt(f))

        elif np.max((w0+dw)) > Wbudget:
            if verbose:
                print("Iteration {}, no solution found".format(i))
                print("Current work {} would exceed current budget {}".format((w0+dw), Wbudget))
                print("Last function value {}".format(np.abs(np.sqrt(f))))
                print("\n")
            epsXtextended[0,idx] = epsinitial
            epsXgradextended[0,idx*dim:idx*dim+dim] = epsinitialgrad
            dw = 0
            return None, np.abs(np.sqrt(f))

        elif np.sqrt(np.abs(f)) < dptarget:

            if i == 0:
                print("\n")
                print("Solution found in {:g} iterations at {}".format(i, Xtextended[idx]))
                print(" Current function value: {:g}".format(np.abs(np.sqrt(f))))
                print(" Needed computational work: "+str(w0))
                print(" Associated accuracy: "+str(epsofw((w0),dim)))
                epsXtextended[0,idx] = epsinitial
                epsXgradextended[0,idx*dim:idx*dim+dim] = epsinitialgrad
                return idx, Xtextended[idx], w0, epsofw((w0),dim), np.abs(np.sqrt(f))

            else:
                print("\n")
                print("Solution found in {:g} iterations at {}".format(i, Xtextended[idx]))
                print(" Current function value: {:g}".format(np.abs(np.sqrt(f))))
                print(" Needed computational work: "+str(w0))
                print(" Associated accuracy: "+str(epsofw((w0),dim)))
                epsXtextended[0,idx] = epsinitial
                epsXgradextended[0,idx*dim:idx*dim+dim] = epsinitialgrad
                return idx, Xtextended[idx], w0, epsofw((w0),dim), np.abs(np.sqrt(f))

# =============================================================================
#         print("Current value of f: {}".format(np.sqrt(np.abs(f))))
#         print("Current value of eps: "+str(epsofw((w0),dim)))
# =============================================================================
# =============================================================================
#
#         if np.abs(f-fn) < 1E-10:
#             epsXtextended[0,idx] = epsinitial
#             epsXgradextended[0,idx*dim:idx*dim+dim] = epsinitialgrad
#             print("    Nothing changed between iterations - return")
#             return None, np.abs(np.sqrt(f))
# =============================================================================

        'Nesterov with momentum'
        v = gamma * v - eta * jac
        w0 = w0 + v
        fn = f

def optimizewithoutgi(w0,v,dw,epsinitial,epsinitialgrad,idx,
                      x,Xtextended,Xgradextended,ytextended,ygradextended,HPm,epsXtextended,epsXgradextended,
                      m,dim,epsphys,
                      initsize,dptarget,itermax,threshold,Wbudget,
                      eta=1,gamma=1.05,verbose=0):

    """ Since we added ghost points as gradient data we use them throughout the optimization. This is ok , since
    their accuracies is set to a high value and therefore those points are not visible to the regression"""

    fn = 0
    for i in range(itermax+1):

        #print("Iteration i: {}".format(i))

# =============================================================================
#         6.12.2021 - commented out due to reimplementation of loss / dloss
#         f   = loss(w0, idx, x, Xtextended, Xgradextended, ytextended, ygradextended, epsXtextended, epsXgradextended, HPm , m, dim, epsphys, 1)
#         jac = dloss(w0+gamma*v, idx, x, Xtextended, Xgradextended, ytextended,ygradextended, epsXtextended, epsXgradextended, HPm, m, dim, epsphys,1)
# =============================================================================
        
        f = losswithoutgflag(w0, idx, x, Xtextended, Xgradextended, ytextended, ygradextended, epsXtextended, epsXgradextended, HPm , m, dim, epsphys)
        jac = dlosswithoutgflag(w0+gamma*v, idx, x, Xtextended, Xgradextended, ytextended,ygradextended, epsXtextended, epsXgradextended, HPm, m, dim, epsphys)

        ' The accuracy is bounded from below since simulations cant be arbitrary accurate'
        if np.abs(epsofw((w0+dw),dim)) < threshold:
            if verbose:
                print("Iteration {}, no solution found".format(i))
                print("Accuracy {} is below allowed threshold".format(str(epsofw((w0+dw),dim))))
                print("Last function value {}".format(np.abs(np.sqrt(f))))
                print("\n")
            epsXtextended[0,idx] = epsinitial
            epsXgradextended[0,idx*dim:idx*dim+dim] = epsinitialgrad
            dw = 0
            return None, np.abs(np.sqrt(f))

        elif i == itermax-1:
            if verbose:
                print("Iteration {}, no solution found - maximum number of iterations reached".format(i))
                print("Last function value {}".format(np.abs(np.sqrt(f))))
                print("\n")
            epsXtextended[0,idx] = epsinitial
            epsXgradextended[0,idx*dim:idx*dim+dim] = epsinitialgrad
            dw = 0
            return None, np.abs(np.sqrt(f))

        elif (w0+dw) > Wbudget:
            if verbose:
                print("Iteration {}, no solution found".format(i))
                print("Current work {} would exceed current budget {}".format((w0+dw), Wbudget))
                print("Last function value {}".format(np.abs(np.sqrt(f))))
                print("\n")
            epsXtextended[0,idx] = epsinitial
            epsXgradextended[0,idx*dim:idx*dim+dim] = epsinitialgrad
            dw = 0
            return None, np.abs(np.sqrt(f))

        elif np.sqrt(np.abs(f)) < dptarget:

            if i == 0:
                print("\n")
                print("Startvalue is sufficient enough, no more iteration needed")
                print("Current function value: {:g}".format(np.abs(np.sqrt(f))))
                print("Needed computational work: {:g}".format(((w0))))
                print("Associated accuracy: {:g}".format(epsofw((w0),dim)))
                #print("Point is NOT added as data point, since nothing changed")
                print("\n")
                epsXtextended[0,idx] = epsinitial
                epsXgradextended[0,idx*dim:idx*dim+dim] = epsinitialgrad
                return idx, Xtextended[idx], w0, epsofw((w0),dim),np.abs(np.sqrt(f))

            else:
                print("\n")
                print("Solution found in {:g} iterations at {}".format(i, Xtextended[idx]))
                print("Current function value: {:g}".format(np.abs(np.sqrt(f))))
                print("Needed computational work: {:g}".format(((w0))))
                print("Associated accuracy: {:g}".format(epsofw((w0),dim)))
                print("\n")
                epsXtextended[0,idx] = epsinitial
                epsXgradextended[0,idx*dim:idx*dim+dim] = epsinitialgrad
                return idx, Xtextended[idx], w0, epsofw((w0),dim),np.abs(np.sqrt(f))

# =============================================================================
#         print("Current value of f: {}".format(np.sqrt(np.abs(f))))
#         print("Current value of eps: {:g}".format( epsofw((w0),dim) ))
# =============================================================================

# =============================================================================
#         if np.abs(f-fn) < 1E-10:
#             epsXtextended[0,idx] = epsinitial
#             epsXgradextended[0,idx*dim:idx*dim+dim] = epsinitialgrad
#             print("    Nothing changed between iterations - return")
#             return None, np.abs(np.sqrt(f))
# =============================================================================

        'Nesterov with momentum'
        v = gamma * v - eta * jac
        w0 = w0 + v
        fn = f