import numpy as np

from optimization.utils.loss import *
from optimization.utils.computationalwork import *


def findeps(x,Xtextended,Xgradextended,ytextended,ygradextended,HPm,epsXtextended,epsXgradextended,
                        m,dim,epsphys,initsize,
                        dptarget,itermax,threshold,Wbudget,
                        eta=1,gamma=1.05,verbose=0):

    n = Xtextended.shape[0]
    solution,index,computationalcost,accuracy = np.array([]), 0, 0.0, 0.0
    currentcomputationalcost = 1E20

    for idx in range(n):

        """
        With given index idx it is tested here whether with or
        without gradient information the parameter error can be brought to the desired tolerance.
        Before that it is necessary to check if gradient information is available at the point x[idx].
        """

        dw,v = 0.0, 1.0
        eps = np.sqrt(epsXtextended[0,idx])

        if idx >= initsize:
            w0 = W(1E-1, dim)
            print("Index: {}, x: {}, type: (g)".format(idx,Xtextended[idx,:]))
            print("Initial work value {:g}, current eps: {:g}".format(w0, eps))
            epsinitial = 1E20

        else:
            w0 = W(eps, dim)
            print("Index: {}, x: {}, type: (s)".format(idx,Xtextended[idx,:]))
            print("Initial work value {:g}, current eps: {:g}".format(w0, eps))
            epsinitial = epsXtextended[0,idx]

        for i in range(itermax+1):

            f = losswithoutgflag(w0, idx, x, Xtextended, Xgradextended, ytextended, ygradextended, epsXtextended, epsXgradextended, HPm , m, dim, epsphys)
            jac = dlosswithoutgflag(w0+gamma*v, idx, x, Xtextended, Xgradextended, ytextended,ygradextended, epsXtextended, epsXgradextended, HPm, m, dim, epsphys)

            ' The accuracy is bounded from below since simulations cant be arbitrary accurate'
            if np.abs(epsofw((w0+dw),dim)) < threshold:
                if verbose:
                    print("  Iteration {}, no solution found".format(i))
                    print("  Accuracy is below allowed threshold")
                    print("  Accuracy: {}".format(str(epsofw((w0+dw),dim))))
                    print("  f: {:g}".format(np.abs(np.sqrt(f))))
                    print("\n")
                epsXtextended[0,idx] = epsinitial
                dw = 0
                break

            elif i == itermax-1:
                if verbose:
                    print("  Iteration {}, no solution found - maximum number of iterations reached".format(i))
                    print("  Last function value {:g}".format(np.abs(np.sqrt(f))))
                    print("\n")
                epsXtextended[0,idx] = epsinitial
                dw = 0
                break

            elif (w0+dw) > Wbudget:
                if verbose:
                    print("  Iteration {}, no solution found".format(i))
                    print("  Computational work would exceed current budget")
                    print("  Calculated work {}, current budget {}".format((w0+dw), Wbudget))
                    print("  f: {:g}".format(np.abs(np.sqrt(f))))
                    print("\n")
                epsXtextended[0,idx] = epsinitial
                dw = 0
                break

            elif epsofw((w0+dw),dim) < 0:
                if verbose:
                    print("  Iteration {}, no solution found".format(i))
                    print("  Negative value occured...")
                    print("  f: {}".format(np.abs(np.sqrt(f))))
                    print("\n")
                epsXtextended[0,idx] = epsinitial
                dw = 0
                break

            elif np.sqrt(np.abs(f)) < dptarget:

                if i == 0:
                    print("  Solution found in {:g} iterations at {}".format(i, Xtextended[idx]))
                    print("  Startvalue is sufficient enough")
                    print("  Current function value: {:g}".format(np.abs(np.sqrt(f))))
                    print("  Calculated computational work: {:g}".format(((w0))))
                    print("  Associated accuracy eps: {:g}".format(epsofw((w0),dim)))
                    print("\n")

                    if epsofw((w0),dim) > epsinitial:
                        print("Something went wrong, the solution is discarded")
                        epsXtextended[0,idx] = epsinitial
                        break

                    if w0 <= currentcomputationalcost:
                        solution = Xtextended[idx]
                        index = idx
                        computationalcost = w0
                        accuracy = epsofw((w0),dim)

                        epsXtextended[0,idx] = epsinitial
                        currentcomputationalcost = w0
                        break

                    epsXtextended[0,idx] = epsinitial
                    break

                else:
                    print("  Solution found in {:g} iterations at {}".format(i, Xtextended[idx]))
                    print("  Current function value: {:g}".format(np.abs(np.sqrt(f))))
                    print("  Calculated computational work: {:g}".format(((w0))))
                    print("  Associated accuracy eps: {:g}".format(epsofw((w0),dim)))
                    print("\n")

                    if epsofw((w0),dim) > epsinitial:
                        print("Something went wrong, the solution is discarded")
                        epsXtextended[0,idx] = epsinitial
                        break

                    if w0 <= currentcomputationalcost:
                        solution = Xtextended[idx]
                        index = idx
                        computationalcost = w0
                        accuracy = epsofw((w0),dim)

                        epsXtextended[0,idx] = epsinitial
                        currentcomputationalcost = w0
                        break

                    epsXtextended[0,idx] = epsinitial
                    break
            'Nesterov with momentum'
            v = gamma * v - eta * jac
            w0 = w0 + v

    return solution,index,computationalcost,accuracy


