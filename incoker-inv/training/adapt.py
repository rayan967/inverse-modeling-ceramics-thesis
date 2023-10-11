from simlopt.gpr.gaussianprocess import *

from simlopt.basicfunctions.utils.creategrid import createPD
from simlopt.optimization.errormodel_new import MCGlobalEstimate, acquisitionfunction, estiamteweightfactors
import matplotlib.pyplot as plt
from adaptive_training import accuracy_test


def adapt_inc(gp, parameterranges, TOL, TOLAcqui, TOLrelchange, epsphys, Xt, yt, X_test, y_test, selected_indices):
    # Initialization
    counter = 0
    N = gp.getdata[0]
    dim = gp.getdata[2]
    m = yt.shape[1]
    NMC = 200
    totaltime = 0
    totalFEM = 0
    global_errors = []
    accuracies = []

    print("---------------------------------- Start adaptive phase")
    print("Number of initial points:          "+str(len(selected_indices)))
    print("Desired tolerance:                 "+str(TOL))
    print("\n")

    XGLEE_f = X_test

    # Main adaptive loop
    while True:
        counter += 1
        print(f"--- Iteration {counter}")

        # Generate candidate points
        XGLEE = createPD(NMC, dim, "random", parameterranges)
        dfGLEE = gp.predictderivative(XGLEE, True)
        varGLEE = gp.predictvariance(XGLEE, True)
        normvar = np.linalg.norm(np.sqrt(np.abs(varGLEE)), 2, axis=0) ** 2
        w = estiamteweightfactors(dfGLEE, epsphys)


        # MC global error estimation
        dfGLEE_f = gp.predictderivative(XGLEE_f, True)
        varGLEE_f = gp.predictvariance(XGLEE_f, True)
        normvar_f = np.linalg.norm(np.sqrt(np.abs(varGLEE_f)), 2, axis=0) ** 2
        w_f = estiamteweightfactors(dfGLEE_f, epsphys)
        mcglobalerrorbefore = MCGlobalEstimate(w_f, normvar_f, NMC,
                                               parameterranges)
        print("Global error estimate before optimization:   {:1.5f}".format(mcglobalerrorbefore))

        """ ------------------------------Acquisition phase ------------------------------ """
        'Add new candidate points'
        XC = np.array([])
        # Acquisition phase
        XC, index, value = acquisitionfunction(gp, dfGLEE, normvar, w, XGLEE, epsphys,
                                               TOLAcqui)  # Use your acquisition function

        # Find closest point in the training data and add it to the GP model
        if XC.size != 0:
            print(" Number of possible candidate points: {}".format(XC.shape[0]))
            print(" Found canditate point(s):            {}".format(XC[0]))
            print(" Use ith highest value   :            {}".format(index))
            print(" Value at index          :            {}".format(value))
            print("\n")

            closest_point, closest_point_value, selected_indices = find_closest_point(Xt, yt, XC[0], selected_indices)
            print("Point closest to Xc:", str(closest_point))
            print("Yc:", str(closest_point_value))


            # y = simulate(XC[0], 'thermal_expansion')
            # XC[0] = [-0.31546006  0.27063561  1.00704592], y = [5.5]


            epsXc = 1E-4 * np.ones((1, XC.shape[0]))  # eps**2
            gp.adddatapoint(closest_point)
            gp.adddatapointvalue(closest_point_value)
            gp.addaccuracy(epsXc)
            print("Size of data: ", str(len(selected_indices)))
        else:
            print("Something went wrong, no candidate point was found.")
            print("\n")


        # A posteriori MC global error estimation
        dfGLEE_f = gp.predictderivative(XGLEE_f, True)
        varGLEE_f = gp.predictvariance(XGLEE_f, True)
        wpost_f = estiamteweightfactors(dfGLEE_f, epsphys)
        normvar_f = np.linalg.norm(np.sqrt(np.abs(varGLEE_f)), 2, axis=0) ** 2
        mcglobalerrorafter = MCGlobalEstimate(wpost_f, normvar_f, NMC, parameterranges)
        global_errors.append(mcglobalerrorafter)

        accuracies.append(accuracy_test(gp, X_test, y_test))

        # Check convergence
        if mcglobalerrorafter <= TOL:
            print("--- Convergence")
            print(" Desired tolerance is reached, adaptive phase is done.")
            plot_global_errors(global_errors)
            plot_accuracy(accuracies)
            return gp

        # Adjust budget
        relchange = np.abs(mcglobalerrorbefore - mcglobalerrorafter) / mcglobalerrorbefore * 100
        if relchange < TOLrelchange:
            TOLAcqui *= 0.9999
            print("Relative change is below set threshold. Adjusting TOLAcqui.")

        # Check number of points
        if len(selected_indices) >= 1815:
            print("--- Maximum number of points reached")
            plot_global_errors(global_errors)
            plot_accuracy(accuracies)
            return gp

        Nmax = 50
        N = gp.getdata[0]
        if N < Nmax:
            print("--- A priori hyperparameter adjustment")
            region = [(0.01, 2) for _ in range(dim)]
            gp.optimizehyperparameter(region, "mean", False)
        else:
            print("--- A priori hyperparameter adjustment")
            print("Number of points is higher then "+str(Nmax))
            print("No optimization is performed")
        print("\n")


def find_closest_point(Xt, yt, point, selected_indices):
    distances = np.linalg.norm(Xt - point, axis=1)
    while True:
        index = np.argmin(distances)
        if index not in selected_indices:
            selected_indices.append(index)
            return Xt[index].reshape(1,-1), yt[index].reshape(1,-1), selected_indices
        else:
            distances[index] = np.inf


def plot_global_errors(global_errors):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(global_errors) + 1), global_errors, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('MC Global Error Estimate')
    plt.title('MC Global Error Estimate per Iteration')
    plt.grid(True)
    plt.savefig('mc_global_error_plot.png')


def plot_accuracy(accuracies):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(accuracies) + 1), accuracies, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per Iteration')
    plt.grid(True)
    plt.savefig('accuracy_plot.png')