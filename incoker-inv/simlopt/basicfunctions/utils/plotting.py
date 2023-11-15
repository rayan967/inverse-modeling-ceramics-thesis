import os
import numpy as np
from simlopt.basicfunctions.utils.creategrid import *
from simlopt.optimization.errormodel_new import estiamteweightfactors
import matplotlib.tri as tri
import scipy.stats as stats
from matplotlib import pyplot as plt, ticker as mticker

def plotErrorOverCost(directory,dataformat):
    fig, axs = plt.subplots(1, 1)
    for filename in os.listdir(directory+"/logs/"):
        f = os.path.join(directory+"/"+"logs/", filename)
        if os.path.isfile(f):
            print("  Using: "+filename)
            data = np.loadtxt(f)
            label = "\u0394W="+filename.strip(".txt")
            #label = "$\epsilon$="+filename.strip(".txt")
            numberofitervalues = data.shape[0]
            #axs.plot(data[:,0]*2.5, data[:,1], '--', label=label)
            axs.plot(data[:,0], data[:,1], '--', label=label)
            axs.set_yscale('log')
            axs.set_xscale('log')
            axs.grid(True)
            axs.tick_params( labelsize='medium', width=3)
            axs.set_xticks([1E2,1E3,1E4,1E5,1E6,1E7,1E8,1E9,1E10,1E11])
            axs.xaxis.set_major_locator(mticker.LogLocator(numticks=999))
            axs.xaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs="auto"))
            #axs.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            axs.set_ylabel(r'$E(D)$')
            axs.set_xlabel("Used comp. work")
            axs.legend()
    fig.savefig(directory+"/"+"postprocessing_plots/"+"ErrorOverCost"+"."+dataformat, format =dataformat, dpi=300)


def plotErrorContour(gp,Xinitial,parameterranges,epsphys,directory,dataformat):
    X = gp.getX
    Xgrad = gp.getXgrad
    dim = gp.getdata[2]
    N = Xinitial.shape[0]

    p1lower = parameterranges[0,0]
    p1upper = parameterranges[0,1]
    p2lower = parameterranges[1,0]
    p2upper = parameterranges[1,1]

    delta = 0.05

    NMC = 25
    XGLEE = createPD(NMC, dim, "grid", parameterranges)
    NGLEE = XGLEE.shape[0]

    v = 1/gp.epsxt[gp.epsxt<1E20]

    'Derivative and variance'
    dfGLEE = gp.predictderivative(XGLEE, True)
    varGLEE = gp.predictvariance(XGLEE,True)
    normedvariance = np.linalg.norm(np.sqrt(np.abs(varGLEE)),2,axis = 0)
    w = estiamteweightfactors(dfGLEE,epsphys)

    ' Plot distribution of error '
    erroratx = w*normedvariance

    ' Scaling by v'
    maxval = np.max(v)
    minval = np.min(v)

    minpointsize = 2
    maxpointsize = 20
    
    levels = np.linspace(0,0.008,40)

    sizes = v*((maxpointsize-minpointsize)/(minval-maxval))+(maxpointsize*maxval-minpointsize*minval) / (maxval-minval)

    figcontour, axcontour = plt.subplots()

    'Initial points'
    axcontour.scatter(X[:N,0], X[:N,1],c="red",zorder=2,s = sizes[:N])

    if Xgrad is not None:
        vgrad = 1/gp.epsxgrad[gp.epsxgrad<1E20]
        sizesgrad = vgrad*((maxpointsize-minpointsize)/(minval-maxval))+(maxpointsize*maxval-minpointsize*minval) / (maxval-minval)
        axcontour.scatter(Xgrad[:,0], Xgrad[:,1],marker='^',c="red",zorder=3,s = sizesgrad[::2])

    'Newly added points'
    axcontour.scatter(X[N:v.shape[0],0], X[N:v.shape[0],1],c="black",zorder=2,s = sizes[N:v.shape[0]])

    ' Contourplot '
    triang = tri.Triangulation(XGLEE[:,0], XGLEE[:,1])
    refiner = tri.UniformTriRefiner(triang)
    tri_refi, z_test_refi = refiner.refine_field(erroratx, subdiv=2)
    z_test_refi = np.abs(z_test_refi)

    #axcontour.tricontour(tri_refi, z_test_refi, linewidths=0.5, colors='k')
    cntr2 = axcontour.tricontourf(tri_refi, z_test_refi, cmap="RdBu_r")
    axcontour.set(xlim=(p1lower-delta, p1upper+delta), ylim=(p2lower-delta, p2upper+delta))
    figcontour.colorbar(cntr2, ax=axcontour,location = "left")
    axcontour.set_xlabel("$p_1$")
    axcontour.set_ylabel("$p_2$")
    figcontour.savefig(directory+"/"+"postprocessing_plots/"+"Contourplot"+"."+dataformat, format =dataformat, dpi=300)


def plotPropContour(gp,Xinitial,parameterranges,epsphys,directory,dataformat):
    X = gp.getX
    Xgrad = gp.getXgrad
    dim = gp.getdata[2]
    N = Xinitial.shape[0]


    NMC = 25
    XGLEE = createPD(NMC, dim, "grid", parameterranges)
    NGLEE = XGLEE.shape[0]

    v = 1/gp.epsxt[gp.epsxt<1E20]

    'Derivative and variance'
    dfGLEE = gp.predictderivative(XGLEE, True)
    w = estiamteweightfactors(dfGLEE,epsphys)

    ' Plot distribution of error '
    erroratx = np.abs(w.reshape((-1,1)))

    ' Scaling by v'
    maxval = np.max(v)
    minval = np.min(v)

    minpointsize = 2
    maxpointsize = 20
    
    p1lower = parameterranges[0,0]
    p1upper = parameterranges[0,1]
    p2lower = parameterranges[1,0]
    p2upper = parameterranges[1,1]

    delta = 0.05

    sizes = v*((maxpointsize-minpointsize)/(minval-maxval))+(maxpointsize*maxval-minpointsize*minval) / (maxval-minval)

    figcontour, axcontour = plt.subplots()

    'Initial points'
    axcontour.scatter(X[:N,0], X[:N,1],c="red",zorder=2,s = sizes[:N])

    if Xgrad is not None:
        axcontour.scatter(Xgrad[:,0], Xgrad[:,1],marker='x',c="green",zorder=2,s = 20)

    'Newly added points'
    axcontour.scatter(X[N:v.shape[0],0], X[N:v.shape[0],1],c="black",zorder=2,s = sizes[N:v.shape[0]])

    ' Contourplot '
    triang = tri.Triangulation(XGLEE[:,0], XGLEE[:,1])
    refiner = tri.UniformTriRefiner(triang)
    tri_refi, z_test_refi = refiner.refine_field(erroratx[:,0], subdiv=4)
    z_test_refi = np.abs(z_test_refi)

    #axcontour.tricontour(tri_refi, z_test_refi, linewidths=0.5, colors='k')
    cntr2 = axcontour.tricontourf(tri_refi, z_test_refi, cmap="RdBu_r")
    axcontour.set(xlim=(p1lower-delta, p1upper+delta), ylim=(p2lower-delta, p2upper+delta))
    figcontour.colorbar(cntr2, ax=axcontour)
    axcontour.set_xlabel("$p_1$")
    axcontour.set_ylabel("$p_2$")
    figcontour.savefig(directory+"/"+"postprocessing_plots/"+"ContourplotPropagation"+"."+dataformat, format =dataformat, dpi=300)

    
def plotStdContour(gp,Xinitial,parameterranges,epsphys,directory,dataformat):
    X = gp.getX
    Xgrad = gp.getXgrad
    dim = gp.getdata[2]
    N = Xinitial.shape[0]


    NMC = 25
    XGLEE = createPD(NMC, dim, "grid", parameterranges)
    NGLEE = XGLEE.shape[0]

    v = 1/gp.epsxt[gp.epsxt<1E20]

    'Derivative and variance'
    varGLEE = gp.predictvariance(XGLEE,True)
    normedvariance = np.linalg.norm(np.sqrt(np.abs(varGLEE)),2,axis = 0)

    p1lower = parameterranges[0,0]
    p1upper = parameterranges[0,1]
    p2lower = parameterranges[1,0]
    p2upper = parameterranges[1,1]

    delta = 0.05

    ' Scaling by v'
    maxval = np.max(v)
    minval = np.min(v)

    minpointsize = 2
    maxpointsize = 20

    sizes = v*((maxpointsize-minpointsize)/(minval-maxval))+(maxpointsize*maxval-minpointsize*minval) / (maxval-minval)

    figcontour, axcontour = plt.subplots()

    'Initial points'
    axcontour.scatter(X[:N,0], X[:N,1],c="red",zorder=2,s = sizes[:N])

    if Xgrad is not None:
        axcontour.scatter(Xgrad[:,0], Xgrad[:,1],marker='x',c="green",zorder=2,s = 20)

    'Newly added points'
    axcontour.scatter(X[N:v.shape[0],0], X[N:v.shape[0],1],c="black",zorder=2,s = sizes[N:v.shape[0]])

    ' Contourplot '
    triang = tri.Triangulation(XGLEE[:,0], XGLEE[:,1])
    refiner = tri.UniformTriRefiner(triang)
    tri_refi, z_test_refi = refiner.refine_field(normedvariance, subdiv=4)
    z_test_refi = np.abs(z_test_refi)

    #axcontour.tricontour(tri_refi, z_test_refi, linewidths=0.5, colors='k')
    cntr2 = axcontour.tricontourf(tri_refi, z_test_refi, cmap="RdBu_r")
    axcontour.set(xlim=(p1lower-delta, p1upper+delta), ylim=(p2lower-delta, p2upper+delta))
    figcontour.colorbar(cntr2, ax=axcontour)
    axcontour.set_xlabel("$p_1$")
    axcontour.set_ylabel("$p_2$")
    figcontour.savefig(directory+"/"+"postprocessing_plots/"+"ContourplotStandarddeviation"+"."+dataformat, format =dataformat, dpi=300)


def plotGPErrorHist(gp,directory,dataformat):

    eps = np.sqrt(gp.epsxt[gp.epsxt<1E20])
    fighist, axshist = plt.subplots(1, 1)
    axshist.grid(True, linestyle='-.')
    axshist.set_axisbelow(True)
    axshist.set_xlabel(r'$\epsilon$')
    axshist.set_ylabel("Counts")
    axshist.ticklabel_format(style='sci', axis='x', scilimits=(-6,-4))
    axshist.tick_params(labelsize='medium', width=3)
    axshist.hist(eps,cumulative=False,bins = 20,edgecolor = "black")
    
    fighist.savefig(directory+"/"+"postprocessing_plots/"+"histOfGPError"+"."+dataformat, format =dataformat, dpi=300)

def plotMarginalSolutions(dim,postmean,poststd,xreal,directory,dataformat):


    n = 1

    fig, axs = plt.subplots(1, dim)
    filepath = directory+"/"+"postprocessing_plots/"
    filename = 'solution'

    #parameternames = [r"$p_1$",r"$p_2$"]
    parameternames = ["$cd$","$t$"]
    ylabels = [r"$f_{p_1}$",r"$f_{p_2}$"]

    for jj in range(dim):
        mu = postmean[jj]  # Mean of reconstructed parameter
        sigma = poststd[jj]  # Standard deviations of reconstruced parameter
    
        x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)

        # Shrink current axis's height by 10% on the bottom
        box = axs[jj].get_position()
        axs[jj].set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.4])
        axs[jj].plot(x, stats.norm.pdf(x, mu, sigma))

        axs[jj].vlines(mu, 0, np.max(stats.norm.pdf(x, mu, n*sigma)), linestyles='dashed',label=r"$\mu: {:10.4f}$".format(mu))
        axs[jj].vlines(mu + n*sigma, 0, np.max(stats.norm.pdf(x, mu, n*sigma)), linestyles='dashdot', color="orange",label=r"$\pm \sigma: {:10.4f}$".format(sigma))
        axs[jj].vlines(mu - n*sigma, 0, np.max(stats.norm.pdf(x, mu, n*sigma)), linestyles='dashdot', color="orange")

        if xreal is not None:
            axs[jj].vlines(xreal[0,jj], 0, np.max(stats.norm.pdf(x, mu, n*sigma)), linestyles='dashed', color="red",label=r"$f_r: {:10.4f}$".format(xreal[0,jj]))
            axs[jj].vlines(xreal[0,jj], 0, np.max(stats.norm.pdf(x, mu, n*sigma)), linestyles='dashed', alpha = 0,  label=r"$\Delta p: {:10.4f}$".format(np.abs(xreal[0,jj]-mu)))

        
        axs[jj].set_xlabel(parameternames[jj])
        axs[jj].set_ylabel(ylabels[jj])

        axs[jj].tick_params( labelsize='medium', width=3)
        axs[jj].grid(True, linestyle='-.')
        axs[jj].legend(loc='lower center',bbox_to_anchor=(0.5, -0.4), fancybox=True, shadow=False, ncol=2)
        kk = 2  # Keeps every 2th label
        [l.set_visible(False) for (i,l) in enumerate(axs[jj].xaxis.get_ticklabels()) if i % kk != 0]

        fig.tight_layout()
    fig.savefig(filepath+'/'+filename+"."+dataformat, format = dataformat, dpi=300)


def plotHistogramOfLocalError(gp,epsphys,parameterranges,directory,dataformat,cumulative=False):

    dim = gp.getdata[2]
    NMC = 50
    Xhist = createPD(NMC, dim, "grid", parameterranges)
    Nhist = Xhist.shape[0]

    df    = gp.predictderivative(Xhist, True)
    varXC = gp.predictvariance(Xhist,True)

    normvar = np.linalg.norm(np.sqrt(np.abs(varXC)),2,axis=0)
    w = estiamteweightfactors(df,epsphys)

    localerror= normvar*w
    
    fighist, axshist = plt.subplots()
    axshist.grid(True, linestyle='-.')
    axshist.set_axisbelow(True)
    axshist.hist(np.abs(localerror),bins=100, density=False,cumulative=cumulative,edgecolor = "black", alpha=0.75,zorder = 2)
    axshist.set_xlabel(r'$|e_{\mathcal{D}}(p_i)|$')
    axshist.set_ylabel("Counts")
    #axshist.ticklabel_format(style='sci')
    axshist.tick_params( labelsize='medium', width=3)
    axshist.ticklabel_format(style='sci', axis='x', scilimits=(-4,-3))
    #fighist.savefig(directory+"/"+"postprocessing_plots/"+"histoflocalerror"+'.pdf', format = 'pdf', dpi=300)
    fighist.savefig(directory+"/"+"postprocessing_plots/"+"histoflocalerror"+"."+dataformat, format = dataformat, dpi=300)
