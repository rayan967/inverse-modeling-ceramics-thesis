import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from tabulate import tabulate

def plotiteration(gp,w,var,N,Ngrad,XGLEE,XC,mcglobalerrorbefore,parameterranges,figurepath,counter):

    
    p1lower = parameterranges[0,0]
    p1upper = parameterranges[0,1]
    p2lower = parameterranges[1,0]
    p2upper = parameterranges[1,1]

    delta = 0.05

    ' Plot w*var '
    erroratx = w*var

    X = gp.getX

    figcontour, axcontour = plt.subplots()
    figvar, axvar = plt.subplots()
    figw, axw = plt.subplots()
    
    axcontour.scatter(X[0:N,0], X[0:N,1],c="black",zorder=2)

    if XC.size!=0:
        axcontour.scatter(XC[:,0], XC[:,1],marker='x',c="green",zorder=2,s = 30)
        axvar.scatter(XC[:,0], XC[:,1],marker='x',c="green",zorder=2,s = 30)
        axw.scatter(XC[:,0], XC[:,1],marker='x',c="green",zorder=2,s = 30)

    if isinstance(Ngrad, int) and Ngrad>0:
        axcontour.scatter(gp.getXgrad[:Ngrad,0], gp.getXgrad[:Ngrad,1],marker='x',c="green",zorder=2,s = 20)


    triang = tri.Triangulation(XGLEE[:,0], XGLEE[:,1])
    refiner = tri.UniformTriRefiner(triang)
    tri_refi, z_test_refi = refiner.refine_field(erroratx, subdiv=4)
    z_test_refi = np.abs(z_test_refi)

    cntr2 = axcontour.tricontourf(tri_refi, z_test_refi, 25, cmap="RdBu_r")
    #axcontour.tricontour(tri_refi, z_test_refi, 15, linewidths=0.5, colors='k')
    axcontour.set(xlim=(p1lower-delta, p1upper+delta), ylim=(p2lower-delta, p2upper+delta))
    
    axcontour.set_title("Global error estimat: "+str(mcglobalerrorbefore))
    figcontour.colorbar(cntr2, ax=axcontour)
    axcontour.set_xlabel("$p_1$")
    axcontour.set_ylabel("$p_2$")
    figcontour.savefig(figurepath+"lee_iter_"+str(counter)+'.png')
    #figcontour.savefig(figurepath+"lee_iter_"+str(counter)+'.svg', format = 'svg', dpi=300)


    'Plot variance '
    tri_refi, z_test_refi = refiner.refine_field(var, subdiv=4)
    
    axvar.scatter(X[0:N,0], X[0:N,1],c="black",zorder=2)
    cntrvar = axvar.tricontourf(tri_refi,z_test_refi, 15, cmap="RdBu_r")
    #cntrvar = axvar.tricontourf(XGLEE[:,0], XGLEE[:,1], var, 15, cmap="RdBu_r")
    #axvar.tricontour(XGLEE[:,0], XGLEE[:,1], var, 15, linewidths=0.5, colors='k')
    axvar.set(xlim=(p1lower-delta, p1upper+delta), ylim=(p2lower-delta, p2upper+delta))
    figvar.colorbar(cntrvar, ax=axvar)
    figvar.savefig(figurepath+"varplot_iter_"+str(counter)+'.png')

    'Plot w'
    tri_refi, z_test_refi = refiner.refine_field(w, subdiv=4)
    
    axw.scatter(X[0:N,0], X[0:N,1],c="black",zorder=2)
    cntrw = axw.tricontourf(tri_refi,z_test_refi, 15, cmap="RdBu_r")
    #cntrw = axw.tricontourf(XGLEE[:,0], XGLEE[:,1], w, 15, cmap="RdBu_r")
    #axvar.tricontour(XGLEE[:,0], XGLEE[:,1], w, 15, linewidths=0.5, colors='k')
    axw.set(xlim=(p1lower-delta, p1upper+delta), ylim=(p2lower-delta, p2upper+delta))
    figw.colorbar(cntrw, ax=axw)
    figw.savefig(figurepath+"w_iter_"+str(counter)+'.png')
    

    
def plotderivativedifference(gp,fun,N,XGLEE,parameterranges,figurepath,counter):

    
    p1lower = parameterranges[0,0]
    p1upper = parameterranges[0,1]
    p2lower = parameterranges[1,0]
    p2upper = parameterranges[1,1]

    delta = 0.05

    ' Estimate derivatives '
    df = gp.predictderivative(XGLEE,True)[...,0]
    grad = fun["gradient"](XGLEE,4)
    norm = np.linalg.norm((df-grad),axis=1)

    nn = int(np.sqrt(XGLEE[...,0].shape[0]))
    dfx = df[...,0]
    dfy = df[...,1]
    
    "--------------"
    figcontour, axcontour = plt.subplots()
    triang = tri.Triangulation(XGLEE[:,0], XGLEE[:,1])
    refiner = tri.UniformTriRefiner(triang)
    tri_refi, z_test_refi = refiner.refine_field(dfx, subdiv=4)
    z_test_refi = np.abs(z_test_refi)
    cntr2 = axcontour.tricontourf(tri_refi, z_test_refi, 25, cmap="RdBu_r")
    
    refiner = tri.UniformTriRefiner(triang)
    tri_refi, z_test_refi = refiner.refine_field(grad[...,0], subdiv=4)
    z_test_refi = np.abs(z_test_refi)
    gradXreal = axcontour.tricontour(tri_refi, z_test_refi, 25,colors = "black",alpha = 0.5)
    
    figcontour.savefig(figurepath+"dfx_"+str(counter)+'.png')
    
    "--------------"
    figcontourY, axcontourY = plt.subplots()
    triang = tri.Triangulation(XGLEE[:,0], XGLEE[:,1])
    refiner = tri.UniformTriRefiner(triang)
    tri_refi, z_test_refi = refiner.refine_field(dfy, subdiv=4)
    z_test_refi = np.abs(z_test_refi)
    cntr2 = axcontourY.tricontourf(tri_refi, z_test_refi, 25, cmap="RdBu_r")
    
    refiner = tri.UniformTriRefiner(triang)
    tri_refi, z_test_refi = refiner.refine_field(grad[...,1], subdiv=4)
    z_test_refi = np.abs(z_test_refi)
    gradYreal = axcontourY.tricontour(tri_refi, z_test_refi, 25,colors = "black",alpha = 0.5)
    
    figcontourY.savefig(figurepath+"dfy_"+str(counter)+'.png')
    
    "--------------"
# =============================================================================
#     figcontournorm, axcontournorm = plt.subplots()
#     triang = tri.Triangulation(XGLEE[:,0], XGLEE[:,1])
#     refiner = tri.UniformTriRefiner(triang)
#     tri_refi, z_test_refi = refiner.refine_field(norm, subdiv=4)
#     z_test_refi = np.abs(z_test_refi)
#     cntr2 = axcontournorm.tricontourf(tri_refi, z_test_refi, 25, cmap="RdBu_r")
#     figcontournorm.colorbar(cntr2, ax=axcontournorm)    
#     figcontournorm.savefig(figurepath+"norm_"+str(counter)+'.png')
# =============================================================================

    
# =============================================================================
#     figcontour, axcontour = plt.subplots()
#     axcontour.scatter(X[0:N,0], X[0:N,1],c="black",zorder=2)
#     
#     axcontour.set_title("Global error estimat: "+str(mcglobalerrorbefore))
#     figcontour.colorbar(cntr2, ax=axcontour)
#     axcontour.set_xlabel("$p_1$")
#     axcontour.set_ylabel("$p_2$")
#     figcontour.savefig(figurepath+"lee_iter_"+str(counter)+'.png')
#     #figcontour.savefig(figurepath+"lee_iter_"+str(counter)+'.svg', format = 'svg', dpi=300)
# =============================================================================


def prettyprintvector(v,dim,graddata):
    if graddata:
        headers = ["Point", "component", "v value", "e value"]

        pointidx = np.repeat(np.arange(int(v.size/dim)),dim).reshape((1,-1))
        gradidx = np.arange(dim).reshape((1,-1))
        gradtile  = np.tile(gradidx, int(v.size/dim)).reshape((1,-1))
        tmp1 = np.concatenate((pointidx,gradtile),axis=0)
        v = v.reshape((1,-1))
        tmp = np.concatenate((tmp1,v),axis=0).T
        v=  np.concatenate((tmp,np.sqrt(v.T)**(-1)),axis=1)

    else:
        headers = ["Point", "v value", "e value"]

        index = np.arange(v.size).reshape((1,-1))
        v = v.reshape((1,-1))
        eps = np.sqrt(v**(-1))
        v = np.concatenate((index,v),axis=0).T
        v = np.concatenate((v,eps.T),axis=1)
    table = tabulate(v, headers)
    print(table)