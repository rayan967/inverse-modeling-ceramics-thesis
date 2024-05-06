from optimization.utils.computationalwork import *
import numpy as np

def plotiterationinformation(idx,dim,Xtextended,epsXtextended,epsXgradextended,w0,w0grad,ptype):

    tmp  = np.array2string(w0grad, precision=2, separator=', ',suppress_small=True)
    tmp2 = np.array2string(epsXgradextended[0,idx*dim:idx*dim+dim], precision=7, separator=', ',suppress_small=True)
    tmp3 = np.array2string(np.array([w0]), precision=7, separator=', ',suppress_small=True)
    tmp4 = np.array2string(np.array([epsXtextended[0,idx]]), precision=7, separator=', ',suppress_small=True)

    output1 = " Initial work value for gradients: "
    output2 = " Initial work value for data:      "

    ss  = len(output1 + tmp)
    ss2 = len(output2 + tmp3)

    if ss > ss2:
        print("=======================================================================================")
        print("Index: {}, x: {}({})".format(idx, Xtextended[idx,:],ptype))
        print(" Initial work value for data:      " +tmp3+' '*np.abs((ss-ss2))+", current acc: "+tmp4)
        print(" Initial work value for gradients: " +tmp+", current acc: "+tmp2)
    else:
        print("=======================================================================================")
        print("Index: {}, x: {}({})".format(idx, Xtextended[idx,:],ptype))
        print(" Initial work value for data:      " +tmp3+", current acc: "+tmp4)
        print(" Initial work value for gradients: " +tmp+' '*np.abs((ss-ss2))+", current acc: "+tmp2)