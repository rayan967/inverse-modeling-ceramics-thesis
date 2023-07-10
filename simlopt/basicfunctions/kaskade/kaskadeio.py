import subprocess
import os
import numpy as np
import json

from os import path

def runkaskade(execpath, execname, parameter):
    """

    Parameters
    ----------
    execpath : string
        Path to kaskade file.
    execname : string
        Name of executable.
    parameter : dictionary
        Dictionary with all parameters which are parsed to the kaskade executable.

    Returns
    -------
    Outputs and errors

    """
    
    if path.isdir(execpath):

        if path.isfile(os.path.join(execpath,execname)):
            cmd = execname

            for key, value in parameter.items():
                if "--" in key:
                    cmd +=' '+key+'='+str(value)
                else:
                    cmd +=' '+"--"+key+'='+str(value)

            print("Starting: " + execname)
            print("Commands: " + cmd.split(' ', 1)[1])

            proc = subprocess.Popen(cmd, cwd=execpath, shell=True)

            try:
                outs, errs = proc.communicate()
                print("output = {}".format(outs))
                print("errors = {}".format(errs))
                return (outs,errs)
            except subprocess.TimeoutExpired:
                proc.kill()
                outs, errs = proc.communicate()
                print("output = {}".format(outs))
                print("errors = {}".format(errs))
                return (outs,errs)

        else:
            print("File not found")
            return None
    else:
        print("Filepath not found")
        return None


def readsimdatafromfile(execpath, filename, dim):
    """
    Helper function to read the data from the simulation log file. Current file structure

    x << y << ... << d << f(x,y,...,d) << dxf << dyf << ... << ddf << epsf << epsdxf << ... << epsddf << reached(bool)

    Parameters
    ----------
    execpath : TYPE
        DESCRIPTION.
    filname : TYPE
        DESCRIPTION.
    dim : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    path = os.path.join(execpath, filename)
    simulationdata = np.loadtxt(path)

    if simulationdata.ndim == 1:  # There is only on line in the file
        reached = int(simulationdata.reshape((1, -1))[0, -1])
        epsx = simulationdata.reshape((1, -1))[0, -2-dim]
        epsxgrad = simulationdata.reshape((1, -1))[-dim-1:-dim+1]
        ytnew = simulationdata.reshape((1, -1))[0, dim]
        ygradnew = simulationdata.reshape((1, -1))[0, dim+1:dim+dim]

    else:
        'Check if simulation reached accuracy'
        reached = int(simulationdata[-1][-1])
        epsx = simulationdata[-1][-2-dim]
        epsxgrad = simulationdata[-1][-dim-1:-dim+1]
        ytnew = simulationdata[-1][dim]
        ygradnew = simulationdata[-1][dim+1:dim+dim+1]

    return ytnew, ygradnew, epsx, epsxgrad, reached

def readtodict(datapath,dataname):

    if path.isdir(datapath):

        last_line = ""
        if path.isfile(os.path.join(datapath,dataname)):

           # reading the data from the file
           with open(os.path.join(datapath,dataname)) as f:

               first_line = f.readline()
               for last_line in f:
                   pass

               if last_line == "":
                   #Last line is empy
                   datadict = json.loads(first_line)
                   return datadict
               else:
                   datadict = json.loads(last_line)
                   return datadict


        else:
            print("File not found")
            return None

    else:
        print("Filepath not existent")
        return None

def createandrun(point,eps,epsgrad,execpath,execname):

    ' Create dict from function parameters '
    nums = np.arange(1,point.shape[1]+1)
    points = ['--x'+s for s in list(map(str, list(nums)))]
    tmpzip = zip(points, point[0,:].tolist())
    parameter = dict(tmpzip)

    if epsgrad is None:
        parameter["--eps"] = eps

    else:
        parameter["--eps"] = eps
        parameter["--epsgrad"] = epsgrad

    runkaskade(execpath, execname, parameter)
    print("\n")

    'Read simulation data and get function value'
    simulationdata = readtodict(execpath, "dump.log")

    #TODO: Reached for gradients....
    reached = np.asarray(simulationdata["flag"])

    epsXtnew = np.asarray(simulationdata["accuracy"])
    epsXtnew = epsXtnew.reshape((1, -1))

    if reached[0] == 0:
        print("Accuracy during simulation reached with: {}".format(epsXtnew[0, 0]))
    elif reached[0] == 1:
        print("Accuracy during simulation was not reached with: {}")
        print("Set accuracy to: ".format(epsXtnew[0, 0]))

    if epsgrad is None:
        ' We are only interested in the eps value for the given point '
        epsXtnew = np.asarray(simulationdata["accuracy"])
        epsXtnew = epsXtnew.reshape((1, -1))

        ' Reshape for concatenate data '
        ytnew = np.asarray(simulationdata["value"])
        ytnew = ytnew.reshape((1, -1))
        print("Simulation value: {}".format(ytnew[0, 0]))

        ygradnew = None
        epsXgradnew = None

    else:
     ' We are interested in the eps and epsgrad value for the given point '
     epsXtnew = np.asarray(simulationdata["accuracy"])
     epsXtnew = epsXtnew.reshape((1, -1))

     ' Reshape for concatenate data '
     ytnew = np.asarray(simulationdata["value"])
     ytnew = ytnew.reshape((1, -1))
     print("Simulation value: {}".format(ytnew[0, 0]))

     'Gradient data'
     epsXgradnew = np.asarray(simulationdata["gradientaccuracy"])
     epsXgradnew = epsXgradnew.reshape((1, -1))

     ' Reshape for concatenate data '
     ygradnew = np.asarray(simulationdata["gradient"])
     ygradnew = ygradnew.reshape((-1, 1))
     print("Simulation value: {}".format(ygradnew))


    return ytnew,epsXtnew,ygradnew,epsXgradnew
