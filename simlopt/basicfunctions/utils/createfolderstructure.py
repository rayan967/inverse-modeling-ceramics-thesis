import os
from os import path
from pathlib import Path
from datetime import datetime
from datetime import date

def createfolders(execpath,foldername = None,commondir = None):

    if foldername == None:
        today = date.today()
        d1 = today.strftime("%d_%m_%Y")

        now = datetime.now()
        current_time = now.strftime("%H_%M_%S")
        execpathex = execpath + "/Sim_"+str(d1)+"_"+str(current_time)
    else:
        execpathex = execpath + foldername

    if commondir == None:
        common_dir = ["iteration_plots", "postprocessing_plots", "saved_data", "logs", "iteration_log"]
    if os.path.exists(execpath):
        for dir2 in common_dir:
            try: os.makedirs(os.path.join(execpathex,dir2))
            except OSError: pass
        return execpathex
    else:
        print("Executable path does not exist")
        return None
    
def createfoldername(*args):
    foldername = ''.join(str(arg)+"_" for arg in args)
    return foldername[:-1]