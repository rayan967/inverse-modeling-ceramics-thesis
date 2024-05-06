from pathlib import Path
from datetime import datetime
from datetime import date


def createfolders(execpath, foldername=None, commondir=None):
    if foldername is None:
        today = date.today()
        d1 = today.strftime("%d_%m_%Y")
        now = datetime.now()
        current_time = now.strftime("%H_%M_%S")
        execpath.mkdir(parents=True, exist_ok=True)
        execpathex = execpath / f"Sim_{d1}_{current_time}"
    else:
        execpath.mkdir(parents=True, exist_ok=True)
        execpathex = execpath / foldername

    if commondir is None:
        commondir = ["iteration_plots", "postprocessing_plots", "saved_data", "logs", "iteration_log"]

    if not execpath.exists():
        print("Executable path does not exist")
        return None

    for dir2 in commondir:
        (execpathex / dir2).mkdir(parents=True, exist_ok=True)

    return str(execpathex)

def createfoldername(*args):
    foldername = ''.join(str(arg)+"_" for arg in args)
    return foldername[:-1]