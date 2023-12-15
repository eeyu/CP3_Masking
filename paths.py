import os
import tkinter.filedialog
from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename
import torch

HOME_PATH = os.path.dirname(os.path.abspath(__file__)) + "/"
PARAM_PATH = HOME_PATH + "parameters" + "/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check if gpu/tpu is available
def select_file(init_dir=HOME_PATH, choose_file=True):
    Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    if choose_file:
        filename = askopenfilename(initialdir=init_dir,
                                   defaultextension="txt")  # show an "Open" dialog box and return the path to the selected file
        return filename
    else:
        foldername = tkinter.filedialog.askdirectory(initialdir=init_dir)
        return foldername

def get_favorite_model_path(name):
    path = PARAM_PATH + name + "/"
    contents = os.listdir(path)[0]
    file_name = path + contents
    return file_name
