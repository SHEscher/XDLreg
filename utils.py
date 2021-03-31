"""
Utilities for XDLreg

Author: Simon M. Hofmann | <[firstname].[lastname][at]pm.me> | 2021
"""

# %% Imports
import sys
import os
import pickle
import platform
import subprocess
from datetime import datetime, timedelta
from functools import wraps
from math import floor, log
from pathlib import Path
import gzip
import numpy as np


# %% OS & Paths << o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

class DisplayablePath(object):
    """
    With honourable mention to 'abstrus':
    https://stackoverflow.com/questions/9727673/list-directory-tree-structure-in-python

    Notes
        * This uses recursion. It will raise a RecursionError on really deep folder trees
        * The tree is lazily evaluated. It should behave well on really wide folder trees.
          Immediate children of a given folder are not lazily evaluated, though.
    """

    display_filename_prefix_middle = '├──'
    display_filename_prefix_last = '└──'
    display_parent_prefix_middle = '    '
    display_parent_prefix_last = '│   '

    def __init__(self, path, parent_path, is_last):
        self.path = Path(str(path))
        self.parent = parent_path
        self.is_last = is_last
        if self.parent:
            self.depth = self.parent.depth + 1
        else:
            self.depth = 0

    @property
    def displayname(self):
        if self.path.is_dir():
            return self.path.name + '/'
        return self.path.name

    @classmethod
    def make_tree(cls, root, parent=None, is_last=False, criteria=None):
        root = Path(str(root))
        criteria = criteria or cls._default_criteria

        displayable_root = cls(root, parent, is_last)
        yield displayable_root

        children = sorted(list(path
                               for path in root.iterdir()
                               if criteria(path)),
                          key=lambda s: str(s).lower())
        count = 1
        for path in children:
            is_last = count == len(children)
            if path.is_dir():
                yield from cls.make_tree(path,
                                         parent=displayable_root,
                                         is_last=is_last,
                                         criteria=criteria)
            else:
                yield cls(path, displayable_root, is_last)
            count += 1

    @classmethod
    def _default_criteria(cls, path):
        return True

    @property
    def displayname(self):
        if self.path.is_dir():
            return self.path.name + '/'
        return self.path.name

    def displayable(self):
        if self.parent is None:
            return self.displayname

        _filename_prefix = (self.display_filename_prefix_last
                            if self.is_last
                            else self.display_filename_prefix_middle)

        parts = ['{!s} {!s}'.format(_filename_prefix,
                                    self.displayname)]

        parent = self.parent
        while parent and parent.parent is not None:
            parts.append(self.display_parent_prefix_middle
                         if parent.is_last
                         else self.display_parent_prefix_last)
            parent = parent.parent

        return ''.join(reversed(parts))


def tree(directory):
    """
    Use the same way as shell command `tree`.
    This leads to output such as:

        directory/
        ├── _static/
        │   ├── embedded/
        │   │   ├── deep_file
        │   │   └── very/
        │   │       └── deep/
        │   │           └── folder/
        │   │               └── very_deep_file
        │   └── less_deep_file
        ├── about.rst
        ├── conf.py
        └── index.rst

    """
    paths = DisplayablePath.make_tree(Path(directory))
    for path in paths:
        print(path.displayable())


def find(fname, folder=".", typ="file",
         exclusive=True, fullname=True, abs_path=False, verbose=True):
    """
    Find file(s) in given folder

    :param fname: full filename OR consecutive part of it
    :param folder: root folder to search
    :param typ: 'file' or folder 'dir'
    :param exclusive: only return path when only one file was found
    :param fullname: True: consider only files which exactly match the given fname
    :param abs_path: False: return relative path(s); True: return absolute path(s)
    :param verbose: Report findings

    :return: path to file OR list of paths, OR None
    """

    ctn_found = 0
    findings = []
    for root, dirs, files in os.walk(folder):
        search_in = files if typ.lower() == "file" else dirs
        for f in search_in:
            if (fname == f) if fullname else (fname in f):
                ffile = os.path.join(root, f)  # found file

                if abs_path:
                    ffile = os.path.abspath(ffile)

                findings.append(ffile)
                ctn_found += 1

    if exclusive and len(findings) > 1:
        if verbose:
            cprint(f"\nFound several {typ}s for given fname='{fname}', please specify:", 'y')
            print("", *findings, sep="\n\t>> ")
        return None

    elif not exclusive and len(findings) > 1:
        if verbose:
            cprint(f"\nFound several {typ}s for given fname='{fname}', return list of {typ} paths", 'y')
        return findings

    elif len(findings) == 0:
        if verbose:
            cprint(f"\nDid not find any {typ} for given fname='{fname}', return None", 'y')
        return None

    else:
        if verbose:
            cprint(f"\nFound this {typ}: '{findings[0]}'", 'y')
        return findings[0]


def open_folder(path):
    """Open specific folder in Finder. Can also be a file"""
    if platform.system() == "Windows":  # for Windows
        os.startfile(path)
    elif platform.system() == 'Darwin':  # ≈ sys.platform = 'darwin' | for Mac
        subprocess.Popen(["open", path])
    else:  # for 'Linux'
        subprocess.Popen(["xdg-open", path])


def browse_files(initialdir=None, filetypes=None):
    from tkinter import Tk
    from tkinter.filedialog import askopenfilename
    root = Tk()
    root.withdraw()

    kwargs = {}
    if initialdir:
        kwargs.update({"initialdir": initialdir})
    if filetypes:
        kwargs.update({"filetypes": [(filetypes + " File", "*." + filetypes.lower())]})

    filename = askopenfilename(parent=root, **kwargs)
    return filename


def delete_dir_and_files(parent_path):
    """
    Delete given folder and all subfolders and files.

    os.walk() returns three values on each iteration of the loop:
        i)    The name of the current folder: dirpath
        ii)   A list of folders in the current folder: dirnames
        iii)  A list of files in the current folder: files

    :param parent_path: path to parent folder
    :return:
    """
    # Print the effected files and subfolders
    if Path(parent_path).exists():
        print(f"\nFollowing (sub-)folders and files of parent folder '{parent_path}' would be deleted:")
        for file in Path(parent_path).glob("**/*"):
            cprint(f"{file}", "b")

        # Double checK: Ask whether to delete
        delete = ask_true_false("Do you want to delete this tree and corresponding files?", "r")

        if delete:
            # Delete all folders and files in the tree
            for dirpath, dirnames, files in os.walk(parent_path, topdown=False):  # start from bottom
                cprint(f"Remove folder: {dirpath}", "r")
                for file_name in files:
                    cprint(f"Remove file: {file_name}", "r")  # f style  (for Python > 3.5)
                    os.remove(os.path.join(dirpath, file_name))
                os.rmdir(dirpath)
        else:
            cprint("Tree and files won't be deleted!", "b")

    else:
        cprint(f"Given folder '{parent_path}' doesn't exist.", col="r")


# %% Timer << o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

def chop_microseconds(delta):
    return delta - timedelta(microseconds=delta.microseconds)


def function_timed(dry_funct=None, ms=None):
    """
    Times the processing duration of wrapped function.

    Way to use:

    Following returns duration without micro-seconds:

    @function_timed
    def abc():
        return 2+2

    Following returns also micro-seconds:

    @function_timed(ms=True)
    def abcd():
        return 2+2

    :param dry_funct: parameter can be ignored. Results in output without micro-seconds
    :param ms: if micro-seconds should be printed, set to True
    :return:
    """

    def _function_timed(funct):

        @wraps(funct)
        def wrapper(*args, **kwargs):
            start_timer = datetime.now()

            # whether to suppress wrapper: use functimer=False in main funct
            w = kwargs.pop("functimer", True)

            output = funct(*args, **kwargs)

            duration = datetime.now() - start_timer

            if w:
                if ms:
                    print("\nProcessing time of {}: {} [h:m:s:ms]".format(funct.__name__, duration))

                else:
                    print("\nProcessing time of {}: {} [h:m:s]".format(funct.__name__,
                                                                       chop_microseconds(duration)))

            return output

        return wrapper

    if dry_funct:
        return _function_timed(dry_funct)

    return _function_timed


def loop_timer(start_time, loop_length, loop_idx, loop_name: str = None, add_daytime=False):
    """
    Estimates remaining time to run through given loop.
    Function must be placed at the end of the loop inside.
    Before the loop, take start time by  start_time=datetime.now()
    Provide position within in the loop via enumerate()
    In the form:
        '
        start = datetime.now()
        for idx, ... in enumerate(iterable):
            ... operations ...

            loop_timer(start_time=start, loop_length=len(iterable), loop_idx=idx)
        '
    :param start_time: time at start of the loop
    :param loop_length: total length of loop-object
    :param loop_idx: position within loop
    :param loop_name: provide name of loop for print
    :param add_daytime: add leading day time to print-out
    """
    _idx = loop_idx
    ll = loop_length

    duration = datetime.now() - start_time
    rest_duration = chop_microseconds(duration / (_idx + 1) * (ll - _idx - 1))

    loop_name = "" if loop_name is None else " of " + loop_name

    nowtime = f"{datetime.now().replace(microsecond=0)} | " if add_daytime else ""
    string = f"{nowtime}Estimated time to loop over rest{loop_name}: {rest_duration} [hh:mm:ss]\t " \
             f"[ {'*' * int((_idx + 1) / ll * 30)}{'.' * (30 - int((_idx + 1) / ll * 30))} ] " \
             f"{(_idx + 1) / ll * 100:.2f} %"

    print(string, '\r' if (_idx + 1) != ll else '\n', end="")

    if (_idx + 1) == ll:
        cprint(f"{nowtime}Total duration of loop{loop_name.split(' of')[-1]}: "
               f"{chop_microseconds(duration)} [hh:mm:ss]\n", 'b')


# %% Normalizer & numerics << o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<

def normalize(array, lower_bound, upper_bound, global_min=None, global_max=None):
    """
    Min-Max-Scaling: Normalizes Input Array to lower and upper bound
    :param array: To be transformed array
    :param upper_bound: Upper Bound b
    :param global_min: if array is part of a larger tensor, normalize w.r.t. glabal min and ...
    :param global_max: ... global max (i.e. tensor min/max)
    :return: normalized array
    """

    assert lower_bound < upper_bound, "lower_bound must be < upper_bound"

    a, b = lower_bound, upper_bound
    mini = np.nanmin(array) if global_min is None else global_min
    maxi = np.nanmax(array) if global_max is None else global_max

    normed_array = (b - a) * ((array - mini) / (maxi - mini)) + a

    return normed_array


def denormalize(array, denorm_minmax, norm_minmax):
    """
    :param array: array to be de-normalized
    :param denorm_minmax: tuple of (min, max) of de-normalized (target) vector
    :param norm_minmax: tuple of (min, max) of normalized vector
    :return: de-normalized value
    """
    array = np.array(array)

    dnmin, dnmax = denorm_minmax
    nmin, nmax = norm_minmax

    assert nmin < nmax, "norm_minmax must be tuple (min, max), where min < max"
    assert dnmin < dnmax, "denorm_minmax must be tuple (min, max), where min < max"

    denormed_array = (array - nmin) / (nmax - nmin) * (dnmax - dnmin) + dnmin

    return denormed_array


def z_score(array):
    """
    Create z-score
    :return: z-score array
    """
    sub_mean = np.nanmean(array)
    sub_std = np.nanstd(array)
    z_array = (array - sub_mean) / sub_std

    return z_array


def getfactors(n):
    # Create an empty list for factors
    factors = []

    # Loop over all factors
    for i in range(1, n + 1):
        if n % i == 0:
            factors.append(i)

    # Return the list of
    return factors


def oom(number):
    """Return order of magnitude of given number"""
    return floor(log(number, 10))


# %% Sorter << o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<

def get_n_cols_and_rows(n_plots, squary=True):
    """Define figure grid-size: with rpl x cpl cells """
    facs = getfactors(n_plots)
    if len(facs) <= 2 or squary:  # prime or squary
        rpl = 1
        cpl = 1
        while (rpl * cpl) < n_plots:
            if rpl == cpl:
                rpl += 1
            else:
                cpl += 1
    else:
        rpl = facs[len(facs) // 2]
        cpl = n_plots // rpl
    ndiff = rpl * cpl - n_plots
    if ndiff > 0:
        cprint(f"There will {ndiff} empty plot.", 'y')
    return rpl, cpl


# %% Color prints & I/O << o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

class Bcolors:
    """
    Colours print-commands in Console
    Usage:
    print(Bcolors.HEADER + "Warning: No active frommets remain. Continue?" + Bcolors.ENDC)
    print(Bcolors.OKBLUE + "Warning: No active frommets remain. Continue?" + Bcolors.ENDC)

    For more:

        CSELECTED = '\33[7m'

        CBLACK  = '\33[30m'
        CRED    = '\33[31m'
        CGREEN  = '\33[32m'
        CYELLOW = '\33[33m'
        CBLUE   = '\33[34m'
        CVIOLET = '\33[35m'
        CBEIGE  = '\33[36m'
        CWHITE  = '\33[37m'

        CBLACKBG  = '\33[40m'
        CREDBG    = '\33[41m'
        CGREENBG  = '\33[42m'
        CYELLOWBG = '\33[43m'
        CBLUEBG   = '\33[44m'
        CVIOLETBG = '\33[45m'
        CBEIGEBG  = '\33[46m'
        CWHITEBG  = '\33[47m'

        CGREY    = '\33[90m'
        CBEIGE2  = '\33[96m'
        CWHITE2  = '\33[97m'

        CGREYBG    = '\33[100m'
        CREDBG2    = '\33[101m'
        CGREENBG2  = '\33[102m'

        CYELLOWBG2 = '\33[103m'
        CBLUEBG2   = '\33[104m'
        CVIOLETBG2 = '\33[105m'
        CBEIGEBG2  = '\33[106m'
        CWHITEBG2  = '\33[107m'

    # For preview type:
    for i in [1, 4, 7] + list(range(30, 38)) + list(range(40, 48)) + list(range(90, 98)) + list(
            range(100, 108)):  # range(107+1)
        print(i, '\33[{}m'.format(i) + "ABC & abc" + '\33[0m')
    """

    HEADERPINK = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    UNDERLINE = '\033[4m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'  # this is necessary in the end to reset to default print

    DICT = {'p': HEADERPINK, 'b': OKBLUE, 'g': OKGREEN, 'y': WARNING, 'r': FAIL,
            'ul': UNDERLINE, 'bo': BOLD}


def cprint(string, col=None, fm=None, ts=False):
    """
    Colorize and format print-out. Add leading time-stamp (fs) if required
    :param string: print message
    :param col: color:'p'(ink), 'b'(lue), 'g'(reen), 'y'(ellow), OR 'r'(ed)
    :param fm: format: 'ul'(:underline) OR 'bo'(:bold)
    :param ts: add leading time-stamp
    :return:
    """

    if col:
        col = col[0].lower()
        assert col in ['p', 'b', 'g', 'y', 'r'], \
            "col must be 'p'(ink), 'b'(lue), 'g'(reen), 'y'(ellow), 'r'(ed)"
        col = Bcolors.DICT[col]

    if fm:
        fm = fm[0:2].lower()
        assert fm in ['ul', 'bo'], "fm must be 'ul'(:underline), 'bo'(:bold)"
        fm = Bcolors.DICT[fm]

    if ts:
        pfx = ""  # collecting leading indent or new line
        while string.startswith('\n') or string.startswith('\t'):
            pfx += string[:1]
            string = string[1:]
        string = f"{pfx}{datetime.now():%Y-%m-%d %H:%M:%S} | " + string

    # print given string with formatting
    print(f"{col if col else ''}{fm if fm else ''}{string}{Bcolors.ENDC}")
    # print("{}{}".format(col if col else "",
    #                     fm if fm else "") + string + Bcolors.ENDC)


def cinput(string, col=None):
    if col:
        col = col[0].lower()
        assert col in ['p', 'b', 'g', 'y', 'r'], \
            "col must be 'p'(ink), 'b'(lue), 'g'(reen), 'y'(ellow), 'r'(ed)"
        col = Bcolors.DICT[col]

    # input(given string) with formatting
    return input("{}".format(col if col else "") + string + Bcolors.ENDC)


def true_false_request(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        func(*args, **kwargs)  # should be only a print command

        tof = input("(T)rue or (F)alse: ").lower()
        assert tof in ["true", "false", "t", "f"], "Must be 'T', 't' or 'T/true', or 'F', 'f', 'F/false'"
        output = True if tof in "true" else False

        return output

    return wrapper


@true_false_request
def ask_true_false(question, col="b"):
    """
    Ask user for input for given True-or-False question
    :param question: str
    :param col: print-colour of question
    :return: answer
    """
    cprint(question, col)


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# %% Save objects externally & load them << o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<

def save_obj(obj, name, folder='./TEMP/', hp=True, as_zip=False):

    # Remove suffix here, if there is e.g. "*.gz.pkl":
    if name.endswith(".gz"):
        name = name[:-3]
    if name.endswith(".pkl"):
        name = name[:-4]

    p2save = os.path.join(folder, name)

    # Create parent folder if not available
    parent_dir = "/".join(p2save.split("/")[:-1])
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    open_it, suffix = (gzip.open, ".pkl.gz") if as_zip else (open, ".pkl")

    with open_it(p2save + suffix, 'wb') as f:
        if hp:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        else:
            p = pickle.Pickler(f)
            p.fast = True
            p.dump(obj)


def load_obj(name, folder='./TEMP/'):

    if not (name.endswith(".pkl") or name.endswith(".pkl.gz")):
        name += ".pkl"

    # Check whether also zipped version is available: "*.pkl.gz"
    name = list(Path(folder).glob(name + "*"))

    if len(name) == 0:
        raise FileNotFoundError(f"In '{folder}' no file with given name was found!")
    elif len(name) > 1:  # len() == 2
        # There is a zipped & unzipped version, take the unzipped version
        name = [p2f for p2f in name if ".gz" not in str(p2f)]
    name = name[0]  # un-list

    open_it = gzip.open if str(name).endswith(".gz") else open

    with open_it(name, 'rb') as f:
        return pickle.load(f)


def end():
    cprint("\n" + "*<o>*" * 9 + "  END  " + "*<o>*" * 9 + "\n", col='p', fm='bo')


# %% Config project paths << o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<

# TODO check whether necessary
root_path = "XDLreg"

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
root_path = list(filter(lambda x: root_path in x, sys.path))[0].split(root_path)[0] + root_path

p2data = os.path.join(root_path, "Data")
p2results = os.path.join(root_path, "Results")

# <<<<<<<<<<< ooo >>>>>>>>>>>>>> ooo <<<<<<<<<<< ooo >>>>>>>>>>>>>> ooo <<<<<<<<<<< ooo >>>>>>>>>>>>>> END
