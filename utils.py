"""
Utilities for XDLreg

Author: Simon M. Hofmann | <[firstname].[lastname][at]pm.me> | 2021
"""

# %% Imports
import sys
import os
import pickle
import platform
import smtplib
import ssl
import subprocess  # i.a. execute shell commands from python
from datetime import datetime, timedelta
# import time
from functools import wraps
from math import floor, log
from pathlib import Path
import gzip
import numpy as np
import psutil
# import getpass


# %% Paths << o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

def setwd(new_dir):
    _root_path = "XDLreg"

    assert _root_path in os.getcwd(), \
        f'Current working dir "{os.getcwd()}" is outside of project "{_root_path}".'

    _root_path = os.getcwd().split(_root_path)[0] + _root_path

    # Remove '/' if new_dir == 'folder/' OR '/folder'
    new_dir = "".join(new_dir.split("/"))

    cprint(f"Current working dir:\t{os.getcwd()}", 'b')

    found = False if new_dir != os.getcwd().split("/")[-1] else True

    # First look down the tree
    if not found:
        # Note: This works only for unique folder names
        for path, _, _ in os.walk(_root_path):  # 2. '_' == files
            # print(path, j, files)
            if new_dir in path:
                os.chdir(path)
                found = True
                break

        if found:
            cprint(f"New working dir:\t{os.getcwd()}\n", 'y')
        else:
            cprint(f"Given folder not found. Working dir remains:\t{os.getcwd()}\n", 'r')


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


# %% Timer << o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
# Use also ipython %timeit funct(), or %prun funct()
# or %load_ext line_profiler; %lprun -f func -f sub_func -f func()  # line-by-line (here multi fct output)
# or %load_ext memory_profiler; %mprun -f func -f sub_func -f func()  # for memory
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


def average_time(list_of_timestamps, in_timedelta=True):
    """
    Method to average time of a list of time-stamps. Necessary for Python 2.
    In Python3 simply: np.mean([timedelta(0, 20), ... , timedelta(0, 32)])

    :param list_of_timestamps: list of time-stamps
    :param in_timedelta: whether to return in timedelta-format.
    :return: average time
    """
    mean_time = sum(list_of_timestamps, timedelta()).total_seconds() / len(list_of_timestamps)

    if in_timedelta:
        mean_time = timedelta(seconds=mean_time)

    return mean_time


def try_funct(funct):
    """
    try wrapped function, if exception: tell user, but continue

    Usage:
        @try_funct
        def abc(a, b, c):
            return a+b+c

        abc(1, 2, 3)  # runs normally
        abc(1, "no int", 3)  # throws exception
    """

    @wraps(funct)
    def wrapper(*args, **kwargs):

        try:
            output = funct(*args, **kwargs)  # == function()

            return output

        except Exception:
            cprint(f"Function {funct.__name__} couldn't be successfully executed!", "r")

    return wrapper


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

def sort_mat_by_mat(mat, mat_idx):
    """
    mat         mat_idx         sorted mat
    [[1,2,3],   [[1,2,0],  ==>  [[2,3,1],
     [4,5,6]]    [2,0,1]]  ==>   [6,4,5]]

    :param mat: matrix to be sorted by rows of mat_idx
    :param mat_idx: matrix with corresponding indices
    :return: sorted matrix
    """
    mat_idx = mat_idx.astype(int)
    assert mat.shape == mat_idx.shape, "Matrices must be the same shape"
    n_rows = mat.shape[0]

    sorted_mat = np.zeros(shape=mat.shape)

    for row in range(n_rows):
        sorted_mat[row, :] = inverse_indexing(arr=mat[row, :], idx=mat_idx[row, :])
        # sorted_mat[row, :] = mat[row, :][mat_idx[row, :]]

    return sorted_mat


def inverse_indexing(arr, idx):
    """
    Inverse indexing of array (e.g., [16.,2.,4.]) to its origin (e.g., [2.,4.,16.])
    given the former index-vector (here: [2,0,1]).
    :param arr: altered array
    :param idx: former indexing vector
    :return: recovered array
    """
    inv_arr = np.repeat(np.nan, len(arr))
    for i, ix in enumerate(idx):
        inv_arr[ix] = arr[i]
    return inv_arr


def interpolate_nan(arr_with_nan):
    """
    Return array with linearly interpolated values.
    :param arr_with_nan: array with missing values
    :return: updated array
    """
    missing_idx = np.where(np.isnan(arr_with_nan))[0]

    if len(missing_idx) == 0:
        print("There are no nan-values in the given vector.")

    else:
        for midx in missing_idx:
            # if the first value is missing take average of the next 5sec
            if midx == 0:
                arr_with_nan[midx] = np.nanmean(arr_with_nan[midx + 1: midx + 1 + 5])
            # Else Take the mean of the two adjacent values
            else:  # midx > 0
                if np.isnan(arr_with_nan[midx]):  # Check whether still missing (see linspace fill below)
                    if not np.isnan(arr_with_nan[midx + 1]):  # we coming from below
                        arr_with_nan[midx] = np.mean([arr_with_nan[midx - 1], arr_with_nan[midx + 1]])
                    else:  # next value is also missing
                        count = 0
                        while True:
                            if np.isnan(arr_with_nan[midx + 1 + count]):
                                count += 1
                            else:
                                break

                        fillvec = np.linspace(start=arr_with_nan[midx - 1], stop=arr_with_nan[midx + 1 + count],
                                              num=3 + count)[1:-1]

                        assert len(fillvec) == 1 + count, "Implementation error at interpolation"

                        arr_with_nan[midx: midx + count + 1] = fillvec

        print("Interpolated {} values.".format(len(missing_idx)))

    updated_array = arr_with_nan

    return updated_array


def split_in_n_bins(a, n=5, attribute_remainder=True):
    """split in 3 bins and attribute remainder equally: [1,2,3,4,5,6,7,8] => [1,2,7], [3,4,8], [5,6]"""
    size = len(a)//n
    split = np.split(a, np.arange(size, len(a), size))

    if attribute_remainder and (len(split) != n):
        att_i = 0
        remainder = list(split.pop(-1))
        while len(remainder) > 0:
            split[att_i] = np.append(split[att_i], remainder.pop(0))
            att_i += 1  # can't overflow
    elif len(split) != n:
        cprint(f"{len(split[-1])} remainder were put in extra bin. Return {len(split)} bins instead of "
               f"{n}.", col='y')

    return split


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

def ol(string, wide_bar=True):
    """Create overline over given string or character"""
    bw = "\u0305" if wide_bar else "\u0304"  # 0305: wide; 0304: smaller
    return "".join([f"{char}{bw}" for char in string])


def ss(string_with_nr, sub=True):
    """Translates these '0123456789()' into subscript or superscript and vice versa"""

    # TODO also for chars
    # https://en.wikipedia.org/wiki/Unicode_subscripts_and_superscripts
    # s = "\u2082" if sub else "\2074"  # 2082: subscript; 2074: sup-script

    subs = str.maketrans("0123456789()₀₁₂₃₄₅₆₇₈₉₍₎", "₀₁₂₃₄₅₆₇₈₉₍₎0123456789()")
    sups = str.maketrans("0123456789()⁰¹²³⁴⁵⁶⁷⁸⁹⁽⁾", "⁰¹²³⁴⁵⁶⁷⁸⁹⁽⁾0123456789()")

    return string_with_nr.translate(subs if sub else sups)


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


# print(Bcolors.HEADER + "Header: No active frommets remain. Continue?" + Bcolors.ENDC)
# print(Bcolors.OKBLUE + "Ok Blue: No active frommets remain. Continue?" + Bcolors.ENDC)
# print(Bcolors.OKGREEN + "Ok Green: No active frommets remain. Continue?" + Bcolors.ENDC)
# print(Bcolors.WARNING + "Warning: No active frommets remain. Continue?" + Bcolors.ENDC)
# print(Bcolors.FAIL + "Fail: No active frommets remain. Continue?" + Bcolors.ENDC)
# print(Bcolors.ENDC + "Endc: No active frommets remain. Continue?" + Bcolors.ENDC)
# print(Bcolors.BOLD + "Bold: No active frommets remain. Continue?" + Bcolors.ENDC)
# print(Bcolors.UNDERLINE + "Underline: No active frommets remain. Continue?" + Bcolors.ENDC)
# print(Bcolors.UNDERLINE + Bcolors.BOLD + Bcolors.WARNING +
#       "Underline: No active frommets remain. Continue?" + Bcolors.ENDC)


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


def block_print():
    """Disable print outs"""
    sys.stdout = open(os.devnull, 'w')


def enable_print():
    """Restore & enable print outs"""
    sys.stdout = sys.__stdout__

# print('This will print')
# block_print()
# print("This won't")
# enable_print()
# print("This will too")


def suppress_print(func):
    """Suppresses print within given function"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        block_print()
        output = func(*args, **kwargs)  # should be only a print command
        enable_print()

        return output

    return wrapper

# @suppress_print
# def abc():
#     print("hello")
#     return 2
# abc()
# print("Na sowas der grüßt ja gar nicht!")


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


def check_system():
    if platform.system().lower() == "linux" and platform.node() != "experience":  # == sys.platform
        current_system = "MPI"

    elif platform.node() == "experience":
        current_system = "HHI"

    elif platform.system().lower() == "darwin":
        current_system = "mymac"  # mymac

    else:
        raise SystemError("Unknown compute system.")

    return current_system


def only_mpi(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if check_system() == "MPI":
            return func(*args, **kwargs)  # should be only a print command
        else:
            raise EnvironmentError(f"Function '{func.__name__}()' can only be executed on MPI server!")

    return wrapper


def check_executor(return_bash_bool=False):
    # Check whether scipt is run via bash
    ppid = os.getppid()
    bash = psutil.Process(ppid).name() == "bash"
    print("Current script{} executed via: {}{}{}".format(
        ' {}{}{}'.format(Bcolors.WARNING, platform.sys.argv[0], Bcolors.ENDC) if bash else "",
        Bcolors.OKBLUE, psutil.Process(ppid).name(), Bcolors.ENDC))

    if return_bash_bool:
        return bash


def cln(factor=1):
    """Clean the console"""
    print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n" * factor)


# %% Email << o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

def email(subject, message, receiver="simon.hofmann@cbs.mpg.de"):
    """
    Sending emails with Python
    For gmail this must be activated: https://www.google.com/settings/security/lesssecureapps
    """

    sender_email = "bzoo.info@gmail.com"  # sender_email = "bzoo.info@googlemail.com"

    # getpass() prevents showing the typed input in shell in contrast to input()
    # pazwoerd = getpass.getpass("Type your password & press enter: ")
    with open("./TEMP/ohne_titel.txt", "r") as file:
        paz = file.read()
        paz = paz[::-1]
    with open("./TEMP/ohne_titel2.txt", "r") as file:
        woerd = file.read()
        woerd = woerd[::-1]
    pazwoerd = paz + woerd

    port = 465  # For SSL
    # Create a secure SSL context
    context = ssl.create_default_context()

    # Create email text and its subject
    email_text = f"""Subject: {subject}


    {message}."""
    # 2 empty lines after 'Subject:' ensures separation of header and body

    # Connect to server and send email
    with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
        # server.ehlo()
        server.login(sender_email, pazwoerd)
        server.sendmail(sender_email, receiver, email_text)

# email(subject="Test mail", "This is a test mail! Tell me whether you like it.")


# %% OS << o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

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
        cprint("Given folder '{}' doesn't exist.".format(parent_path), "r")


# %% Save objects externally & load them << o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<

@function_timed
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


@function_timed
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


def memory_in_use():
    # psutil.virtual_memory() returns bytes
    print(f"{psutil.virtual_memory().used / (10 ** 9):.3f} GB ({psutil.virtual_memory().percent} %) of "
          f"memory are used.")


def free_memory(variable=None, verbose=False):
    """
    This functions frees memory of (unsigned) objects. If variable (i.e. object) is given, it deletes it
    from namespace and memory.
    Source: https://stackoverflow.com/questions/1316767/how-can-i-explicitly-free-memory-in-python

    Alternatively use subprocesses like this:
        import concurrent.futures
        def dfprocessing_func(data):
            ...
            return df

        with concurrent.futures.ThreadPoolExecutor(1) as executor:
            df = executor.map(dfprocessing_func, [data])[0]

    """
    import gc

    if verbose:
        print("Before cleaning memory ...")
        memory_in_use()

    if variable is not None:
        if isinstance(variable, (tuple, list)):
            for var in variable:
                del var
        if isinstance(variable, dict):
            for key in list(variable.keys()):
                del variable[key]
        del variable
    gc.collect()

    if verbose:
        print("After cleaning memory ...")
        memory_in_use()


# %% Compute tests << o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<

def run_gpu_test():
    """Test GPU implementation"""
    import tensorflow as tf

    print("GPU device is available:",
          tf.test.is_gpu_available())  # Returns True iff a gpu device of the requested kind is available.

    # TODO fine-tune
    compute_on = '/gpu:0' if tf.test.is_gpu_available() else '/cpu:0'

    with tf.device(compute_on):  # '/cpu:0'
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
        c = tf.matmul(a, b)
    # Creates a session with log_device_placement set to True.
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    # Runs the op.
    print(sess.run(c))

    if 'gpu' not in compute_on:
        print("Can only be run on CPU")

    # Another test
    # In case there is a GPU: test it against CPU
    device_name = ["/gpu:0", "/cpu:0"] if tf.test.is_gpu_available() else ["/cpu:0"]

    for device in device_name:
        for shape in [4500, 6000, 12000]:
            with tf.device(device):
                random_matrix = tf.random_uniform(shape=(shape, shape), minval=0, maxval=1)
                dot_operation = tf.matmul(random_matrix, tf.transpose(random_matrix))
                sum_operation = tf.reduce_sum(dot_operation)

            startTime = datetime.now()
            with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
                result = session.run(sum_operation)
                print(result)

            print("\n" * 3)
            print("Shape:", (shape, shape), "Device:", device)
            print("Time taken:", datetime.now() - startTime)
            print("\n" * 3)


def end():
    cprint("\n" + "*<o>*" * 9 + "  END  " + "*<o>*" * 9 + "\n", col='p', fm='bo')



# %% Config project paths << o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<

root_path = "/XDLreg"
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
root_path = list(filter(lambda x: root_path in x, sys.path))[0].split(root_path)[0] + root_path

# for subfold in ["PumpkinNet", "LRP"]:
#     if os.path.join(root_path, subfold) not in sys.path:
#         sys.path.append(os.path.join(root_path, subfold))

# setwd("PumpkinNet")

# <<<<<<<<<<< ooo >>>>>>>>>>>>>> ooo <<<<<<<<<<< ooo >>>>>>>>>>>>>> ooo <<<<<<<<<<< ooo >>>>>>>>>>>>>> END
