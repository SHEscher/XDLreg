# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Some functions to visualize images and relevance maps.

Author: Simon M. Hofmann | <[firstname].[lastname][at]pm.me> | 2021
"""
# %% Import
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# %% Global vars << o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

planes = ["sagittal/longitudinal", "transverse/superior/horizontal", "coronal/posterior/frontal"]


# %% ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<

def find_edges(x3d, sl=False):
    # print("x3d shape", x3d.shape)

    bg = x3d.min()  # usually: 0

    # # Find planes with first brain-data (i.e. being not black)
    # Find 'lower' planes (i.e. low, left, back, respectively)
    il, jl, kl = 0, 0, 0
    while np.all(x3d[il, :, :] == bg):  # sagittal slide
        il += 1
    while np.all(x3d[:, jl, :] == bg):  # transverse slide
        jl += 1
    while np.all(x3d[:, :, kl] == bg):  # coronal/posterior/frontal
        kl += 1

    # Find 'upper' planes (i.e. upper, right, front, respectively)
    iu, ju, ku = np.array(x3d.shape) - 1
    while np.all(x3d[iu, :, :] == bg):  # sagittal/longitudinal
        iu -= 1
    while np.all(x3d[:, ju, :] == bg):  # transverse/inferior/horizontal
        ju -= 1
    while np.all(x3d[:, :, ku] == bg):  # coronal/posterior/frontal
        ku -= 1

    # plt.imshow(x3d[:, :, kl])  # first brain
    # plt.imshow(x3d[:, :, kl-1])  # black
    # plt.imshow(x3d[:, :, ku])  # first brain
    # plt.imshow(x3d[:, :, ku+1])  # black
    if sl:  # return slices
        return slice(il, iu + 1), slice(jl, ju + 1), slice(kl, ku + 1)

    else:  # return coordinates
        return il, iu, jl, ju, kl, ku


def get_brain_axes_length(x3d):

    tp = False
    sic = None
    if type(x3d) == tuple:  # (sic, brain)
        tp = True
        sic = x3d[0]
        x3d = x3d[1]

    il, iu, jl, ju, kl, ku = find_edges(x3d)
    axes_lengths = [iu + 1 - il, ju + 1 - jl, ku + 1 - kl]

    if tp:
        return sic, axes_lengths
    else:
        return axes_lengths


def max_of_axes(x3d):
    tp = False
    sic = None
    if type(x3d) == tuple:  # (sic, brain)
        tp = True
        sic = x3d[0]
        x3d = x3d[1]

    if isinstance(x3d, np.ndarray):
        longest_axis = np.max(get_brain_axes_length(x3d))
    else:
        longest_axis = np.nan

    if tp:
        return sic, longest_axis
    else:
        return longest_axis


def prep_save_folder(save_folder):
    save_folder = os.path.join(save_folder, "")  # adds '/' if not there

    if (not os.path.exists(save_folder)) and (len(save_folder) > 0):
        print("Create save folder:", os.path.abspath(save_folder))
        os.makedirs(save_folder)

    return save_folder


def plot_slice(mri, axis, idx_slice, edges=True, c_range=None, **kwargs):
    # In general: axis-call could work like the following:
    # axis==0: aslice = (idx_slice, slice(None), slice(None))  == (idx_slice, ...)
    # axis==1: aslice = (..., idx_slice, slice(None))
    # axis==2: aslice = (..., idx_slice)
    # mri[aslice]

    if edges:
        edges = find_edges(mri if mri.shape[-1] > 4 else mri[..., -1])
        # works for transparent (!) RGB image (x,y,z,4) and volume (x,y,z)

    # Set color range
    if c_range == "full":  # takes full possible spectrum
        imax = 255 if np.max(mri) > 1 else 1.
        imin = 0 if np.min(mri) >= 0 else -1.
    elif c_range == "single":  # takes min/max of given brain
        imax = np.max(mri)
        imin = np.min(mri)
    else:  # c_range=None
        assert c_range is None, "c_range must be 'full', 'single' or None."
        imax, imin = None, None

    # Get kwargs (which are not for imshow)
    crosshairs = kwargs.pop("crosshairs", True)
    ticks = kwargs.pop("ticks", False)

    if axis == 0:  # sagittal
        im = plt.imshow(mri[idx_slice, :, :], vmin=imin, vmax=imax, **kwargs)
        if edges:
            plt.hlines(edges[2] - 1, 2, mri.shape[1] - 2, colors="darkgrey", alpha=.3)  # == max edges
            plt.hlines(edges[3] + 1, 2, mri.shape[1] - 2, colors="darkgrey", alpha=.3)
            plt.vlines(edges[4] - 1, 2, mri.shape[0] - 2, colors="darkgrey", alpha=.3)
            plt.vlines(edges[5] + 1, 2, mri.shape[0] - 2, colors="darkgrey", alpha=.3)

    elif axis == 1:  # transverse / superior
        im = plt.imshow(np.rot90(mri[:, idx_slice, :], axes=(0, 1)), vmin=imin, vmax=imax,
                        **kwargs)
        if edges:
            plt.hlines(mri.shape[0] - edges[5] - 1, 2, mri.shape[0] - 2, colors="darkgrey", alpha=.3)
            plt.hlines(mri.shape[0] - edges[4] + 1, 2, mri.shape[0] - 2, colors="darkgrey", alpha=.3)
            plt.vlines(edges[0] - 1, 2, mri.shape[1] - 2, colors="darkgrey", alpha=.3)
            plt.vlines(edges[1] + 1, 2, mri.shape[1] - 2, colors="darkgrey", alpha=.3)

    elif axis == 2:  # # coronal / posterior
        im = plt.imshow(np.rot90(mri[:, :, idx_slice], axes=(1, 0)), vmin=imin, vmax=imax,
                        **kwargs)
        if edges:
            plt.hlines(edges[2] - 1, 2, mri.shape[0] - 2, colors="darkgrey", alpha=.3)
            plt.hlines(edges[3] + 1, 2, mri.shape[0] - 2, colors="darkgrey", alpha=.3)
            plt.vlines(mri.shape[1] - edges[1] - 1, 2, mri.shape[1] - 2, colors="darkgrey", alpha=.3)
            plt.vlines(mri.shape[1] - edges[0] + 1, 2, mri.shape[1] - 2, colors="darkgrey", alpha=.3)

    # Add mid-cross ('crosshairs')
    if crosshairs:
        plt.hlines(int(mri.shape[axis] / 2), 2, len(mri) - 2, colors="red", alpha=.3)
        plt.vlines(int(mri.shape[axis] / 2), 2, len(mri) - 2, colors="red", alpha=.3)

    if not ticks:
        plt.axis("off")

    return im


def plot_mid_slice(mri, axis=None, figname=None, edges=True, c_range=None,
                   save=False, save_folder="./TEMP/", **kwargs):
    """
    Plots mid 2d-slice of a given 3D-MRI. If no axis is given, plots for each axis its mid-slice.
    :param mri: 3D MRI
    :param axis: None: all 3 axes
    :param figname: name of figure
    :param edges: if True, draw edges around brain
    :param c_range: "full", "single", or None
    :param save: True/False
    :param save_folder: Where to save
    :param kwargs: arguments for plt.imshow(), & crosshairs: bool = True, ticks: bool = False)
    """

    save_folder = prep_save_folder(save_folder)

    # Get colorbar kwargs (if any)
    cbar = kwargs.pop("cbar", False)
    cbar_range = kwargs.pop("cbar_range") if ("cbar_range" in kwargs.keys() and cbar) else None
    # only if cbar is active
    suptitle = kwargs.pop("suptitle", None)

    # Set mid-slice index
    mid = int(np.round(mri.shape[0] / 2))

    # Begin plotting
    if axis is None:
        _fs = {"size": 10}  # define font size

        fig = plt.figure(num=f"{figname if figname else ''} MRI mid-slice", figsize=(12, 4))
        if suptitle:
            fig.suptitle(suptitle, fontsize=14)

        # # Planes
        ims = []
        axs = []
        for ip, plane in enumerate(planes):
            axs.append(fig.add_subplot(1, 3, ip + 1))
            ims.append(plot_slice(mri, axis=ip, idx_slice=mid, edges=edges, c_range=c_range, **kwargs))
            plt.title(planes[ip], fontdict=_fs)

            divider = make_axes_locatable(axs[ip])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cax.axis("off")
            if cbar and ip == len(planes) - 1:
                caxbar = fig.colorbar(ims[-1], ax=cax, fraction=0.048, pad=0.04)  # shrink=0.8, aspect=50)
                if cbar_range:
                    caxbar.set_ticks(np.linspace(0, 1, 7), True)
                    caxbar.ax.set_yticklabels(
                        labels=[f"{tick:.2g}" for tick in np.linspace(cbar_range[0], cbar_range[1],
                                                                      len(caxbar.get_ticks()))])

        if save:
            fig.savefig(fname=f"{save_folder}{figname if figname else ''} MRI mid-slice.png")
            plt.close()
        else:
            plt.show()

    else:  # If specific axis to plot
        assert axis in range(3), f"axis={axis} is not valid. Take 0, 1 or 2."

        axis_name = "sagittal"
        if axis != 0:
            axis_name = "transverse" if axis == 1 else "coronal"
        fig = plt.figure(f"{figname if figname else ''} {axis_name} mid-slice")
        im = plot_slice(mri, axis, idx_slice=mid, edges=edges, c_range=c_range, **kwargs)
        if cbar:
            caxbar = fig.colorbar(im, fraction=0.048, pad=0.04)  # shrink=0.8, aspect=50)
            if cbar_range:
                caxbar.set_ticks(np.linspace(0, 1, 7), True)
                caxbar.ax.set_yticklabels(
                    labels=[f"{tick:.2g}" for tick in np.linspace(cbar_range[0], cbar_range[1],
                                                                  len(caxbar.get_ticks()))])

        plt.tight_layout()

        if save:
            fig.savefig(fname=f"{save_folder}{figname if figname else ''} {axis_name} mid-slice.png")
            plt.close()
        else:
            plt.show()

# << o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< END
