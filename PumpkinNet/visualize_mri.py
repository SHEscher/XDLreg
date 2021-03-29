# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Some functions to visualize MRIs

Author: Simon M. Hofmann | <[firstname].[lastname][at]cbs.mpg.de> | 2018-2020
"""
# %% Import
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from prune_image import *


# %% ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<
planes = ["sagittal/longitudinal", "transverse/superior/horizontal", "coronal/posterior/frontal"]


# %% ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<

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
        # planes = ["sagittal/longitudinal", "transverse/superior/horizontal","coronal/posterior/frontal"]
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


# TODO For fancy slicing, check https://docs.pyvista.org/examples/01-filter/slicing.html
def slice_through(mri, every=2, axis=0, figname=None, edges=True, c_range=None,
                  save=False, save_folder="./TEMP/", **kwargs):
    save_folder = prep_save_folder(save_folder)

    assert axis in range(3), f"axis={axis} is not valid. Take 0, 1 or 2."

    nrow_sq_grid = 5
    n_slices = int(mri.shape[0] / every)
    n_figs = int(np.round(n_slices / nrow_sq_grid ** 2))

    axis_name = "sagittal"
    if axis != 0:
        axis_name = "transverse" if axis == 1 else "coronal"

    fig_n = 1
    sub_n = 1
    fig = None  # init
    for scl in range(n_slices):
        if scl % (nrow_sq_grid ** 2) == 0:
            fig = plt.figure(f"{figname if figname else ''} {axis_name} slice-through {fig_n}|{n_figs}",
                             figsize=(10, 10))

        plt.subplot(nrow_sq_grid, nrow_sq_grid, sub_n)

        plot_slice(mri, axis, idx_slice=scl + (every - 1), edges=edges, c_range=c_range, **kwargs)

        plt.tight_layout()

        sub_n += 1

        if ((sub_n - 1) % (nrow_sq_grid ** 2) == 0) or (scl + 1 == n_slices):
            if save:
                fig.savefig(fname=f"{save_folder}{figname + ' ' if figname else ''}{axis_name} "
                                  f"slice-through {fig_n}|{n_figs}.png")
                plt.close()

            fig_n += 1
            sub_n = 1


def multi_slice_viewer(mri, axis=0, **kwargs):
    """Source: https://www.datacamp.com/community/tutorials/matplotlib-3d-volumetric-data"""

    # TODO [Optional!] 3 axes plot, similar to plot_mid_slice()

    def remove_keymap_conflicts(new_keys_set):
        for prop in plt.rcParams:
            if prop.startswith('keymap.'):
                keys = plt.rcParams[prop]
                remove_list = set(keys) & new_keys_set
                for key in remove_list:
                    keys.remove(key)

    def apply_rot(vol_slice):
        if axis > 0:
            axes = (0, 1) if axis == 1 else (1, 0)
            vol_slice = np.rot90(vol_slice, axes=axes)
        return vol_slice

    def previous_slice(ax_):
        volume = ax_.volume
        ax_.index = (ax_.index - 1) % volume.shape[axis]  # wrap around using %
        slice_ax = [slice(None)] * 3
        slice_ax[axis] = slice(ax_.index, ax_.index+1, None)
        ax_.images[0].set_array(apply_rot(volume[slice_ax].squeeze()))

    def next_slice(ax_):
        volume = ax_.volume
        ax_.index = (ax_.index + 1) % volume.shape[axis]
        slice_ax = [slice(None)] * 3
        slice_ax[axis] = slice(ax_.index, ax_.index + 1, None)
        ax_.images[0].set_array(apply_rot(volume[slice_ax].squeeze()))

    def process_key(event):
        _fig = event.canvas.figure
        _ax = _fig.axes[0]
        if event.key == 'j':
            previous_slice(_ax)
        elif event.key == 'k':
            next_slice(_ax)
        print(f"At slice: {_ax.index}\r", end="")
        _ax.set_title(_ax.get_title().split("slice")[0] + f"slice {_ax.index}")
        _fig.canvas.draw()

    # Execute
    remove_keymap_conflicts({'j', 'k'})
    cprint("\nPress 'j' and 'k' to slide through the volume.", 'y')

    # Prepare plot
    fig, ax = plt.subplots()

    # Unpack kwargs
    window_title = kwargs.pop('window_title', "MRI Slice Slider")
    fig.canvas.set_window_title(window_title)  # fig.canvas.get_window_title()

    cbar = kwargs.pop("cbar", False)
    cbar_range = kwargs.pop("cbar_range") if ("cbar_range" in kwargs.keys() and cbar) else None
    # cbar_range: only if cbar is active

    ax.volume = mri
    ax.index = mri.shape[axis] // 2
    ax.set_title(f"{planes[axis]} | slice {ax.index}")

    # Plot
    # im = ax.imshow(mri[ax.index], **kwargs)
    im = plot_slice(mri=mri, axis=axis, idx_slice=ax.index, **kwargs)

    if cbar:
        axbar = fig.colorbar(im)
        if cbar_range:
            axbar.ax.set_yticklabels(
                labels=[f"{tick:.2g}" for tick in np.linspace(cbar_range[0], cbar_range[1],
                                                              len(axbar.get_ticks()))])
    fig.canvas.mpl_connect('key_press_event', process_key)
    # plt.show()


# TODO get atlas and apply: see z_playground
# TODO if MNI space, allow to underlie mni template
def apply_atlas(mri):
    pass


# TODO transparent 3D plot: see z_playground
def plot_3d(mri):
    """Adapt: https://terbium.io/2017/12/matplotlib-3d/"""
    pass


def plot_transparent(plot_fct, img, cmap=None, **kwargs):
    cm_fct = eval(f'plt.cm.{plt.get_cmap().name if cmap is None else cmap}')
    colimg = cm_fct(img)  # == plt.cm.viridis(img), if default cmap (i.e., None):
    # set alpha to zero, where image zero
    colimg[..., -1][np.where(img == 0.0)] = 0.0  # TODO check if possible for rescaled images (-1, 1)
    plot_fct(colimg, **kwargs)


def close_plots():
    plt.close()
# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<  END
