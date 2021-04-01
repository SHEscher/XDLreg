"""
Different colormaps for heatmap plots.

Adaptation from apply_heatmap.py by Sebastian L. & Leander Weber (FH HHI, Berlin)

Author: Simon M. Hofmann | <[firstname].[lastname][at]pm.me> | 2021
"""
# %% Import
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from xdlreg.utils import cprint, normalize


# %% Create colored heatmap ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

def gregoire_black_firered(robj: np.ndarray):
    """
    Create RGB version of given relevance/analyzer object.
    :param robj: relevance object
    :return: RBG version of relevance object
    """
    x = robj.copy()  # there 'mutable' issues
    x /= np.max(np.abs(x))

    hrp = np.clip(x - 0.00, 0, 0.25) / 0.25
    hgp = np.clip(x - 0.25, 0, 0.25) / 0.25
    hbp = np.clip(x - 0.50, 0, 0.50) / 0.50

    hbn = np.clip(-x - 0.00, 0, 0.25) / 0.25
    hgn = np.clip(-x - 0.25, 0, 0.25) / 0.25
    hrn = np.clip(-x - 0.50, 0, 0.50) / 0.50

    return np.concatenate([(hrp + hrn)[..., None], (hgp + hgn)[..., None], (hbp + hbn)[..., None]],
                          axis=x.ndim)


def create_cmap(color_fct, res=4999):
    """
    Creates cmap for given color-function, such as gregoire_black_firered, which can be used for
    colorbars and other purposes.
    The function creates a color-dict in the following form and
    feeds it to matplotlib.colors.LinearSegmentedColormap

    cdict_gregoire_black_firered = {
        "red": [
            [0., 1., 1.],
            [.25, 0., 0.],
            [.5, 0., 0.],
            [.625, 1., 1.],
            [1., 1., 1.]
        ],
        "green": [
            [0., 1., 1.],
            [.25, 1., 1.],
            [.375, .0, .0],
            [.5, 0., 0.],
            [.625, .0, .0],
            [.75, 1., 1.],
            [1., 1., 1.]
        ],
        "blue": [
            [0., 1., 1.],
            [.375, 1., 1.],
            [.5, 0., 0.],
            [.75, 0., 0.],
            [1., 1., 1.]
        ]
    }
    """

    # Prep resolution (res):
    assert float(res).is_integer(), "'res' must be a positive natural number."
    if res < 10:
        cprint(f"res={res} is too small in order to create a detailed cmap. res was set to 999, instead.",
               'y')
        res = 999
    if res % 2 == 0:
        res += 1
        print("res was incremented by 1 to zero center the cmap.")

    linear_space = np.linspace(-1, 1, res)
    linear_space_norm = normalize(linear_space, 0., 1.)

    colored_linear_space = color_fct(linear_space)
    red = colored_linear_space[:, 0]
    green = colored_linear_space[:, 1]
    blue = colored_linear_space[:, 2]

    cdict = {"red": [[linear_space_norm[i_], col_, col_] for i_, col_ in enumerate(red)],
             "green": [[linear_space_norm[i_], col_, col_] for i_, col_ in enumerate(green)],
             "blue": [[linear_space_norm[i_], col_, col_] for i_, col_ in enumerate(blue)]}

    _cmap = LinearSegmentedColormap(name=color_fct.__name__, segmentdata=cdict)

    return _cmap


def symmetric_clip(analyzer_obj, percentile=1-1e-2, min_sym_clip=True):
    """
    Clip relevance object symmetrically around zero.
    :param analyzer_obj: LRP analyser object
    :param percentile: default: keep very small values at border of range out. percentile=1: no change
    :param min_sym_clip: True: finds the min(abs(R.min), R.max) to clip symmetrically
    :return symmetrically clipped analyzer object
    """

    assert .5 <= percentile <= 1, "percentile must be in range (.5, 1)!"

    if not (analyzer_obj.min() < 0. < analyzer_obj.max()) and min_sym_clip:
        cprint("Relevance object has only values larger OR smaller than 0., "
               "thus 'min_sym_clip' is switched off!", 'y')
        min_sym_clip = False

    # Get cut-off values for lower and upper percentile
    if min_sym_clip:
        # min_clip example: [-7, 10] => clip(-7, 7) | [-10, 7] => clip(-7, 7)
        max_min_q = min(abs(analyzer_obj.min()), analyzer_obj.max())  # > 0
        min_min_q = -max_min_q  # < 0

    if percentile < 1:
        # max_q = np.percentile(a=analyzer_obj, q=percentile)
        max_q = -np.percentile(a=-analyzer_obj, q=1-percentile)
        min_q = np.percentile(a=analyzer_obj, q=1-percentile)

        # Clip-off at max-abs percentile value
        max_q = max(abs(min_q), abs(max_q))  # > 0
        # Does opposite of min_clip, example: [-7, 10] => clip(-10, 10) | [-10, 7] => clip(-10, 10)
        if min_sym_clip:
            # However, when both option active, 'percentile' is prioritized
            max_q = min(max_q, max_min_q)
        min_q = -max_q  # < 0

        return np.clip(a=analyzer_obj, a_min=min_q, a_max=max_q)

    elif percentile == 1. and min_sym_clip:
        return np.clip(a=analyzer_obj, a_min=min_min_q, a_max=max_min_q)

    else:
        return analyzer_obj


def apply_colormap(robj: np.ndarray, inputimage: np.ndarray = None, cintensifier: float = 1.,
                   clipq: float = 1e-2, min_sym_clip: bool = False, gamma: float = 0.2,
                   true_scale: bool = False):
    """
    Merge relevance tensor with input image to receive a heatmap over the input space.

    :param robj: relevance map/tensor
    :param inputimage: input image
    :param cintensifier: [1, ...[ increases the color strength by multiplying
    :param clipq: clips off given percentile of relevance symmetrically around zero. range: [0, .5]
    :param min_sym_clip: True: finds the min(abs(R.min), R.max) to clip symmetrically around zero
    :param gamma: the smaller the gamma (< 1.) the brighter, for gamma > 1., the image gets darker
    :param true_scale: True: return min/max robj value (after clip) for true color-scaling in e.g. cbar
    :return: heatmap merged with input, the RGB version, and if true_scale, also the max-value of robj
    """

    # # Prep input image
    img = inputimage.copy()
    x = robj.copy()  # there 'mutable' issues
    # Check whether image has RGB(A) channels
    if img.shape[-1] <= 4:
        img = np.mean(img, axis=-1)  # removes rgb channels
    # Following creates a grayscale image
    img = np.concatenate([img[..., None]] * 3, axis=-1)  # (X,Y,Z, [r,g,b]), where r=g=b (i.e., grayscale)
    # Normalize image (0, 1)
    img /= np.max(np.abs(img))

    # # Symmetrically clip relevance values around zero
    assert 0. <= clipq <= .5, "clipq must be in range (0, .5)!"
    x = symmetric_clip(analyzer_obj=x, percentile=1-clipq, min_sym_clip=min_sym_clip)
    rmax = np.abs(x).max()  # symmetric: rmin = -rmax

    # # Normalize relevance tensor
    x /= np.max(np.abs(x))
    # norm to [-1, 1] for real numbers, or [0, 1] for R+, where zero remains zero

    # # Apply chosen cmap
    r_rgb = gregoire_black_firered(x)  # other cmaps are possible

    # Increase col-intensity
    if cintensifier != 1.:
        assert cintensifier >= 1., "cintensifier must be 1 (i.e. no change) OR greater (intensify color)!"
        r_rgb *= cintensifier
        r_rgb = r_rgb.clip(0., 1.)

    # Merge input image with heatmap via inverse alpha channels
    alpha = np.abs(x[..., None])  # as alpha channel, use (absolute) relevance map amplitude.
    alpha = np.concatenate([alpha] * 3, axis=-1) ** gamma  # (X, Y, Z, 3)
    heat_img = (1 - alpha) * img + alpha * r_rgb

    if true_scale:
        return heat_img, r_rgb, rmax
    else:
        return heat_img, r_rgb

# <<<<<<<<<<< ooo >>>>>>>>>>>>>> ooo <<<<<<<<<<< ooo >>>>>>>>>>>>>> ooo <<<<<<<<<<< ooo >>>>>>>>>>>>>> END
