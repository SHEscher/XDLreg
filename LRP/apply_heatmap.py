"""
Adaptation from apply_heatmap.py by Sebastian L. & Leander Weber (FH HHI)
"""
# %% Import
# import sys
# import os
# sys.path.append((os.path.abspath(".").split("DeepAge")[0] + "DeepAge/Analysis/Modelling/MRInet/"))

from utils import *
import numpy as np
from matplotlib import cm
from imageio import imwrite as imsave
# import cv2


# %% ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<

def produce_supported_maps():
    # return a list of names and extreme color values.
    print(*(list(custom_maps.keys()) + matplotlib_maps), sep="\n")
    return list(custom_maps.keys()) + matplotlib_maps


def colorize_matplotlib(R, cmapname):
    # fetch color mapping function by string
    cmap = cm.__dict__[cmapname]

    # bring data to [-1 1]
    R = R / np.max(np.abs(R))

    # push data to [0 1] to avoid automatic color map normalization
    R = (R + 1) / 2

    sh = R.shape

    return cmap(R.flatten())[:, 0:3].reshape([*sh, 3])


# %% ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<
# Functions to create colored heatmap

def gregoire_gray_red(R):
    basegray = 0.8  # floating point gray

    maxabs = np.max(R)
    RGB = np.ones([*R.shape, 3]) * basegray  # uniform gray image.

    tvals = np.maximum(np.minimum(R / maxabs, 1.0), -1.0)
    negatives = R < 0

    RGB[negatives, 0] += tvals[negatives] * basegray
    RGB[negatives, 1] += tvals[negatives] * basegray
    RGB[negatives, 2] += -tvals[negatives] * (1 - basegray)

    positives = R >= 0
    RGB[positives, 0] += tvals[positives] * (1 - basegray)
    RGB[positives, 1] += -tvals[positives] * basegray
    RGB[positives, 2] += -tvals[positives] * basegray

    return RGB


def gregoire_black_green(R):
    maxabs = np.max(R)
    RGB = np.zeros([*R.shape, 3])

    negatives = R < 0
    RGB[negatives, 2] = -R[negatives] / maxabs

    positives = R >= 0
    RGB[positives, 1] = R[positives] / maxabs

    return RGB


def gregoire_black_firered(R):
    # normalize to [-1, 1] for Real numbers, or [0, 1] for R+, where zero remains zero:
    R /= np.max(np.abs(R))
    x = R

    hrp = np.clip(x - 0.00, 0, 0.25) / 0.25  # all pos. values(+) above 0 get red, above .25 full red(=1.)
    hgp = np.clip(x - 0.25, 0, 0.25) / 0.25  # all above .25 get green, above .50 full green
    hbp = np.clip(x - 0.50, 0, 0.50) / 0.50  # all above .50 get blue until full blue at 1. (mix 2 white)

    hbn = np.clip(-x - 0.00, 0, 0.25) / 0.25  # all neg. values(-) below 0 get blue ...
    hgn = np.clip(-x - 0.25, 0, 0.25) / 0.25  # ... green ....
    hrn = np.clip(-x - 0.50, 0, 0.50) / 0.50  # ... red ... mixes to white (1.,1.,1.)

    return np.concatenate([(hrp + hrn)[..., None],
                           (hgp + hgn)[..., None],
                           (hbp + hbn)[..., None]], axis=x.ndim)


def gregoire_gray_red2(R):
    v = np.var(R)
    R[R > 10 * v] = 0
    R[R < 0] = 0
    R = R / np.max(R)
    # (this is copypasta)
    x = R

    # positive relevance
    hrp = 0.9 - np.clip(x - 0.3, 0, 0.7) / 0.7 * 0.5
    hgp = 0.9 - np.clip(x - 0.0, 0, 0.3) / 0.3 * 0.5 - np.clip(x - 0.3, 0, 0.7) / 0.7 * 0.4
    hbp = 0.9 - np.clip(x - 0.0, 0, 0.3) / 0.3 * 0.5 - np.clip(x - 0.3, 0, 0.7) / 0.7 * 0.4

    # negative relevance
    hrn = 0.9 - np.clip(-x - 0.0, 0, 0.3) / 0.3 * 0.5 - np.clip(-x - 0.3, 0, 0.7) / 0.7 * 0.4
    hgn = 0.9 - np.clip(-x - 0.0, 0, 0.3) / 0.3 * 0.5 - np.clip(-x - 0.3, 0, 0.7) / 0.7 * 0.4
    hbn = 0.9 - np.clip(-x - 0.3, 0, 0.7) / 0.7 * 0.5

    hr = hrp * (x >= 0) + hrn * (x < 0)
    hg = hgp * (x >= 0) + hgn * (x < 0)
    hb = hbp * (x >= 0) + hbn * (x < 0)

    return np.concatenate([hr[..., None], hg[..., None], hb[..., None]], axis=R.ndim)


def alex_black_yellow(R):
    maxabs = np.max(R)
    RGB = np.zeros([*R.shape, 3])

    negatives = R < 0
    RGB[negatives, 2] = -R[negatives] / maxabs

    positives = R >= 0
    RGB[positives, 0] = R[positives] / maxabs
    RGB[positives, 1] = R[positives] / maxabs

    return RGB


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
    from matplotlib.colors import LinearSegmentedColormap

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


# %% ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<
# List of supported color map names (the maps need to be implemented above this line)
custom_maps = {'gray-red': gregoire_gray_red,
               'gray-red2': gregoire_gray_red2,
               'black-green': gregoire_black_green,
               'black-firered': gregoire_black_firered,
               'blue-black-yellow': alex_black_yellow}

matplotlib_maps = ['afmhot', 'jet', 'seismic', 'bwr', "cool"]


# %% ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<

def symmetric_clip(analyzer_obj, percentile=1-1e-2, min_sym_clip=True):
    """
    Clip relevance object symmetrically around zero.
    :param analyzer_obj: LRP analyser object
    :param percentile: default: keep very small values at border of range out. percentile=1: no change
    :param min_sym_clip: True: finds the min(abs(R.min), R.max) to clip symmetrically
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


def apply_colormap(R, inputimage=None, cmapname='black-firered', cintensifier=1., clipq=1e-2,
                   min_sym_clip=False, gamma=0.2, gblur=0, true_scale=False):
    """
    Merge relevance tensor with input image to receive a heatmap over the input space.
    :param R: relevance map/tensor
    :param inputimage: input image
    :param cmapname: name of color-map (cmap) to be applied
    :param cintensifier: [1, ...[ increases the color strength by multiplying + clipping [DEPRECATED]
    :param clipq: clips off given percentile of relevance symmetrically around zero. range: [0, .5]
    :param min_sym_clip: True: finds the min(abs(R.min), R.max) to clip symmetrically around zero
    :param gamma: the smaller the gamma (< 1.) the brighter, for gamma > 1., the image gets darker
    :param gblur: ignore for now
    :param true_scale: True: return min/max R value (after clipping) for true col scaling in e.g. cbar
    :return: heatmap merged with input
    """

    # # Prep Input Image
    img = inputimage.copy()
    _R = R.copy()  # since mutable
    # Check whether image has RGB(A) channels
    if img.shape[-1] <= 4:
        img = np.mean(img, axis=-1)  # removes rgb channels
    # Following creates a grayscale image (for MRI case, no difference)
    img = np.concatenate([img[..., None]] * 3, axis=-1)  # (X,Y,Z, [r,g,b]), where r=g=b (i.e., grayscale)
    # normalize image (0, 1)
    if img.min() < 0.:  # for img range (-1, 1)
        img += np.abs(img.min())
    img /= np.max(np.abs(img))

    # Symmetrically clip relevance values around zero
    assert 0. <= clipq <= .5, "clipq must be in range (0, .5)!"
    _R = symmetric_clip(analyzer_obj=_R, percentile=1-clipq, min_sym_clip=min_sym_clip)
    rmax = np.abs(_R).max()  # symmetric: rmin = -rmax

    # # Normalize relevance tensor
    _R /= np.max(np.abs(_R))
    # norm to [-1, 1] for real numbers, or [0, 1] for R+, where zero remains zero

    # # Apply chosen cmap
    if cmapname in custom_maps:
        r_rgb = custom_maps[cmapname](_R)
    elif cmapname in matplotlib_maps:
        r_rgb = colorize_matplotlib(_R, cmapname)
    else:
        raise Exception(f'You have managed to smuggle in the unsupported colormap {cmapname} into method '
                        f'apply_colormap. Supported mappings are:\n\t{produce_supported_maps()}')

    # Increase col-intensity
    if cintensifier != 1.:
        assert cintensifier >= 1., "cintensifier must be 1 (i.e. no change) OR greater (intensify color)!"
        r_rgb *= cintensifier
        r_rgb = r_rgb.clip(0., 1.)

    # Merge input image with heatmap via inverse alpha channels
    alpha = np.abs(_R[..., None])  # as alpha channel, use (absolute) relevance map amplitude.
    alpha = np.concatenate([alpha] * 3, axis=-1) ** gamma  # (X,Y,Z, 3)
    heat_img = (1 - alpha) * img + alpha * r_rgb

    # Apply Gaussian blur
    if gblur > 0:  # there is a bug in opencv which causes an error with this command
        raise ValueError("'gaussblur' currently not activated, keep 'gaussblur=0' for now!")
        # hm = cv2.GaussianBlur(HM, (gaussblur, gaussblur), 0)

    if true_scale:
        return heat_img, r_rgb, rmax
    else:
        return heat_img, r_rgb


# %% ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<
def write_heatmap_image(rgb, outputpath):
    # RGB (from apply_colormap()) is still filled with floating point values.
    # convert, then save.
    rgb *= 255.
    rgb = rgb.astype(np.uint8)
    imsave(outputpath, rgb)

# %% ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o  END
