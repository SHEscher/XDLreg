#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to transform MRI

Author: Simon M. Hofmann | <[firstname].[lastname][at]cbs.mpg.de> | 2018-2020
"""
# %% Import
import ast
import string
import concurrent.futures
import datetime
from shutil import copyfile
# from random import sample

import nibabel as nb
import ants
from scipy import ndimage

from utils import cprint, chop_microseconds, np, os
from prune_image import find_edges  # TODO integrate prune here !!
# import nibabel as nb  # Read/write access to some common neuroimaging file formats
# from nilearn import image

# %% ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<
# # Global image orientation (according to nibabel)
# Note: We use FreeSurfer output as reference space, according to nibabel: ('L', 'I', 'A'),
# whereas nibabel (canonical) standard is: ('R', 'A', 'S') [RAS+]
# For more, see: https://nipy.org/nibabel/coordinate_systems.html
global_orientation_space = "LIA"  # Note for ANTsPy this is vice versa 'RSP'

# Set global variable
all_manip_opt = ['rotation', 'translation', 'noise', 'none']  # all implemented options for manipulation
# all_manip_opt = ['rotation', 'translation', 'noise', 'iloss', 'contrast', 'none']  # planed
# 'flip': biologically implausible


# %% ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<
# NiBabel based re-orientation functions

def get_orientation_transform(affine, reference_space: str = global_orientation_space):
    return nb.orientations.ornt_transform(start_ornt=nb.orientations.io_orientation(affine),
                                          end_ornt=nb.orientations.axcodes2ornt(reference_space))


def file_to_ref_orientation(image_file, reference_space: str = global_orientation_space):
    """Takes a Nibabel NifTi-file (not array) and returns a reoriented version (to global ref space"""
    # assert isinstance(image_file, nb.freesurfer.mghformat.MGHImage), "image_file must be nibabel.NifTi."
    ornt_trans = get_orientation_transform(affine=image_file.affine, reference_space=reference_space)
    return image_file.as_reoriented(ornt_trans)


def mri_to_ref_orientation(image, affine, reference_space: str = global_orientation_space):
    """Takes an MRI array + corresponding affine matrix and returns it, reoriented to global ref space"""

    # Create orientation transform object first
    ornt_trans = get_orientation_transform(affine=affine, reference_space=reference_space)
    return nb.apply_orientation(image, ornt_trans)


def deg2rad(degree):
    radians = degree * np.pi / 180
    return radians


def get_rotation_affine(degree, axis: int, scale=1, affine4d: bool = False):
    """Computes the rotation affine for 3D tensor for given angle (in degree)"""

    assert axis in [0, 1, 2], "axis must be 0, 1, or 2 (int)!"

    rad = deg2rad(degree=degree)

    if axis == 0:
        rot_affine = np.array(
            [[1, 0, 0, 0],
             [0, np.cos(rad), -np.sin(rad), 0],
             [0, np.sin(rad), np.cos(rad), 0],
             [0, 0, 0, 1]])

    elif axis == 1:
        rot_affine = np.array(
            [[np.cos(rad), 0, np.sin(rad), 0],
             [0, 1, 0, 0],
             [-np.sin(rad), 0, np.cos(rad), 0],
             [0, 0, 0, 1]])

    else:  # axis == 2:
        rot_affine = np.array(
            [[np.cos(rad), -np.sin(rad), 0, 0],
             [np.sin(rad), np.cos(rad), 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]])

    rot_affine[:3, :3] = rot_affine[:3, :3] * scale

    if not affine4d:
        rot_affine = rot_affine[:3, :3]

    return rot_affine


# %% ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<
# Operate with external affines
def load_external_transform_affines(fname):
    """Load affine transform from file
    Parameters
    ----------
    fname : str or None
        Filename of an LTA or FSL-style MAT transform file.
        If ``None``, return an identity transform
    Returns
    -------
    affine : (4, 4) numpy.ndarray
    """
    if fname is None:
        return np.eye(4)

    if fname.endswith('.mat'):
        return np.loadtxt(fname)

    elif fname.endswith('.lta'):
        with open(fname, 'rb') as fobj:
            for line in fobj:
                if line.startswith(b'1 4 4'):
                    break
            lines = fobj.readlines()[:4]
        return np.genfromtxt(lines)

    raise ValueError("Unknown transform type; pass FSL (.mat) or LTA (.lta)")


def write_ants_mat_to_txt(path_2_ants_mat, savepath=None):
    """
    This function writes a text file in the following form:

        #Insight Transform File V1.0
        #Transform 0
        Transform: AffineTransform_double_3_3
        Parameters: 0.9 0.0 -0.01 -0.01 0.9 0.2 0.01 -0.1 0.8 0.1 2.5 22.5
        FixedParameters: 0 0 0

    This is the equivalent to ANTS's `ConvertTransformFile` output:
        `$ANTSPATH/ConvertTransformFile 3 3D_antsAffine.mat 3D_antsAffine.txt`
    Which basically just converts the binary to a text file.

    This can then be used by FreeSurfer's: `lta_convert`:
        `lta_convert --initk 3D_antsAffine.txt --outlta 3D_antsAffine.lta --src src.nii --trg trg.nii`

    :param path_2_ants_mat: path to ANTs AffineMatrix
    :param savepath: path ending with *.txt; if None: save next to original
    """

    assert path_2_ants_mat.endswith(".mat"), f"Provide ANTs' *.mat file!\nGiven file is {path_2_ants_mat}"

    ants_mat = ants.read_transform(filename=path_2_ants_mat)

    print("Convert to *.txt file: ... \n\n", ants_mat)

    params = np.array2string(ants_mat.parameters, suppress_small=True).strip(' []').replace('\n', '')
    params = params. replace("  ", " ")
    fix = np.array2string(ants_mat.fixed_parameters, suppress_small=True).strip(' []')  # .replace('.','')
    dim = ants_mat.dimension

    convert = ["#Insight Transform File V1.0\n",
               "#Transform 0\n",
               f"Transform: {ants_mat.type}_double_{dim}_{dim}\n",
               f"Parameters: {params}\n",
               f"FixedParameters: {fix}"]

    # Save
    savepath = path_2_ants_mat.replace(".mat", ".txt") if savepath is None else savepath
    print(f"Save to '.../{savepath.split('/')[-1]}'")

    with open(savepath, "w") as file:
        file.writelines(convert)


def ants_mat_to_affine_matrix(ants_mat, s=1, inverse=False, to_LIA=False):
    """
    Transform the ANTs/ITK specific *.mat file to an 4x4 affine transformation matrix:

    This is the equivalent to ANTS's `ConvertTransformFile` with 'hm' it outputs the 4x4 matrix):
    `$ANTSPATH/ConvertTransformFile 3 3D_antsAffine.mat 3D_antsAffine.txt --hm`

    [OUTPUT OF BOTH FUNCTIONS WERE CHECKED]

    Check out:
    * Problem raised: https://sourceforge.net/p/advants/discussion/840261/thread/9fbbaab7/?limit=25#1783
    * Solution in matlab: https://github.com/netstim/leaddbs/blob/master/helpers/ea_antsmat2mat.m
    This is just a python adaptation
    * https://www.neuro.polymtl.ca/tips_and_tricks/how_to_use_ants

    What we need & what we get from input arg 'ants_mat':
    * ants_mat.parameters: (12, ): [a-i; l-n]
        * 3x3-matrix (the first 9 parameters, [a-i])
        * translation (10-12 param, [l,m,n])
    * center [x, y, z] := ants_mat.fixed_parameters: (3, )
    * The offset [o, p, q] needs to be computed.

    return:
        M = [
        [a, b, c, o],
        [d, e, f, p],
        [g, h, i, q],
        [0, 0, 0, 1]
        ]
    """

    n_dim = ants_mat.dimension  # 3

    # Init empty matrix
    M = np.zeros((n_dim + 1, n_dim + 1))  # 4x4

    # Set corner piece
    M[-1, -1] = 1

    # Fill 3x3 matrix [a-i] part
    M[:n_dim, :n_dim] = ants_mat.parameters[0:n_dim ** 2].reshape(n_dim, n_dim) * s  # 3x3 * scalar

    # Get translation params, center
    translation = ants_mat.parameters[n_dim ** 2:]  # (3, )
    center = ants_mat.fixed_parameters  # (3, )

    # Compute offset [o, p, q] (3,)
    offset = translation + center  # step 1
    offset -= M[:n_dim, :n_dim] @ center  # step 2 (@ : dot product)

    # Place offeset in 4x4 affine matrix
    M[:n_dim, n_dim] = offset

    if inverse:
        M = np.linalg.inv(M)

    # TODO revisit whether "to_LIA" makes sense here:
    if to_LIA:
        M *= np.array([[1,   1, -1, -1],  # Turner
                       [1,   1, -1, -1],
                       [-1, -1,  1,  1],
                       [1,   1,  1,  1]])

    return M


# %% ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<
# ANTspy based warping function

def save_ants_warpers(tx, folder_path, image_name):

    assert 'fwdtransforms' in list(tx.keys()) and 'invtransforms' in list(tx.keys()), \
        "tx object misses forward and/or inverse transformation files."

    # # Set paths
    # for forward warper
    save_path_name_fwd = os.path.join(folder_path, f"{image_name}1Warp.nii.gz")
    # for inverse warper
    save_path_name_inv = os.path.join(folder_path, f"{image_name}1InverseWarp.nii.gz")
    # # Save also linear transformation .mat file
    save_path_name_mat = os.path.join(folder_path, f"{image_name}0GenericAffine.mat")

    # # Copy warper files from temporary tx folder file to new location
    copyfile(tx['fwdtransforms'][0], save_path_name_fwd)
    copyfile(tx['invtransforms'][1], save_path_name_inv)
    copyfile(tx['invtransforms'][0], save_path_name_mat)  # == ['fwdtransforms'][1]


def get_list_ants_warper(folderpath, inverse=False, only_linear=False):
    warp_fn = "1Warp.nii.gz" if not inverse else "1InverseWarp.nii.gz"
    lin_mat_fn = "0GenericAffine.mat"

    warp_found = False  # init
    mat_found = False  # init

    # Search for transformation files
    for file in os.listdir(folderpath):

        # Look for non-linear warper file
        if warp_fn in file:
            warp_fn = os.path.join(folderpath, file)
            if not warp_found:
                warp_found = True
            else:
                raise FileExistsError(f"Too many files of type '*{warp_fn}' exist in folderpath "
                                      f"'{folderpath}'.")

        # Look for linear transformation (affine) *.mat
        if lin_mat_fn in file:
            lin_mat_fn = os.path.join(folderpath, file)
            if not mat_found:
                mat_found = True
            else:
                raise FileExistsError(f"Too many files of type '*{lin_mat_fn}' exist in folderpath "
                                      f"'{folderpath}'.")

    if mat_found and warp_found:
        transformlist = [lin_mat_fn, warp_fn] if inverse else [warp_fn, lin_mat_fn]

    elif only_linear and mat_found:
        transformlist = [lin_mat_fn]

    else:
        transformlist = None
        cprint("Not all necessary transformation files were found in given path.", 'r')

    return transformlist


# %% ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<
# # Masking
# Set globabl paths
p2data = "/data/pt_02238/DeepAge/Data/mri/"
fn_mask = "T1_brain_mask.nii.gz"  # filename of T1-brain mask
p2mni = "/data/pt_life_restingstate/LIFE/preprocessed/{sic}/structural/"   # folder of F. Liem's study


def create_mask(mri, nl=False):
    """Create brain mask (1:=brain; 0:=background) from skull-stripped image"""
    if nl:  # via nilearn (slightly slower)
        from nilearn import masking
        return masking.compute_background_mask(mri)  # ~ 190 ms

    else:
        mri.get_fdata()[mri.get_fdata() > 0] = 1  # ~ 105 ms
        return mri


def create_t1_brainmask(sic: str):

    # Link to corrsponding brain-mask in F. Liem's folder OR create brain mask one
    dir_mask = f"{p2data}{sic}/{fn_mask}"  # in SIC parent folder (not in sequence-specific dir)
    if not os.path.isfile(dir_mask):
        if os.path.isfile(p2mni.format(sic=sic) + fn_mask):
            os.symlink(src=p2mni.format(sic=sic) + fn_mask, dst=dir_mask)

        else:  # Create mask
            from load_mri_data import load_sic_mri

            sic, t1_mask = load_sic_mri(_sic=sic, mri_sequence="T1", bm=True, norm=None, regis=False,
                                        dtype=np.float16, as_numpy=False, raiserr=False)

            t1_mask = create_mask(mri=t1_mask, nl=True)

            # Save mask
            nb.Nifti1Image(dataobj=t1_mask.get_fdata().astype(np.uint8),
                           affine=t1_mask.affine).to_filename(dir_mask)  # save as nii.gz
            # t1_mask.to_filename(dir_mask.replace("nii.gz", "mgz"))  # save as *.mgz


def create_raw_brainmask(sic: str, mri_sequence: str):
    from load_mri_data import load_sic_raw_mri

    cprint(f"Create raw brainmask for {mri_sequence.upper()} of {sic} ...\n")

    p2reg = f"{p2data}{sic}/{mri_sequence.lower()}/"
    brain_mask = get_t1_brainmask(sic=sic)
    raw_move = load_sic_raw_mri(_sic=sic, mri_sequence=mri_sequence, brain_masked=False, reorient=False,
                                path_only=False)

    transformlist = get_list_ants_warper(folderpath=p2reg, only_linear=True)
    brain_mask_native = ants.apply_transforms(fixed=ants.from_nibabel(raw_move),
                                              moving=ants.from_nibabel(brain_mask),
                                              transformlist=transformlist,
                                              whichtoinvert=[1], verbose=False).astype('uint8')

    # Save mask
    brain_mask_native.to_file(filename=f"{p2reg}raw/{fn_mask.replace('T1', mri_sequence.upper())}")


def get_t1_brainmask(sic: str):
    """Get brain mask (1:=brain; 0:=background) of given SIC in T1-FreeSurfer Space"""
    create_t1_brainmask(sic)  # this creates/links brain mask only if not available yet
    dir_mask = f"{p2data}{sic}/{fn_mask}"  # in SIC parent folder (not in sequence-specific dir)
    return nb.load(filename=dir_mask)


def get_raw_brainmask(sic: str, mri_sequence: str):
    """Get brain mask (1:=brain; 0:=background) of given SIC in raw/native space of given sequence"""

    p2reg = f"{p2data}{sic}/{mri_sequence.lower()}/"
    brain_mask_fname = f"{p2reg}raw/{fn_mask.replace('T1', mri_sequence.upper())}"

    if not os.path.isfile(brain_mask_fname):
        from mri_registration import register_native_to_t1_space
        if not register_native_to_t1_space(sic=sic, mri_sequence=mri_sequence, save_move=True,
                                           verbose=True):
            create_raw_brainmask(sic=sic, mri_sequence=mri_sequence)

    if not os.path.isfile(brain_mask_fname):
        cprint(f"No brain mask for native {mri_sequence.upper()} of {sic} could be found nor created!",
               'y')
        return None

    else:
        return nb.load(filename=brain_mask_fname)


# %% ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<
# Transformations

def clip_img(mri, high_clip=None):
    """
    Clip background & tiny deviations from it to the corresponding (global) background value (usually: 0)
    For image data in float scale, (0,1) OR (-1,1), (scipy) spline-interpolation (specifically for
    rotation) is not clean, i.e. it pushes the intensity for some images beyond the initial image range.
    For rotation, we use also 'high_clip' to clip the max of the rotated image to the initial image-max.
    :param mri: MRI image
    :param high_clip: clip image to given value
    :return: MRI with clipped background
    """

    assert mri.min() >= -1., "Image scale should be either (0, 255), (0, 1), OR (-1, 1)."

    # # Clip tiny deviations from background to background (can also become negative)
    # For scale (0, 255) no clipping necessary
    if mri.max() <= 1.1:  # due to scipy.rotate()-spline-interpolation img.max can become bigger than 1.

        if mri.min() < -.9:
            # For image range (-1, 1)
            bg = -1.
            mri[mri < (-1 + 1/256)] = bg  # clip
        else:
            # For image range (0, 1)
            bg = 0.
            mri[mri < 1/256] = bg  # clip

        # Clip high
        high_clip = 1. if high_clip is None else high_clip

    else:
        high_clip = 255 if high_clip is None else high_clip

    mri[mri > high_clip] = high_clip  # ==np.clip(a=mri, a_min=bg, a_max=high_clip)

    return mri


def rotate_mri(mri, axes: tuple, degree):
    """
    Rotate given MRI
    :param mri: 3D MRI
    :param axes: either (0,1), (0,2), or (1,2)
    :param degree: -/+ ]0, 360[
    :return: rotated MRI
    """
    return clip_img(mri=ndimage.interpolation.rotate(input=mri, angle=degree, axes=axes, reshape=False,
                                                     mode="constant",  # (default: constant)
                                                     order=3,  # spline interpolation (default: 3)
                                                     cval=mri.min()),  # cval = background voxel value
                    high_clip=mri.max())


def max_axis_translation(mri, axis):
    """
    Find the max possible translation along given axis, where brain is not cut-off at boarder
    :param mri: given MRI
    :param axis: translation Axis
    :return: tuple of absolute (!) max shift-sizes in both directions (-, +)
    """
    edges = find_edges(mri)
    len_ax = mri.shape[axis]
    return edges[0 + axis * 2], len_ax - edges[1 + axis * 2] - 1


def translate_mri(mri, axis, by_n):
    """
    Translate given MRI along given axis by n steps.
    :param mri: given MRI
    :param axis: 0, 1, or 2 (3D)
    :param by_n: shift by n steps, sign indicates direction of shift
    :return: translated MRI
    """

    edges = find_edges(mri)
    ledg = edges[0 + axis * 2]
    hedg = edges[1 + axis * 2]

    # If translation too big, make shift size smaller
    if ((hedg + by_n) >= mri.shape[axis]) or ((ledg + by_n) < 0):
        max_shift = max_axis_translation(mri, axis)[0 if np.sign(by_n) < 0 else 1]
        print(f"Max shift-size is {max_shift*np.sign(by_n)}. Given 'by_n' is adapted accordingly.")
        by_n = max_shift * np.sign(by_n)
    # Alternatively, implement: raise Error

    shift_axis = [slice(None)] * 3  # init
    new_shift_axis = [slice(None)] * 3  # init
    shift_axis[axis] = slice(ledg, hedg + 1)
    new_shift_axis[axis] = slice(ledg + by_n, hedg + by_n + 1)

    # Shift/translate brain
    trans_mri = np.zeros(mri.shape, mri.dtype)  # Create empty cube to store shifted brain
    trans_mri[:] = mri.min()  # for non-zero backgroun (e.g., after normalization)
    old_pos = tuple(shift_axis)
    new_pos = tuple(new_shift_axis)
    trans_mri[new_pos] = mri[old_pos]

    return trans_mri


def noisy_mri(mri, noise_type: str, noise_rate: float = None):
    """See also: http://scipy-lectures.org/advanced/image_processing/#image-filtering"""

    noise_types = ["random_knockout", "random_swap", "local_disturb", "image_blur"]
    noise_type = noise_type.lower()
    assert noise_type in noise_types, f"'noise_type' must be in {noise_types}"

    n_all_vox = len(mri.flatten())  # number of all voxels
    bg = mri.min()  # image-value of background (assumption that for all given images bg == img.min)

    if noise_rate is None:
        noise_rate = dict(zip(noise_types, [.01, .01, .1, .5]))[noise_type]  # defaults (order important!)
        # print(f"Set 'noise_rate' to default value: {noise_rate}")

    assert 0. <= noise_rate <= 1., f"'noise_rate' for  {noise_type} must be between [0.-1.]!"
    n_noise = round(noise_rate * n_all_vox)  # number of voxels which will be perturbed

    # # Knock out 1% of all (non-background) voxels (information loss)
    if noise_type == "random_knockout":
        # Find indices of non-background voxels
        xs, ys, zs = (mri + abs(bg)).nonzero()  # abs(...) necessary for re-normed data, e.g. between -1,1
        # Choose random voxels which are be manipulated
        noise_idx = np.random.choice(a=range(len(xs)), size=n_noise, replace=False)
        # Apply knockout
        mri[xs[noise_idx], ys[noise_idx], zs[noise_idx]] = bg  # TODO could be any number (e.g. via arg)

    # # Swap two random non-background voxels
    # (partial information loss, intensity distribution. remains same, global noise addition)
    if noise_type == "random_swap":
        xs, ys, zs = (mri + abs(bg)).nonzero()
        noise_idx = np.random.choice(a=range(len(xs)), size=n_noise, replace=False)
        # Copy indices and shuffle them for swap
        noise_idx2 = noise_idx.copy()
        np.random.shuffle(noise_idx)
        mri[xs[noise_idx], ys[noise_idx], zs[noise_idx]] = mri[xs[noise_idx2],
                                                               ys[noise_idx2],
                                                               zs[noise_idx2]]

    # # Disturb pixel values (local noise addition)
    if noise_type == "local_disturb":
        # Create a blured copy of given MRI:
        mri_med_filt = ndimage.median_filter(mri, 3)  # we sample from this
        xs, ys, zs = (mri + abs(bg)).nonzero()
        # Swap random voxels with blured MRI, i.e. on-spot distortion
        noise_idx = np.random.choice(a=range(len(xs)), size=n_noise, replace=False)
        mri[xs[noise_idx], ys[noise_idx], zs[noise_idx]] = mri_med_filt[xs[noise_idx],
                                                                        ys[noise_idx],
                                                                        zs[noise_idx]]

    # # Blur whole image
    if noise_type == "image_blur":
        # Create blured copy of given MRI
        blur_mri = ndimage.median_filter(mri, 2)  # 1: now filter; 3: too strong
        # Mix blured version in original
        mri = (mri * (1.-noise_rate) + blur_mri * noise_rate).astype(mri.dtype)  # keep datatype !

    return mri


def add_background_noise(mri, noise_scalar=.015):
    """
    Add noise drawn from absolute normal distribution abs((0, sd))
    :param mri: MRI
    :param noise_scalar: after some testing: recommended between [.008, .02[
    :return: MRI with noisy background
    """

    # # Check mri-format: (sic, brain) OR brain
    tp = isinstance(mri, tuple)  # (sic, brain)
    sic = mri[0] if tp else None
    mri = mri[1] if tp else mri

    # # Prepare constants & variables
    img_max = 255 if mri.max() > 1. else 1.  # should be either in range (0, 255), (0, 1) OR (-1, 1)

    # Get number of background voxels
    bg = mri.min()  # background
    n_bg_vox = len(mri[mri == bg])

    # Define image data type ('keep it small')
    img_dtype = np.uint8 if img_max == 255 else np.float16

    # Define scalar for normalal distribution
    sl = img_max * noise_scalar

    # # Add noise to background,
    # Noise is drawn from half-normal distribution, ie., abs(norm-distr) == scipy.stats.halfnorm.rvs()
    noise = abs(np.random.RandomState().normal(loc=0, scale=sl, size=n_bg_vox))
    mri[mri == bg] = bg + noise

    if tp:
        return sic, clip_img(mri.astype(img_dtype))
    else:
        return clip_img(mri.astype(img_dtype))


def add_bg_noise_mriset(mriset, noise_scalar=.015):

    assert isinstance(mriset, dict), "'mriset' is expected to be dict in the from {sic: brain}"

    cprint("\nAdd noise to background on whole dataset...", 'b')
    start_time_load = datetime.datetime.now()
    with concurrent.futures.ThreadPoolExecutor(100) as executor:
        mriset = executor.map(add_background_noise,
                              mriset.items(),
                              [noise_scalar]*len(mriset))

    mriset = dict(tuple(mriset))

    print(f"Duration of adding background noise to all images of dataset (via threading) "
          f"{chop_microseconds(datetime.datetime.now() - start_time_load)} [h:m:s]")

    return mriset


# %% ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<

def random_transform_mri(mri, manipulation=None, n_manips: int = 1):
    """
    Transform given MRI by random or given type of manipulation.
    :param mri: a 3D-MRI [tuple](sic, mri), or [array](mri)
    :param manipulation: if specific image manipulation should be applied, indicate which, str OR list
    :param n_manips: number of manipulations
    :return: (sic, transformed mri) [tuple] or only transformed mri (depends on input)
    """

    # TODO Transform data with variants of these samples:
    #  Check: https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-deep-learning-tips-and-tricks

    tp = type(mri) == tuple
    sic = mri[0] if tp else None  # tp is True -> mri: (sic, brain), else -> mri: brain
    mri = mri[1] if tp else mri

    # # Create suffix for transformed SIC data: SIC_{mkey}
    mkey = np.random.RandomState().choice(list(string.ascii_letters)) + str(
        np.random.RandomState().randint(0, 10)) + np.random.RandomState().choice(
        list(string.ascii_letters))
    # RandomState necessary for parallization of concurrent.futures, otherwise same value
    # manipulation_suffix
    # (Note: A neglectable small probability remains that suffix is the same for two transformed MRIs)

    # # Choose type of image manipulation (implemented so far )
    if manipulation is not None:
        manipulation = manipulation if isinstance(manipulation, (list, np.ndarray)) else [manipulation]
        manipulation = [manip.lower() for manip in manipulation]
        assert all([manip in all_manip_opt for manip in manipulation]), \
            f"'manipulation' must be None or subset of {all_manip_opt}."
    else:
        manipulation = all_manip_opt

    # Check for the number of manipulations to be applied
    n_manips = np.clip(a=n_manips, a_min=1, a_max=len(manipulation))
    # at least one, and maximally each manipulation shall be applied just once

    # Randomly pick image manipulation (Note: Each manipulation is applied just once, 'replace=False')
    manips = np.random.RandomState().choice(a=manipulation, size=n_manips, replace=False)

    for manip in manips:
        # # Rotation
        if manip == 'rotation':
            # e.g., -40,40 degrees (Cole et al., 2017; Jonsson et al., 2019)
            # However, (after testing) seems too strong. Limit angle depending on rotation axis

            mkey = "rot" + mkey

            n_rot = np.random.RandomState().choice([1, 2, 3])  # number of rotations: 1, 2, or on all axes
            # With sequential rotations we can rotate the brain in all directions
            # choose rnd axes (don't rotate on same axis: replace=False)

            # Define angle-range for each axis
            axes_angles_dict = {'(0,1)': (-15, 15),
                                '(0,2)': (10, 10),
                                '(1,2)': (-35, 35)}

            # Choose random axis/axes
            _axes = np.random.RandomState().choice(np.array(list(axes_angles_dict.keys())), n_rot,
                                                   replace=False)

            # for ax in _axes:
            for ax in _axes:
                # Choose random angle and apply rotation
                while True:
                    rot_angle = np.random.RandomState().randint(low=axes_angles_dict[ax][0],
                                                                high=axes_angles_dict[ax][1]+1)
                    if rot_angle != 0:
                        break

                mri = rotate_mri(mri=mri, degree=rot_angle, axes=ast.literal_eval(ax))

                # plot_mid_slice(mri, figname=sic + f" Rotation_ax{ax}_by{rot_angle}")  # TEST

        # # Translation
        if manip == 'translation':
            # e.g., -10,10 voxels (Cole et al., 2017; Jonsson et al., 2019)
            # However, there seems to be no good reason not to translate brain to the boarders of the MRI

            mkey = "tran" + mkey

            # Translate on 1, 2 or all 3 axes
            n_ax = np.random.RandomState().randint(1, 3 + 1)
            _axes = np.random.RandomState().choice(a=[0, 1, 2], size=n_ax, replace=False)

            for ax in _axes:

                direction = np.random.RandomState().choice([-1, 1])  # random direction of shift

                max_shift = max_axis_translation(mri=mri, axis=ax)[0 if direction < 0 else 1]

                move = np.random.RandomState().randint(1, np.maximum(2, max_shift))
                # or randint(1, 10+1) as e.g. Cole et al. (2017)
                move *= direction

                # Shift/translate brain
                mri = translate_mri(mri=mri, axis=ax, by_n=move)

                # plot_mid_slice(mri, figname=sic + f" Translation_ax{ax}_by{move}")  # TEST

        # # Add noise
        if manip == "noise":
            mkey = "noi" + mkey
            # TODO discuss default noise_type here: 'image_blur'
            mri = noisy_mri(mri=mri, noise_type="image_blur", noise_rate=None)

        # TODO
        #  - Information loss (e.g., whole areas or bigger slices)
        #  - intensity shift/contrast change: since neg. corr between age and max&mean MRI-intensity,
        #       see(z_playground: exp_mri), i.e., intensity distributions is a function of age.

        # No manipulation
        if manip == "none":
            # create a pure copy
            pass

    if tp:
        # Save manipulated mri in (training-)set with key: sic + manipulation_suffix
        aug_key = "_".join([sic, mkey])
        return aug_key, mri  # augmented MRI

    else:
        return mri  # augmented MRI

# %% ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o  END
