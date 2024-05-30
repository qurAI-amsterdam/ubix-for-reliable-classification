# -*- coding: utf-8 -*-
"""An example to read in a image file, and resample the image to a new grid.
   code from deepmedic:https://raw.githubusercontent.com/deepmedic/SimpleITK-examples/master/examples/resample_isotropically.py
"""
import os

import SimpleITK as sitk
import numpy as np

# https://github.com/SimpleITK/SlicerSimpleFilters/blob/master/SimpleFilters/SimpleFilters.py
_SITK_INTERPOLATOR_DICT = {
    "nearest": sitk.sitkNearestNeighbor,
    "linear": sitk.sitkLinear,
    "gaussian": sitk.sitkGaussian,
    "label_gaussian": sitk.sitkLabelGaussian,
    "bspline": sitk.sitkBSpline,
    "hamming_sinc": sitk.sitkHammingWindowedSinc,
    "cosine_windowed_sinc": sitk.sitkCosineWindowedSinc,
    "welch_windowed_sinc": sitk.sitkWelchWindowedSinc,
    "lanczos_windowed_sinc": sitk.sitkLanczosWindowedSinc,
}


def resample_sitk_image(sitk_image, spacing=None, interpolator=None, fill_value=0):
    """Resamples an ITK image to a new grid. If no spacing is given,
    the resampling is done isotropically to the smallest value in the current
    spacing. This is usually the in-plane resolution. If not given, the
    interpolation is derived from the input data type. Binary input
    (e.g., masks) are resampled with nearest neighbors, otherwise linear
    interpolation is chosen.

    Parameters
    ----------
    sitk_image : SimpleITK image or str
      Either a SimpleITK image or a path to a SimpleITK readable file.
    spacing : tuple
      Tuple of integers
    interpolator : str
      Either `nearest`, `linear` or None.
    fill_value : int

    Returns
    -------
    SimpleITK image.
    """

    if isinstance(sitk_image, str):
        sitk_image = sitk.ReadImage(sitk_image)
    num_dim = sitk_image.GetDimension()

    if not interpolator:
        interpolator = "linear"
        pixelid = sitk_image.GetPixelIDValue()

        if pixelid not in [1, 2, 4]:
            raise NotImplementedError(
                "Set `interpolator` manually, "
                "can only infer for 8-bit unsigned or 16, "
                "32-bit signed integers"
            )
        if pixelid == 1:  # 8-bit unsigned int
            interpolator = "nearest"

    orig_pixelid = sitk_image.GetPixelIDValue()
    orig_origin = sitk_image.GetOrigin()
    orig_direction = sitk_image.GetDirection()
    orig_spacing = np.array(sitk_image.GetSpacing())
    orig_size = np.array(sitk_image.GetSize(), dtype=np.int64)

    if not spacing:
        min_spacing = orig_spacing.min()
        new_spacing = [min_spacing] * num_dim
    else:
        new_spacing = [float(s) for s in spacing]

    assert (
        interpolator in _SITK_INTERPOLATOR_DICT.keys()
    ), "`interpolator` should be one of {}".format(_SITK_INTERPOLATOR_DICT.keys())

    sitk_interpolator = _SITK_INTERPOLATOR_DICT[interpolator]

    new_size = orig_size * (orig_spacing / new_spacing)
    new_size = np.ceil(new_size).astype(np.int64)  # Image dimensions are in integers
    new_size = [int(s) for s in new_size]  # SimpleITK expects lists, not ndarrays

    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetTransform(sitk.Transform())
    resample_filter.SetInterpolator(sitk_interpolator)
    resample_filter.SetSize(new_size)
    resample_filter.SetOutputSpacing(new_spacing)
    resample_filter.SetOutputOrigin(orig_origin)
    resample_filter.SetOutputDirection(orig_direction)
    resample_filter.SetDefaultPixelValue(fill_value)

    resampled_sitk_image = resample_filter.Execute(sitk_image)  # orig_pixelid)

    return resampled_sitk_image


def resample_image(
    input_filename=None, output_filename=None, factor=2, interpolator="linear"
):
    """
    Resample the input filename to new voxel size.
    Image is downsampled using linear interpolator (linear) and segmentations
    as nearest neighbour (nearest).
    Input:
       - input_filename: filename of the image to be downsampled
       - factor: factor by which the image needs to rescaled. >1 is downsample
         and <1 is upsample
    Output:
       -  output_filename: filename of the output image
    """
    sitk_image = sitk.ReadImage(input_filename)
    orig_spacing = np.array(sitk_image.GetSpacing())
    req_spacing = factor * orig_spacing
    req_spacing = tuple([float(s) for s in req_spacing])
    resampled_image = resample_sitk_image(
        sitk_image, spacing=req_spacing, interpolator=interpolator, fill_value=0
    )

    sitk.WriteImage(resampled_image, output_filename)

    assert os.path.exists(output_filename)
