"""
This module provides utilities for applying color filters to point cloud data. It includes functions
to apply filters to HLS colors, convert and filter RGB colors, test color filters on a range of
RGB colors, and filter point cloud data based on color ranges.
"""

import cv2
import numpy as np
from  sklearn import linear_model


def apply_filters(hls_colors, hsl_filters):
    """
    Applies a series of filters to HLS colors and returns a mask indicating the filtered points.

    :param numpy.ndarray hls_colors: HLS colors to be filtered.
    :param list hsl_filters: List of HLS filter ranges to be applied.
    :return: A mask indicating the filtered points.
    :rtype: numpy.ndarray
    """

    masks= []
    for hsl_filter in hsl_filters:
        masks.append(np.logical_and(
            (hls_colors >= hsl_filter[0]).all(axis=1),
            (hls_colors <= hsl_filter[1]).all(axis=1)))

    if len(masks) > 0:
        mask = np.logical_or.reduce(masks)
    else:
        mask = np.tile(True, hls_colors.shape[0])

    return mask

def filter_on_color_rgb(rgb_colors, hsl_filters):
    """
    Converts RGB to HLS colors and applies a series of filters in HSL space,
    returning a mask indicating the filtered points.

    :param numpy.ndarray rgb_colors: RGB colors to be filtered.
    :param list hsl_filters: List of filter ranges to be applied in HLS space.
    :return: A mask indicating the filtered points.
    :rtype: numpy.ndarray
    """

    hls_colors = cv2.cvtColor(
        (rgb_colors * 255).astype(np.uint8).reshape(-1, 1, 3), cv2.COLOR_RGB2HLS)
    hls_colors = (hls_colors/[180, 255, 255]).reshape(-1, 3)

    mask = apply_filters(hls_colors, hsl_filters)

    return mask

def test_pcd_color_filter(hsl_filters):

    """
    Tests the point cloud color filter on a high range of RGB colors.

    :param list hsl_filters: List of filter ranges to be applied.
    :return: None
    :rtype: None
    """

    rgb_colors = np.array(np.mgrid[0:1:1/256, 0:1:1/256, 0:1:1/256]).reshape(3, -1).T
    mask = filter_on_color_rgb(rgb_colors, hsl_filters=hsl_filters)
    return mask

def pcd_color_filter(pcd, hsl_filters):

    """
    Filters a point cloud based on HSL color ranges.

    :param numpy.ndarray pcd: The point cloud data to filter.
    :param list filters: Optional. List of filter ranges to be applied.
        Default is an empty list.
    :return: A tuple containing the filtered inlier point cloud and
        the filtered outlier point cloud.
    :rtype: tuple
    """
    rgb_colors = pcd[:, -3:]
    mask = filter_on_color_rgb(
        rgb_colors, hsl_filters=hsl_filters)

    in_pcd = pcd[mask]
    out_pcd = pcd[~mask]

    return in_pcd, out_pcd

def luminosity_normalization(hls_colors, normalization):
    """
    Normalize the luminosity channel of HLS colors.

    :param numpy.ndarray hls_colors: HLS colors to be normalized.
    :param tuple normalization: A tuple containing the mean and standard deviation
        for normalization.
    :return: The normalized HLS colors.
    :rtype: numpy.ndarray
    """

    mean_lum = hls_colors[:, 1].mean()
    std_lum = hls_colors[:, 1].std()

    hls_colors[:, 1] = (hls_colors[:, 1] - mean_lum) / std_lum
    hls_colors[:, 1] = (hls_colors[:, 1] * normalization[1] + normalization[0])

    hls_colors[:, 1] = np.clip(hls_colors[:, 1], 0, 1)

    return hls_colors

def pcd_color_normalization(pcd, normalization):
    """
    Normalize the color of a point cloud based on luminosity statistics.

    :param numpy.ndarray pcd: The point cloud data to normalize.
    :param tuple normalization: A tuple containing the mean and standard deviation
        for normalization.
    :return: The normalized point cloud data.
    :rtype: numpy.ndarray
    """

    rgb_colors = pcd[:, -3:]
    hls_colors = cv2.cvtColor(
        (rgb_colors * 255).astype(np.uint8).reshape(-1, 1, 3), cv2.COLOR_RGB2HLS)
    hls_colors = (hls_colors/[180, 255, 255]).reshape(-1, 3)

    hls_colors = luminosity_normalization(hls_colors, normalization)

    hls_colors = hls_colors*[180, 255, 255]

    norm_rgb_colors = cv2.cvtColor(
        hls_colors.astype(np.uint8).reshape(-1, 1, 3),
        cv2.COLOR_HLS2RGB)

    pcd[:, -3:] = (norm_rgb_colors/255).reshape(-1, 3)
    return pcd

def pcd_std(pcd):
    """
    Calculate the standard deviation of the luminosity channel in HLS colors of a point cloud.

    :param numpy.ndarray pcd: The point cloud data.
    :return: The standard deviation of the luminosity channel.
    :rtype: float
    """

    rgb_colors = pcd[:, -3:]
    hls_colors = cv2.cvtColor(
        (rgb_colors * 255).astype(np.uint8).reshape(-1, 1, 3), cv2.COLOR_RGB2HLS)
    hls_colors = (hls_colors/[180, 255, 255]).reshape(-1, 3)
    std_lum = hls_colors[:, 1].std()
    return std_lum

def generate_dynamic_white_filter(std_lum):
    """
    Generate a dynamic white HSL filter based on the standard deviation of luminosity.

    :param float std_lum: Standard deviation of the luminosity channel.
    :return: The generated HSL filter definition as a nested list of ranges.
    :rtype: list
    """

    regr = linear_model.LinearRegression()
    regr.coef_ = np.array([[3.6367276727176723]])
    regr.intercept_ = np.array([[0.09241015239560368]])

    lum_filter = regr.predict([[std_lum]])[0,0]
    hsl_filters = [[[0, lum_filter, 0], [1, 1, 1]]]
    return hsl_filters
