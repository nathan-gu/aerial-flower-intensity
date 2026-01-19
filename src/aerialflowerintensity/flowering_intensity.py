"""
This module provides functions to measure the flowering intensity of trees point cloud data
(PCD) for a given plot or a site. It includes functionalities to process PCD files,
and compute flowering intensities sequentially or in parallel.
"""

import logging
import multiprocessing
import os
from functools import partial

import numpy as np
import pandas
import tqdm

from . import flowering_visualization
from . import laspy_utils
from . import voxelize_pcd
from . import ground_filtering as grf
from . import pcd_color_filter as clrf

_LOGGER = logging.getLogger(__name__)

def segment_voxelize_tree(
        pcd, hsl_filters, voxel_size, ground_threshold):
    """
    Segment and voxelize a tree point cloud.

    :param numpy.ndarray pcd: The point cloud data.
    :param list hsl_filters: HSL Color filters to identify flowers and canopy.
    :param list voxel_size: Voxel size for voxelize the point cloud data.
    :param float ground_threshold: The height relative to the ground plane to filter
        out ground points.
    :return: The segmented voxels points, the segmented pcds, the ground plane equation,
    the segmented voxels and the pitch and roll of ground plane with horizon.
    :rtype: tuple
    """
    _LOGGER.debug("Clean tree point cloud from ground")
    tree_pcd, ground_pcd, plane_model, theta_pitch, theta_roll = grf.clean_tree_pcd(
        pcd, ground_threshold)

    if max(theta_pitch, theta_roll) > 20:
        return "GroundAngle", None, None, None, None, None, None, theta_pitch, theta_roll
    if tree_pcd.shape[0] == 0:
        return "NoTree", None, None, ground_pcd, plane_model, None, None, theta_pitch, theta_roll

    _LOGGER.debug("Segment flower and non flower points from HSL color")
    flower_pcd, non_flower_pcd = clrf.pcd_color_filter(
        tree_pcd,
        hsl_filters=hsl_filters)

    _LOGGER.debug("Voxelize point cloud and count classified points per voxel")
    min_bounds = [tree_pcd[:,i].min()for i in range(3)]
    pcds_voxel_count = voxelize_pcd.point_count_per_voxels(
        min_bounds, voxel_size, flower_pcd, non_flower_pcd)

    tree_voxels = pcds_voxel_count[["idx_voxel_x", "idx_voxel_y", "idx_voxel_z"]].to_numpy()
    flower_voxels = tree_voxels[(pcds_voxel_count["flower_count"] > 0)]
    non_flower_voxels = tree_voxels[~(pcds_voxel_count["flower_count"] > 0)]

    return (pcds_voxel_count, flower_pcd, non_flower_pcd, ground_pcd, plane_model,
        flower_voxels, non_flower_voxels, theta_pitch, theta_roll)

def measure_flowering(
        pcd, hsl_filters, voxel_size, ground_threshold):
    """
    Computes the flowering intensities of a tree point cloud.

    :param numpy.ndarray pcd: The point cloud data.
    :param list hsl_filters: HSL Color filters to identify flowers and canopy.
    :param list voxel_size: Voxel size for voxelize the point cloud data.
    :param float ground_threshold: The height relative to the ground plane to filter
        out ground points.
    :return: The floral voxel ratio, voxel flower intensity mean, flower point density and flower
        point ratio the segmented pcds, the ground plane equation, the segmented voxels and the
        pitch and roll of ground plane with horizon.
    :rtype: tuple
    """

    (pcds_voxel_count, flower_pcd, non_flower_pcd, ground_pcd, plane_model, flower_voxels,
    non_flower_voxels, theta_pitch, theta_roll) = segment_voxelize_tree(
        pcd, hsl_filters, voxel_size=voxel_size, ground_threshold=ground_threshold)

    if isinstance(pcds_voxel_count,(str)):
        return (pcds_voxel_count, None, None, None, None, None, flower_pcd,
                non_flower_pcd, ground_pcd, plane_model,
                flower_voxels, non_flower_voxels, theta_pitch, theta_roll)

    #Voxel counts
    flower_voxels_count = len(flower_voxels)
    non_flower_voxels_count = len(non_flower_voxels)

    #Voxel volume
    tree_voxel_count = flower_voxels_count + non_flower_voxels_count
    tree_voxel_volume = tree_voxel_count * np.prod(voxel_size)

    #Floral voxel ratio
    floral_voxel_ratio = flower_voxels_count / (flower_voxels_count + non_flower_voxels_count)

    #Voxel flower intensity mean
    voxel_flower_intensity_mean = (
        pcds_voxel_count["flower_count"]
        / (pcds_voxel_count["flower_count"] + pcds_voxel_count["canopy_count"])
        ).mean()

    #Flower point density
    flower_point_density = len(flower_pcd) / tree_voxel_volume

    #Flower Point Ratio
    flowering_point_ratio = len(flower_pcd) / (len(flower_pcd) + len(non_flower_pcd))

    return (pcds_voxel_count, floral_voxel_ratio, voxel_flower_intensity_mean, flower_point_density,
        flowering_point_ratio,  tree_voxel_volume, flower_pcd, non_flower_pcd, ground_pcd,
        plane_model, flower_voxels, non_flower_voxels, theta_pitch, theta_roll)


def measure_plot_flowering_intensity(
        las_plot_path, hsl_filters, voxel_size, ground_threshold, dynamic_white_filter,
        visualization_output_path):

    """
    Measures the flowering intensity for a single LAS plot file.

    :param pathlib.Path las_plot_path: The path to the LAS plot file.
    :param list hsl_filters: HSl Color filters to identify flowers and canopy.
    :param list voxel_size: Voxel size for partitioning the point cloud data.
    :param float ground_threshold: The height relative to the ground plane to filter
        out ground points.
    :return: A DataFrame containing the plot name and its flowering intensity.
    :rtype: pandas.DataFrame
    """

    las_plot_name = las_plot_path.parent.name

    _LOGGER.debug("Plot: %s", las_plot_name)
    _LOGGER.debug("%s: Read las file point cloud : %s", las_plot_name, las_plot_path)
    pcd, soil_buffer = laspy_utils.read_pcd_from_las(las_plot_path)

    if dynamic_white_filter:
        lum_std = clrf.pcd_std(pcd)
        hsl_filters = clrf.generate_dynamic_white_filter(lum_std)

    #Empty pcd
    if pcd.shape[0] == 0:
        _LOGGER.debug("%s: Empty point cloud", las_plot_name)
        return pandas.DataFrame(
            [[
                las_plot_name, pandas.NA, pandas.NA, pandas.NA, pandas.NA,
                pandas.NA, pandas.NA, pandas.NA,  pandas.NA, pandas.NA
            ]],
            columns=[
                "plot_id", "floral_voxel_ratio", "voxel_flower_intensity_mean",
                "flower_point_density", "flowering_point_ratio", "tree_voxel_volume",
                "flower_voxels_count", "non_flower_voxels_count", "pitch_y", "roll_x"])

    (
        pcds_voxel_count, floral_voxel_ratio, voxel_flower_intensity_mean, flower_point_density,
        flowering_point_ratio, tree_voxel_volume, flower_pcd, non_flower_pcd, ground_pcd,
        plane_model, flower_voxels, non_flower_voxels, theta_pitch, theta_roll
    ) = measure_flowering(pcd, hsl_filters, voxel_size=voxel_size,
                           ground_threshold=ground_threshold)

    flowering_visualization.plot_flowering_visualization(
        visualization_output_path, las_plot_name, flower_pcd, non_flower_pcd, ground_pcd,
        plane_model, flower_voxels, non_flower_voxels, voxel_size, soil_buffer, theta_pitch,
        theta_roll, sample_size=5000)

    #Invalid pcd
    if isinstance(pcds_voxel_count,(str)):
        if pcds_voxel_count == "NoTree":
            _LOGGER.debug("%s: No tree in plot", las_plot_name)
        if pcds_voxel_count == "GroundAngle":
            _LOGGER.debug(
                "%s: High ground angle, pitch: %s°, roll: %s°",
                las_plot_name, theta_pitch, theta_roll)
        return pandas.DataFrame(
            [[
                las_plot_name, pandas.NA, pandas.NA, pandas.NA, pandas.NA, pandas.NA, pandas.NA,
                pandas.NA, theta_pitch, theta_roll
            ]],
            columns=[
                "plot_id", "floral_voxel_ratio", "voxel_flower_intensity_mean",
                "flower_point_density", "flowering_point_ratio", "tree_voxel_volume",
                "flower_voxels_count", "non_flower_voxels_count", "pitch_y", "roll_x"])

    #Voxel counts
    flower_voxels_count = len(flower_voxels)
    non_flower_voxels_count = len(non_flower_voxels)

    return pandas.DataFrame(
            [[
                las_plot_name, floral_voxel_ratio, voxel_flower_intensity_mean,
                flower_point_density, flowering_point_ratio, tree_voxel_volume,
                flower_voxels_count, non_flower_voxels_count, theta_pitch, theta_roll
            ]],
            columns=[
                "plot_id", "floral_voxel_ratio", "voxel_flower_intensity_mean",
                "flower_point_density", "flowering_point_ratio", "tree_voxel_volume",
                "flower_voxels_count", "non_flower_voxels_count", "pitch_y", "roll_x"])

def sequential_site_flowering_intensity(
        las_folder_path, hsl_filters, voxel_size, ground_threshold, dynamic_white_filter,
        visualization_output_path):
    """
    Sequentially calculates the flowering intensity for all LAS plot files in a site.

    :param pathlib.Path las_folder_path: The path to the folder containing LAS plot files.
    :param list hsl_filters: HSl Color filters to identify flowers and canopy.
    :param list voxel_size: Voxel size for partitioning the point cloud data.
    :param float ground_threshold: The height relative to the ground plane to filter
        out ground points.
    :return: A DataFrame containing all the site plot names and their corresponding flowering
        intensities.
    :rtype: pandas.DataFrame
    """

    _LOGGER.info("Start sequential flower intensity 3D")
    site_flowering_intensity = []
    las_plot_paths = list(las_folder_path.rglob("*.la[sz]"))
    for las_plot_path in tqdm.tqdm(las_plot_paths, total=len(las_plot_paths),ncols=100):
        # las_plot_path = list(las_folder_path.rglob("*_10_56_*.laz"))[0]
        flowering_intensity = measure_plot_flowering_intensity(
            las_plot_path, hsl_filters, voxel_size, ground_threshold,
            dynamic_white_filter, visualization_output_path)
        site_flowering_intensity.append(flowering_intensity)
    site_flowering_intensity = pandas.concat(site_flowering_intensity)
    return site_flowering_intensity

def parrallel_site_flowering_intensity(
        las_folder_path, hsl_filters, voxel_size, ground_threshold, dynamic_white_filter,
        visualization_output_path, nb_thread):

    """
    Parallelizes the calculation of flowering intensity for all LAS plot files in a site.

    :param pathlib.Path las_folder_path: The path to the folder containing LAS plot files.
    :param list hsl_filters: HSl Color filters to identify flowers and canopy.
    :param list voxel_size: Voxel size for partitioning the point cloud data.
    :param float ground_threshold: The height relative to the ground plane to filter
        out ground points.
    :param int nb_thread: Number of threads to use for parallel processing.
    :return: A DataFrame containing all the site plot names and their corresponding flowering
        intensities.
    :rtype: pandas.DataFrame
    """

    _LOGGER.info("Start parrallel flower intensity 3D")
    las_plot_paths = list(las_folder_path.rglob("*.la[sz]"))
    if not las_plot_paths:
        raise FileNotFoundError("No las/laz have been found")

    pool = multiprocessing.get_context("forkserver").Pool(min(len(las_plot_paths), nb_thread))

    site_flowering_intensity = []
    for flowering_intensity in tqdm.tqdm(
                pool.imap(
                    partial(
                        measure_plot_flowering_intensity,
                        hsl_filters=hsl_filters, voxel_size=voxel_size,
                        ground_threshold=ground_threshold,
                        dynamic_white_filter=dynamic_white_filter,
                        visualization_output_path=visualization_output_path,
                    ),
                    las_plot_paths),
                total=len(las_plot_paths), ncols=100, leave=False):

        site_flowering_intensity.append(flowering_intensity)
    site_flowering_intensity = pandas.concat(site_flowering_intensity)
    return site_flowering_intensity

def measure_site_flowering_intensity(
        las_folder_path, hsl_filters, voxel_size, ground_threshold, dynamic_white_filter,
        visualization_output_path=None, nb_thread=None):

    """
    Calculate the flowering intensity for all LAS plot files in a site,
    sequentially or in parallel.

    :param pathlib.Path las_folder_path: The path to the folder containing LAS plot files.
    :param list hsl_filters: HSl Color filters to identify flowers and canopy.
    :param list voxel_size: Voxel size for partitioning the point cloud data.
    :param float ground_threshold: The height relative to the ground plane to filter
        out ground points.
    :param int nb_thread: Number of threads to use for parallel processing. Default is None.
    :return: A DataFrame containing all the site plot names and their corresponding flowering
        intensities.
    :rtype: pandas.DataFrame
    """

    las_plot_names = list(las_folder_path.rglob("*.la[sz]"))

    visualization_output_path.mkdir(parents=True, exist_ok=True)

    if nb_thread is None:
        nb_thread = os.cpu_count()

    _LOGGER.info("Start flower intensity 3D on %s CPU core", nb_thread)

    if len(las_plot_names) == 1 or nb_thread == 1:
        return sequential_site_flowering_intensity(
            las_folder_path, hsl_filters, voxel_size, ground_threshold, dynamic_white_filter,
            visualization_output_path)
    return parrallel_site_flowering_intensity(
        las_folder_path, hsl_filters, voxel_size, ground_threshold, dynamic_white_filter,
        visualization_output_path, nb_thread)
