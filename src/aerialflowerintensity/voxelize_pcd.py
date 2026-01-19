
"""
This module provides functions for voxelizing point cloud data and counting points
in each voxel.
"""

from functools import reduce

import numpy as np
import pandas

def voxelize_pcd(pcd, min_bounds, voxel_size):
    """
    Voxelizes a point cloud data.

    :param numpy.ndarray pcd: The input point cloud data.
    :param numpy.ndarray min_bounds: The minimum bounds of the voxel grid.
    :param numpy.ndarray voxel_size: The three-dimensional size of voxels.
    :return: The voxelized point cloud data.
    :rtype: numpy.ndarray
    """

    points = pcd[:, :3]
    voxelized_pcd = (np.floor_divide((points-min_bounds),voxel_size)).astype(int)
    return voxelized_pcd

def voxels_points_count(voxelized_pcd):
    """
    Calculates the count of points in each voxel of a voxelized point cloud.

    :param numpy.ndarray voxelized_pcd: The voxelized point cloud data.
    :param numpy.ndarray voxel_size: The three-dimensional size of voxels.
    :return: A tuple containing the voxelized points and the count of points in each voxel.
    :rtype: tuple
    """
    voxels, voxels_count = np.unique(voxelized_pcd, return_counts=True, axis=0)
    return voxels, voxels_count

def get_pcd_count(pcd_voxels, pcd_voxels_count, column_count_name):
    """
    Creates a pandas DataFrame containing voxel indices and their corresponding point counts.

    :param numpy.ndarray pcd_voxels: The voxelized points.
    :param numpy.ndarray pcd_voxels_count: The count of points in each voxel.
    :param str column_count_name: The name of the column for the point counts.
    :return: A pandas DataFrame containing voxel indices and point counts.
    :rtype: pandas.DataFrame
    """

    df_pcd_voxels = pandas.DataFrame(
        pcd_voxels, columns=["idx_voxel_x","idx_voxel_y","idx_voxel_z"])
    df_pcd_voxels_count = pandas.DataFrame(pcd_voxels_count,columns=[column_count_name])
    pcd_count = pandas.concat((df_pcd_voxels,df_pcd_voxels_count),axis=1)
    return pcd_count


def point_count_per_voxels(min_bounds, voxel_size, *pcds):
    """
    Voxelize multiple point cloud data on the same grid and count the points per voxel.

    :param numpy.ndarray min_bounds: The minimum bounds of the voxel grid.
    :param numpy.ndarray voxel_size: The three-dimensional size of voxels.
    :param numpy.ndarray pcds: Variable-length argument list of point cloud data.
    :return: A pandas DataFrame containing voxel indices and point counts for each point cloud.
    :rtype: pandas.DataFrame
    """

    voxelized_pcds = [voxelize_pcd(pcd, min_bounds, voxel_size) for pcd in pcds]
    pcds_voxels_n_counts = [
        voxels_points_count(
            voxelized_pcd)
        for voxelized_pcd in voxelized_pcds]

    column_count_names = ["flower_count", "canopy_count"]

    pcds_count = [
        get_pcd_count(
            pcd_voxels_n_counts[0],
            pcd_voxels_n_counts[1],
            column_count_name=column_count_names[i])
        for i, pcd_voxels_n_counts in enumerate(pcds_voxels_n_counts)]

    pcds_count = reduce(
        lambda  left,right: pandas.merge(
            left,right,
            on=["idx_voxel_x", "idx_voxel_y", "idx_voxel_z"],
            how="outer"), pcds_count)
    pcds_count = pcds_count.fillna(value=0.0)

    return (
        pcds_count[["idx_voxel_x", "idx_voxel_y", "idx_voxel_z", *column_count_names]])
