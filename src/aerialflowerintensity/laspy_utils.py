"""
    This module provides utilities for reading, writing, and manipulating LAS point cloud files.
    It includes functions to append points to LAS files, read LAS files, convert LAS data to NumPy
    arrays, read PCD (Point Cloud Data) files, and handle chunking of LAS data for
    processing large files.
"""

import laspy
import numpy as np

from . import geo_utils


def las_data_to_np(las_data):
    """
    Converts LAS data to a NumPy array.

    :param laspy.LasData las_data: The LAS data to convert.
    :return: The point cloud data as a NumPy array.
    :rtype: numpy.ndarray
    """
    points_record = las_data.points

    #If trans vect extract it
    try:
        trans_vect = np.frombuffer(
            las_data.header.vlrs.get_by_id("TransVect")[0].record_data_bytes(), dtype=float)
    except ValueError:
        trans_vect = 0

    points = point_record_to_np(points_record).astype(np.float64)
    points[:, :3] = (points[:, :3] * points_record.scales) + points_record.offsets + trans_vect

    try:
        projection = las_data.vlrs.get_by_id(
            "LASF_Projection")[2].record_data_bytes().decode("utf-8")
    except (ValueError, IndexError):
        projection = None

    if projection is None:
        #Do nothing
        #Future: Try to estimate projection
        pass
    elif "WGS 84" in projection:
        points[:, :2] = geo_utils.convert_coordinates(points[:, :2], source_crs= "EPSG:4326").T

    return points

def read_pcd_from_las(las_plot_path):
    """
    Reads a PCD (Point Cloud Data) from a .las file.

    :param pathlib.Path las_plot_path: The path to the .las file.
    :return: The point cloud data as a NumPy array.
    :rtype: numpy.ndarray
    """

    las_data = laspy.read(las_plot_path)
    pcd = las_data_to_np(las_data)

    try:
        trans_vect = np.frombuffer(las_data.vlrs.get_by_id("TransVect")[0].record_data)
        soil_buffer = np.frombuffer(
            las_data.vlrs.get_by_id("SoilBufBB")[0].record_data).reshape((2, 4)).T
        pcd[:, :3] = pcd[:, :3] - trans_vect
    except ValueError:
        pass

    pcd[:, -3:] = (pcd[:, -3:] / np.iinfo(np.uint8).max)
    return pcd, soil_buffer

def point_record_to_np(point_record):
    """
    Converts a point record to a NumPy array.

    :param laspy.LasData point_record: The point record to convert.
    :return: The point record data as a NumPy array.
    :rtype: numpy.ndarray
    """
    array_feature = [
        point_record.array[name].reshape(-1, 1) for name in point_record.array.dtype.names]
    array = np.concatenate(array_feature, axis=1)
    return array
