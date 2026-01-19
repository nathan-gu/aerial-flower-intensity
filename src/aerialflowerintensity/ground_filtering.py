"""
This module provides functions for estimating and removing the ground plane from UTM
point cloud data.
"""
import logging

import numpy as np
import open3d as o3d
import pandas

_LOGGER = logging.getLogger(__name__)

def find_plane(
        utm_points, distance_threshold=0.01, quantile=0.1, num_iterations=1000):
    """
    Estimates a plane from a set of UTM points using the RANSAC algorithm.

    :param numpy.ndarray utm_points: The UTM points data.
    :param float distance_threshold: The maximum distance a point can be from the plane
        to be considered an inlier. Default is 0.01.
    :param float quantile: The quantile to reduce the point cloud for ground plane search.
        Default is 0.1.
    :param int num_iterations: The number of RANSAC iterations to perform. Default is 1000.
    :return: The estimated plane model and the angle to the XY plane.
    :rtype: tuple
    """

    #Reduce point cloud in which to search the ground plane, to the lower half of points
    utm_points = utm_points[utm_points[:,2] < np.quantile(utm_points[:,2], quantile)]

    o3pcd = o3d.geometry.PointCloud()
    o3pcd.points = o3d.utility.Vector3dVector(utm_points)

    # Plane Estimation
    plane_model, _inliers = o3pcd.segment_plane(
        distance_threshold=distance_threshold, ransac_n=3, num_iterations=num_iterations)
    plane_model = plane_model * np.sign(plane_model[2])

    nx, ny, nz = plane_model[:3]
    theta_pitch = np.degrees(np.arctan2(-nx, nz))
    theta_roll = np.degrees(np.arctan2(ny, nz))

    return plane_model, theta_pitch, theta_roll

def find_ground_plane(pcd, ransac_repetition):
    """
    Finds the ground plane in a point cloud using RANSAC algorithm.

    :param np.ndarray pcd: The input point cloud data.
    :param int ransac_repetition: The number of RANSAC repetitions.
    :return: A tuple containing the plane model and the angle to the XY plane.
    :rtype: tuple
    """

    if ransac_repetition <= 1:
        plane_model, theta_pitch, theta_roll = find_plane(
            pcd[:, :3], quantile=0.45)
    else:
        quantile_angle = []
        for q in np.linspace(0.05, 0.5, ransac_repetition):
            plane_model, theta_pitch, theta_roll = find_plane(
                pcd[:, :3], quantile=q)
            quantile_angle.append([plane_model, theta_pitch, theta_roll])
        quantile_angle = pandas.DataFrame(
            quantile_angle, columns=["plane_model", "theta_pitch", "theta_roll"])
        mean_absolute_angle = quantile_angle[["theta_pitch", "theta_roll"]].abs().mean(axis=1)
        plane_model, theta_pitch, theta_roll = quantile_angle[
            mean_absolute_angle == mean_absolute_angle.quantile(interpolation="nearest")].values[0]

    return plane_model, theta_pitch, theta_roll

def segment_ground(utm_pcd, plane_model, ground_threshold):
    """
    Segment the ground from a UTM point cloud.

    :param numpy.ndarray utm_pcd: The UTM point cloud data.
    :param numpy.ndarray plane_model: The estimated plane model of the ground.
    :param float ground_threshold: The height relative to the ground plane from which to
        remove ground plane points.
    :return: The point cloud data representing trees without the ground.
    :rtype: numpy.ndarray
    """

    #Altitude to ground
    ground_z = -(np.dot(utm_pcd[:, :2], plane_model[:2]) + plane_model[3]) / plane_model[2]
    znorm_utm_pcd = utm_pcd[:, 2] - ground_z

    #Remove ground
    tree_pcd = utm_pcd[znorm_utm_pcd >= ground_threshold]
    ground_pcd = utm_pcd[znorm_utm_pcd < ground_threshold]

    return tree_pcd, ground_pcd

def clean_tree_pcd(pcd, ground_threshold, ransac_repetition=5):
    """
    Cleans the tree point cloud data by removing the ground.

    :param numpy.ndarray pcd: The input point cloud data.
    :param float ground_threshold: The altitude threshold for classifying points as ground.
    :param int ransac_repetition: The number of RANSAC repetitions to perform. Default is 5.
    :return: The cleaned tree point cloud data without the ground and the angle
        to the XY plane.
    :rtype: tuple
    """

    _LOGGER.debug("Find ground plane")
    plane_model, theta_pitch, theta_roll = find_ground_plane(pcd, ransac_repetition)

    _LOGGER.debug("Segment tree and ground points")
    tree_pcd, ground_pcd = segment_ground(pcd, plane_model, ground_threshold)

    return tree_pcd, ground_pcd, plane_model, theta_pitch, theta_roll
