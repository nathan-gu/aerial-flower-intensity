
"""
This module provides utilities for creating visualization.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import is_color_like
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def build_scalar_map(float_array, color_map, v=None, norm="linear"):
    """
    Build a scalar map for mapping numbers to colors.

    :param numpy.ndarray float_array: The array of floating-point numbers to map to colors.
    :param matplotlib.colors.Colormap color_map: The colormap for mapping the numbers to colors.
    :param list v: Optional. The range of values for normalization. Default is [None, None].
    :param str norm: Optional. The normalization method. Options: "linear" or "log".
        Default is "linear".
    :return: The scalar map for mapping numbers to colors.
    :rtype: matplotlib.cm.ScalarMappable

    https://stackoverflow.com/questions/15140072/how-to-map-number-to-color-using-matplotlibs-colormap
    """
    if v is None:
        v = [None, None]
    vmin, vmax = v
    if vmin is None:
        vmin = float_array.min()
    if vmax is None:
        vmax = float_array.max()

    if norm == "log":
        norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    norm_scalar_map = cm.ScalarMappable(norm=norm, cmap=color_map)
    return norm_scalar_map

def scatter_colorful_pcd(pcd, ax=None, colors=None, sample_size=None, point_size=1):
    """
    Displays a colorful point cloud.

    :param numpy.ndarray pcd: Point cloud data.
    :param matplotlib.axes._subplots.Axes3DSubplot ax: Optional. The matplotlib 3D axis
        to plot on. If None, a new figure will be created. Default is None.
    :param numpy.ndarray colors: Optional. Color values for the points.
        If None, the colors will be derived from the point cloud data. Default is None.
    :param int sample_size: Optional. Number of points to sample for visualization.
        If None, display all points. Default is None.
    :param int point_size: Optional. Size of the points in the plot. Default is 1.
    """

    if sample_size is None or sample_size > pcd.shape[0]:
        sub_sample = np.tile(True, pcd.shape[0])
    else:
        sub_sample = np.random.choice(pcd.shape[0], sample_size, replace=False)

    pcd = pcd[sub_sample, :]

    points = pcd[:, :3]
    if colors is None:
        colors = pcd[:, -3:]

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    ax.scatter(points[:, 0],points[:, 1],points[:, 2], c=colors, s=point_size)

    if ax is None:
        ax.axis("equal")
        fig.show()

    return pcd

def cuboid_data(origin, size=(1, 1, 1)):
    """
    Returns the coordinates of the vertices of a cuboid.

    :param tuple origin: The origin point of the cuboid.
    :param tuple size: The size of the cuboid in each dimension. Default is (1, 1, 1).
    :return: An array containing the coordinates of the vertices of the cuboid.
    :rtype: numpy.ndarray

    https://stackoverflow.com/questions/49277753/plotting-cuboids
    """

    vertices = [[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
         [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
         [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
         [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
         [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
         [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]
    vertices = np.array(vertices).astype(float)
    for i in range(3):
        vertices[:, :, i] *= size[i]
    vertices += np.array(origin)
    return vertices

def plot_cuboids(positions, sizes=None, colors=None, **kwargs):
    """
    Plots multiple cuboids given their positions, sizes, and colors.

    :param list positions: List or array of the positions of the cuboids.
    :param list|np.array sizes: Optional. List or array of the sizes of the cuboids.
        If not provided, default size (1, 1, 1) is used for all cuboids.
    :param list|np.array colors: Optional. List or array of the colors of the cuboids.
        If not provided, default color "C0" is used for all cuboids.
    :param **kwargs: Additional keyword arguments to pass to the Poly3DCollection object.
    :return: A Poly3DCollection object representing the plotted cuboids.
    :rtype: matplotlib.collections.Poly3DCollection

    https://stackoverflow.com/questions/49277753/plotting-cuboids
    """
    if not isinstance(colors,(list,np.ndarray)):
        colors = ["C0"] * len(positions)
    if not isinstance(sizes,(list,np.ndarray)):
        sizes = [(1,1,1)] * len(positions)
    g = []
    for p,s,_c in zip(positions, sizes, colors):
        g.append( cuboid_data(p, size=s))
    return Poly3DCollection(np.concatenate(g), facecolors=np.repeat(colors, 6, axis=0), **kwargs)

def plot_voxels(voxels, voxel_size, colors=None, ax=None, color_map=cm.hot): # pylint: disable=no-member

    """
    Plots a set of voxels.

    :param numpy.ndarray voxels: The voxel data.
    :param float voxel_size: The size of each voxel.
    :param numpy.ndarray colors: Optional. The colors of the voxels. If not provided, default
        colors will be used.
    :param matplotlib.axes._subplots.Axes3DSubplot ax: Optional. The matplotlib 3D axis to
        plot on. If None, a new figure will be created. Default is None.
    :param matplotlib.colors.Colormap color_map: Optional. The color map to use for mapping
        voxel values to colors. Default is cm.hot.
    :return: The axis and the normalized scalar map.
    :rtype: tuple
    """
    passed_ax = True
    norm_scalar_map = None

    if colors is None or len(colors) == 0:
        colors=[[0.3, 0.3, 0.3, 0.5]] * len(voxels)
    elif len(colors) == len(voxels):
        if not is_color_like(colors[0]):
            colors = colors.reshape(-1)
            if isinstance(colors[0], (str, list, np.ndarray)):
                colors = [[0.3, 0.3, 0.3, 0.5]] * len(voxels)
            else:
                norm_scalar_map = build_scalar_map(colors, color_map)
                colors = norm_scalar_map.to_rgba(colors)
    elif len(colors) == 1 and is_color_like(colors[0]):
        colors = [colors[0]] * len(voxels)
    else:
        colors = [[0.3, 0.3, 0.3, 0.5]] * len(voxels)

    if ax is None:
        passed_ax = False
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    sizes = [voxel_size] * len(voxels)
    if len(sizes) > 0:
        poly_collection = plot_cuboids(voxels, sizes=sizes, colors=colors)
        ax.add_collection3d(poly_collection)

    if len(voxels) > 0:
        lims = np.concatenate(([voxels.min()] * 3,[voxels.max()] * 3)).reshape(2, 3).T
        ax.set_xlim(lims[0])
        ax.set_ylim(lims[1])
        ax.set_zlim(lims[2])

    if not passed_ax :
        plt.colorbar(norm_scalar_map, ax=ax)
        fig.show()

    return ax, norm_scalar_map
