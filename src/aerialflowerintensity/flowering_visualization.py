"""
Module for visualizing the different step of flowering intensity measurements including
tree segmentation and flowering voxels from point cloud data.
"""


import pathlib

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea



from . import visualization_img
from . import image_utils as iu
from .visualization_utils import scatter_colorful_pcd, plot_voxels

def approx_2d_minimal_square(points):
    """
    Approximates the minimal square in 2D from a set of points.

    :param np.ndarray points: The input points.
    :return: The vertices of the minimal square.
    :rtype: np.ndarray
    """
    surface_vertices = np.concatenate((
        points[[points[:, 0].argmin()]],
        points[[points[:, 1].argmin()]],
        points[[points[:, 0].argmax()]],
        points[[points[:, 1].argmax()]]))[:,:2]

    return surface_vertices

def calc_plane_model_z(points, plane_model):
    """
    Calculates the z-coordinate of point in a plane given points and the
    plane model coefficients.

    :param np.ndarray points: The input points.
    :param tuple plane_model: The coefficients of the plane model equation.
    :return: The x, y, and z coordinates of the point in the plane.
    :rtype: tuple
    """
    x, y = points[:, 0], points[:, 1]
    z = -(plane_model[0] * x + plane_model[1] * y + plane_model[3]) / plane_model[2]
    return x, y, z

def approx_3d_minimal_surface(points, plane_model):
    """
    Approximates the minimal surface in 3D from a set of points and a plane model.

    :param np.ndarray points: The input points.
    :param tuple plane_model: The coefficients of the plane model equation.
    :return: The x, y, and z coordinates of the point in the plane.
    :rtype: tuple
    """
    surface_vertices = approx_2d_minimal_square(points)
    surface_vertices = closing_surface_loop(surface_vertices)
    return calc_plane_model_z(surface_vertices, plane_model)

def closing_surface_loop(vertices):
    """
    Closes the surface loop by duplicating the first vertex.

    :param np.ndarray vertices: The vertices of the surface.
    :return: The closed surface loop.
    :rtype: np.ndarray
    """
    return np.vstack([vertices, vertices[0]])

def tree_segmentation_visualization(
        flower_pcd, non_flower_pcd, ground_pcd, plane_model, soil_buffer, min_z_pcds,
        sample_size=None, view_angle=(0,0,0), dist=7, lims=None, ax=None):

    """
    Visualizes the segmentation of a tree based on point cloud data.
    :param np.ndarray flower_pcd: Flower points.
    :param np.ndarray non_flower_pcd: Non-flower points.
    :param np.ndarray ground_pcd: Ground points.
    :param tuple plane_model: The coefficients of the plane model equation.
    :param np.ndarray soil_buffer: The soil buffer corners points.
    :param float min_z_pcds: The minimum z-coordinate of the point clouds.
    :param int sample_size: Sample size for visualization.
        Default is None, all point are taken.
    :param tuple view_angle: View angle for visualization. Default is (0,0,0).
    :param float dist: Distance parameter for visualization. Default is 7.
    :param matplotlib.axes._subplots.Axes3DSubplot ax: The subplot to draw the visualization.
        Default is None, create a new figure and axis.
    :return: The axis with the visualization.
    :rtype: matplotlib.axes._subplots.Axes3DSubplot
    """

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    view_angle_name_dict = {0: "side view", 90:"top view"}
    if view_angle[0] in view_angle_name_dict:
        view_angle_name = view_angle_name_dict[view_angle[0]]
    else:
        view_angle_name = f"{view_angle[0]}°"

    if sample_size is not None:
        sample_size = int(sample_size/3)

    if flower_pcd is not None:
        _ = scatter_colorful_pcd(
            flower_pcd, ax, "tab:red", sample_size, 1)[:, :3]
    if non_flower_pcd is not None:
        _ = scatter_colorful_pcd(
            non_flower_pcd, ax, "tab:green", sample_size, 1)[:, :3]
    if ground_pcd is not None:
        sample_ground_points = scatter_colorful_pcd(
            ground_pcd, ax, "tab:brown", sample_size, 1)[:, :3]

        x, y, z = approx_3d_minimal_surface(sample_ground_points, plane_model)
        ax.plot_trisurf(x, y, z, color="tab:pink", alpha=0.7)

    soil_buffer = closing_surface_loop(soil_buffer)

    ax.plot(
        soil_buffer[:, 0], soil_buffer[:, 1], np.array([min_z_pcds] * soil_buffer.shape[0]),
        linewidth=3.0, color="tab:olive", linestyle="--", label="Soil Buffer")

    flower_patch = Patch(color="tab:red", alpha=1, label="Flower Points")
    non_flower_patch = Patch(color="tab:green", alpha=1, label="Non flower Points")
    ground_patch = Patch(color="tab:brown", alpha=1, label="Ground Points")
    surface_patch = Patch(color="tab:pink", alpha=0.7, label="Ground Plane")
    soil_buffer_patch = Patch(
        fill=False, edgecolor="tab:olive", linestyle="dashed", label="Soil Buffer", linewidth=1)

    ax.legend(handles=[
        flower_patch, non_flower_patch, ground_patch,
        surface_patch, soil_buffer_patch], loc="best")
    if lims is not None:
        ax.set_xlim(lims[0])
        ax.set_ylim(lims[1])
        ax.set_zlim(lims[2])
    ax.view_init(elev=view_angle[0], azim=view_angle[1], roll=view_angle[2])
    ax.dist = dist
    ax.grid(False)
    ax.axis("off")
    ax.set_aspect("equal")
    ax.set_title(f"Plot point cloud segmentation from {view_angle_name}")

    return ax

def flowering_voxels_visualization(
        flower_voxels, non_flower_voxels, voxel_size, soil_buffer, min_z_pcds, view_angle=(0,0,0),
        dist=7, lims=None, ax=None):
    """
    Visualizes flowering voxels from point cloud data.

    :param np.ndarray flower_voxels: The voxelized flower points.
    :param np.ndarray non_flower_voxels: The voxelized non flower points.
    :param float voxel_size: The size of the voxels.
    :param np.ndarray soil_buffer: The soil buffer corners points.
    :param float min_z_pcds: The minimum z-coordinate of the point clouds.
    :param tuple view_angle: View angle for visualization. Default is (0,0,0).
    :param float dist: Distance parameter for visualization. Default is 7.
    :param tuple lims: Limits for the visualization. Default is None.
    :param matplotlib.axes._subplots.Axes3DSubplot ax: The subplot to draw the visualization.
        Default is None, create a new figure and axis.
    :return: The axis with the visualization.
    :rtype: matplotlib.axes._subplots.Axes3DSubplot
    """

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    view_angle_name_dict = {0: "side view", 90:"top view"}
    if view_angle[0] in view_angle_name_dict:
        view_angle_name = view_angle_name_dict[view_angle[0]]
    else:
        view_angle_name = f"{view_angle[0]}°"

    if flower_voxels is not None:
        ax, _norm_scalar_map = plot_voxels(
            non_flower_voxels, voxel_size, colors=["tab:green"], ax=ax,
            color_map=get_flower_color_map())
    if non_flower_voxels is not None:
        ax, _norm_scalar_map = plot_voxels(
            flower_voxels, voxel_size, colors=["tab:red"], ax=ax,
            color_map=get_flower_color_map())

    soil_buffer = closing_surface_loop(soil_buffer)
    ax.plot(
        soil_buffer[:, 0], soil_buffer[:, 1], np.array([min_z_pcds] * soil_buffer.shape[0]),
        linewidth=3.0, color="tab:olive", linestyle="--", label="Soil Buffer")

    ax.view_init(elev=view_angle[0], azim=view_angle[1], roll=view_angle[2])
    if lims is not None:
        ax.set_xlim(lims[0])
        ax.set_ylim(lims[1])
        ax.set_zlim(lims[2])
    ax.dist = dist
    ax.grid(False)
    ax.axis("off")
    ax.set_aspect("equal")
    ax.set_title(f"Tree point cloud voxelization from {view_angle_name}")

    return ax

def get_flower_color_map():
    """
    Returns a custom color map for flower intensity purposes.

    :return: The custom color map.
    :rtype: matplotlib.colors.LinearSegmentedColormap
    """
    return LinearSegmentedColormap.from_list(
        "flowers", ["tab:green","yellow","tab:orange","tab:red"])

def plot_colorfull_pcd_visualization(
        plot_colorfull_pcd, point_size=1, dist=7, sample_size=None, view_angle=(0, 0, 0), ax=None):

    """
    Visualizes a colorfull plot point cloud.

    :param np.ndarray plot_colorfull_pcd: The colorfull plot point cloud.
    :param int point_size: Size of the points. Default is 1.
    :param float dist: Distance parameter for visualization. Default is 7.
    :param int sample_size: Sample size for visualization.
        Default is None, all point are taken.
    :param tuple view_angle: View angle for visualization. Default is (0,0,0).
    :param matplotlib.axes._subplots.Axes3DSubplot ax: The subplot to draw the visualization.
        Default is None, create a new figure and axis.
    :return: The axis with the visualization.
    :rtype: matplotlib.axes._subplots.Axes3DSubplot
    """

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    if plot_colorfull_pcd is not None:
        scatter_colorful_pcd(plot_colorfull_pcd, ax, None, sample_size, point_size)

    ax.view_init(elev=view_angle[0], azim=view_angle[1], roll=view_angle[2])
    ax.dist = dist
    ax.grid(False)
    ax.axis("off")
    ax.set_aspect("equal")
    ax.set_title("Colorfull plot point cloud")

    return ax

def ground_angle_visualization(
        pitch_img_path, roll_img_path, theta_pitch, theta_roll, x_box, y_box,
        zoom_icons, size_txt, ax=None):

    """
    Visualizes the ground plane pitch and roll.

    :param pathlib.Path pitch_img_path: Path to the pitch image.
    :param pathlib.Path roll_img_path: Path to the roll image.
    :param float theta_pitch: The pitch angle.
    :param float theta_roll: The roll angle.
    :param list x_box: The x-box coordinates.
    :param list y_box: The y-box coordinates.
    :param float zoom_icons: Zoom factor for the icons.
    :param float size_txt: Size of the text.
    :param matplotlib.axes._subplots.Axes3DSubplot ax: The subplot to draw the visualization.
        Default is None, create a new figure and axis.
    """

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    if theta_pitch is None:
        theta_pitch = np.nan
    if theta_roll is None:
        theta_roll = np.nan

    img_pitch = plt.imread(pitch_img_path)
    img_roll = plt.imread(roll_img_path)
    imagebox_pitch = OffsetImage(img_pitch, zoom=zoom_icons)
    imagebox_roll = OffsetImage(img_roll, zoom=zoom_icons)
    x_y = [0.5, 0.5]
    pitch = AnnotationBbox(
        imagebox_pitch,
        x_y,
        xybox=(x_box[0], y_box[0]),
        xycoords="data",
        boxcoords="offset points",
        pad=0,
        frameon=False)
    roll = AnnotationBbox(
        imagebox_roll,
        x_y,
        xybox=(x_box[1], y_box[0]),
        xycoords="data",
        boxcoords="offset points",
        pad=0,
        frameon=False)
    offsetbox_pitch = TextArea(
        "Pitch - Y", textprops={"size": size_txt})
    text_pitch = AnnotationBbox(
        offsetbox_pitch,
        x_y,
        xybox=(x_box[0], y_box[1]),
        xycoords="data",
        boxcoords="offset points",
        frameon=False)
    offsetbox_roll = TextArea(
        "Roll - X", textprops={"size": size_txt})
    text_roll = AnnotationBbox(
        offsetbox_roll,
        x_y,
        xybox=(x_box[1], y_box[1]),
        boxcoords="offset points",
        frameon=False)
    offsetbox_theta_y = TextArea(
        f"{theta_pitch:.3f}°", textprops={"size": size_txt})
    text_theta_y = AnnotationBbox(
        offsetbox_theta_y,
        x_y,
        xybox=(x_box[0], y_box[2]),
        xycoords="data",
        boxcoords="offset points",
        frameon=False)
    offsetbox_theta_x = TextArea(
        f"{theta_roll:.3f}°", textprops={"size": size_txt})
    text_theta_x = AnnotationBbox(
        offsetbox_theta_x,
        x_y,
        xybox=(x_box[1], y_box[2]),
        xycoords="data",
        boxcoords="offset points",
        frameon=False)
    ax.add_artist(pitch)
    ax.add_artist(roll)
    ax.add_artist(text_pitch)
    ax.add_artist(text_roll)
    ax.add_artist(text_theta_y)
    ax.add_artist(text_theta_x)
    ax.axis("off")
    ax.set_title("Ground plane pitch and roll")

def plot_flowering_visualization(
        visualization_output_path, las_plot_name, flower_pcd, non_flower_pcd, ground_pcd,
        plane_model, flower_voxels, non_flower_voxels, voxel_size, soil_buffer, theta_pitch,
        theta_roll, sample_size=None):

    """
    Generate and save a visualization of the flowering assessment process of a tree.

    :param pathlib.Path visualization_output_path: Path to save the visualization.
    :param str las_plot_name: Name of the plot.
    :param np.ndarray flower_pcd: Flower points.
    :param np.ndarray non_flower_pcd: Non-flower points.
    :param np.ndarray ground_pcd: Ground points.
    :param tuple plane_model: The coefficients of the plane model equation.
    :param np.ndarray flower_voxels: The voxelized flower points.
    :param np.ndarray non_flower_voxels: The voxelized non flower points.
    :param float voxel_size: The size of the voxels.
    :param np.ndarray soil_buffer: The soil buffer corners points.
    :param float theta_pitch: The pitch angle.
    :param float theta_roll: The roll angle.
    :param int sample_size: Sample size for visualization. Default is None, all points are taken.

    :return: None
    """

    plot_pcds = [arr for arr in (flower_pcd, non_flower_pcd, ground_pcd) if arr is not None]
    if not plot_pcds:
        plot_pcds = None
        soil_buffer_z = 0
        lims = None
    else:
        plot_pcds = np.concatenate(plot_pcds)
        plot_points = plot_pcds[:, :3]
        soil_buffer_z = plot_points[:, 2].min()
        lims = np.concatenate(
            (
                plot_points.min(axis=0),
                plot_points.min(axis=0) + (plot_points.max(axis=0) - plot_points.min(axis=0)
            ).max())).reshape(2, 3).T

    tree_pcds = [arr for arr in (flower_pcd, non_flower_pcd) if arr is not None]
    if tree_pcds:
        tree_pcds = np.concatenate(tree_pcds)
        tree_pcds_min = tree_pcds[:, :3].min(axis=0)
        flower_voxels = flower_voxels * voxel_size + tree_pcds_min
        non_flower_voxels = non_flower_voxels * voxel_size + tree_pcds_min

    visualization_img_path = pathlib.Path(visualization_img.__file__).parent
    pitch_img_path = visualization_img_path/"pitch.jpg"
    roll_img_path = visualization_img_path/"roll.jpg"

    fig, axs = plt.subplots(2, 3, subplot_kw={"projection": "3d"}, figsize=(16, 9))
    axs[1,0].remove()
    axs[1,0] = fig.add_subplot(2, 3, 4)
    fig.subplots_adjust(hspace=0, wspace=0, left=0.02, right=0.98, top=0.90, bottom=0.05)
    fig.suptitle(las_plot_name)


    plot_colorfull_pcd_visualization(
        plot_pcds, point_size=3, sample_size=sample_size, view_angle=(0, 0, 0), ax=axs[0, 0])

    tree_segmentation_visualization(
        flower_pcd, non_flower_pcd, ground_pcd, plane_model, soil_buffer, soil_buffer_z,
        sample_size=sample_size, view_angle=(0, 0, 0), lims=lims, ax=axs[0, 1])

    tree_segmentation_visualization(
        flower_pcd, non_flower_pcd, ground_pcd, plane_model, soil_buffer, soil_buffer_z,
        sample_size=sample_size, view_angle=(90, 0, 0), lims=lims, ax=axs[0, 2])

    flowering_voxels_visualization(
        flower_voxels, non_flower_voxels, voxel_size, soil_buffer, soil_buffer_z,
        view_angle=(0, 0, 0), dist=7, lims=lims, ax=axs[1,1])

    flowering_voxels_visualization(
        flower_voxels, non_flower_voxels, voxel_size, soil_buffer, soil_buffer_z,
        view_angle=(90, 0, 0), dist=7, lims=lims, ax=axs[1,2])

    ground_angle_visualization(
        pitch_img_path, roll_img_path, theta_pitch, theta_roll, x_box = [-50, 50],
        y_box = [0, -40, -60], zoom_icons=0.25, size_txt=15.0, ax=axs[1,0])

    img = iu.get_img_from_fig(fig)
    plt.close(fig)
    iu.save_visualisation(img, visualization_output_path/f"plot_{las_plot_name}.webp")
