"""
Utility functions for image handling using matplotlib, numpy, and opencv.
"""

import os
import io
import logging

import numpy as np
import cv2

_LOGGER = logging.getLogger(__name__)


def get_img_from_fig(fig, dpi=180):
    """
    Convert a matplotlib figure to a numpy array with RGB channels and return it.

    :param matplotlib.figure.Figure fig: Figure to convert.
    :param int dpi: Resolution of the figure.
    :return: Image as a numpy array with RGB channels.
    :rtype: numpy.ndarray
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    return img


def _save(img, filename, params=None):
    """
    Saves the image with appropriate parameters.

    :param np.ndarray img: Image to save.
    :param pathlib.Path filename: Path where the image will be generated.
    :param tuple | None params: Additional parameters to pass to the image writer.
    """
    filename.parent.mkdir(parents=True, exist_ok=True)
    filename = os.fspath(filename)
    cv2.imwrite(filename, img, params=params)


def save_visualisation(img, filename, params=(cv2.IMWRITE_WEBP_QUALITY, 50)):
    """
    Saves the visualisation figure with appropriate parameters.

    :param np.ndarray img: Image to save.
    :param pathlib.Path filename: Path where the image will be generated.
    :param Union [tuple, None] params: Additional parameters to pass to the image writer.
        Default is (cv2.IMWRITE_WEBP_QUALITY, 50).
    """
    if not filename.suffix == ".webp":
        _LOGGER.warning("filename suffix is not .webp, it will be replaced.")
        filename = filename.with_suffix(".webp")
    _save(img, filename, params=params)
