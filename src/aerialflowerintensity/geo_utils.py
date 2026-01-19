"""Utility classes and functions helpers to work with geographic coordinates."""

import numpy as np
import pyproj

def convert_coordinates(points, source_crs="EPSG:4326", target_crs="utm"):
    """
    Convert coordinates between different coordinate reference systems.

    :param points: Points to convert in the form ``[longitude, latitude, altitude]``.
    :type points: numpy.ndarray or list
    :param str source_crs: Source CRS identifier (EPSG code).
    :param str target_crs: Target CRS identifier (EPSG code or "utm").
        If "utm", automatically determines the appropriate UTM zone.
    :return: Converted points.
    :rtype: numpy.ndarray
    """
    points = np.atleast_2d(points)

    if source_crs.lower() == "utm":
        raise ValueError("source cannot take the special utm value")
    if target_crs.lower() == "utm":
        # This will not work if source is not lon, lat
        target_crs = _get_utm_crs_code(*points[0, :2])

    transformer = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)
    return np.asarray(transformer.transform(*points.T, errcheck=True)).squeeze()

def _get_utm_crs_code(longitude, latitude):
    """
    Get the EPSG code for the UTM zone covering a point.

    :param float longitude: Longitude of the point.
    :param float latitude: Latitude of the point.
    :return: UTM EPSG code identifier in the form ``EPSG:xxxxx``.
    :rtype: str
    """

    zone = _calculate_utm_zone(longitude, latitude)
    hemisphere = "N" if latitude >= 0 else "S"
    prefix = "326" if hemisphere == "N" else "327"
    return f"EPSG:{prefix}{zone}"

def _calculate_utm_zone(longitude, latitude):
    """
    Calculate the UTM zone number for a given point.

    :param float longitude: Longitude of the point.
    :param float latitude: Latitude of the point.
    :return: UTM zone identifier.
    :rtype: str
    """

    if not -80 < latitude < 84:
        raise ValueError(f"Latitude {latitude} outside valid UTM range (-80, 84)")

    zone_number = int(np.floor(((longitude + 180) / 6) % 60) + 1)

    # Special cases for specific regions
    if 56 <= latitude < 64 and 3 <= longitude < 12:
        zone_number = 32
    elif 72 <= latitude < 84:
        if 0 <= longitude < 9:
            zone_number = 31
        elif 9 <= longitude < 21:
            zone_number = 33
        elif 21 <= longitude < 33:
            zone_number = 35
        elif 33 <= longitude < 42:
            zone_number = 37

    return zone_number
