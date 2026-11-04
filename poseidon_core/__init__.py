"""Photogrammetric processing pipeline for POSEIDON.

Classes
-------
GridGenerator
    Build X/Y/Z elevation grids from LiDAR point clouds.
ImageRectifier
    Project camera images onto an elevation grid.
DepthMapProcessor
    Turn rectified water masks into water surface and depth maps.
DepthMapPlotter
    Plot georeferenced depth maps and virtual sensor time series.
RoadwayAnalyzer
    Summarize flooding inside a LabelMe roadway line or polygon.

Each class is imported on first access, so ``import poseidon_core``
does not require optional GPU or geospatial plotting libraries.
"""

import importlib

_LAZY_ATTRIBUTES = {
    "GridGenerator": "grid_generator",
    "ImageRectifier": "image_rectifier",
    "DepthMapProcessor": "depth_map_processor",
    "DepthMapPlotter": "depth_map_plotter",
    "RoadwayAnalyzer": "roadway_analyzer",
}

__all__ = list(_LAZY_ATTRIBUTES)


def __getattr__(name):
    """
    Import a public class on first access.

    Parameters
    ----------
    name : str
        Attribute requested from the package.

    Returns
    -------
    type
        The requested class.

    Raises
    ------
    AttributeError
        If `name` is not a public class of the package.
    """
    if name in _LAZY_ATTRIBUTES:
        module = importlib.import_module(f".{_LAZY_ATTRIBUTES[name]}",
                                         __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
