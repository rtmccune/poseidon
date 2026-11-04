"""Summarize flooding on a roadway region of interest over a flood event.

A region of interest (ROI) is drawn in LabelMe on a rectified image as
either a line/linestrip (a transect along the road) or a polygon (the
road surface). For every depth and water surface elevation (WSE) map
in an event, the values inside the ROI are extracted and reduced to a
per-timestep CSV:

* ``<roi>_accessibility_time_series.csv`` - flooded pixel counts,
  impassability flag, and depth statistics.
* ``<roi>_wse_time_series.csv`` - WSE statistics.

The raw ROI samples are also saved as ``zarr/<roi>_depths.zip`` and
``zarr/<roi>_wse.zip`` for later analysis.
"""

import json
import logging
import os
import warnings

import numpy as np
import pandas as pd
from skimage.draw import polygon

from poseidon_utils.naming import extract_timestamp
from poseidon_utils.zarr_io import (
    array_names,
    open_zip_group,
    write_array,
    write_zip_group,
)

logger = logging.getLogger(__name__)

#: LabelMe shape types treated as transects.
LINE_SHAPES = ("line", "linestrip")


def roi_pixels_from_labelme(json_path, label, step_size=1.0):
    """
    Read a LabelMe shape and return the image pixels it covers.

    Parameters
    ----------
    json_path : str or os.PathLike
        LabelMe JSON file drawn on a rectified image.
    label : str
        Label of the shape to use.
    step_size : float, optional
        Spacing in pixels between samples along a line. Default 1.0.

    Returns
    -------
    coords : numpy.ndarray
        ``(N, 2)`` integer ``(row, col)`` pixels in image orientation
        (row 0 at the top / northern edge).
    shape_type : str
        LabelMe shape type (``"polygon"``, ``"line"``, ``"linestrip"``).

    Raises
    ------
    ValueError
        If no shape has `label` or its type is unsupported.
    """
    with open(json_path) as handle:
        shapes = json.load(handle).get("shapes", [])

    shape = next((s for s in shapes if s.get("label") == label), None)
    if shape is None:
        found = sorted({s.get("label", "?") for s in shapes})
        raise ValueError(f"No shape labeled '{label}' in {json_path}; "
                         f"found {found}")

    points = np.asarray(shape["points"], dtype=float)  # (x, y) vertices
    shape_type = shape["shape_type"]

    if shape_type in LINE_SHAPES:
        # Resample the polyline at even arc-length spacing
        seg_lengths = np.hypot(*np.diff(points, axis=0).T)
        arc = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        samples = np.arange(0.0, arc[-1], step_size)
        if samples.size == 0 or samples[-1] < arc[-1]:
            samples = np.append(samples, arc[-1])  # include the end vertex
        cols = np.interp(samples, arc, points[:, 0])
        rows = np.interp(samples, arc, points[:, 1])
        coords = np.column_stack((np.round(rows), np.round(cols)))
        return coords.astype(int), shape_type

    if shape_type == "polygon":
        rows, cols = polygon(points[:, 1], points[:, 0])
        return np.column_stack((rows, cols)).astype(int), shape_type

    raise ValueError(f"Unsupported shape_type '{shape_type}' for label "
                     f"'{label}'")


def sample_roi(grid_map, coords):
    """
    Sample a north-up image ROI from a south-up grid map.

    Grid maps are stored with row 0 at the southern edge, while
    LabelMe coordinates come from images displayed north-up, so rows
    are flipped before sampling.

    Parameters
    ----------
    grid_map : numpy.ndarray
        2-D map in grid orientation.
    coords : numpy.ndarray
        ``(N, 2)`` ``(row, col)`` pixels in image orientation.

    Returns
    -------
    numpy.ndarray
        ``(N,)`` float32 samples; NaN for pixels outside the map.
    """
    height, width = grid_map.shape
    rows = (height - 1) - coords[:, 0]
    cols = coords[:, 1]
    inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)

    values = np.full(len(coords), np.nan, dtype=np.float32)
    values[inside] = grid_map[rows[inside], cols[inside]]
    return values


def accessibility_table(times, depths):
    """
    Summarize ROI depth samples per timestep.

    Parameters
    ----------
    times : pandas.DatetimeIndex
        Timestamp of each row of `depths`.
    depths : numpy.ndarray
        ``(T, N)`` ROI depth samples; NaN where there is no water.

    Returns
    -------
    pandas.DataFrame
        Columns ``Time``, ``Impassable`` (1 if any ROI pixel has depth
        above zero), ``FloodedPixels``, ``TotalPixels``, ``MeanDepth``,
        ``MedianDepth``, ``MaxDepth``, ``MinDepth``, sorted by time.
    """
    flooded = depths > 0.0
    with warnings.catch_warnings():
        # Dry timesteps have all-NaN rows; NaN is the right answer
        warnings.simplefilter("ignore", RuntimeWarning)
        table = pd.DataFrame({
            "Time": times,
            "Impassable": flooded.any(axis=1).astype(int),
            "FloodedPixels": flooded.sum(axis=1),
            "TotalPixels": depths.shape[1],
            "MeanDepth": np.nanmean(depths, axis=1),
            "MedianDepth": np.nanmedian(depths, axis=1),
            "MaxDepth": np.nanmax(depths, axis=1),
            "MinDepth": np.nanmin(depths, axis=1),
        })
    return table.sort_values("Time")


def wse_table(times, wse):
    """
    Summarize ROI water surface elevation samples per timestep.

    Parameters
    ----------
    times : pandas.DatetimeIndex
        Timestamp of each row of `wse`.
    wse : numpy.ndarray
        ``(T, N)`` ROI WSE samples; NaN where there is no water.

    Returns
    -------
    pandas.DataFrame
        Columns ``Time``, ``MeanWSE``, ``MedianWSE``, ``MaxWSE``,
        sorted by time.
    """
    with warnings.catch_warnings():
        # Dry timesteps have all-NaN rows; NaN is the right answer
        warnings.simplefilter("ignore", RuntimeWarning)
        table = pd.DataFrame({
            "Time": times,
            "MeanWSE": np.nanmean(wse, axis=1),
            "MedianWSE": np.nanmedian(wse, axis=1),
            "MaxWSE": np.nanmax(wse, axis=1),
        })
    return table.sort_values("Time")


class RoadwayAnalyzer:
    """
    Extract and summarize depth and WSE on a roadway ROI for one event.

    Parameters
    ----------
    target_event_dir : str or os.PathLike
        Event folder containing ``zarr/depth_maps.zip``.
    labelme_json_path : str or os.PathLike
        LabelMe JSON defining the ROI on a rectified image.
    roi_label : str, optional
        Label of the ROI shape. Default is ``"roadway"``.
    step_size : float, optional
        Sample spacing in pixels along line ROIs. Default is 1.0.
    statistic : str, optional
        Depth/WSE map statistic to analyze (``"95_perc"``,
        ``"90_perc"``, ``"mean"``, ``"median"``). Default ``"95_perc"``.

    Attributes
    ----------
    roi_name : str
        ROI name taken from the JSON filename; prefixes all outputs.
    roi_coords : numpy.ndarray
        ``(N, 2)`` ROI pixels in image orientation.
    geom_type : str
        LabelMe shape type of the ROI.
    """

    def __init__(self, target_event_dir, labelme_json_path,
                 roi_label="roadway", step_size=1.0, statistic="95_perc"):
        """
        Load the ROI geometry for an event.

        Parameters
        ----------
        target_event_dir : str or os.PathLike
            Event folder containing ``zarr/depth_maps.zip``.
        labelme_json_path : str or os.PathLike
            LabelMe JSON defining the ROI.
        roi_label : str, optional
            Label of the ROI shape. Default is ``"roadway"``.
        step_size : float, optional
            Sample spacing along line ROIs. Default is 1.0.
        statistic : str, optional
            Map statistic to analyze. Default is ``"95_perc"``.
        """
        self.event_path = os.fspath(target_event_dir)
        self.event_name = os.path.basename(os.path.normpath(self.event_path))
        self.statistic = statistic
        self.roi_name = os.path.splitext(
            os.path.basename(labelme_json_path))[0]
        self.roi_coords, self.geom_type = roi_pixels_from_labelme(
            labelme_json_path, roi_label, step_size)
        logger.info("RoadwayAnalyzer: %s ROI '%s' with %d pixels "
                    "(statistic %s).", self.geom_type, self.roi_name,
                    len(self.roi_coords), statistic)

    def process_single_event(self):
        """
        Extract ROI samples and write the accessibility and WSE CSVs.

        Returns
        -------
        bool
            True if the event's depth maps were found and processed.
        """
        logger.info("Analyzing roadway for %s", self.event_name)
        depth = self._extract_roi_series("depth")
        if depth is None:
            return False
        times, depths = depth
        self._save_csv(accessibility_table(times, depths),
                       "accessibility_time_series")

        wse = self._extract_roi_series("wse")
        if wse is not None:
            self._save_csv(wse_table(*wse), "wse_time_series")
        return True

    def _extract_roi_series(self, kind):
        """
        Sample the ROI from every depth or WSE map in the event.

        Parameters
        ----------
        kind : {'depth', 'wse'}
            Which map family to read.

        Returns
        -------
        tuple of (pandas.DatetimeIndex, numpy.ndarray) or None
            Timestamps and ``(T, N)`` samples, or None if the event has
            no matching maps.
        """
        zarr_dir = os.path.join(self.event_path, "zarr")
        in_path = os.path.join(zarr_dir, "depth_maps.zip")
        if not os.path.exists(in_path):
            logger.warning("Depth maps not found: %s", in_path)
            return None

        suffix = f"{kind}_map_{self.statistic}"
        with open_zip_group(in_path) as group:
            names = array_names(group, suffix=suffix)
            if not names:
                logger.warning("No maps ending in '%s' in %s",
                               suffix, in_path)
                return None
            samples = np.stack([sample_roi(group[name][:], self.roi_coords)
                                for name in names])

        stamps = [extract_timestamp(name) for name in names]
        times = pd.to_datetime(stamps, format="%Y%m%d%H%M%S", utc=True)

        # Keep the raw samples next to the event's other Zarr products
        out_name = "depths" if kind == "depth" else "wse"
        out_path = os.path.join(zarr_dir, f"{self.roi_name}_{out_name}.zip")
        with write_zip_group(out_path) as out_group:
            stamp_text = np.array(times.astype(str), dtype="U30")
            write_array(out_group, "timestamps", stamp_text)
            write_array(out_group, f"{self.roi_name}_{out_name}", samples)
        return times, samples

    def _save_csv(self, table, suffix):
        """
        Write a summary table to the event folder.

        Parameters
        ----------
        table : pandas.DataFrame
            Table to save.
        suffix : str
            Filename suffix after ``<roi_name>_``.

        Returns
        -------
        str
            Path of the written CSV.
        """
        path = os.path.join(self.event_path, f"{self.roi_name}_{suffix}.csv")
        table.to_csv(path, index=False)
        logger.info("Saved %s", path)
        return path
