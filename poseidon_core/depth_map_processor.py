"""Estimate water surface elevation and depth from rectified water masks.

For each rectified label image, connected water regions ("ponds") are
identified and their waterline is taken to be the pond's inner
boundary: pond pixels that touch a non-pond neighbor. The water
surface elevation (WSE) of a pond is a statistic (mean, median, 90th
or 95th percentile) of the ground elevation along that waterline, and
depth is the WSE minus the ground elevation inside the pond, floored
at zero.

Runs on the GPU with CuPy/cuCIM when available and falls back to
NumPy/scikit-image otherwise.
"""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np

from poseidon_utils.logging_utils import progress_interval, should_report
from poseidon_utils.zarr_io import (
    array_names,
    open_zip_group,
    write_array,
    write_zip_group,
)

from .backend import get_xp, resolve_use_gpu, to_numpy

logger = logging.getLogger(__name__)

#: Waterline statistics computed for every pond, by map-name suffix.
STATISTICS = {
    "mean": np.mean,
    "median": np.median,
    "95_perc": lambda values: np.percentile(values, 95),
    "90_perc": lambda values: np.percentile(values, 90),
}


def _image_ops(use_gpu):
    """
    Return the labeling and morphology functions for a backend.

    Parameters
    ----------
    use_gpu : bool
        True for cuCIM (GPU), False for scikit-image (CPU).

    Returns
    -------
    label : callable
        Connected-component labeling.
    binary_closing : callable
        Binary morphological closing.
    find_boundaries : callable
        Label boundary detection.
    """
    if use_gpu:
        from cucim.skimage.measure import label
        from cucim.skimage.morphology import binary_closing
        from cucim.skimage.segmentation import find_boundaries
    else:
        from skimage.measure import label
        from skimage.morphology import binary_closing
        from skimage.segmentation import find_boundaries
    return label, binary_closing, find_boundaries


class DepthMapProcessor:
    """
    Compute WSE and depth maps from rectified segmentation labels.

    Parameters
    ----------
    elevation_grid : array_like
        2-D ground elevation grid (meters) matching the rectified
        label images.
    plot_edges : bool, optional
        Save waterline elevation histograms for every image. Default
        is True.
    use_gpu : bool or None, optional
        True for CuPy, False for NumPy, None to use the GPU when
        available. Default is None.
    min_pond_size : int, optional
        Ponds smaller than this many pixels are discarded. Default 55.
    water_class : int, optional
        Label value that marks water. Default is 1.
    pond_edge_elev_plot_dir : str, optional
        Default output folder for histograms. Default is
        ``"data/edge_histograms"``.
    hist_range : tuple of float, optional
        Elevation range (meters) of histogram bins and axes. Default
        is ``(0, 2)``.

    Attributes
    ----------
    elev_grid : numpy.ndarray or cupy.ndarray
        Elevation grid on the active backend.
    """

    def __init__(self, elevation_grid, plot_edges=True, use_gpu=None,
                 min_pond_size=55, water_class=1,
                 pond_edge_elev_plot_dir="data/edge_histograms",
                 hist_range=(0, 2)):
        """
        Store the elevation grid and processing options.

        Parameters
        ----------
        elevation_grid : array_like
            2-D ground elevation grid (meters).
        plot_edges : bool, optional
            Save waterline elevation histograms. Default is True.
        use_gpu : bool or None, optional
            Backend choice; see the class docstring. Default is None.
        min_pond_size : int, optional
            Minimum pond size in pixels. Default is 55.
        water_class : int, optional
            Label value that marks water. Default is 1.
        pond_edge_elev_plot_dir : str, optional
            Default histogram folder. Default ``"data/edge_histograms"``.
        hist_range : tuple of float, optional
            Histogram elevation range in meters. Default ``(0, 2)``.
        """
        self.use_gpu = resolve_use_gpu(use_gpu)
        self.xp = get_xp(self.use_gpu)
        self.elev_grid = self.xp.asarray(elevation_grid, dtype=float)
        self.plot_edges = plot_edges
        self.min_pond_size = min_pond_size
        self.water_class = water_class
        self.pond_edge_elev_plot_dir = pond_edge_elev_plot_dir
        self.hist_range = hist_range
        self._label, self._closing, self._boundaries = _image_ops(
            self.use_gpu)

    def process_single_depth_map(self, label_array, file_name):
        """
        Compute every WSE and depth map for one rectified label image.

        Parameters
        ----------
        label_array : array_like
            Rectified labels, shape ``(rows, cols)`` or
            ``(rows, cols, 1)``.
        file_name : str
            Base name used for output map names and histogram files.

        Returns
        -------
        dict of str to array
            Maps named ``<file_name>_wse_map_<stat>`` and
            ``<file_name>_depth_map_<stat>`` for each statistic in
            :data:`STATISTICS`. Cells outside any pond are NaN.
        """
        labels = self._label_ponds(self.xp.asarray(label_array))
        edge_elevs = self._extract_edge_elevations(labels)

        if self.plot_edges:
            self._plot_pond_edge_elevations(edge_elevs, file_name)

        maps = self._calculate_all_depths(labels, edge_elevs)
        return {f"{file_name}_{suffix}": array
                for suffix, array in maps.items()}

    def process_depth_maps(self, labels_zarr_zip_path,
                           depth_map_zarr_zip_path,
                           pond_edge_elev_plot_dir=None):
        """
        Compute depth maps for every label array in a ZIP Zarr store.

        Parameters
        ----------
        labels_zarr_zip_path : str or os.PathLike
            Input store of rectified label arrays.
        depth_map_zarr_zip_path : str or os.PathLike
            Output store; replaced if it already exists.
        pond_edge_elev_plot_dir : str, optional
            Histogram folder for this run. Default keeps the folder set
            at initialization.

        Returns
        -------
        int
            Number of label arrays processed successfully.
        """
        if pond_edge_elev_plot_dir is not None:
            self.pond_edge_elev_plot_dir = pond_edge_elev_plot_dir

        if not os.path.exists(labels_zarr_zip_path):
            logger.error("Label store not found: %s", labels_zarr_zip_path)
            return 0

        logger.info("Computing depth maps: %s -> %s",
                    labels_zarr_zip_path, depth_map_zarr_zip_path)
        with open_zip_group(labels_zarr_zip_path) as in_group:
            names = array_names(in_group)
            if not names:
                logger.warning("No label arrays in %s; nothing written.",
                               labels_zarr_zip_path)
                return 0

            total = len(names)
            interval = progress_interval(total)
            processed = 0
            with write_zip_group(depth_map_zarr_zip_path) as out_group:
                for i, name in enumerate(names):
                    if should_report(i, total, interval):
                        logger.info("  Label array %d/%d: %s",
                                    i + 1, total, name)
                    try:
                        maps = self.process_single_depth_map(
                            in_group[name][:], name)
                    except Exception:  # noqa: BLE001 - keep the batch going
                        logger.exception("  Failed on %s; skipping.", name)
                        continue
                    for map_name, array in maps.items():
                        write_array(out_group, map_name, to_numpy(array))
                    processed += 1

        logger.info("Processed %d/%d label arrays.", processed, total)
        return processed

    def _label_ponds(self, label_array):
        """
        Label connected water regions and drop small ones.

        A 3x3 binary closing joins water pixels split by thin gaps
        before 8-connected labeling. Ponds below ``min_pond_size``
        pixels are removed and the rest renumbered 1..N with a lookup
        table, avoiding a second labeling pass.

        Parameters
        ----------
        label_array : numpy.ndarray or cupy.ndarray
            Rectified labels on the active backend.

        Returns
        -------
        numpy.ndarray or cupy.ndarray
            Integer pond IDs, 0 for background.
        """
        xp = self.xp
        water = label_array.squeeze() == self.water_class
        closed = self._closing(water, footprint=xp.ones((3, 3), dtype=bool))
        labeled, n_ponds = self._label(closed, return_num=True)
        if n_ponds == 0:
            return labeled

        sizes = xp.bincount(labeled.ravel(), minlength=n_ponds + 1)
        all_ids = xp.arange(n_ponds + 1)
        kept_ids = all_ids[(sizes >= self.min_pond_size) & (all_ids != 0)]

        # Map kept IDs to 1..N; small ponds and background map to 0
        lookup = xp.zeros(n_ponds + 1, dtype=xp.int32)
        lookup[kept_ids] = xp.arange(1, kept_ids.size + 1, dtype=xp.int32)
        return lookup[labeled]

    def _extract_edge_elevations(self, labels):
        """
        Collect finite ground elevations along each pond's waterline.

        The waterline is the pond's inner boundary: pond pixels with at
        least one 4-connected neighbor outside the pond. Pixels on the
        grid border are not treated as waterline.

        Parameters
        ----------
        labels : numpy.ndarray or cupy.ndarray
            Pond IDs from :meth:`_label_ponds`.

        Returns
        -------
        dict of int to numpy.ndarray
            Waterline elevations per pond ID. Ponds whose waterline
            has no finite elevation are omitted.
        """
        edges = self._boundaries(labels, mode="inner") & (labels > 0)
        pond_ids = to_numpy(labels[edges])
        elevations = to_numpy(self.elev_grid[edges])

        finite = np.isfinite(elevations)
        pond_ids, elevations = pond_ids[finite], elevations[finite]

        # Group elevations by pond with one sort instead of N masks
        order = np.argsort(pond_ids, kind="stable")
        pond_ids, elevations = pond_ids[order], elevations[order]
        unique_ids, starts = np.unique(pond_ids, return_index=True)
        groups = np.split(elevations, starts[1:])
        return {int(pid): values for pid, values in zip(unique_ids, groups)}

    def _calculate_all_depths(self, labels, edge_elevs):
        """
        Build WSE and depth maps for every statistic.

        Parameters
        ----------
        labels : numpy.ndarray or cupy.ndarray
            Pond IDs from :meth:`_label_ponds`.
        edge_elevs : dict of int to numpy.ndarray
            Waterline elevations per pond.

        Returns
        -------
        dict of str to array
            ``wse_map_<stat>`` and ``depth_map_<stat>`` arrays on the
            active backend; NaN outside ponds with a valid waterline.
        """
        xp = self.xp
        n_ponds = int(labels.max()) if labels.size else 0
        maps = {}
        for suffix, statistic in STATISTICS.items():
            # Per-pond WSE lookup; ID 0 (background) stays NaN
            wse_by_id = np.full(n_ponds + 1, np.nan)
            for pond_id, values in edge_elevs.items():
                wse_by_id[pond_id] = statistic(values)

            wse = xp.asarray(wse_by_id)[labels]
            maps[f"wse_map_{suffix}"] = wse
            maps[f"depth_map_{suffix}"] = xp.maximum(wse - self.elev_grid,
                                                     0.0)
        return maps

    def _plot_pond_edge_elevations(self, edge_elevs, file_name):
        """
        Save waterline elevation histograms per pond and combined.

        Parameters
        ----------
        edge_elevs : dict of int to numpy.ndarray
            Waterline elevations per pond.
        file_name : str
            Base name for the saved figures.
        """
        if not edge_elevs:
            logger.debug("No pond waterlines in %s; no histograms.",
                         file_name)
            return

        all_dir = os.path.join(self.pond_edge_elev_plot_dir, "all_ponds")
        ind_dir = os.path.join(self.pond_edge_elev_plot_dir, "ind_ponds")
        os.makedirs(all_dir, exist_ok=True)
        os.makedirs(ind_dir, exist_ok=True)
        bins = np.linspace(*self.hist_range, 51)

        for pond_id, values in edge_elevs.items():
            self._plot_histogram(
                values, bins, f"Elevation Histogram for Pond {pond_id}",
                os.path.join(ind_dir, f"{file_name}_Pond_{pond_id}"),
                color="lightblue")

        self._plot_histogram(
            np.concatenate(list(edge_elevs.values())), bins,
            "Elevation Histogram - All Ponds Combined",
            os.path.join(all_dir, f"{file_name}_All_Ponds_Histogram"),
            color="lightcoral")

    def _plot_histogram(self, data, bins, title, save_path, color):
        """
        Save one histogram with mean, median, and 95th percentile lines.

        Parameters
        ----------
        data : numpy.ndarray
            1-D elevations to plot.
        bins : numpy.ndarray
            Histogram bin edges.
        title : str
            Figure title.
        save_path : str
            Output path; matplotlib adds the ``.png`` extension.
        color : str
            Bar color.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(data, bins=bins, color=color, edgecolor="black", alpha=0.7)
        for value, line_color, style, label in (
                (np.mean(data), "red", "dashed", "Mean"),
                (np.median(data), "green", "solid", "Median"),
                (np.percentile(data, 95), "purple", "dashdot", "95th %ile")):
            ax.axvline(value, color=line_color, linestyle=style,
                       linewidth=2, label=f"{label}: {value:.2f}")

        ax.set_xlim(*self.hist_range)
        ax.set_xlabel("Elevation")
        ax.set_ylabel("Frequency")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)
        fig.savefig(save_path)
        plt.close(fig)
