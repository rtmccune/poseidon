"""Make two-site publication figures, one site per side.

Redraws the panels of ``publication_figures.py`` (and plot 1c of
``access_plotter.py``) for two sites side by side, the first ``--sites``
entry on the left. Each site uses its own config (ROI: the stem of
``camera_roi_json``, else ``roi_json``) and the files stage 99 reads
under ``<data_root>/<site>/``: ``flood_events/``,
``sensor_data_comparison.csv`` and ``segmentation_labels.csv``. Only
well-segmented images are used. Figures have no titles (captions name the
sites) and are saved as PNG (300 dpi) and PDF in ``--out_dir``.

``combined_fig1_aligned_timing``
    Each site's figure 1, panels (a)-(c) on the left and (d)-(f) on
    the right. Panel (a) is aligned on the in-situ sensor's peak (hour
    0), so it holds only events where the sensor saw water, and dry
    times count as 0 m (the sensor reads 0 m outside the event sheet's
    start to end), so each line is the expected depth at that time
    over every event rather than the mean among events still wet.
    Every line in (a) averages the same events: those with a sensor
    peak and at least one well-segmented image (the in-situ sensor
    line is limited to the events the camera lines use). Times are
    binned every ``--bin_minutes`` (default 30, five images at one
    every 6 min); a line breaks where a bin has fewer than 3 events.
    Panels (b) and (c) can hold fewer events: they also need water at
    the camera location and the sensor while the camera has images.
    The log gives each line's event count for the caption. The
    single-site script defaults to start alignment, 15 min bins, dry
    times left out and every sensor event (``--align``,
    ``--bin_minutes``, ``--dry``, ``--sensor_with_camera``).

``combined_fig2_depth_by_coverage``
    Per site, plot 1c for the mean roadway depth (logistic fit of
    depth on coverage with its median and 80 % prediction band; the fit's CI is in the fit summary CSV, not drawn)
    above figure 2 (mean, maximum and sensor depth by
    coverage bin; the depth axis starts just below 0 m so markers at
    0 m stay visible): (a), (b) on the left and (c), (d) on the right.
    Both panels use the same images: well segmented, with water on the
    road and an in-situ sensor reading within ``--match_minutes``.

Each row shares its y-axis across the two sites, so the sites compare
directly. The CSVs behind the panels that differ from the single-site
outputs are written beside the figures, prefixed with the site name:
the peak-aligned means of panel (a), the 1c fit summary, and
``_event_membership.csv``, which lists per event and camera location
whether it is in panel (a), panels (b)-(c), both or neither, and why
(a summary is logged). The
statistics in each site's ``<roi>_pub_metrics.md`` still apply; they do
not depend on the alignment.

Example
-------
``python combined_publication_figures.py --sites carolina_beach
down_east --data_root data --out_dir data/publication_combined``
"""

import argparse
import logging
import os

import matplotlib

matplotlib.use("Agg")  # headless HPC nodes

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from poseidon_utils.constants import FEET_TO_METERS  # noqa: E402
from poseidon_utils.logging_utils import setup_logging  # noqa: E402

import access_plotter as ap  # noqa: E402
import publication_figures as pub  # noqa: E402
import sensor_camera_comparison as sc  # noqa: E402
from plot_style import apply_style  # noqa: E402

logger = logging.getLogger(__name__)

#: Depth metric of the 1c panels.
FIT_METRIC = "MeanDepth"

#: Panel grid margins within each site's half of a figure.
MARGINS = {"left": 0.08, "right": 0.98, "top": 0.95, "bottom": 0.08}


def load_site(site, args, roi=None):
    """
    Load everything one site's panels need.

    Parameters
    ----------
    site : str
        Site config name.
    args : argparse.Namespace
        Parsed arguments.
    roi : str, optional
        ROI name. Default: from the site config (:func:`site_roi
        <publication_figures.site_roi>`).

    Returns
    -------
    dict
        ``site``, ``roi``, ``sensor``, ``series``, ``timing``
        and ``images`` (None if the ROI has no accessibility CSVs).

    Raises
    ------
    SystemExit
        If the site has no ROI, labels or sensor readings.
    """
    site_dir = os.path.join(args.data_root, os.path.basename(site))
    roi = roi or pub.site_roi(site)
    good = ap.load_good_timestamps(
        os.path.join(site_dir, "segmentation_labels.csv"))
    sensor = ap.load_sensor_data(
        os.path.join(site_dir, "sensor_data_comparison.csv"))
    if roi is None or not good or sensor is None:
        raise SystemExit(f"{site}: needs an ROI JSON in its config, "
                         f"segmentation_labels.csv and "
                         f"sensor_data_comparison.csv in {site_dir}")
    events_dir = os.path.join(site_dir, "flood_events")
    _, timing, series = sc.collect(
        events_dir, roi, sensor, good, args.match_minutes,
        args.padding_hours, wet_m=args.sensor_wet_ft * FEET_TO_METERS)
    images = pub.load_roi_images(events_dir, roi, good, sensor,
                                 args.padding_hours)
    logger.info("%s: ROI %s, %d events with camera series, %d images",
                site, roi, series["event"].nunique() if len(series) else 0,
                0 if images is None else len(images))
    return {"site": site, "roi": roi,
            "sensor": sensor, "series": series, "timing": timing,
            "images": images}


def _share_ylim(axes, bottom=0.0):
    """
    Give axes the same y-limits: from `bottom` to the highest top.

    Parameters
    ----------
    axes : list of matplotlib.axes.Axes
        Axes to align.
    bottom : float, optional
        Lower limit. Default 0. None keeps the lowest bottom.
    """
    lows, highs = zip(*(ax.get_ylim() for ax in axes))
    low = min(lows) if bottom is None else bottom
    for ax in axes:
        ax.set_ylim(low, max(highs))


def figure_one(sites, args, stem):
    """
    Make the two-site figure 1 (peak-aligned).

    Parameters
    ----------
    sites : list of dict
        Output of :func:`load_site` per side, left first.
    args : argparse.Namespace
        Parsed arguments (``align`` is ``'peak'``).
    stem : str
        Output filename stem.

    Returns
    -------
    bool
        Whether the figure was made.
    """
    if any(s["series"].empty for s in sites):
        logger.warning("A site has no well-segmented camera series; "
                       "figure 1 skipped.")
        return False
    fig = plt.figure(figsize=(17, 8.5))
    subfigs = fig.subfigures(1, 2, wspace=0)
    panels = []
    for i, (subfig, site) in enumerate(zip(subfigs, sites)):
        axes = pub.aligned_timing_axes(subfig, hspace=0.32, **MARGINS)
        result = pub.draw_aligned_timing(
            axes, site["series"], site["timing"], site["sensor"], args,
            letters="abcdef"[3 * i:3 * i + 3])
        counts = result["composites"].groupby("source")[
            "source_events"].first()
        logger.info("%s figure 1 (a) events per line: %s", site["site"],
                    counts.to_dict())
        result["composites"].to_csv(
            os.path.join(args.out_dir,
                         f"{site['site']}_{stem}_a_peak_means.csv"),
            index=False)
        pub.write_event_membership(
            site["series"], site["timing"], args,
            os.path.join(args.out_dir,
                         f"{site['site']}_{stem}_event_membership.csv"))
        panels.append(axes)
    _share_ylim([axes[0] for axes in panels])
    _share_ylim([ax for axes in panels for ax in axes[1:]], bottom=None)
    pub._save(fig, args.out_dir, stem)
    return True


def figure_two(sites, args, stem):
    """
    Make the two-site figure 2: plot 1c (mean depth) above figure 2.

    Parameters
    ----------
    sites : list of dict
        Output of :func:`load_site` per side, left first.
    args : argparse.Namespace
        Parsed arguments.
    stem : str
        Output filename stem.

    Returns
    -------
    bool
        Whether the figure was made.
    """
    prepared = []
    for site in sites:
        images = site["images"]
        if images is None:
            logger.warning("%s has no accessibility CSVs; figure 2 "
                           "skipped.", site["site"])
            return False
        # The fit uses the same sensor-matched images as the binned panel
        paired = pub.paired_images(images)
        if len(paired) <= 3:
            logger.warning("%s: only %d paired images; figure 2 skipped.",
                           site["site"], len(paired))
            return False
        logger.info("%s figure 2: %d images, %d events", site["site"],
                    len(paired), paired["event"].nunique())
        prepared.append((paired, pub.binned_depth_tables(paired, args)))

    # One depth scale per row: the deepest image (fit row) and the
    # highest band or CI (binned row) of either site
    fit_top = 1.05 * max(float(np.nanmax(good[FIT_METRIC]))
                         for good, _ in prepared)
    binned_top = max(pub.binned_depth_top(tables) for _, tables in prepared)

    fig = plt.figure(figsize=(17, 10.5))
    subfigs = fig.subfigures(1, 2, wspace=0)
    for i, (subfig, site, (good, tables)) in enumerate(
            zip(subfigs, sites, prepared)):
        grid = subfig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.22,
                                   **MARGINS)
        ax_fit = subfig.add_subplot(grid[0])
        fit = ap.draw_good_only_fit(
            ax_fit, good, FIT_METRIC, ap.PRETTY_METRICS[FIT_METRIC],
            args.n_boot, args.seed, ylim=(0, fit_top), legend_fontsize=8,
            name=site["site"], brief=True)
        if fit is not None:
            ap.write_fit_summary(fit, args.out_dir,
                                 f"{site['site']}_{stem}_1c_mean_depth")
        ax_fit.set_xlim(0, 100)
        ax, ax_full = pub.binned_depth_axes(subfig, grid[1])
        # A little room below 0 m so markers at 0 m stay visible
        pub.draw_binned_depth(ax, ax_full, tables, args,
                              ylim=(-0.04 * binned_top, binned_top))
        for ax_, letter in zip((ax_fit, ax), "abcd"[2 * i:2 * i + 2]):
            pub.panel_letter(ax_, letter)
    pub._save(fig, args.out_dir, stem)
    return True


def parse_args(argv=None):
    """
    Parse command-line arguments.

    Parameters
    ----------
    argv : list of str, optional
        Arguments to parse. Default reads ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments, with ``align`` set to ``'peak'``, ``dry`` to
        ``'zero'`` and ``sensor_with_camera`` on.
    """
    parser = argparse.ArgumentParser(
        description="Make two-site publication figures.")
    parser.add_argument("--sites", nargs=2, required=True,
                        metavar=("LEFT", "RIGHT"),
                        help="Site config names, left side first.")
    parser.add_argument("--data_root", default="data",
                        help="Folder holding one data folder per site.")
    parser.add_argument("--out_dir", required=True, help="Output folder.")
    parser.add_argument("--rois", nargs=2, default=None,
                        metavar=("LEFT", "RIGHT"),
                        help="ROI names, left first. Default: each site "
                             "config's camera_roi_json, else roi_json.")
    parser.add_argument("--bin_minutes", type=float, default=30,
                        help="Figure 1 (a): time bin width (minutes).")
    pub.add_figure_arguments(parser)
    args = parser.parse_args(argv)
    args.align, args.dry = "peak", "zero"
    args.sensor_with_camera = True
    return args


def main(argv=None):
    """
    Make both two-site figures.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments. Default reads ``sys.argv``.

    Returns
    -------
    int
        Process exit code: 1 if neither figure was made.
    """
    setup_logging()
    apply_style()
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    sites = [load_site(site, args, roi) for site, roi in
             zip(args.sites, args.rois or (None, None))]
    made = sum((figure_one(sites, args, "combined_fig1_aligned_timing"),
                figure_two(sites, args, "combined_fig2_depth_by_coverage")))
    logger.info("%d combined figures written to %s", made, args.out_dir)
    return 0 if made else 1


if __name__ == "__main__":
    raise SystemExit(main())
