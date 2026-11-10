"""Export the example images for the methods process diagram.

Picks one well-segmented image from one flood event (or the ``--event``
and ``--timestamp`` given) and writes the figure panels for that moment
to ``--out_dir``, plus ``manifest.json`` (what was used, and each
image's pixel size) for ``methods_process_diagram.py``:

=========================  ==================================================
``01_original.jpg``        The camera image, downsized.
``02_overlay.jpg``         Segmentation overlay from stage 04 (else drawn
                           from the label image).
``03_classes.png``         Predicted class per pixel, in the label colours.
``04_water_mask.png``      Water class only (``--water_class``) on the dimmed
                           image.
``05_camera_roi.jpg``      The camera-view roadway polygon on the image
                           (site ``camera_roi_json``), if set.
``06_dem.png``             Ground elevation grid from LiDAR (stage 06),
                           shaded, north up.
``07_rectified.png``       The image rectified onto the grid (stage 07).
``08_rectified_water.png`` Rectified water mask on the rectified image.
``09_depth.png``           Depth map (stage 08, ``--statistic``) on the
                           rectified image.
``09_depth_colorbar.svg``  Its colour bar, as an editable vector.
``10_roi_sensors.png``     Roadway ROI outline and virtual sensors on the
                           depth map.
=========================  ==================================================

Plan-view images (06-10) share one north-up extent and pixel size, so
they stack exactly in Inkscape. Images have no axes or text; labels go
in the diagram. A panel whose input is missing is skipped and listed
in the manifest.

The default example is the well-segmented image of the site's roadway
ROI whose coverage is closest to ``--target_coverage`` (partial flooding
shows water, road and waterline at once).

Example
-------
``python methods_figure_assets.py --site carolina_beach --events_dir
data/carolina_beach/flood_events --images_dir
data/carolina_beach/images --labels
data/carolina_beach/segmentation_labels.csv --sensor
data/carolina_beach/sensor_data_comparison.csv --grid
data/grids/shared_grid_carolina_beach.npz --out_dir
data/methods_figure/assets``
"""

import argparse
import glob
import json
import logging
import os

import matplotlib

matplotlib.use("Agg")  # headless HPC nodes

import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LightSource, to_rgb  # noqa: E402

from poseidon_utils.logging_utils import setup_logging  # noqa: E402
from poseidon_utils.naming import extract_timestamp  # noqa: E402
from poseidon_utils.site_config import load_site_config  # noqa: E402
from poseidon_utils.zarr_io import array_names, open_zip_group  # noqa: E402

import access_plotter as ap  # noqa: E402
import publication_figures as pub  # noqa: E402
from plot_style import BLUE, SEQUENTIAL_BLUE, apply_style  # noqa: E402

logger = logging.getLogger(__name__)

#: Segmentation colours of classes 1, 2, ... (``pred_label_generator``).
CLASS_COLORS = ("#3366CC", "#DC3912", "#FF9900", "#109618", "#990099",
                "#0099C6", "#DD4477", "#66AA00", "#B82E2E", "#316395")

#: Longest side (pixels) of the exported camera-view images.
MAX_SIDE = 1600


def _stamp_files(folder, stamp):
    """Files in `folder` (recursively) whose name holds `stamp`."""
    if not folder or not os.path.isdir(folder):
        return []
    return sorted(p for p in glob.glob(os.path.join(folder, "**", "*"),
                                       recursive=True)
                  if os.path.isfile(p) and extract_timestamp(
                      os.path.basename(p)) == stamp)


def _zip_array(path, stamp, contains=""):
    """
    The array of a ZIP Zarr store named with `stamp` (and `contains`).

    Parameters
    ----------
    path : str
        ``.zip`` store.
    stamp : str
        14-digit image timestamp.
    contains : str, optional
        Text the array name must also contain.

    Returns
    -------
    numpy.ndarray or None
        The array, or None if the store or array is missing.
    """
    if not os.path.exists(path):
        return None
    with open_zip_group(path) as group:
        names = [n for n in array_names(group)
                 if extract_timestamp(n) == stamp and contains in n]
        return group[names[0]][:] if names else None


def _resize(image, max_side=MAX_SIDE):
    """Downsize an image so its longest side is at most `max_side`."""
    scale = max_side / max(image.shape[:2])
    if scale >= 1:
        return image
    return cv2.resize(image, None, fx=scale, fy=scale,
                      interpolation=cv2.INTER_AREA)


def _colorize(labels):
    """RGB image of a class map in the segmentation colours (0 = black)."""
    rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for value, color in enumerate(CLASS_COLORS, start=1):
        rgb[labels == value] = (np.array(to_rgb(color)) * 255).astype(
            np.uint8)
    return rgb


def _north_up(array):
    """Flip a grid array (row 0 at the southern edge) to north up."""
    return np.asarray(array)[::-1]


def _save_rgb(path, rgb):
    """Write an RGB uint8 image (PNG or JPEG by extension)."""
    params = ([cv2.IMWRITE_JPEG_QUALITY, 92]
              if path.lower().endswith((".jpg", ".jpeg")) else [])
    cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), params)


def _save_rgba(path, rgba):
    """Write an RGBA float (0-1) or uint8 image as PNG."""
    rgba = np.asarray(rgba)
    if rgba.dtype != np.uint8:
        rgba = (np.clip(rgba, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))


def _upscale(rgb, factor):
    """Enlarge a small grid image by an integer factor without smoothing."""
    return np.repeat(np.repeat(rgb, factor, axis=0), factor, axis=1)


def pick_example(args, site):
    """
    Choose the event and image timestamp to show.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.
    site : poseidon_utils.site_config.SiteConfig
        Site configuration.

    Returns
    -------
    tuple of str
        Event folder name and 14-digit UTC timestamp.

    Raises
    ------
    SystemExit
        If no well-segmented image with water on the roadway is found.
    """
    if args.event and args.timestamp:
        return args.event, args.timestamp
    roi = args.roi or pub.site_roi(args.site)
    good = ap.load_good_timestamps(args.labels)
    sensor = ap.load_sensor_data(args.sensor)
    images = pub.load_roi_images(args.events_dir, roi, good, sensor, 3.0)
    if images is None:
        raise SystemExit(f"No accessibility CSVs for ROI {roi}; give "
                         f"--event and --timestamp.")
    keep = (images["Segmentation"] == "good") & (images["Coverage"] > 0)
    if args.event:
        keep &= images["event"] == args.event
    rows = images[keep].dropna(subset=["Coverage", "MeanDepth"])
    if rows.empty:
        raise SystemExit("No well-segmented image with water on the "
                         "roadway; give --event and --timestamp.")
    rows = rows.assign(gap=(rows["Coverage"] - args.target_coverage).abs())
    best = rows.sort_values(["gap", "MeanDepth"],
                            ascending=[True, False]).iloc[0]
    time = pd.Timestamp(best["Time"])
    time = time.tz_convert("UTC") if time.tzinfo else time
    stamp = time.strftime("%Y%m%d%H%M%S")
    logger.info("Example: %s at %s (coverage %.0f%%, mean depth %.3f m)",
                best["event"], stamp, best["Coverage"], best["MeanDepth"])
    return best["event"], stamp


def export_camera_view(event_dir, stamp, args, site, out, made):
    """
    Write panels 01-05 (camera view).

    Parameters
    ----------
    event_dir : str
        Event folder.
    stamp : str
        Image timestamp.
    args : argparse.Namespace
        Parsed arguments.
    site : poseidon_utils.site_config.SiteConfig
        Site configuration.
    out : str
        Output folder.
    made : dict
        Filled with ``{filename: source}`` for the manifest.

    Returns
    -------
    numpy.ndarray or None
        The full-size RGB camera image.
    """
    originals = _stamp_files(os.path.join(event_dir, "orig_images"), stamp)
    if not originals:
        logger.error("No original image for %s in %s", stamp, event_dir)
        return None
    image = cv2.cvtColor(cv2.imread(originals[0]), cv2.COLOR_BGR2RGB)
    _save_rgb(os.path.join(out, "01_original.jpg"), _resize(image))
    made["01_original.jpg"] = originals[0]

    label_paths = _stamp_files(os.path.join(event_dir, "labels"), stamp)
    labels = (cv2.imread(label_paths[0], cv2.IMREAD_UNCHANGED)
              if label_paths else None)
    if labels is not None and labels.ndim == 3:
        labels = labels[..., 0]
    if labels is not None and labels.shape != image.shape[:2]:
        labels = cv2.resize(labels, image.shape[1::-1],
                            interpolation=cv2.INTER_NEAREST)

    overlays = _stamp_files(args.overlay_dir, stamp)
    if overlays:
        overlay = cv2.cvtColor(cv2.imread(overlays[0]), cv2.COLOR_BGR2RGB)
        made["02_overlay.jpg"] = overlays[0]
    elif labels is not None:
        overlay = (0.4 * image + 0.6 * _colorize(labels)).astype(np.uint8)
        made["02_overlay.jpg"] = f"drawn from {label_paths[0]}"
    else:
        overlay = None
    if overlay is not None:
        _save_rgb(os.path.join(out, "02_overlay.jpg"), _resize(overlay))

    if labels is not None:
        _save_rgb(os.path.join(out, "03_classes.png"),
                  _resize(_colorize(labels)))
        made["03_classes.png"] = label_paths[0]
        water = labels == args.water_class
        dim = (image * 0.45).astype(np.uint8)
        dim[water] = (0.35 * image[water] + 0.65 * np.array(
            to_rgb(BLUE)) * 255).astype(np.uint8)
        _save_rgb(os.path.join(out, "04_water_mask.png"), _resize(dim))
        made["04_water_mask.png"] = f"class {args.water_class} of " \
                                    f"{label_paths[0]}"
    else:
        logger.warning("No label image for %s; panels 02-04 limited.",
                       stamp)

    roi_json = site.resolve(site.camera_roi_json)
    if roi_json is not None and roi_json.exists():
        with open(roi_json) as handle:
            shapes = json.load(handle)["shapes"]
        drawn = image.copy()
        width = max(4, image.shape[1] // 300)
        for shape in shapes:
            if shape.get("label") != args.roi_label:
                continue
            points = np.round(shape["points"]).astype(np.int32)
            fill = drawn.copy()
            cv2.fillPoly(fill, [points], (255, 153, 0))
            drawn = cv2.addWeighted(fill, 0.3, drawn, 0.7, 0)
            cv2.polylines(drawn, [points], True, (255, 153, 0), width)
        _save_rgb(os.path.join(out, "05_camera_roi.jpg"), _resize(drawn))
        made["05_camera_roi.jpg"] = str(roi_json)
    return image


def export_plan_view(event_dir, stamp, args, site, out, made):
    """
    Write panels 06-10 (plan view, north up, one shared extent).

    Parameters
    ----------
    event_dir : str
        Event folder.
    stamp : str
        Image timestamp.
    args : argparse.Namespace
        Parsed arguments.
    site : poseidon_utils.site_config.SiteConfig
        Site configuration.
    out : str
        Output folder.
    made : dict
        Filled with ``{filename: source}`` for the manifest.

    Returns
    -------
    dict
        Plan-view extent and depth range, for the manifest.
    """
    info = {}
    factor = args.plan_scale
    zarr = os.path.join(event_dir, "zarr")
    rect = _zip_array(os.path.join(zarr, "orig_image_rects.zip"), stamp)
    if rect is not None:
        rect = _north_up(rect).astype(np.uint8)
        if rect.ndim == 2:
            rect = np.dstack([rect] * 3)
        rect = rect[..., :3]
        _save_rgb(os.path.join(out, "07_rectified.png"),
                  _upscale(rect, factor))
        made["07_rectified.png"] = "zarr/orig_image_rects.zip"
    grey = (rect.mean(axis=2, keepdims=True).repeat(3, axis=2) / 255
            if rect is not None else None)

    if os.path.exists(args.grid):
        grid = np.load(args.grid)
        z = _north_up(grid["z"]).astype(float)
        shade = LightSource(azdeg=315, altdeg=45).shade(
            np.nan_to_num(z, nan=np.nanmin(z)),
            cmap=plt.get_cmap("gist_earth"),
            vert_exag=3, blend_mode="soft",
            vmin=np.nanpercentile(z, 1), vmax=np.nanpercentile(z, 99))
        shade[np.isnan(z)] = [1, 1, 1, 0]
        _save_rgba(os.path.join(out, "06_dem.png"),
                   _upscale((shade * 255).astype(np.uint8), factor))
        made["06_dem.png"] = args.grid
        info["elevation_range_m"] = [float(np.nanpercentile(z, 1)),
                                     float(np.nanpercentile(z, 99))]
    info["extent_m"] = {key: getattr(site.grid, key) for key in (
        "min_x", "max_x", "min_y", "max_y", "resolution")}

    water = _zip_array(os.path.join(zarr, "labels_rects.zip"), stamp)
    if water is not None and grey is not None:
        water = _north_up(np.squeeze(water)) == args.water_class
        layer = grey * 0.6
        layer[water] = 0.35 * layer[water] + 0.65 * np.array(to_rgb(BLUE))
        _save_rgb(os.path.join(out, "08_rectified_water.png"),
                  _upscale((layer * 255).astype(np.uint8), factor))
        made["08_rectified_water.png"] = "zarr/labels_rects.zip"

    depth = _zip_array(os.path.join(zarr, "depth_maps.zip"), stamp,
                       f"depth_map_{args.statistic}")
    if depth is not None and grey is not None:
        depth = _north_up(np.squeeze(depth)).astype(float)
        if args.depth_max:
            top = args.depth_max
        elif (depth > 0).any():
            top = float(np.nanpercentile(depth[depth > 0], 99))
        else:
            top = 0.1
        colored = SEQUENTIAL_BLUE(np.clip(depth / top, 0, 1))[..., :3]
        layer = grey * 0.6
        wet = depth > 0
        layer[wet] = colored[wet]
        _save_rgb(os.path.join(out, "09_depth.png"),
                  _upscale((layer * 255).astype(np.uint8), factor))
        made["09_depth.png"] = f"zarr/depth_maps.zip (depth_map_" \
                               f"{args.statistic})"
        info["depth_colorbar_max_m"] = top
        fig, ax = plt.subplots(figsize=(0.6, 3))
        fig.colorbar(plt.cm.ScalarMappable(
            norm=plt.Normalize(0, top), cmap=SEQUENTIAL_BLUE), cax=ax,
            label="Water depth (m)")
        fig.savefig(os.path.join(out, "09_depth_colorbar.svg"),
                    bbox_inches="tight", transparent=True)
        plt.close(fig)
        made["09_depth_colorbar.svg"] = "colour scale of 09_depth.png"

        roi = args.roi or pub.site_roi(args.site)
        masks = glob.glob(os.path.join(event_dir, f"{roi}_roi_mask.png"))
        drawn = (layer * 255).astype(np.uint8).copy()
        if masks:
            mask = cv2.imread(masks[0], cv2.IMREAD_GRAYSCALE) > 0
            if mask.shape == drawn.shape[:2]:
                contours, _ = cv2.findContours(mask.astype(np.uint8),
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(drawn, contours, -1, (255, 153, 0), 2)
                made["10_roi_sensors.png"] = masks[0]
        drawn = _upscale(drawn, factor)
        height = depth.shape[0]
        sensors = ([] if site.virtual_sensors is None
                   else site.virtual_sensors)
        for col, row in sensors:
            center = (int((col + 0.5) * factor),
                      int((height - 1 - row + 0.5) * factor))
            cv2.circle(drawn, center, 5 * factor, (255, 255, 255), -1)
            cv2.circle(drawn, center, 5 * factor, (20, 20, 20), factor)
        _save_rgb(os.path.join(out, "10_roi_sensors.png"), drawn)
        made.setdefault("10_roi_sensors.png", "virtual sensors only")
    return info


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
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Export example images for the methods diagram.")
    parser.add_argument("--site", required=True, help="Site config name.")
    parser.add_argument("--events_dir", required=True,
                        help="Folder of event subfolders.")
    parser.add_argument("--images_dir", required=True,
                        help="Site image folder (holds "
                             "daylight_all_events_overlays/).")
    parser.add_argument("--labels", required=True,
                        help="segmentation_labels.csv.")
    parser.add_argument("--sensor", required=True,
                        help="sensor_data_comparison.csv.")
    parser.add_argument("--grid", required=True,
                        help="shared_grid_<site>.npz from stage 06.")
    parser.add_argument("--out_dir", required=True, help="Output folder.")
    parser.add_argument("--event", default=None,
                        help="Event folder name. Default: chosen by "
                             "--target_coverage.")
    parser.add_argument("--timestamp", default=None,
                        help="14-digit UTC image timestamp (with --event).")
    parser.add_argument("--roi", default=None,
                        help="Roadway ROI name. Default: from the site.")
    parser.add_argument("--roi_label", default="roadway",
                        help="LabelMe label of the camera ROI polygon.")
    parser.add_argument("--target_coverage", type=float, default=60,
                        help="Roadway coverage (%%) of the default example.")
    parser.add_argument("--water_class", type=int, default=1,
                        help="Label value of water.")
    parser.add_argument("--statistic", default="95_perc",
                        help="Depth map statistic shown (stage 10 uses "
                             "95_perc).")
    parser.add_argument("--depth_max", type=float, default=None,
                        help="Top of the depth colour scale (m). Default: "
                             "99th percentile of the map.")
    parser.add_argument("--plan_scale", type=int, default=2,
                        help="Integer enlargement of plan-view images.")
    args = parser.parse_args(argv)
    args.overlay_dir = os.path.join(args.images_dir,
                                    "daylight_all_events_overlays")
    return args


def main(argv=None):
    """
    Export every panel and the manifest.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments. Default reads ``sys.argv``.

    Returns
    -------
    int
        Process exit code: 1 if the original image was not found.
    """
    setup_logging()
    apply_style()
    args = parse_args(argv)
    site = load_site_config(args.site)
    os.makedirs(args.out_dir, exist_ok=True)
    event, stamp = pick_example(args, site)
    event_dir = os.path.join(args.events_dir, event)
    made = {}
    image = export_camera_view(event_dir, stamp, args, site, args.out_dir,
                               made)
    if image is None:
        return 1
    info = export_plan_view(event_dir, stamp, args, site, args.out_dir, made)
    sizes = {}
    for name in made:
        path = os.path.join(args.out_dir, name)
        if name.endswith(".svg") or not os.path.exists(path):
            continue
        height, width = cv2.imread(path, cv2.IMREAD_UNCHANGED).shape[:2]
        sizes[name] = [width, height]
    manifest = {"site": args.site, "event": event, "timestamp_utc": stamp,
                "files": made, "pixel_sizes": sizes, **info}
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2)
    missing = sorted({"01_original.jpg", "02_overlay.jpg", "03_classes.png",
                      "04_water_mask.png", "05_camera_roi.jpg", "06_dem.png",
                      "07_rectified.png", "08_rectified_water.png",
                      "09_depth.png", "10_roi_sensors.png"} - set(made))
    if missing:
        logger.warning("Not made (input missing): %s", ", ".join(missing))
    logger.info("%d panels written to %s", len(made), args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
