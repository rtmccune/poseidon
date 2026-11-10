"""Make the site description figure: North Carolina, community and site.

Layout (first ``--sites`` entry on the left, as in the other combined
figures):

(a) North Carolina with both sites' community extents outlined.
(b), (c) Each community on aerial imagery, with the site's elevation
    grid extent and the camera.
(d), (f) Each site's elevation grid (LiDAR, stage 06), shaded, with
    the camera and its field of view, the roadway region and the
    virtual sensors.
(e), (g) A photo from each camera, with the roadway polygon.

Maps carry a zebra frame, scale bar and north arrow
(``map_decorations.py``, adapted from the shift repository).

Two steps, because HPC compute nodes have no internet:

``fetch`` (run where there is internet, e.g. locally)
    Downloads the background layers into ``--layers_dir``: US state
    boundaries (US Census cartographic boundaries, 1:5,000,000) as
    ``states.gpkg``, a light basemap of North Carolina (Esri World Light
    Gray Canvas; CARTO tiles now need an API key) as
    ``context_nc.tif`` and aerial imagery of each community as
    ``community_<site>.tif`` (Web Mercator GeoTIFFs via contextily).
``render`` (works offline, locally or on the HPC)
    Draws the figure from those layers, each site's config, its grid
    (``<grid_dir>/shared_grid_<site>.npz``) and a camera photo. A
    missing input leaves its panel with a note instead of failing.

Writes ``site_description.{png,pdf,svg}`` to ``--out_dir``. Imagery
attribution for the caption is printed and saved in
``attribution.txt``.

Example
-------
``python site_description_figure.py fetch --sites carolina_beach
down_east --layers_dir data/site_figure/layers``

``python site_description_figure.py render --sites carolina_beach
down_east --layers_dir data/site_figure/layers --grid_dir data/grids
--out_dir data/site_figure``
"""

import argparse
import glob
import json
import logging
import os
import urllib.request
import zipfile

import matplotlib

matplotlib.use("Agg")  # headless HPC nodes

import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LightSource, Normalize  # noqa: E402
from pyproj import Transformer  # noqa: E402

from poseidon_utils.logging_utils import setup_logging  # noqa: E402
from poseidon_utils.site_config import load_site_config  # noqa: E402

import map_decorations as md  # noqa: E402
from plot_style import INK, apply_style  # noqa: E402

logger = logging.getLogger(__name__)

#: Web Mercator, the CRS of the downloaded tiles and the regional maps.
WEB = "EPSG:3857"

#: North Carolina map extent (lon_min, lon_max, lat_min, lat_max).
NC_EXTENT = (-84.4, -75.3, 33.7, 36.7)

#: Census cartographic boundary file of US states (1:5,000,000).
STATES_URL = ("https://www2.census.gov/geo/tiger/GENZ2023/shp/"
              "cb_2023_us_state_5m.zip")

#: States kept for the context map.
STATES = ("NC", "SC", "VA", "TN", "GA")

#: Highlight colour of zoom boxes and site markers (as in shift).
HIGHLIGHT = "#FFB000"

#: Imagery credits, for the caption.
ATTRIBUTION = {
    "context": "Basemap: Esri, HERE, Garmin, OpenStreetMap contributors, "
               "and the GIS User Community (World Light Gray Canvas).",
    "community": "Imagery: Esri, Maxar, Earthstar Geographics, and the "
                 "GIS User Community.",
    "states": "State boundaries: US Census Bureau cartographic boundary "
              "files (2023).",
}


# ----- Shared helpers --------------------------------------------------------

def site_title(site):
    """Display name of a site, e.g. ``carolina_beach`` -> Carolina Beach."""
    return site.replace("_", " ").title()


def community_extent(site, half_km):
    """
    Lon/lat box of side ``2 * half_km`` centred on the site's camera.

    Parameters
    ----------
    site : poseidon_utils.site_config.SiteConfig
        Site configuration (``latitude`` / ``longitude``).
    half_km : float
        Half the box side in kilometres.

    Returns
    -------
    tuple of float
        ``(lon_min, lon_max, lat_min, lat_max)``.
    """
    dlat = half_km / 111.0
    dlon = half_km / (111.32 * np.cos(np.radians(site.latitude)))
    return (site.longitude - dlon, site.longitude + dlon,
            site.latitude - dlat, site.latitude + dlat)


def to_web(lons, lats):
    """Transform lon/lat to Web Mercator x / y arrays."""
    transformer = Transformer.from_crs("EPSG:4326", WEB, always_xy=True)
    return transformer.transform(np.asarray(lons), np.asarray(lats))


def web_box(extent):
    """Web Mercator ``(x0, x1, y0, y1)`` of a lon/lat extent."""
    xs, ys = to_web([extent[0], extent[1]], [extent[2], extent[3]])
    return xs[0], xs[1], ys[0], ys[1]


def grid_outline_web(site):
    """Web Mercator outline (x, y) of a site's grid extent."""
    g = site.grid
    xs = [g.min_x, g.max_x, g.max_x, g.min_x, g.min_x]
    ys = [g.min_y, g.min_y, g.max_y, g.max_y, g.min_y]
    transformer = Transformer.from_crs(site.crs, WEB, always_xy=True)
    return transformer.transform(xs, ys)


def note(ax, text):
    """Write a centred note on an empty panel."""
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center",
            va="center", fontsize=8, color="#7A7F79", wrap=True)
    ax.set_xticks([])
    ax.set_yticks([])


def panel_letter(ax, letter):
    """Label a panel in its top-left corner, inside the frame."""
    ax.text(0.03, 0.97, f"({letter})", transform=ax.transAxes, ha="left",
            va="top", fontsize=10, weight="bold", zorder=30,
            path_effects=[pe.withStroke(linewidth=3, foreground="white")])


def read_raster(path):
    """
    Read an RGB(A) GeoTIFF for imshow.

    Returns
    -------
    tuple
        ``(image, (left, right, bottom, top))`` in the raster's CRS.
    """
    import rasterio

    with rasterio.open(path) as src:
        data = src.read()
        bounds = src.bounds
    image = np.moveaxis(data[:3], 0, -1)
    return image, (bounds.left, bounds.right, bounds.bottom, bounds.top)


# ----- fetch -----------------------------------------------------------------

def download_tiles(extent, path, source, zoom, args, label):
    """
    Download web map tiles of a lon/lat extent into a GeoTIFF.

    Retries failed tiles (tile servers return occasional 502 errors)
    and, if a zoom level still fails, tries up to two coarser levels.
    The GeoTIFF is written only once every tile has arrived, so a
    failed run leaves no partial file and the next run tries again.

    Parameters
    ----------
    extent : tuple of float
        ``(lon_min, lon_max, lat_min, lat_max)``.
    path : str
        Output GeoTIFF.
    source : xyzservices.TileProvider
        Tile source.
    zoom : int
        Preferred zoom level.
    args : argparse.Namespace
        Parsed arguments (``retries``, ``wait``).
    label : str
        Name for the log.

    Raises
    ------
    RuntimeError
        If every zoom level fails.
    """
    import contextily as ctx
    import requests

    partial = path + ".partial.tif"
    for level in range(zoom, zoom - 3, -1):
        logger.info("Downloading %s (zoom %d)", label, level)
        try:
            ctx.bounds2raster(extent[0], extent[2], extent[1], extent[3],
                              partial, ll=True, zoom=level, source=source,
                              wait=args.wait, max_retries=args.retries)
        except (requests.HTTPError, requests.ConnectionError) as err:
            logger.warning("%s failed at zoom %d: %s", label, level, err)
            if os.path.exists(partial):
                os.remove(partial)
            continue
        os.replace(partial, path)
        logger.info("Wrote %s", path)
        return
    raise RuntimeError(f"Could not download {label}; the tile server may "
                       f"be down. Run fetch again later (finished layers "
                       f"are kept).")


def fetch(args):
    """
    Download the background layers (needs internet).

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments of the ``fetch`` command.
    """
    import contextily as ctx
    import geopandas as gpd

    os.makedirs(args.layers_dir, exist_ok=True)
    states_path = os.path.join(args.layers_dir, "states.gpkg")
    if args.force or not os.path.exists(states_path):
        zip_path = os.path.join(args.layers_dir, "states_5m.zip")
        logger.info("Downloading %s", STATES_URL)
        urllib.request.urlretrieve(STATES_URL, zip_path)
        with zipfile.ZipFile(zip_path) as archive:
            shp = [n for n in archive.namelist() if n.endswith(".shp")][0]
        states = gpd.read_file(f"zip://{zip_path}!{shp}")
        states = states[states["STUSPS"].isin(STATES)].to_crs(WEB)
        states[["STUSPS", "NAME", "geometry"]].to_file(states_path)
        os.remove(zip_path)
        logger.info("Wrote %s", states_path)

    context_path = os.path.join(args.layers_dir, "context_nc.tif")
    if args.force or not os.path.exists(context_path):
        download_tiles(NC_EXTENT, context_path,
                       ctx.providers.Esri.WorldGrayCanvas,
                       args.context_zoom, args, "the North Carolina basemap")

    for name in args.sites:
        site = load_site_config(name)
        path = os.path.join(args.layers_dir, f"community_{name}.tif")
        if args.force or not os.path.exists(path):
            download_tiles(community_extent(site, args.community_km), path,
                           ctx.providers.Esri.WorldImagery,
                           args.community_zoom, args,
                           f"imagery around {name}")
    logger.info("Layers ready in %s; copy that folder to the HPC to "
                "render there.", args.layers_dir)


# ----- render ----------------------------------------------------------------

def draw_context(ax, sites, args):
    """Draw panel (a): North Carolina with the community extents."""
    import geopandas as gpd

    x0, x1, y0, y1 = web_box(NC_EXTENT)
    context = os.path.join(args.layers_dir, "context_nc.tif")
    if os.path.exists(context):
        image, bounds = read_raster(context)
        ax.imshow(image, extent=bounds, interpolation="bilinear", zorder=0)
    states_path = os.path.join(args.layers_dir, "states.gpkg")
    if os.path.exists(states_path):
        states = gpd.read_file(states_path)
        nc = states[states["STUSPS"] == "NC"]
        states.plot(ax=ax, facecolor="none", edgecolor="#8C918B",
                    linewidth=0.6, zorder=2)
        nc.plot(ax=ax, facecolor="#FFFFFF", alpha=0.35, edgecolor=INK,
                linewidth=0.9, zorder=3)
    if not (os.path.exists(context) or os.path.exists(states_path)):
        note(ax, "Run the fetch step for the North Carolina layers")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    for site in sites:
        bx0, bx1, by0, by1 = web_box(community_extent(
            site["config"], args.community_km))
        pad = 4 * (bx1 - bx0)  # keep the small box visible at state scale
        ax.add_patch(mpatches.Rectangle(
            (bx0 - pad, by0 - pad), bx1 - bx0 + 2 * pad,
            by1 - by0 + 2 * pad, facecolor="none", edgecolor=HIGHLIGHT,
            linewidth=1.8, zorder=10))
        # Labels sit west of the boxes, over land, clear of the coast
        ax.text(bx0 - 1.5 * pad, (by0 + by1) / 2,
                f"{site['title']} ({site['letter']})", fontsize=8,
                va="center", ha="right", weight="bold", zorder=11,
                path_effects=[pe.withStroke(linewidth=3,
                                            foreground="white")])
    ax.text(0.30, 0.55, "North Carolina", transform=ax.transAxes,
            fontsize=10, color=INK, style="italic", ha="center")
    ax.text(0.93, 0.12, "Atlantic\nOcean", transform=ax.transAxes,
            fontsize=8, color="#5F655F", style="italic", ha="center")
    _, xs, _, ys = md.add_graticule_ticks(ax, WEB, args.context_tick_deg)
    md.add_zebra_frame(ax, xs, ys, width=4)
    md.add_scale_bar(ax, WEB, location=(0.04, 0.08))
    md.add_north_arrow(ax, location=(0.95, 0.78), scale=0.16)


def draw_community(ax, site, args):
    """Draw a community panel: imagery, grid extent and camera."""
    config = site["config"]
    x0, x1, y0, y1 = web_box(community_extent(config, args.community_km))
    path = os.path.join(args.layers_dir, f"community_{site['name']}.tif")
    if os.path.exists(path):
        image, bounds = read_raster(path)
        ax.imshow(image, extent=bounds, interpolation="bilinear", zorder=0)
    else:
        note(ax, f"Run the fetch step for\ncommunity_{site['name']}.tif")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    gx, gy = grid_outline_web(config)
    ax.plot(gx, gy, color=HIGHLIGHT, linewidth=1.6, zorder=10)
    cx, cy = to_web(config.longitude, config.latitude)
    ax.plot(cx, cy, marker="o", markersize=11, markerfacecolor="none",
            markeredgecolor=HIGHLIGHT, markeredgewidth=1.6, zorder=11)
    _, xs, _, ys = md.add_graticule_ticks(ax, WEB, args.community_tick_deg,
                                          fontsize=7)
    md.add_zebra_frame(ax, xs, ys, width=4)
    md.add_scale_bar(ax, WEB, location=(0.06, 0.07), color="white",
                     outline="black")
    md.add_north_arrow(ax, location=(0.9, 0.82), scale=0.11)


def load_grid(site, args):
    """Load a site's elevation grid, or None if it is missing."""
    path = os.path.join(args.grid_dir, f"shared_grid_{site['name']}.npz")
    if not os.path.exists(path):
        logger.warning("No grid %s; run stage 06 (run_prep_grid.py) or "
                       "copy it from the HPC.", path)
        return None
    return dict(np.load(path))


def camera_roi_outline(site, grid):
    """
    North-up grid mask of the camera-view roadway polygon.

    Returns
    -------
    numpy.ndarray or None
        Boolean mask (row 0 north), or None without a camera ROI.
    """
    from poseidon_core.image_rectifier import ImageRectifier
    from poseidon_core.roadway_analyzer import (
        roi_mask, roi_pixels_from_camera_labelme)

    config = site["config"]
    path = config.resolve(config.camera_roi_json)
    if path is None or not path.exists():
        return None
    try:
        rectifier = ImageRectifier(config.intrinsics, config.extrinsics,
                                   grid["x"], grid["y"], grid["z"],
                                   use_gpu=False)
        coords = roi_pixels_from_camera_labelme(path, "roadway", rectifier)
        return roi_mask(coords, grid["z"].shape)
    except (ValueError, OSError) as err:
        logger.warning("Roadway outline skipped for %s: %s", site["name"],
                       err)
        return None


def draw_site(ax, site, grid, norm, args):
    """Draw a site panel: shaded elevation grid and site features."""
    config = site["config"]
    g = config.grid
    if grid is None:
        note(ax, f"No grid for {site['title']}:\nrun stage 06 or copy\n"
                 f"shared_grid_{site['name']}.npz")
        return
    z = np.asarray(grid["z"], dtype=float)
    cmap = plt.get_cmap(args.elevation_cmap)
    filled = np.where(np.isfinite(z), z, norm.vmin)
    shaded = LightSource(azdeg=315, altdeg=40).shade(
        filled, cmap=cmap, norm=norm, vert_exag=2, blend_mode="soft",
        dx=g.resolution, dy=g.resolution)
    shaded[~np.isfinite(z)] = (1, 1, 1, 1)
    extent = (g.min_x, g.max_x, g.min_y, g.max_y)
    ax.imshow(shaded, origin="lower", extent=extent, interpolation="none")
    ax.set_xlim(g.min_x, g.max_x)
    ax.set_ylim(g.min_y, g.max_y)
    ax.set_aspect("equal")

    mask = camera_roi_outline(site, grid)
    if mask is not None:
        ax.contour(np.flipud(mask).astype(float), levels=[0.5],
                   origin="lower", extent=extent, colors="white",
                   linewidths=1.6, zorder=5)
        ax.contour(np.flipud(mask).astype(float), levels=[0.5],
                   origin="lower", extent=extent, colors=INK,
                   linewidths=0.7, linestyles="--", zorder=6)

    # Camera and its horizontal field of view (CIRN azimuth: clockwise
    # from north)
    cam_x, cam_y, _, azimuth = config.extrinsics[:4]
    fov = 2 * np.arctan(config.intrinsics[0] / (2 * config.intrinsics[4]))
    reach = 0.45 * np.hypot(g.max_x - g.min_x, g.max_y - g.min_y)
    angles = np.linspace(azimuth - fov / 2, azimuth + fov / 2, 30)
    wedge = np.column_stack((
        np.r_[cam_x, cam_x + reach * np.sin(angles), cam_x],
        np.r_[cam_y, cam_y + reach * np.cos(angles), cam_y]))
    ax.add_patch(mpatches.Polygon(wedge, facecolor="white", alpha=0.18,
                                  edgecolor="white", linewidth=0.8,
                                  zorder=4))
    ax.plot(cam_x, cam_y, marker="^", markersize=8, color=HIGHLIGHT,
            markeredgecolor=INK, zorder=8, clip_on=False)

    for k, (col, row) in enumerate(
            [] if config.virtual_sensors is None else config.virtual_sensors,
            start=1):
        x = g.min_x + (col + 0.5) * g.resolution
        y = g.min_y + (row + 0.5) * g.resolution
        ax.plot(x, y, marker="o", markersize=5.5, color="white",
                markeredgecolor=INK, markeredgewidth=1, zorder=9)
        ax.text(x + 0.03 * (g.max_x - g.min_x), y, f"VS{k}", fontsize=7,
                va="center", zorder=9,
                path_effects=[pe.withStroke(linewidth=2.5,
                                            foreground="white")])

    breaks_x = md.clean_ticks(g.min_x, g.max_x, args.site_tick_m)
    breaks_y = md.clean_ticks(g.min_y, g.max_y, args.site_tick_m)
    ax.set_xticks([])
    ax.set_yticks([])
    md.add_zebra_frame(ax, breaks_x, breaks_y, width=3)
    md.add_scale_bar(ax, config.crs, length_m=args.site_scale_m,
                     location=(0.07, 0.05), fontsize=7)
    md.add_north_arrow(ax, location=(0.88, 0.12), scale=0.1, fontsize=8)

    # Colour bar under the panel; each site has its own range
    cax = ax.inset_axes([0.1, -0.11, 0.8, 0.035])
    ax.figure.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                       cax=cax, orientation="horizontal")
    cax.tick_params(labelsize=6.5, length=2, pad=1)
    cax.set_xlabel(args.elevation_label, fontsize=7, labelpad=1)


def draw_photo(ax, site, path):
    """Draw a camera photo with the roadway polygon."""
    config = site["config"]
    if not path or not os.path.exists(path):
        note(ax, f"No photo for {site['title']}")
        return
    image = plt.imread(path)
    ax.imshow(image)
    roi = config.resolve(config.camera_roi_json)
    if roi is not None and roi.exists():
        with open(roi) as handle:
            shapes = json.load(handle)["shapes"]
        for shape in shapes:
            if shape.get("label") == "roadway":
                points = np.asarray(shape["points"])
                closed = np.vstack([points, points[:1]])
                ax.plot(closed[:, 0], closed[:, 1], color="white",
                        linewidth=1.6)
                ax.plot(closed[:, 0], closed[:, 1], color=INK,
                        linewidth=0.7, linestyle="--")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(INK)


def render(args):
    """
    Draw the figure from local files only.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments of the ``render`` command.

    Returns
    -------
    int
        Process exit code.
    """
    apply_style()
    plt.rcParams.update({"axes.grid": False, "savefig.bbox": "tight"})
    sites = []
    for i, name in enumerate(args.sites):
        photo = (args.photos[i] if args.photos else
                 next(iter(sorted(glob.glob(os.path.join(
                     args.data_root, name, "CAM_*.jpg")))), None))
        sites.append({"name": name, "title": site_title(name),
                      "config": load_site_config(name), "photo": photo,
                      "letter": "bc"[i]})
    grids = [load_grid(site, args) for site in sites]
    norms = [Normalize(*(args.elevation_range or (
        np.percentile(g["z"][np.isfinite(g["z"])], [1, 99])
        if g is not None else (0, 1)))) for g in grids]

    fig = plt.figure(figsize=(7.5, 9.0))
    outer = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.25, 1.1],
                             hspace=0.12)
    ax_nc = fig.add_subplot(outer[0])
    draw_context(ax_nc, [s for s in sites], args)
    panel_letter(ax_nc, "a")

    middle = outer[1].subgridspec(1, 2, wspace=0.12)
    bottom = outer[2].subgridspec(1, 4, wspace=0.06,
                                  width_ratios=[1, 0.8, 1, 0.8])
    for i, (site, grid, norm) in enumerate(zip(sites, grids, norms)):
        ax = fig.add_subplot(middle[i])
        draw_community(ax, site, args)
        panel_letter(ax, "bc"[i])
        ax_site = fig.add_subplot(bottom[2 * i])
        draw_site(ax_site, site, grid, norm, args)
        panel_letter(ax_site, "df"[i])
        ax_photo = fig.add_subplot(bottom[2 * i + 1])
        draw_photo(ax_photo, site, site["photo"])
        panel_letter(ax_photo, "eg"[i])

    os.makedirs(args.out_dir, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(os.path.join(args.out_dir, f"site_description.{ext}"),
                    dpi=300)
    plt.close(fig)
    credits = "\n".join(ATTRIBUTION.values())
    with open(os.path.join(args.out_dir, "attribution.txt"), "w") as handle:
        handle.write(credits + "\n")
    logger.info("Figure written to %s. For the caption:\n%s", args.out_dir,
                credits)
    return 0


# ----- CLI -------------------------------------------------------------------

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
        Parsed arguments, with ``command`` set.
    """
    parser = argparse.ArgumentParser(
        description="Site description figure (fetch layers, then render).")
    commands = parser.add_subparsers(dest="command", required=True)
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--sites", nargs=2, required=True,
                        metavar=("LEFT", "RIGHT"),
                        help="Site config names, left first.")
    shared.add_argument("--layers_dir", default="data/site_figure/layers",
                        help="Folder of downloaded background layers.")
    shared.add_argument("--community_km", type=float, default=2.0,
                        help="Half-width (km) of each community map.")

    get = commands.add_parser("fetch", parents=[shared],
                              help="Download background layers (internet).")
    get.add_argument("--context_zoom", type=int, default=8,
                     help="Tile zoom of the North Carolina basemap.")
    get.add_argument("--community_zoom", type=int, default=16,
                     help="Tile zoom of the community imagery.")
    get.add_argument("--force", action="store_true",
                     help="Download again even if a layer exists.")
    get.add_argument("--retries", type=int, default=6,
                     help="Retries per failed tile.")
    get.add_argument("--wait", type=float, default=3,
                     help="Seconds between retries of a failed tile.")

    draw = commands.add_parser("render", parents=[shared],
                               help="Draw the figure (offline).")
    draw.add_argument("--grid_dir", default="data/grids",
                      help="Folder of shared_grid_<site>.npz (stage 06).")
    draw.add_argument("--data_root", default="data",
                      help="Folder of site data folders (default photos: "
                           "the first <site>/CAM_*.jpg).")
    draw.add_argument("--photos", nargs=2, default=None,
                      metavar=("LEFT", "RIGHT"),
                      help="Camera photos, left first.")
    draw.add_argument("--out_dir", default="data/site_figure",
                      help="Output folder.")
    draw.add_argument("--elevation_cmap", default="gist_earth",
                      help="Colour map of the elevation grids.")
    draw.add_argument("--elevation_range", type=float, nargs=2,
                      default=None, metavar=("LOW", "HIGH"),
                      help="Elevation colour range (m) for both sites. "
                           "Default: each grid's 1st-99th percentile.")
    draw.add_argument("--elevation_label", default="Elevation (m)",
                      help="Colour bar label; add the vertical datum.")
    draw.add_argument("--context_tick_deg", type=float, default=1.0,
                      help="Graticule spacing (degrees) of panel (a).")
    draw.add_argument("--community_tick_deg", type=float, default=0.01,
                      help="Graticule spacing (degrees) of (b), (c).")
    draw.add_argument("--site_tick_m", type=float, default=10.0,
                      help="Zebra stripe length (m) of (d), (f).")
    draw.add_argument("--site_scale_m", type=float, default=10.0,
                      help="Scale bar length (m) of (d), (f).")
    return parser.parse_args(argv)


def main(argv=None):
    """
    Run the ``fetch`` or ``render`` command.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments. Default reads ``sys.argv``.

    Returns
    -------
    int
        Process exit code.
    """
    setup_logging()
    args = parse_args(argv)
    if args.command == "fetch":
        fetch(args)
        return 0
    return render(args)


if __name__ == "__main__":
    raise SystemExit(main())
