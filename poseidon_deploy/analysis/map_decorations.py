"""Map decorations for projected matplotlib axes: zebra frame, scale bar,
north arrow and graticule ticks.

Adapted from ``shift/utils/plotting.py`` (``add_zebra_frame``,
``add_scale_bar``, ``add_north_arrow``, ``get_clean_ticks``), rewritten
for ordinary axes in a projected CRS (Web Mercator or a State Plane
system) so they need no cartopy. Stripes and ticks follow geographic
intervals, so they stay put when the extent changes.

All functions draw on an axes whose data coordinates are the projected
x / y of ``crs``; geographic positions are converted with pyproj.
"""

import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
from pyproj import Geod, Transformer

#: Geodesic calculator for true ground distances.
GEOD = Geod(ellps="WGS84")


def clean_ticks(min_val, max_val, interval):
    """
    Tick values on multiples of `interval` that cover a range.

    Parameters
    ----------
    min_val, max_val : float
        Range to cover.
    interval : float
        Tick spacing.

    Returns
    -------
    numpy.ndarray
        Ticks inside ``[min_val, max_val]``.
    """
    start = np.ceil(min_val / interval) * interval
    ticks = np.arange(start, max_val + interval / 2, interval)
    return ticks[(ticks >= min_val) & (ticks <= max_val)]


def _transformers(crs):
    """Forward (lon/lat to `crs`) and inverse transformers."""
    return (Transformer.from_crs("EPSG:4326", crs, always_xy=True),
            Transformer.from_crs(crs, "EPSG:4326", always_xy=True))


def _graticule(ax, crs, interval):
    """
    Projected positions of longitude and latitude ticks on the frame.

    Returns
    -------
    tuple
        ``(lons, x_positions, lats, y_positions)``.
    """
    forward, inverse = _transformers(crs)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ym, xm = (y0 + y1) / 2, (x0 + x1) / 2
    lon0, _ = inverse.transform(x0, ym)
    lon1, _ = inverse.transform(x1, ym)
    _, lat0 = inverse.transform(xm, y0)
    _, lat1 = inverse.transform(xm, y1)
    lons = clean_ticks(min(lon0, lon1), max(lon0, lon1), interval)
    lats = clean_ticks(min(lat0, lat1), max(lat0, lat1), interval)
    xs = [forward.transform(lon, (lat0 + lat1) / 2)[0] for lon in lons]
    ys = [forward.transform((lon0 + lon1) / 2, lat)[1] for lat in lats]
    return lons, np.array(xs), lats, np.array(ys)


def _format_degrees(value, positive, negative):
    """Format a coordinate as e.g. ``77.9°W`` (trailing zeros dropped)."""
    hemisphere = positive if value >= 0 else negative
    return f"{abs(value):.4g}°{hemisphere}"


def add_graticule_ticks(ax, crs, interval, labels=True, fontsize=8):
    """
    Put longitude / latitude ticks on a projected map's frame.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Map axes (data coordinates in `crs`).
    crs : str
        Projected CRS of the axes, e.g. ``"EPSG:3857"``.
    interval : float
        Tick spacing in degrees.
    labels : bool, optional
        Label the ticks. Default True.
    fontsize : float, optional
        Tick label size.

    Returns
    -------
    tuple
        ``(lons, x_positions, lats, y_positions)`` for
        :func:`add_zebra_frame`.
    """
    lons, xs, lats, ys = _graticule(ax, crs, interval)
    ax.set_xticks(xs)
    ax.set_yticks(ys)
    if labels:
        ax.set_xticklabels([_format_degrees(v, "E", "W") for v in lons],
                           fontsize=fontsize)
        ax.set_yticklabels([_format_degrees(v, "N", "S") for v in lats],
                           fontsize=fontsize, rotation=90, va="center")
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    ax.tick_params(length=0, pad=6 if labels else 0)
    return lons, xs, lats, ys


def add_zebra_frame(ax, x_breaks, y_breaks, width=5):
    """
    Draw an alternating black-and-white border on a map's edges.

    Stripes change colour at each break (graticule tick), counted from a
    fixed origin, so the pattern is tied to the coordinates rather than
    to the extent.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Map axes.
    x_breaks, y_breaks : sequence of float
        Projected x / y positions where stripes alternate, e.g. from
        :func:`add_graticule_ticks`.
    width : float, optional
        Stripe width in points. Default 5.
    """
    for spine in ax.spines.values():
        spine.set_visible(False)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.add_patch(mpatches.Rectangle(
        (x0, y0), x1 - x0, y1 - y0, linewidth=width + 1.5,
        edgecolor="black", facecolor="none", zorder=20, clip_on=False))

    def edges(lo, hi, breaks):
        bounds = np.unique(np.clip(np.concatenate([[lo], breaks, [hi]]),
                                   min(lo, hi), max(lo, hi)))
        # Parity from the break's index in a regular sequence when the
        # breaks are evenly spaced, so stripes do not shift with extent
        if len(breaks) > 1:
            step = np.median(np.diff(np.sort(breaks)))
            origin = np.sort(breaks)[0]
            return [(a, b, int(np.floor(((a + b) / 2 - origin) / step)) % 2)
                    for a, b in zip(bounds[:-1], bounds[1:])]
        return [(a, b, i % 2) for i, (a, b) in
                enumerate(zip(bounds[:-1], bounds[1:]))]

    for a, b, odd in edges(x0, x1, x_breaks):
        if odd:
            for y in (y0, y1):
                ax.plot([a, b], [y, y], color="white", linewidth=width,
                        solid_capstyle="butt", zorder=21, clip_on=False)
    for a, b, odd in edges(y0, y1, y_breaks):
        if odd:
            for x in (x0, x1):
                ax.plot([x, x], [a, b], color="white", linewidth=width,
                        solid_capstyle="butt", zorder=21, clip_on=False)


def ground_scale(ax, crs):
    """
    Metres on the ground per projected unit at the map's centre.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Map axes.
    crs : str
        Projected CRS of the axes.

    Returns
    -------
    float
        Ground metres per projected unit along x (1 for a conformal
        State Plane grid near its origin; about cos(latitude) for Web
        Mercator).
    """
    _, inverse = _transformers(crs)
    x0, x1 = ax.get_xlim()
    y = sum(ax.get_ylim()) / 2
    xm = (x0 + x1) / 2
    half = (x1 - x0) / 20
    lon_a, lat_a = inverse.transform(xm - half, y)
    lon_b, lat_b = inverse.transform(xm + half, y)
    _, _, metres = GEOD.inv(lon_a, lat_a, lon_b, lat_b)
    return metres / (2 * half)


def nice_length(target_m):
    """Round a length down to 1, 2 or 5 times a power of ten (metres)."""
    power = 10 ** np.floor(np.log10(target_m))
    for step in (5, 2, 1):
        if step * power <= target_m:
            return step * power
    return power


def add_scale_bar(ax, crs, length_m=None, location=(0.05, 0.06),
                  fontsize=8, color="black", outline="white"):
    """
    Draw a scale bar with true ground length.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Map axes.
    crs : str
        Projected CRS of the axes.
    length_m : float, optional
        Bar length in metres. Default: a round length near a fifth of
        the map width.
    location : tuple of float, optional
        Left end of the bar in axes fractions.
    fontsize : float, optional
        Label size.
    color, outline : str, optional
        Bar and outline colours.

    Returns
    -------
    float
        The bar's length in metres.
    """
    scale = ground_scale(ax, crs)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    if length_m is None:
        length_m = nice_length((x1 - x0) * scale / 5)
    start_x = x0 + location[0] * (x1 - x0)
    bar_y = y0 + location[1] * (y1 - y0)
    end_x = start_x + length_m / scale
    for colour, width in ((outline, 6), (color, 3)):
        ax.plot([start_x, end_x], [bar_y, bar_y], color=colour,
                linewidth=width, solid_capstyle="butt", zorder=22)
    label = (f"{length_m / 1000:g} km" if length_m >= 1000
             else f"{length_m:g} m")
    ax.text((start_x + end_x) / 2, bar_y + 0.025 * (y1 - y0), label,
            ha="center", va="bottom", fontsize=fontsize, color=color,
            weight="bold", zorder=22,
            path_effects=[pe.withStroke(linewidth=2.5, foreground=outline)])
    return length_m


def add_north_arrow(ax, location=(0.92, 0.84), scale=0.09, fontsize=9):
    """
    Draw a two-tone north arrow (axes coordinates, aspect corrected).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes (north is up).
    location : tuple of float, optional
        Arrow centre in axes fractions.
    scale : float, optional
        Arrow height in axes fractions.
    fontsize : float, optional
        Size of the "N".
    """
    x, y = location
    fig_w, fig_h = ax.figure.get_size_inches()
    box = ax.get_position()
    dx = scale * 0.35 * (box.height * fig_h) / (box.width * fig_w)
    tip, base = (x, y + scale / 2), y - scale / 2
    for corner, face in ((x - dx, "black"), (x + dx, "white")):
        ax.add_patch(mpatches.Polygon(
            [tip, (corner, base), (x, base + scale * 0.3)],
            transform=ax.transAxes, facecolor=face, edgecolor="black",
            linewidth=0.8, zorder=23, clip_on=False))
    ax.text(x, tip[1] + 0.01, "N", transform=ax.transAxes, ha="center",
            va="bottom", fontsize=fontsize, weight="bold", zorder=23,
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])
