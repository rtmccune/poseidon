"""Draw the methods process diagram as a layered SVG for Inkscape.

Three panels on one page:

(a) Overview: the workflow from flood event detection to evaluation, as
    a row of steps. The segmentation and georectification steps are
    outlined in their zoom-in colours.
(b) Machine learning zoom-in: model training (once) and per-image
    segmentation with quality review, with example images.
(c) Georectification zoom-in: LiDAR grid, camera model, rectification,
    waterline-based water surface and depth, and roadway / virtual
    sensor sampling, with example images.

Example images come from ``methods_figure_assets.py`` (its
``manifest.json`` gives their sizes). Missing images are drawn as
labelled placeholders, so the diagram can be laid out before the
images exist. Images are linked by relative path (``--embed`` stores
them in the SVG instead), so keep the SVG beside the assets folder or
re-link in Inkscape (Edit > Preferences > Imported Images).

Every part is a named Inkscape layer (Overview, ML zoom, Georect zoom,
Connectors, Images, Panel letters), and all text is live text in one
font, so the layout can be refined by hand. Numbers in the text (grid
spacing, model input size, statistic) are read from the site config
and arguments; check them against the methods.

Example
-------
``python methods_process_diagram.py --site carolina_beach --assets
data/methods_figure/assets --out data/methods_figure/process_diagram.svg``
"""

import argparse
import base64
import json
import logging
import mimetypes
import os
import textwrap
from xml.sax.saxutils import escape

from poseidon_utils.logging_utils import setup_logging
from poseidon_utils.palette import BLUE, INK, ORANGE
from poseidon_utils.site_config import load_site_config

logger = logging.getLogger(__name__)

#: Page size (SVG user units, 96 per inch).
WIDTH, HEIGHT = 1800, 950

#: Height of the example image slots in the zoom panels.
IMAGE_HEIGHT = 300

#: Font of every text element.
FONT = "Arial, Helvetica, sans-serif"

#: Fill of ordinary step boxes and of the zoom panels.
BOX_FILL, PANEL_FILL = "#F4F5F3", "#FFFFFF"

#: Accent colours of the two zoom-ins.
ML_COLOR, GEO_COLOR = BLUE, ORANGE


class Svg:
    """
    Collect SVG elements into named Inkscape layers.

    Parameters
    ----------
    assets : str
        Folder of the example images.
    out_path : str
        Where the SVG will be written (image links are relative to it).
    embed : bool
        Embed images as base64 instead of linking them.
    sizes : dict
        Image ``{filename: [width, height]}`` from the asset manifest.
    """

    def __init__(self, assets, out_path, embed, sizes):
        self.assets, self.out_path, self.embed = assets, out_path, embed
        self.sizes = sizes
        self.layers = {}

    def add(self, layer, element):
        """Append an element string to a layer."""
        self.layers.setdefault(layer, []).append(element)

    def text(self, layer, x, y, lines, size=13, weight="normal",
             anchor="start", color=INK, line_height=1.25, style=""):
        """
        Add a multi-line text element (one tspan per line).

        Returns
        -------
        float
            Y of the line after the last one.
        """
        if isinstance(lines, str):
            lines = [lines]
        spans = "".join(
            f'<tspan x="{x:.1f}" dy="{0 if i == 0 else size * line_height:.1f}"'
            f'>{escape(line)}</tspan>' for i, line in enumerate(lines))
        self.add(layer, f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" '
                        f'font-size="{size}" font-weight="{weight}" '
                        f'text-anchor="{anchor}" fill="{color}" '
                        f'xml:space="preserve" style="{style}">{spans}</text>')
        return y + size * line_height * len(lines)

    def box(self, layer, x, y, w, h, title, body=(), stroke="#9AA09A",
            fill=BOX_FILL, stroke_width=1.2, dash=None, wrap=None, tag=None,
            title_size=14, body_size=11.5):
        """
        Add a rounded step box with a bold title and wrapped body text.

        Returns
        -------
        tuple of float
            The box's ``(x, y, w, h)``.
        """
        dashing = f' stroke-dasharray="{dash}"' if dash else ""
        group = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" '
                 f'fill="{fill}" stroke="{stroke}" '
                 f'stroke-width="{stroke_width}"{dashing}/>']
        self.add(layer, "<g>" + "".join(group))
        wrap = wrap or max(12, int(w / (body_size * 0.52)))
        title_lines = textwrap.wrap(title, max(10, int(w / (title_size * 0.56))))
        y_text = self.text(layer, x + 10, y + 10 + title_size, title_lines,
                           size=title_size, weight="bold")
        lines = [line for item in body for line in textwrap.wrap(item, wrap)]
        if lines:
            self.text(layer, x + 10, y_text + 2, lines, size=body_size,
                      color="#3C403C")
        if tag:
            self.text(layer, x + w - 8, y + h - 8, tag, size=10.5,
                      anchor="end", color=stroke, weight="bold")
        self.add(layer, "</g>")
        return x, y, w, h

    def arrow(self, layer, points, color="#5F655F", width=1.6, dash=None):
        """Add a polyline arrow through `points` ending in an arrowhead."""
        path = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        dashing = f' stroke-dasharray="{dash}"' if dash else ""
        marker = {INK: "head-ink", ML_COLOR: "head-ml",
                  GEO_COLOR: "head-geo"}.get(color, "head")
        self.add(layer, f'<polyline points="{path}" fill="none" '
                        f'stroke="{color}" stroke-width="{width}"'
                        f'{dashing} marker-end="url(#{marker})"/>')

    def image(self, x, y, w, h, name, caption=None, layer="Images"):
        """
        Add an example image fitted inside a slot, or a placeholder.

        Returns
        -------
        tuple of float
            The drawn image's ``(x, y, w, h)``.
        """
        path = os.path.join(self.assets, name)
        size = self.sizes.get(name)
        if name.endswith(".svg") or os.path.exists(path):
            if size:
                scale = min(w / size[0], h / size[1])
                dw, dh = size[0] * scale, size[1] * scale
                x, y, w, h = x + (w - dw) / 2, y + (h - dh) / 2, dw, dh
        if os.path.exists(path):
            if self.embed:
                kind = mimetypes.guess_type(path)[0] or "image/png"
                with open(path, "rb") as handle:
                    href = (f"data:{kind};base64,"
                            + base64.b64encode(handle.read()).decode())
            else:
                href = os.path.relpath(path, os.path.dirname(
                    os.path.abspath(self.out_path)))
            self.add(layer, f'<image x="{x:.1f}" y="{y:.1f}" '
                            f'width="{w:.1f}" height="{h:.1f}" '
                            f'xlink:href="{escape(href)}" '
                            f'preserveAspectRatio="xMidYMid meet" '
                            f'inkscape:label="{escape(name)}"/>')
        else:
            self.add(layer, f'<g inkscape:label="placeholder {escape(name)}">'
                            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" '
                            f'height="{h:.1f}" fill="#ECEDEB" stroke="#B5B9B4" '
                            f'stroke-dasharray="5 4"/>')
            self.text(layer, x + w / 2, y + h / 2, [name, "(not found)"],
                      size=11, anchor="middle", color="#7A7F79")
            self.add(layer, "</g>")
        if caption:
            self.text(layer, x + w / 2, y + h + 15, caption, size=11.5,
                      anchor="middle", color="#3C403C")
        return x, y, w, h

    def write(self):
        """Write the SVG with one Inkscape layer per group."""
        heads = "".join(
            f'<marker id="{name}" viewBox="0 0 10 10" refX="9" refY="5" '
            f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{color}"/></marker>'
            for name, color in (("head", "#5F655F"), ("head-ink", INK),
                                ("head-ml", ML_COLOR),
                                ("head-geo", GEO_COLOR)))
        order = ("Background", "Overview", "ML zoom", "Georect zoom",
                 "Images", "Connectors", "Panel letters")
        body = "".join(
            f'<g inkscape:groupmode="layer" id="layer-{i}" '
            f'inkscape:label="{name}">{"".join(self.layers.get(name, []))}'
            f'</g>' for i, name in enumerate(order))
        svg = (f'<?xml version="1.0" encoding="UTF-8"?>\n'
               f'<svg xmlns="http://www.w3.org/2000/svg" '
               f'xmlns:xlink="http://www.w3.org/1999/xlink" '
               f'xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" '
               f'width="{WIDTH}" height="{HEIGHT}" '
               f'viewBox="0 0 {WIDTH} {HEIGHT}">'
               f'<defs>{heads}</defs>{body}</svg>\n')
        with open(self.out_path, "w") as handle:
            handle.write(svg)


def overview(svg, site, args):
    """
    Draw panel (a): the workflow overview.

    Returns
    -------
    dict
        Box geometry of the segmentation and rectification steps, for
        the zoom connectors.
    """
    layer = "Overview"
    steps = [
        ("Flood events", ["In-situ water level sensor; events detected, "
                          "then quality controlled",
                          f"(± {args.padding_hours:g} h window)"],
         None),
        ("Image retrieval", ["Camera images from each event window "
                             "(stage 00)"], None),
        ("Daylight filter", ["Keep images between local sunrise and "
                             "sunset (stage 01)"], None),
        ("Segmentation", ["Classify every pixel; water mask per image; "
                          "quality review (stages 02-04)"], "ml"),
        ("Event organization", ["Images, labels and sensor series per "
                                "event (stage 05)"], None),
        ("Georectification & depth", ["Water mask onto LiDAR grid; "
                                      "water surface and depth maps "
                                      "(stages 06-08)"], "geo"),
        ("Flood metrics", ["Virtual sensors; roadway coverage and depth "
                           "per image (stages 09-11)"], None),
        ("Evaluation", ["Compare with the in-situ sensor: timing, depth "
                        "by coverage"], None),
    ]
    x0, y0, gap, h = 30, 62, 16, 132
    w = (WIDTH - 2 * x0 - gap * (len(steps) - 1)) / len(steps)
    svg.add("Background", f'<rect width="{WIDTH}" height="{HEIGHT}" '
                          f'fill="#FFFFFF"/>')
    svg.text(layer, x0 + 34, 40, "Workflow overview", size=17,
             weight="bold")
    boxes = {}
    for i, (title, body, zoom) in enumerate(steps):
        x = x0 + i * (w + gap)
        color = {"ml": ML_COLOR, "geo": GEO_COLOR}.get(zoom, "#9AA09A")
        svg.box(layer, x, y0, w, h, title, body, stroke=color,
                stroke_width=2.6 if zoom else 1.2,
                tag={"ml": "see (b)", "geo": "see (c)"}.get(zoom))
        if zoom:
            boxes[zoom] = (x, y0, w, h)
        if i:
            svg.arrow("Connectors", [(x - gap + 1, y0 + h / 2),
                                     (x - 2, y0 + h / 2)])
    return boxes


def ml_zoom(svg, args, frame):
    """Draw panel (b): segmentation zoom-in inside `frame`."""
    layer = "ML zoom"
    x, y, w, h = frame
    svg.add(layer, f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" '
                   f'fill="{PANEL_FILL}" stroke="{ML_COLOR}" '
                   f'stroke-width="2" stroke-dasharray="8 5"/>')
    svg.text(layer, x + 52, y + 30, "Segmentation (machine learning)",
             size=17, weight="bold", color=ML_COLOR)

    # Training, done once
    ty, th, tw = y + 50, 104, (w - 36 - 2 * 24) / 3
    train = [
        ("Labelled images", [f"Hand-labelled training images "
                             f"({args.n_classes} classes, incl. water)"]),
        ("Model training", [f"SegFormer in Segmentation Gym; "
                            f"{args.model_size} px input, augmented"]),
        ("Trained model", ["Fixed weights applied to every image"]),
    ]
    svg.text(layer, x + 18, ty - 2, "Training (once)", size=12.5,
             weight="bold", color="#5F655F")
    for i, (title, body) in enumerate(train):
        bx = x + 18 + i * (tw + 24)
        svg.box(layer, bx, ty + 6, tw, th, title, body)
        if i:
            svg.arrow("Connectors", [(bx - 23, ty + 6 + th / 2),
                                     (bx - 2, ty + 6 + th / 2)])

    # Per image
    py = ty + th + 52
    svg.text(layer, x + 18, py - 8, "Per daylight image", size=12.5,
             weight="bold", color="#5F655F")
    pw, ph = (w - 36 - 3 * 18) / 4, 120
    steps = [
        ("Camera image", ["Daylight image from the event window"]),
        ("Prediction", ["Class of every pixel, resized to the full image"]),
        ("Water mask", [f"Pixels of the water class (label "
                        f"{args.water_class})"]),
        ("Quality review", ["Overlay inspected by eye; images scored good "
                            "or poor. Only good images are analysed"]),
    ]
    tops = []
    for i, (title, body) in enumerate(steps):
        bx = x + 18 + i * (pw + 18)
        svg.box(layer, bx, py, pw, ph, title, body, stroke=ML_COLOR)
        tops.append(bx)
        if i:
            svg.arrow("Connectors", [(bx - 17, py + ph / 2),
                                     (bx - 2, py + ph / 2)], color=ML_COLOR)
    # Model feeds the prediction step
    model_x = x + 18 + 2 * (tw + 24) + tw / 2
    svg.arrow("Connectors", [(model_x, ty + 6 + th), (model_x, py - 22),
                             (tops[1] + pw / 2, py - 22),
                             (tops[1] + pw / 2, py - 2)], color=ML_COLOR)

    # Example images below their steps
    iy, ih = py + ph + 22, IMAGE_HEIGHT
    for bx, name, caption in zip(
            tops, ("01_original.jpg", "02_overlay.jpg", "04_water_mask.png",
                   "05_camera_roi.jpg"),
            ("Original image", "Segmentation overlay", "Water mask",
             "Roadway region (camera view)")):
        svg.image(bx, iy, pw, ih, name, caption)


def geo_zoom(svg, site, args, frame):
    """Draw panel (c): georectification zoom-in inside `frame`."""
    layer = "Georect zoom"
    x, y, w, h = frame
    svg.add(layer, f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" '
                   f'fill="{PANEL_FILL}" stroke="{GEO_COLOR}" '
                   f'stroke-width="2" stroke-dasharray="8 5"/>')
    svg.text(layer, x + 52, y + 30, "Georectification and depth",
             size=17, weight="bold", color=GEO_COLOR)
    res = site.grid.resolution
    span_x = site.grid.max_x - site.grid.min_x
    span_y = site.grid.max_y - site.grid.min_y

    # Inputs, done once
    ty, th, tw = y + 50, 104, (w - 36 - 24) / 2
    svg.text(layer, x + 18, ty - 2, "Site set-up (once)", size=12.5,
             weight="bold", color="#5F655F")
    svg.box(layer, x + 18, ty + 6, tw, th, "Elevation grid",
            [f"Ground LiDAR returns interpolated to a {res * 100:g} cm grid "
             f"({span_x:g} m x {span_y:g} m)"])
    svg.box(layer, x + 18 + tw + 24, ty + 6, tw, th, "Camera model",
            ["Calibrated intrinsics (lens, distortion) and pose "
             "(position, azimuth, tilt, roll)"])

    py = ty + th + 52
    svg.text(layer, x + 18, py - 8, "Per water mask", size=12.5,
             weight="bold", color="#5F655F")
    pw, ph = (w - 36 - 3 * 18) / 4, 120
    steps = [
        ("Rectification", ["Each grid cell projected into the image; its "
                           "pixel's class sampled"]),
        ("Waterline", ["Connected water regions; edge cells touching dry "
                       "ground"]),
        ("Water surface & depth", [f"Surface = {args.statistic_label} of "
                                   f"ground elevation along the waterline; "
                                   f"depth = surface - ground"]),
        ("Sampling", ["Virtual sensors (grid cells) and the roadway region "
                      "per image"]),
    ]
    tops = []
    for i, (title, body) in enumerate(steps):
        bx = x + 18 + i * (pw + 18)
        svg.box(layer, bx, py, pw, ph, title, body, stroke=GEO_COLOR)
        tops.append(bx)
        if i:
            svg.arrow("Connectors", [(bx - 17, py + ph / 2),
                                     (bx - 2, py + ph / 2)], color=GEO_COLOR)
    grid_x, cam_x = x + 18 + tw / 2, x + 18 + tw + 24 + tw / 2
    svg.arrow("Connectors", [(cam_x, ty + 6 + th), (cam_x, py - 22),
                             (tops[0] + pw * 0.7, py - 22),
                             (tops[0] + pw * 0.7, py - 2)], color=GEO_COLOR)
    svg.arrow("Connectors", [(grid_x, ty + 6 + th), (grid_x, py - 30),
                             (tops[2] + pw / 2, py - 30),
                             (tops[2] + pw / 2, py - 2)], color=GEO_COLOR)

    iy, ih = py + ph + 22, IMAGE_HEIGHT
    for bx, name, caption in zip(
            tops, ("07_rectified.png", "08_rectified_water.png",
                   "09_depth.png", "10_roi_sensors.png"),
            ("Rectified image (north up)", "Rectified water mask",
             "Depth map", "Roadway region and virtual sensors")):
        # Leave room beside the depth map for its colour bar
        slot = pw * 0.8 if name == "09_depth.png" else pw
        svg.image(bx, iy, slot, ih, name, caption)
    svg.image(tops[2] + pw * 0.8, iy + ih * 0.15, pw * 0.2, ih * 0.7,
              "09_depth_colorbar.svg")
    return tops, py, pw


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
        description="Draw the methods process diagram as SVG.")
    parser.add_argument("--site", required=True,
                        help="Site config (grid size and spacing).")
    parser.add_argument("--assets", required=True,
                        help="Folder of methods_figure_assets.py output.")
    parser.add_argument("--out", required=True, help="Output .svg path.")
    parser.add_argument("--embed", action="store_true",
                        help="Embed images instead of linking them.")
    parser.add_argument("--n_classes", type=int, default=5,
                        help="Number of segmentation classes.")
    parser.add_argument("--model_size", default="512 x 512",
                        help="Model input size, as written in the text.")
    parser.add_argument("--water_class", type=int, default=1,
                        help="Label value of water.")
    parser.add_argument("--statistic_label", default="95th percentile",
                        help="Waterline statistic, as written in the text.")
    parser.add_argument("--padding_hours", type=float, default=3,
                        help="Hours added around each event.")
    return parser.parse_args(argv)


def main(argv=None):
    """
    Write the diagram.

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
    site = load_site_config(args.site)
    manifest_path = os.path.join(args.assets, "manifest.json")
    sizes = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as handle:
            sizes = json.load(handle).get("pixel_sizes", {})
    else:
        logger.warning("No %s; images drawn as placeholders.", manifest_path)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    svg = Svg(args.assets, args.out, args.embed, sizes)

    boxes = overview(svg, site, args)
    top, gap = 240, 24
    half = (WIDTH - 60 - gap) / 2
    ml_frame = (30, top, half, HEIGHT - top - 24)
    geo_frame = (30 + half + gap, top, half, HEIGHT - top - 24)
    ml_zoom(svg, args, ml_frame)
    geo_zoom(svg, site, args, geo_frame)

    # Zoom connectors from the overview steps to their panels
    for key, frame, color in (("ml", ml_frame, ML_COLOR),
                              ("geo", geo_frame, GEO_COLOR)):
        bx, by, bw, bh = boxes[key]
        svg.add("Connectors",
                f'<polygon points="{bx:.1f},{by + bh:.1f} '
                f'{bx + bw:.1f},{by + bh:.1f} '
                f'{frame[0] + frame[2] - 40:.1f},{frame[1]:.1f} '
                f'{frame[0] + 40:.1f},{frame[1]:.1f}" fill="{color}" '
                f'fill-opacity="0.07" stroke="{color}" stroke-opacity="0.5" '
                f'stroke-dasharray="4 4"/>')
    for letter, (lx, ly) in (("a", (30, 40)), ("b", (48, top + 30)),
                             ("c", (geo_frame[0] + 18, top + 30))):
        svg.text("Panel letters", lx, ly, f"({letter})", size=17,
                 weight="bold")
    svg.write()
    logger.info("Diagram written to %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
