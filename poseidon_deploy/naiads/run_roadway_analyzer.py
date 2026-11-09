"""Stage 10: summarize flooding on a roadway ROI for one event.

Writes ``<roi>_accessibility_time_series.csv``,
``<roi>_wse_time_series.csv`` and ``<roi>_roi_mask.png`` into the
event folder, where ``<roi>`` is the LabelMe JSON filename without its
extension.

``--roi_view rectified`` (default) reads an ROI drawn on a rectified
image: ``--json_path`` or the site config's ``roi_json``.
``--roi_view camera`` reads a polygon drawn on an original camera
image (``--json_path`` or the site's ``camera_roi_json``) and rectifies
it with the site's camera model and ``--grid_dir`` grid.

Example
-------
``python run_roadway_analyzer.py --site down_east --roi_view camera
--grid_dir data/grids
--target_event_dir flood_events/DE_01_20240521203240_20240522064440``
"""

from poseidon_core.depth_map_processor import STATISTICS
from poseidon_core.image_rectifier import ImageRectifier
from poseidon_core.roadway_analyzer import RoadwayAnalyzer

import cli_common


def main(argv=None):
    """
    Analyze roadway flooding for one flood event.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments. Default reads ``sys.argv``.
    """
    parser = cli_common.make_parser(__doc__.splitlines()[0])
    parser.add_argument("--target_event_dir", required=True,
                        help="Event folder to analyze.")
    parser.add_argument("--site", default=None,
                        help="Site whose ROI JSON to use when --json_path "
                             "is omitted; required for --roi_view camera.")
    parser.add_argument("--roi_view", choices=("rectified", "camera"),
                        default="rectified",
                        help="Whether the ROI was drawn on a rectified "
                             "image or on an original camera image.")
    parser.add_argument("--json_path", default=None,
                        help="LabelMe JSON defining the roadway ROI.")
    parser.add_argument("--grid_dir", default=None,
                        help="Folder holding shared_grid_<site>.npz; "
                             "required for --roi_view camera.")
    parser.add_argument("--label", default="roadway",
                        help="Label of the ROI shape in the JSON.")
    parser.add_argument("--step_size", type=float, default=1.0,
                        help="Pixel spacing of samples along line ROIs.")
    parser.add_argument("--statistic", default="95_perc",
                        choices=list(STATISTICS),
                        help="Depth/WSE map statistic to analyze.")
    args = cli_common.start(parser, argv)
    camera = args.roi_view == "camera"
    site = cli_common.load_site(args) if args.site else None
    if camera and (site is None or args.grid_dir is None):
        cli_common.fail("--roi_view camera needs --site and --grid_dir.")

    json_path = args.json_path
    if json_path is None and site is not None:
        json_path = site.resolve(site.camera_roi_json if camera
                                 else site.roi_json)
    if json_path is None:
        key = "camera_roi_json" if camera else "roi_json"
        cli_common.fail(f"Pass --json_path, or --site with a {key}.")

    rectifier = None
    if camera:
        grid = cli_common.load_shared_grid(args.grid_dir, site)
        rectifier = ImageRectifier(site.intrinsics, site.extrinsics,
                                   grid["x"], grid["y"], grid["z"],
                                   use_gpu=False)

    analyzer = RoadwayAnalyzer(args.target_event_dir, json_path,
                               roi_label=args.label,
                               step_size=args.step_size,
                               statistic=args.statistic,
                               rectifier=rectifier)
    if not analyzer.process_single_event():
        cli_common.fail(f"No depth maps to analyze in "
                        f"{args.target_event_dir}")


if __name__ == "__main__":
    main()
