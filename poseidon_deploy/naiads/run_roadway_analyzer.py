"""Stage 10: summarize flooding on a roadway ROI for one event.

Writes ``<roi>_accessibility_time_series.csv`` and
``<roi>_wse_time_series.csv`` into the event folder, where ``<roi>``
is the LabelMe JSON filename without its extension. The ROI comes from
``--json_path`` or, if omitted, the site config's ``roi_json``.

Example
-------
``python run_roadway_analyzer.py --site carolina_beach
--target_event_dir flood_events/CB_03_20231002131313_20231002203713``
"""

from poseidon_core.depth_map_processor import STATISTICS
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
                        help="Site whose roi_json to use when --json_path "
                             "is omitted.")
    parser.add_argument("--json_path", default=None,
                        help="LabelMe JSON defining the roadway ROI.")
    parser.add_argument("--label", default="roadway",
                        help="Label of the ROI shape in the JSON.")
    parser.add_argument("--step_size", type=float, default=1.0,
                        help="Pixel spacing of samples along line ROIs.")
    parser.add_argument("--statistic", default="95_perc",
                        choices=list(STATISTICS),
                        help="Depth/WSE map statistic to analyze.")
    args = cli_common.start(parser, argv)
    json_path = args.json_path
    if json_path is None and args.site:
        site = cli_common.load_site(args)
        json_path = site.resolve(site.roi_json)
    if json_path is None:
        cli_common.fail("Pass --json_path, or --site with a roi_json.")

    analyzer = RoadwayAnalyzer(args.target_event_dir, json_path,
                               roi_label=args.label,
                               step_size=args.step_size,
                               statistic=args.statistic)
    if not analyzer.process_single_event():
        cli_common.fail(f"No depth maps to analyze in "
                        f"{args.target_event_dir}")


if __name__ == "__main__":
    main()
