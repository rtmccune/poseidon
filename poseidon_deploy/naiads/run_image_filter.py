"""Stage 01: keep only images taken in daylight.

By default an image is kept if it was taken between local sunrise and
sunset at the site's camera location (``--method sun``). Alternatively
``--method hours`` keeps a fixed Eastern-time hour window.

Example
-------
``python run_image_filter.py --site carolina_beach --image_dir
images/all_events --dest images/daylight``
"""

from poseidon_utils.image_handler import ImageHandler

import cli_common


def main(argv=None):
    """
    Copy daylight images into a new folder.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments. Default reads ``sys.argv``.
    """
    parser = cli_common.make_parser(__doc__.splitlines()[0])
    cli_common.add_site_argument(parser)
    parser.add_argument("--image_dir", required=True,
                        help="Folder of images to filter.")
    parser.add_argument("--dest", required=True,
                        help="Folder receiving the kept images.")
    parser.add_argument("--method", choices=("sun", "hours"), default="sun",
                        help="Filter by solar position or by clock hours.")
    parser.add_argument("--lat", type=float, default=None,
                        help="Camera latitude; overrides the site config.")
    parser.add_argument("--lon", type=float, default=None,
                        help="Camera longitude; overrides the site config.")
    parser.add_argument("--start", type=int, default=6,
                        help="First Eastern hour kept (--method hours).")
    parser.add_argument("--end", type=int, default=19,
                        help="Last Eastern hour kept (--method hours).")
    parser.add_argument("--workers", type=int, default=None,
                        help="Copy threads.")
    args = cli_common.start(parser, argv)
    handler = ImageHandler()

    if args.method == "hours":
        handler.copy_images_using_hour_window(
            args.image_dir, args.dest, args.start, args.end, args.workers)
        return

    site = cli_common.load_site(args)
    lat = site.latitude if args.lat is None else args.lat
    lon = site.longitude if args.lon is None else args.lon
    if lat is None or lon is None:
        cli_common.fail(f"No camera coordinates for site '{site.name}'; set "
                        f"latitude/longitude in its YAML or pass --lat/--lon.")
    handler.copy_images_using_sun_position(
        args.image_dir, args.dest, latitude=lat, longitude=lon,
        max_workers=args.workers)


if __name__ == "__main__":
    main()
