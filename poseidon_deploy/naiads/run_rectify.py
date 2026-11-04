"""Stage 07: rectify one event's images and labels onto the site grid.

Writes ``<event>/<zarr_base>/<zarr_orig_name>.zip`` and
``<zarr_label_name>.zip``, replacing any earlier output.

Example
-------
``python run_rectify.py --site carolina_beach --grid_dir data/grids
--target_event_dir flood_events/CB_03_20231002131313_20231002203713``
"""

import os

from poseidon_core.image_rectifier import ImageRectifier

import cli_common


def main(argv=None):
    """
    Rectify one flood event.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments. Default reads ``sys.argv``.
    """
    parser = cli_common.make_parser(__doc__.splitlines()[0])
    cli_common.add_site_argument(parser)
    parser.add_argument("--target_event_dir", required=True,
                        help="Event folder to process.")
    parser.add_argument("--grid_dir", required=True,
                        help="Folder holding shared_grid_<site>.npz.")
    parser.add_argument("--image_subfolder", default="orig_images",
                        help="Image subfolder inside the event.")
    parser.add_argument("--label_subfolder", default="labels",
                        help="Label subfolder inside the event.")
    parser.add_argument("--zarr_base", default="zarr",
                        help="Output subfolder inside the event.")
    parser.add_argument("--zarr_orig_name", default="orig_image_rects",
                        help="Store name for rectified images.")
    parser.add_argument("--zarr_label_name", default="labels_rects",
                        help="Store name for rectified labels.")
    cli_common.add_gpu_argument(parser)
    args = cli_common.start(parser, argv)
    site = cli_common.load_site(args)

    images = os.path.join(args.target_event_dir, args.image_subfolder)
    labels = os.path.join(args.target_event_dir, args.label_subfolder)
    cli_common.require_dir(images, "Image folder")
    cli_common.require_dir(labels, "Label folder")

    grid = cli_common.load_shared_grid(args.grid_dir, site)
    rectifier = ImageRectifier(site.intrinsics, site.extrinsics, grid["x"],
                               grid["y"], grid["z"],
                               use_gpu=cli_common.gpu_choice(args))

    out_dir = os.path.join(args.target_event_dir, args.zarr_base)
    n_images = rectifier.merge_rectify_folder(
        images, os.path.join(out_dir, args.zarr_orig_name))
    n_labels = rectifier.merge_rectify_folder(
        labels, os.path.join(out_dir, args.zarr_label_name), labels=True)
    if n_labels == 0:
        cli_common.fail(f"No labels rectified for {args.target_event_dir}")
    cli_common.logger.info("Rectified %d images and %d labels.", n_images,
                           n_labels)


if __name__ == "__main__":
    main()
