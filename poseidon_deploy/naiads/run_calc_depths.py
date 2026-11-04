"""Stage 08: compute water surface and depth maps for one event.

Reads ``<event>/<zarr_base>/<zarr_label_dir>.zip`` and writes
``<event>/<zarr_base>/<zarr_depth_dir>.zip`` plus waterline elevation
histograms under ``<event>/<plot_base_dir>``.

Example
-------
``python run_calc_depths.py --site carolina_beach --grid_dir data/grids
--target_event_dir flood_events/CB_03_20231002131313_20231002203713``
"""

import os

from poseidon_core.depth_map_processor import DepthMapProcessor

import cli_common


def main(argv=None):
    """
    Compute depth maps for one flood event.

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
    parser.add_argument("--zarr_base", default="zarr",
                        help="Zarr subfolder inside the event.")
    parser.add_argument("--zarr_label_dir", default="labels_rects",
                        help="Rectified label store name (no .zip).")
    parser.add_argument("--zarr_depth_dir", default="depth_maps",
                        help="Depth map store name (no .zip).")
    parser.add_argument("--plot_base_dir", default="plots",
                        help="Histogram folder inside the event.")
    parser.add_argument("--no_histograms", action="store_true",
                        help="Skip waterline elevation histograms.")
    cli_common.add_gpu_argument(parser)
    args = cli_common.start(parser, argv)
    site = cli_common.load_site(args)

    zarr_dir = os.path.join(args.target_event_dir, args.zarr_base)
    labels_zip = os.path.join(zarr_dir, f"{args.zarr_label_dir}.zip")
    if not os.path.exists(labels_zip):
        cli_common.fail(f"Rectified labels not found: {labels_zip}")

    grid = cli_common.load_shared_grid(args.grid_dir, site)
    processor = DepthMapProcessor(grid["z"],
                                  plot_edges=not args.no_histograms,
                                  use_gpu=cli_common.gpu_choice(args))
    processed = processor.process_depth_maps(
        labels_zip, os.path.join(zarr_dir, f"{args.zarr_depth_dir}.zip"),
        pond_edge_elev_plot_dir=os.path.join(args.target_event_dir,
                                             args.plot_base_dir))
    if processed == 0:
        cli_common.fail(f"No depth maps computed for "
                        f"{args.target_event_dir}")


if __name__ == "__main__":
    main()
