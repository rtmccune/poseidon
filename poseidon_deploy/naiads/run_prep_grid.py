"""Stage 06: build a site's rectification grid from LiDAR once.

Interpolates the site's ground-classified LiDAR onto its grid extent
and saves ``<grid_dir>/shared_grid_<site>.npz`` (arrays ``x``, ``y``,
``z``) for the per-event rectification and depth jobs, plus Zarr
copies of each grid.

Example
-------
``python run_prep_grid.py --site carolina_beach --grid_dir data/grids``
"""

import os

import numpy as np

from poseidon_core.grid_generator import GridGenerator

import cli_common


def main(argv=None):
    """
    Generate and save a site's shared grid.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments. Default reads ``sys.argv``.
    """
    parser = cli_common.make_parser(__doc__.splitlines()[0])
    cli_common.add_site_argument(parser)
    parser.add_argument("--grid_dir", required=True,
                        help="Folder for the shared grid files.")
    parser.add_argument("--lidar_file", default=None,
                        help="LiDAR file; overrides the site config.")
    args = cli_common.start(parser, argv)
    site = cli_common.load_site(args)

    lidar = args.lidar_file or site.resolve(site.lidar_file)
    if lidar is None or not os.path.exists(lidar):
        cli_common.fail(f"LiDAR file not found: {lidar}")

    grid = site.grid
    generator = GridGenerator(lidar, grid.min_x, grid.max_x, grid.min_y,
                              grid.max_y, extent_units="meters",
                              lidar_units=site.lidar_units)
    grid_x, grid_y, grid_z = generator.gen_grid(
        grid.resolution, generator.create_point_array(), dir=args.grid_dir,
        grid_descriptor=site.name)

    out_path = cli_common.shared_grid_path(args.grid_dir, site)
    np.savez_compressed(out_path, x=grid_x, y=grid_y, z=grid_z)
    cli_common.logger.info("Saved shared grid %s", out_path)


if __name__ == "__main__":
    main()
