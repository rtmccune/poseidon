"""Stage 09: plot one event's depth maps and virtual sensor series.

Uses the site's grid extent, CRS, virtual sensors, and basemap from
its YAML config.

Example
-------
``python run_plotter.py --site carolina_beach --target_event_dir
flood_events/CB_03_20231002131313_20231002203713 --stats 95_perc``
"""

from poseidon_core.depth_map_plotter import DepthMapPlotter

import cli_common


def main(argv=None):
    """
    Plot one flood event.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments. Default reads ``sys.argv``.
    """
    parser = cli_common.make_parser(__doc__.splitlines()[0])
    cli_common.add_site_argument(parser)
    parser.add_argument("--target_event_dir", required=True,
                        help="Event folder to plot.")
    parser.add_argument("--stats", nargs="+", default=["95_perc"],
                        help="Map statistics to plot.")
    parser.add_argument("--basemap", default=None,
                        help="Basemap GeoTIFF; overrides the site config.")
    args = cli_common.start(parser, argv)
    site = cli_common.load_site(args)
    cli_common.require_dir(args.target_event_dir, "Event folder")

    grid = site.grid
    plotter = DepthMapPlotter(
        args.target_event_dir, grid.min_x, grid.max_x, grid.min_y,
        grid.max_y, resolution_m=grid.resolution, bbox_crs=site.crs,
        virtual_sensor_locations=site.virtual_sensors, plot_sensors=True,
        basemap_path=args.basemap or site.resolve(site.basemap))
    plotter.preprocess_single_event()
    plotter.process_single_flood_event(stats_to_plot=args.stats)
    plotter.plot_all_time_series()


if __name__ == "__main__":
    main()
