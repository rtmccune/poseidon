import cmocean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def plot_elev_grid(grid_z, save_fig=False, fig_name="pixel_DEM.png"):
    """
    Plots a digital elevation model (DEM) grid using a truncated version of the cmocean 'topo' colormap
    that emphasizes above-land elevation values.

    Parameters:
    ----------
    grid_z : 2D array-like
        A 2D array representing elevation values, typically generated from a gridded interpolation or DEM.

    save_fig : bool, optional (default=False)
        If True, saves the figure to disk instead of just displaying it.

    fig_name : str, optional (default='pixel_DEM.png')
        Filename to use when saving the figure. Only used if `save_fig` is True.

    Behavior:
    --------
    - Uses the upper half of the cmocean `topo` colormap to emphasize land elevations.
    - Plots the elevation grid with labeled axes and a colorbar.
    - Saves the figure as a high-resolution PNG if `save_fig` is True.
    """

    # Get the topo colormap from cmocean
    cmap = cmocean.cm.topo

    # Truncate the colormap to get only the above-land portion
    # Assuming "above land" is the upper half of the colormap
    above_land_cmap = LinearSegmentedColormap.from_list(
        "above_land_cmap", cmap(np.linspace(0.5, 1, 256))
    )

    plt.imshow(grid_z, origin="lower", cmap=above_land_cmap)
    plt.colorbar(label="Elevation (meters)")
    plt.title("Pixel DEM")
    plt.xlabel("Easting")
    plt.ylabel("Northing")

    if save_fig:
        # Save the figure before showing it
        plt.savefig(fig_name, bbox_inches="tight", pad_inches=0.1, dpi=300)

    plt.show()
