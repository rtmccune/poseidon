"""Shared figure style: a colour-blind-safe palette drawn from feathers.

The colours come from the feathers R package of Australian bird
palettes (Shandiya Balasubramaniam, https://github.com/shandiya/feathers).
feathers makes no colour-blind claims, so the categorical slots below
were chosen from its palettes and checked against white
(``#ffffff``) with the six checks of the dataviz method:

* OKLCH lightness 0.43-0.77 and chroma >= 0.10 for every slot;
* contrast >= 3:1 against white for every slot;
* every pair >= 15 apart for normal vision and >= 8 apart under
  protanopia and deuteranopia (Machado, Oliveira and Fernandes 2009,
  severity 1.0), distances in OKLab x 100.

Results (worst pair over all pairs): normal 19.4, protanopia 11.6,
deuteranopia 9.7; contrasts 4.53, 3.76 and 5.21. Rejected feathers
candidates: bee-eater yellow ``#EDD03E`` and orange ``#F5A200`` (too
light on white), and olive ``#6D8600``, purple ``#A56EB6`` and teal
``#3EBCB6``, which collapse into another slot under protanopia or
deuteranopia. Categorical slots are assigned in this fixed order and
never cycled.
"""

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

# Colours are defined once in poseidon_utils.palette (shared with the
# depth maps); see that module for the checks behind them.
from poseidon_utils.palette import (  # noqa: E402,F401
    BLUE,
    CATEGORICAL,
    CONTEXT_GREY,
    INK,
    INK_STRONG,
    MAGENTA,
    ORANGE,
    VIRTUAL_SENSOR_COLORS,
    virtual_sensor_color,
)

#: Sequential blue ramp, light to dark: bee_eater ``#06ABDF`` and
#: ``#007CBF``, blue_winged_kookaburra ``#02407c`` and ``#06213a``.
#: OKLCH lightness falls monotonically (0.69, 0.56, 0.37, 0.25) and
#: the light end keeps 2.7:1 contrast on white.
SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list(
    "feathers_blue", ["#06ABDF", "#007CBF", "#02407C", "#06213A"])


def apply_style():
    """Set matplotlib defaults: recessive grid and axes, thin marks."""
    mpl.rcParams.update({
        "axes.edgecolor": INK,
        "axes.labelcolor": INK_STRONG,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),
        "grid.color": "#DDE0DC",
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
        "xtick.color": INK,
        "ytick.color": INK,
        "lines.linewidth": 1.8,
        "lines.markersize": 6,
        "legend.frameon": False,
        "font.size": 10,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
