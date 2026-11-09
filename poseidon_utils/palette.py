"""Colour-blind-checked colours drawn from the feathers palettes.

feathers (Shandiya Balasubramaniam, https://github.com/shandiya/feathers)
is an R package of Australian bird palettes; it makes no colour-blind
claims, so every group below was checked with the dataviz six checks:
OKLCH lightness and chroma, contrast against white, and pairwise
distance in OKLab x 100 for normal vision (>= 15) and under protanopia
and deuteranopia (Machado, Oliveira and Fernandes 2009, severity 1.0;
>= 8).

Series colours (worst pair: normal 19.4, CVD 9.7; contrast >= 3.76:1):
blue ``#007CBF`` (bee_eater), orange ``#D36328`` (spotted_pardalote),
magenta ``#BD338F`` (rose_crowned_fruit_dove).

Virtual sensor colours (worst pair: normal 19.5, CVD 12.1; with the
series blue: normal 24.3, CVD 10.0): olive ``#6D8600`` (bee_eater),
gold ``#E19E00`` (yellow_robin), pink ``#D683AD`` (princess_parrot).
They avoid the blue-purple depth colormap (cmocean ``dense``) of the
maps. Gold and pink are below 3:1 against white, so sensors are always
labelled as well (``VS1``...). Do not use them as separate series in a
figure that also uses the series orange: gold and orange are only 14.8
apart for normal vision.
"""

#: Series colours, assigned in this fixed order and never cycled.
BLUE = "#007CBF"
ORANGE = "#D36328"
MAGENTA = "#BD338F"
CATEGORICAL = (BLUE, ORANGE, MAGENTA)

#: Neutral context marks and ink (spotted_pardalote greys).
CONTEXT_GREY = "#B4B9B3"
INK = "#424847"
INK_STRONG = "#000100"

#: Virtual sensor colours: VS1, VS2, VS3, ... in site-config order.
VIRTUAL_SENSOR_COLORS = ("#6D8600", "#E19E00", "#D683AD")


def virtual_sensor_color(index):
    """
    Colour of a virtual sensor, fixed by its position in the site config.

    Sensors beyond :data:`VIRTUAL_SENSOR_COLORS` get the neutral grey
    rather than a recycled colour; their labels still identify them.

    Parameters
    ----------
    index : int
        0-based sensor index (VS1 is 0).

    Returns
    -------
    str
        Hex colour.
    """
    if 0 <= index < len(VIRTUAL_SENSOR_COLORS):
        return VIRTUAL_SENSOR_COLORS[index]
    return CONTEXT_GREY
