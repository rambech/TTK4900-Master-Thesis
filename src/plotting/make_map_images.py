import tilemapbase
import matplotlib.pyplot as plt
import utils
# Latex settings for plot
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{lmodern,amsmath,amsfonts}')


def ravnkloa(show=False, save_file_name=None):
    """
    Makes a map of Ravnkloa for later use

    Based on the work of @MatthewDaws on Github:
    https://github.com/MatthewDaws/TileMapBase/blob/master/notebooks/Example.ipynb

    """
    tilemapbase.start_logging()
    # Don't need if you have run before; DB file will already exist.
    tilemapbase.init(create=True)
    # Use open street map
    t = tilemapbase.tiles.build_OSM()
    # My current office at the University of Leeds
    latlon_origin = (10.391, 63.434)
    aspect = 1.3

    degree_range = 0.003
    extent = tilemapbase.Extent.from_lonlat(
        latlon_origin[0] - degree_range,
        latlon_origin[0] + degree_range,
        latlon_origin[1] - degree_range,
        latlon_origin[1] + degree_range
    )

    extent = extent.to_aspect(aspect)
    print(f"extent: {extent}")

    map_width = aspect*utils.distance_along_great_circle(
        latlon_origin[0] - degree_range,
        latlon_origin[1] - degree_range,
        latlon_origin[0] + degree_range,
        latlon_origin[1] + degree_range
    )
    print(f"map width: {map_width}")

    # On my desktop, DPI gets scaled by 0.75
    fig, ax = plt.subplots(figsize=(8, 6), dpi=400)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    plotter = tilemapbase.Plotter(extent, t, width=1000, height=700)
    plotter.plot(ax, t)

    # x, y = tilemapbase.project(*my_office)
    x, y = tilemapbase.project(*latlon_origin)
    # x_max, y_max = tilemapbase.project(
    #     *(latlon_origin[0] + degree_range - 0.0, channel[1] + degree_range - 0.001))
    origin = x, y
    print(f"(x, y): {(x, y)}")
    # ax.scatter(x, y, marker=".", color="black", linewidth=20)
    # ax.scatter(x_max, y_max, marker=".", color="red", linewidth=20)
    # ax.set(xlim=(-5, 5), ylim=(-8, 8))

    if save_file_name is not None:
        plt.savefig(f'figures/{save_file_name}.png', bbox_inches='tight')

    if show:
        plt.show()


def brattora(show=False, save_file_name=None):
    """
    Makes a map of Brattøra for later use

    Based on the work of @MatthewDaws on Github:
    https://github.com/MatthewDaws/TileMapBase/blob/master/notebooks/Example.ipynb

    """

    tilemapbase.start_logging()
    # Don't need if you have run before; DB file will already exist.
    tilemapbase.init(create=True)
    # Use open street map
    t = tilemapbase.tiles.build_OSM()
    # My current office at the University of Leeds
    lonlat_origin = (10.40052, 63.439309)

    degree_range = 0.001
    extent = tilemapbase.Extent.from_lonlat(
        lonlat_origin[0] - degree_range,
        lonlat_origin[0] + degree_range,
        lonlat_origin[1] - degree_range,
        lonlat_origin[1] + degree_range
    )

    extent = extent.to_aspect(1.3)
    print(f"extent: {extent}")
    # On my desktop, DPI gets scaled by 0.75
    # fig, ax = plt.subplots(figsize=(8, 6), dpi=400)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=400)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    plotter = tilemapbase.Plotter(extent, t, width=1000, height=700)
    plotter.plot(ax, t)

    # x, y = tilemapbase.project(*my_office)
    x, y = tilemapbase.project(*lonlat_origin)
    # x_max, y_max = tilemapbase.project(
    #     *(channel[0] + degree_range - 0.0, channel[1] + degree_range - 0.001))
    origin = x, y
    print(f"(x, y): {(x, y)}")
    # ax.scatter(x, y, marker=".", color="black", linewidth=20)
    # ax.scatter(x_max, y_max, marker=".", color="red", linewidth=20)
    # ax.set(xlim=(-5, 5), ylim=(-8, 8))

    if save_file_name is not None:
        plt.savefig(f'figures/{save_file_name}.png', bbox_inches='tight')

    if show:
        plt.show()


def nidelva(show=False, save_file_name=None):
    """
    Makes a map of Brattøra for later use

    Based on the work of @MatthewDaws on Github:
    https://github.com/MatthewDaws/TileMapBase/blob/master/notebooks/Example.ipynb

    """

    tilemapbase.start_logging()
    # Don't need if you have run before; DB file will already exist.
    tilemapbase.init(create=True)
    # Use open street map
    t = tilemapbase.tiles.build_OSM()
    # My current office at the University of Leeds
    lonlat_origin = (10.4014, 63.42775)

    degree_range = 0.001
    extent = tilemapbase.Extent.from_lonlat(
        lonlat_origin[0] - degree_range,
        lonlat_origin[0] + degree_range,
        lonlat_origin[1] - degree_range,
        lonlat_origin[1] + degree_range
    )

    extent = extent.to_aspect(0.8)
    print(f"extent: {extent}")
    # On my desktop, DPI gets scaled by 0.75
    # fig, ax = plt.subplots(figsize=(8, 6), dpi=400)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=400)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    plotter = tilemapbase.Plotter(extent, t, width=1000, height=700)
    plotter.plot(ax, t)

    # x, y = tilemapbase.project(*my_office)
    x, y = tilemapbase.project(*lonlat_origin)
    # x_max, y_max = tilemapbase.project(
    #     *(channel[0] + degree_range - 0.0, channel[1] + degree_range - 0.001))
    origin = x, y
    print(f"(x, y): {(x, y)}")
    # ax.scatter(x, y, marker=".", color="black", linewidth=20)
    # ax.scatter(x_max, y_max, marker=".", color="red", linewidth=20)
    # ax.set(xlim=(-5, 5), ylim=(-8, 8))

    if save_file_name is not None:
        plt.savefig(f'figures/{save_file_name}.png', bbox_inches='tight')

    if show:
        plt.show()
