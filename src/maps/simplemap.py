import pygame
import numpy as np
from .map import Map

from utils import N2S2D, S2N2D, N2S


class SimpleObs():
    def __init__(self, length: float, width: float, pos: tuple[float, float], scale: float, origin: np.ndarray) -> None:

        xy_s = N2S2D(pos, scale, origin)
        self.surf = pygame.Surface((width*scale, length*scale))
        self.surf.fill((0, 0, 0))
        self.rect = self.surf.get_rect(center=(xy_s[0], xy_s[1]))
        SE = (pos[0] - length/2, pos[1] + width/2)
        SW = (pos[0] - length/2, pos[1] - width/2)
        self.colliding_edge = (SW, SE)


class SimpleQuay(SimpleObs):
    def __init__(self, quay_length: float, quay_width: float, quay_pos: np.ndarray, scale: float, origin: np.ndarray) -> None:
        super(SimpleQuay, self).__init__(
            quay_length, quay_width, quay_pos, scale, origin)
        self.length = quay_length
        self.surf.fill((192, 192, 192))


class SimpleMap(Map):
    # Map parameters
    MAP_SIZE = (25, 30)                 # [m]    Size of map
    scale = 30                          # [px/m] pixels/meter
    BOX_WIDTH = MAP_SIZE[1]*scale       # [px]   Overall box width
    BOX_LENGTH = MAP_SIZE[0]*scale      # [px]   Overall box width
    QUAY_SIZE = (0.75, 10)              # [m]
    # [m] x position of the center of quay in NED
    QUAY_POS = (MAP_SIZE[0]/2 - QUAY_SIZE[0]/2, 0)
    OCEAN_BLUE = (0, 157, 196)          # [RGB]
    BACKGROUND_COLOR = OCEAN_BLUE

    # [px, px, px] Screen offset
    origin = np.array([BOX_WIDTH/2, BOX_LENGTH/2, 0], float)

    # Outer bounds of the map
    bounds = [-MAP_SIZE[0]/2, -MAP_SIZE[1]/2,
              MAP_SIZE[0]/2, MAP_SIZE[1]/2]

    # Map obstacles defined in ned
    quay = SimpleQuay(QUAY_SIZE[0], QUAY_SIZE[1],
                      QUAY_POS, scale, origin)
    extra_wall_width = MAP_SIZE[1]/2-QUAY_SIZE[1]/2
    extra_wall_east = SimpleObs(
        QUAY_SIZE[0], extra_wall_width, (QUAY_POS[0], MAP_SIZE[1]/2 - extra_wall_width/2), scale, origin)
    extra_wall_west = SimpleObs(
        QUAY_SIZE[0], extra_wall_width, (QUAY_POS[0], extra_wall_width/2 - MAP_SIZE[1]/2), scale, origin)

    # Weather
    SIDESLIP = 0  # 30           # [deg]
    CURRENT_MAGNITUDE = 0  # 3   # [0]

    def __init__(self, convex_set) -> None:
        super(SimpleMap, self).__init__(convex_set)
        self.obstacles = [self.extra_wall_east, self.extra_wall_west]
        colliding_edges = []
        for obstacle in self.obstacles:
            colliding_edges.append(obstacle.colliding_edge)

        self.colliding_edges = colliding_edges


def test_map():
    """
    Function for testing the map above
    """
    pygame.init()

    mymap = SimpleMap()

    screen = pygame.display.set_mode([mymap.BOX_WIDTH, mymap.BOX_LENGTH])
    screen.fill(mymap.OCEAN_BLUE)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        for obstacle in mymap.obstacles:
            screen.blit(obstacle.surf, obstacle.rect)

        pygame.display.flip()

    pygame.quit()


# test_map()
