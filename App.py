import sys

import pygame as pygame

from scenes.game.Game import Game
from scenes.menu.Menu import Menu


class App:
    def __init__(self):
        self.screen_width = 600
        self.screen_height = 900
        pygame.init()
        pygame.display.set_caption("TETRIS")
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.scene = None
        self.start_menu()

    def run(self):
        while True:
            events = pygame.event.get()

            for event in events:
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            # update
            self.scene.update(events)

            # draw
            self.screen.fill("black")
            self.scene.draw()
            pygame.display.flip()

            self.clock.tick(60)

    def start_game(self):
        self.scene = Game(self, self.screen)

    def start_menu(self):
        self.scene = Menu(self, self.screen)
