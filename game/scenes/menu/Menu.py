import sys
from typing import List

import pygame
from pygame.event import Event

from game.Controls import Controls


class Menu:
    def __init__(self, app, screen):
        self.app = app
        self.screen = screen
        self.title_font = pygame.font.Font("PressStart2P-Regular.ttf", 64)
        self.font = pygame.font.Font("PressStart2P-Regular.ttf", 32)
        self.font_small = pygame.font.Font("PressStart2P-Regular.ttf", 26)

        self.start_button = pygame.Rect(150, 400, 300, 100)
        self.controls_button = pygame.Rect(150, 540, 300, 100)
        self.exit_button = pygame.Rect(150, 680, 300, 100)

        self.title_text = self.title_font.render('Tetris', True, "white")
        self.start_text = self.font.render('Start', True, "white")
        self.controls_text = self.font_small.render('Controls:', True, "white")
        self.controls_text2 = self.font_small.render(Controls.name, True, "white")
        self.exit_text = self.font.render('Exit', True, "white")

    def update(self, events: List[Event]):
        for event in events:
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                if self.start_button.collidepoint(event.pos):
                    self.app.start_game()
                if self.controls_button.collidepoint(event.pos):
                    Controls.change_controls()
                    self.controls_text2 = self.font_small.render(Controls.name, True, "white")
                if self.exit_button.collidepoint(event.pos):
                    pygame.quit()
                    sys.exit()

    def draw(self):
        self.screen.blit(self.title_text, (300-self.title_text.get_width()/2, 200))

        pygame.draw.rect(self.screen, pygame.Color("white"), self.start_button, border_radius=8, width=4)
        pygame.draw.rect(self.screen, pygame.Color("white"), self.controls_button, border_radius=8, width=4)
        pygame.draw.rect(self.screen, pygame.Color("white"), self.exit_button, border_radius=8, width=4)

        self.screen.blit(self.start_text, (300-self.start_text.get_width()/2, 436))
        self.screen.blit(self.controls_text, (300-self.controls_text.get_width()/2, 556))
        self.screen.blit(self.controls_text2, (300-self.controls_text2.get_width()/2, 594))
        self.screen.blit(self.exit_text, (300-self.exit_text.get_width()/2, 716))


