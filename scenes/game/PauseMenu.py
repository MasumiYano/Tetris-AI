import pygame


class PauseMenu:
    def __init__(self, screen):
        self.screen: pygame.Surface = screen
        self.title_font = pygame.font.Font("PressStart2P-Regular.ttf", 64)
        self.button_font = pygame.font.Font("PressStart2P-Regular.ttf", 32)

        self.overlay = pygame.Surface((self.screen.get_width(), self.screen.get_height()), pygame.SRCALPHA)
        self.overlay.fill((0, 0, 0, 220))

        self.resume_button = pygame.Rect(150, 400, 300, 100)
        self.menu_button = pygame.Rect(150, 540, 300, 100)

        self.title_text = self.title_font.render('Game', True, "white")
        self.title_text2 = self.title_font.render('Paused', True, "white")
        self.resume_text = self.button_font.render('Resume', True, "white")
        self.menu_text = self.button_font.render('Menu', True, "white")

    def draw(self):
        self.screen.blit(self.overlay, (0, 0))

        self.screen.blit(self.title_text, (300-self.title_text.get_width()/2, 150))
        self.screen.blit(self.title_text2, (300-self.title_text2.get_width()/2, 220))

        pygame.draw.rect(self.screen, pygame.Color("white"), self.resume_button, border_radius=8, width=4)
        pygame.draw.rect(self.screen, pygame.Color("white"), self.menu_button, border_radius=8, width=4)

        self.screen.blit(self.resume_text, (300-self.resume_text.get_width()/2, 436))
        self.screen.blit(self.menu_text, (300-self.menu_text.get_width()/2, 576))

