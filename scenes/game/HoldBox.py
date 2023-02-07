import pygame

from scenes.game.Block import OBlock, IBlock


class HoldBox:
    def __init__(self, screen, x, y):
        self.screen: pygame.Surface = screen
        self.x = x
        self.y = y
        self.font = pygame.font.Font("PressStart2P-Regular.ttf", 24)
        self.box = pygame.Rect(x, y, 136, 128)
        self.text = self.font.render("Hold", True, "white")
        self.block = None
        self.previous_block = None

    def draw(self):
        pygame.draw.rect(self.screen, pygame.Color("white"), self.box, width=2)
        self.screen.blit(self.text, (self.x + 16, self.y-28))
        if self.block:
            if isinstance(self.block, (IBlock, OBlock)):
                self.block.draw(self.x+5, self.y)
            else:
                self.block.draw(self.x + 21, self.y)

    def swap(self, block):
        self.previous_block = self.block
        self.block = block
        return self.previous_block

    def can_swap(self, block):
        if block is self.previous_block:
            return False
        else:
            return True
