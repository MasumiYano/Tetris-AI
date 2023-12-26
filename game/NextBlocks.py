import random

import pygame

from game.Block import IBlock, JBlock, LBlock, OBlock, SBlock, TBlock, ZBlock


class NextBlocks:
    def __init__(self, screen, board, x, y):
        self.screen: pygame.Surface = screen
        self.board = board
        self.x = x
        self.y = y
        self.box = pygame.Rect(x, y, 136, 512)
        self.font = pygame.font.Font("PressStart2P-Regular.ttf", 24)
        self.text = self.font.render("Next", True, "white")
        self.bag = self.create_new_bag()
        self.next_blocks = [self.take_next_from_bag() for _ in range(4)]

    def draw(self):
        pygame.draw.rect(self.screen, pygame.Color("white"), self.box, width=2)
        self.screen.blit(self.text, (self.x + 16, self.y - 28))
        for i in range(len(self.next_blocks)):
            if isinstance(self.next_blocks[i], (IBlock, OBlock)):
                self.next_blocks[i].draw(self.x+5, self.y + i*128)
            else:
                self.next_blocks[i].draw(self.x + 21, self.y + i * 128)

    def take_next(self):
        self.next_blocks.append(self.take_next_from_bag())
        return self.next_blocks.pop(0)

    def take_next_from_bag(self):
        next = self.bag.pop(0)
        if len(self.bag) == 0:
            self.bag = self.create_new_bag()
        return next

    def create_new_bag(self):
        bag = [IBlock(self.board, self.screen),
               JBlock(self.board, self.screen),
               LBlock(self.board, self.screen),
               OBlock(self.board, self.screen),
               SBlock(self.board, self.screen),
               TBlock(self.board, self.screen),
               ZBlock(self.board, self.screen)]
        random.shuffle(bag)
        return bag
