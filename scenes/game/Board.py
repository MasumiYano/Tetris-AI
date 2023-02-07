import pygame
from pygame import Color, Surface

from scenes.game.Block import Block, ZBlock, TBlock, SBlock, IBlock, JBlock, LBlock, OBlock


class Board:
    def __init__(self, screen, x, y, cell_size):
        self.screen: Surface = screen
        self.x = x
        self.y = y
        self.cell_size = cell_size
        self.horizontal_lines = 22
        self.vertical_lines = 10
        self.grid_width = self.vertical_lines * self.cell_size
        self.grid_height = self.horizontal_lines * self.cell_size
        self.grid_color = Color((120, 120, 120))
        self.grid = [[0 for _ in range(self.vertical_lines)] for _ in range(self.horizontal_lines)]
        self.colors = {
            1: IBlock.color,
            2: JBlock.color,
            3: LBlock.color,
            4: OBlock.color,
            5: SBlock.color,
            6: TBlock.color,
            7: ZBlock.color
        }

    def add_block(self, block: Block):
        lock_out = True
        block_grid = block.get_block_grid()
        for y in range(len(block_grid)):
            for x in range(len(block_grid[0])):
                if block_grid[y][x] > 0:
                    self.grid[block.grid_y + y][block.grid_x + x] = block_grid[y][x]
                    if lock_out and block.grid_y + y > 1:
                        lock_out = False
        return lock_out

    def clear_lines(self):
        lines_cleared = 0
        for line_no in range(len(self.grid)-1, -1, -1):
            line = self.grid[line_no]
            if self.is_line_full(line):
                self.grid.pop(line_no)
                lines_cleared += 1

        for i in range(lines_cleared):
            self.grid.insert(0, [0 for _ in range(self.vertical_lines)])

        return lines_cleared

    def is_line_full(self, line):
        for i in line:
            if i == 0:
                return False
        return True

    def draw(self):
        self.draw_grid_lines()
        self.draw_blocks()
        self.draw_border()

    def draw_grid_lines(self):
        for vl in range(self.vertical_lines + 1):
            pygame.draw.line(self.screen, self.grid_color, (vl * self.cell_size + self.x, 64 + self.y),
                             (vl * self.cell_size + self.x, self.grid_height + self.y))
        for hl in range(2, self.horizontal_lines + 1):
            pygame.draw.line(self.screen, self.grid_color, (0 + self.x, hl * self.cell_size + self.y),
                             (self.grid_width + self.x, hl * self.cell_size + self.y))

    def draw_border(self):
        # Left border
        pygame.draw.line(self.screen, "white", (self.x, self.y+64),
                         (self.x, self.grid_height + self.y), 2)
        # Right border
        pygame.draw.line(self.screen, "white", (self.vertical_lines * self.cell_size + self.x, self.y+64),
                         (self.vertical_lines * self.cell_size + self.x, self.grid_height + self.y), 2)
        # Bottom border
        pygame.draw.line(self.screen, "white", (0 + self.x, self.horizontal_lines * self.cell_size + self.y),
                         (self.grid_width + self.x, self.horizontal_lines * self.cell_size + self.y), 2)

    def draw_blocks(self):
        for _y in range(len(self.grid)):
            for _x in range(len(self.grid[0])):
                if self.grid[_y][_x] > 0:
                    real_x = _x * self.cell_size + self.x
                    real_y = _y * self.cell_size + self.y
                    rect = pygame.Rect(real_x, real_y, self.cell_size, self.cell_size)
                    pygame.draw.rect(self.screen, self.colors[self.grid[_y][_x]], rect)
