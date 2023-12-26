import pygame


class Block:
    def __init__(self, board, screen):
        self.board = board
        self.screen_grid = board.grid
        self.screen = screen
        self.block_size = self.board.cell_size
        self.grid_x = 3
        self.grid_y = -1
        self.locked = False
        self.rotation_index = 0
        self.rotation_test = {
            0: {
                1: [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
                3: [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],
            },
            1: {
                0: [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
                2: [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
            },
            2: {
                1: [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
                3: [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],
            },
            3: {
                2: [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
                0: [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
            }
        }

    def get_block_grid(self):
        return self.rotations[self.rotation_index]

    def reset(self):
        self.grid_x = 3
        self.grid_y = -1
        self.rotation_index = 0

    def hard_drop(self):
        while self.can_move_down():
            self.move_down()
        self.locked = True

    def is_blocked_out(self):
        return not self.can_move(self.grid_x, self.grid_y, self.get_block_grid())

    # rotation
    def rotate_cw(self):
        target_rotation_index = (self.rotation_index + 1) % 4
        kick = self.can_rotate(target_rotation_index)
        if kick:
            self.grid_x += kick[0]
            self.grid_y += kick[1]
            self.rotation_index = target_rotation_index
            return True
        else:
            return False

    def rotate_ccw(self):
        target_rotation_index = (self.rotation_index - 1) % 4
        kick = self.can_rotate(target_rotation_index)
        if kick:
            self.grid_x += kick[0]
            self.grid_y += kick[1]
            self.rotation_index = target_rotation_index
            return True
        else:
            return False

    def can_rotate(self, target_rotation_index):
        temp_piece_grid = self.rotations[target_rotation_index]
        tests = self.rotation_test[self.rotation_index][target_rotation_index]
        for x, y in tests:
            if self.can_move(self.grid_x + x, self.grid_y - y, temp_piece_grid):
                return x, -y
        return False

    # movement
    def move_down(self):
        if self.can_move_down():
            self.grid_y += 1
            return True
        else:
            return False

    def move_right(self):
        if self.can_move_right():
            self.grid_x += 1
            return True
        else:
            return False

    def move_left(self):
        if self.can_move_left():
            self.grid_x -= 1
            return True
        else:
            return False

    def can_move_down(self):
        return self.can_move(self.grid_x, self.grid_y + 1, self.get_block_grid())

    def can_move_right(self):
        return self.can_move(self.grid_x + 1, self.grid_y, self.get_block_grid())

    def can_move_left(self):
        return self.can_move(self.grid_x - 1, self.grid_y, self.get_block_grid())

    def can_move(self, grid_x, grid_y, piece_grid):
        for y in range(len(piece_grid)):
            for x in range(len(piece_grid)):
                if piece_grid[y][x] > 0:
                    try:
                        if self.screen_grid[grid_y + y][grid_x + x] != 0 or grid_x + x < 0:
                            return False
                    except IndexError:
                        return False
        return True

    # drawing
    def draw_ghost_piece(self):
        ghost_y = None
        block_grid = self.get_block_grid()
        for i in range(1, 50):
            if not self.can_move(self.grid_x, self.grid_y + i, block_grid):
                ghost_y = self.grid_y + i - 1
                break
        for y in range(len(block_grid)):
            for x in range(len(block_grid[0])):
                if block_grid[y][x] > 0:
                    real_x = x * self.block_size + self.grid_x * self.block_size + self.board.x
                    real_y = y * self.block_size + ghost_y * self.block_size + self.board.y

                    rect = pygame.Surface((self.block_size, self.block_size))
                    rect.set_alpha(128)
                    rect.fill(self.color)
                    self.screen.blit(rect, (real_x, real_y))

    def draw_on_board(self):
        block_grid = self.get_block_grid()
        for y in range(len(block_grid)):
            for x in range(len(block_grid[0])):
                if block_grid[y][x] > 0:
                    real_x = x * self.block_size + self.grid_x * self.block_size + self.board.x
                    real_y = y * self.block_size + self.grid_y * self.block_size + self.board.y
                    rect = pygame.Rect(real_x, real_y, self.block_size, self.block_size)
                    pygame.draw.rect(self.screen, self.color, rect)

    def draw(self, x_offset, y_offset):
        block_grid = self.get_block_grid()
        for block_grid_y in range(len(block_grid)):
            for block_grid_x in range(len(block_grid[0])):
                if block_grid[block_grid_y][block_grid_x] > 0:
                    real_x = block_grid_x * self.block_size + x_offset
                    real_y = block_grid_y * self.block_size + y_offset
                    rect = pygame.Rect(real_x, real_y, self.block_size, self.block_size)
                    pygame.draw.rect(self.screen, self.color, rect)


class IBlock(Block):
    color = pygame.Color("Turquoise")

    def __init__(self, board, screen):
        super().__init__(board, screen)
        self.kick_data = {
            0: {
                1: [(-2, 0), (1, 0), (-2, -1), (1, 2)],
                3: [(-1, 0), (2, 0), (-1, 2), (2, -1)],
            },
            1: {
                0: [(2, 0), (-1, 0), (2, 1), (-1, 2)],
                2: [(-1, 0), (2, 0), (-1, 2), (2, -1)],
            },
            2: {
                1: [(1, 0), (-2, 0), (1, -2), (-2, 2)],
                3: [(2, 0), (-1, 0), (2, 1), (-1, -2)],
            },
            3: {
                2: [(-2, 0), (1, 0), (-2, -1), (1, 2)],
                0: [(1, 0), (-2, 0), (1, -2), (-2, 1)],
            }
        }
        self.rotations = [
            [[0, 0, 0, 0],
             [1, 1, 1, 1],
             [0, 0, 0, 0],
             [0, 0, 0, 0]],

            [[0, 0, 1, 0],
             [0, 0, 1, 0],
             [0, 0, 1, 0],
             [0, 0, 1, 0]],

            [[0, 0, 0, 0],
             [0, 0, 0, 0],
             [1, 1, 1, 1],
             [0, 0, 0, 0]],

            [[0, 1, 0, 0],
             [0, 1, 0, 0],
             [0, 1, 0, 0],
             [0, 1, 0, 0]],
        ]


class JBlock(Block):
    color = pygame.Color("RoyalBlue")

    def __init__(self, board, screen):
        super().__init__(board, screen)
        self.rotations = [
            [[0, 0, 0, 0],
             [2, 0, 0, 0],
             [2, 2, 2, 0],
             [0, 0, 0, 0]],

            [[0, 0, 0, 0],
             [0, 2, 2, 0],
             [0, 2, 0, 0],
             [0, 2, 0, 0]],

            [[0, 0, 0, 0],
             [0, 0, 0, 0],
             [2, 2, 2, 0],
             [0, 0, 2, 0]],

            [[0, 0, 0, 0],
             [0, 2, 0, 0],
             [0, 2, 0, 0],
             [2, 2, 0, 0]],
        ]


class LBlock(Block):
    color = pygame.Color("Coral")

    def __init__(self, board, screen):
        super().__init__(board, screen)
        self.rotations = [
            [[0, 0, 0, 0],
             [0, 0, 3, 0],
             [3, 3, 3, 0],
             [0, 0, 0, 0]],

            [[0, 0, 0, 0],
             [0, 3, 0, 0],
             [0, 3, 0, 0],
             [0, 3, 3, 0]],

            [[0, 0, 0, 0],
             [0, 0, 0, 0],
             [3, 3, 3, 0],
             [3, 0, 0, 0]],

            [[0, 0, 0, 0],
             [3, 3, 0, 0],
             [0, 3, 0, 0],
             [0, 3, 0, 0]],
        ]


class OBlock(Block):
    color = pygame.Color("Gold")

    def __init__(self, board, screen):
        super().__init__(board, screen)
        self.rotations = [
            [[0, 0, 0, 0],
             [0, 4, 4, 0],
             [0, 4, 4, 0],
             [0, 0, 0, 0]],

            [[0, 0, 0, 0],
             [0, 4, 4, 0],
             [0, 4, 4, 0],
             [0, 0, 0, 0]],

            [[0, 0, 0, 0],
             [0, 4, 4, 0],
             [0, 4, 4, 0],
             [0, 0, 0, 0]],

            [[0, 0, 0, 0],
             [0, 4, 4, 0],
             [0, 4, 4, 0],
             [0, 0, 0, 0]],
        ]


class SBlock(Block):
    color = pygame.Color("LightGreen")

    def __init__(self, board, screen):
        super().__init__(board, screen)
        self.rotations = [
            [[0, 0, 0, 0],
             [0, 5, 5, 0],
             [5, 5, 0, 0],
             [0, 0, 0, 0]],

            [[0, 0, 0, 0],
             [0, 5, 0, 0],
             [0, 5, 5, 0],
             [0, 0, 5, 0]],

            [[0, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 5, 5, 0],
             [5, 5, 0, 0]],

            [[0, 0, 0, 0],
             [5, 0, 0, 0],
             [5, 5, 0, 0],
             [0, 5, 0, 0]],
        ]


class TBlock(Block):
    color = pygame.Color("Orchid")

    def __init__(self, board, screen):
        super().__init__(board, screen)
        self.rotations = [
            [[0, 0, 0, 0],
             [0, 6, 0, 0],
             [6, 6, 6, 0],
             [0, 0, 0, 0]],

            [[0, 0, 0, 0],
             [0, 6, 0, 0],
             [0, 6, 6, 0],
             [0, 6, 0, 0]],

            [[0, 0, 0, 0],
             [0, 0, 0, 0],
             [6, 6, 6, 0],
             [0, 6, 0, 0]],

            [[0, 0, 0, 0],
             [0, 6, 0, 0],
             [6, 6, 0, 0],
             [0, 6, 0, 0]],
        ]


class ZBlock(Block):
    color = pygame.Color("OrangeRed")

    def __init__(self, board, screen):
        super().__init__(board, screen)
        self.rotations = [
            [[0, 0, 0, 0],
             [7, 7, 0, 0],
             [0, 7, 7, 0],
             [0, 0, 0, 0]],

            [[0, 0, 0, 0],
             [0, 0, 7, 0],
             [0, 7, 7, 0],
             [0, 7, 0, 0]],

            [[0, 0, 0, 0],
             [0, 0, 0, 0],
             [7, 7, 0, 0],
             [0, 7, 7, 0]],

            [[0, 0, 0, 0],
             [0, 7, 0, 0],
             [7, 7, 0, 0],
             [7, 0, 0, 0]],
        ]
