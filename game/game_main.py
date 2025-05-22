import math

import pygame

from game.Block import Block, IBlock, OBlock
from game.Board import Board
from game.HoldBox import HoldBox
from game.NextBlocks import NextBlocks
from game.ScoreBoard import ScoreBoard
from enum import Enum
import numpy as np

from config import FPS


class Movement(Enum):
    PLACE_HOLDER = 0
    SOFT_DROP = 1
    MOVE_RIGHT = 2
    MOVE_LEFT = 3
    ROTATE_CW = 4
    ROTATE_CCW = 5
    HARD_DROP = 6
    HOLD = 7


class TetrisAI:
    def __init__(self):
        pygame.init()
        self.font = pygame.font.Font("game/PressStart2P-Regular.ttf", 32)
        self.screen_width = 600
        self.screen_height = 900
        pygame.display.set_caption("TETRIS")
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.screen.fill("black")
        self.clock = pygame.time.Clock()
        self.clock.tick(FPS)
        self.gravity_levels = {
            1: 0.01667,
            2: 0.021017,
            3: 0.026977,
            4: 0.035256,
            5: 0.04693,
            6: 0.06361,
            7: 0.0879,
            8: 0.1236,
            9: 0.1775,
            10: 0.2598,
            11: 0.388,
            12: 0.59,
            13: 0.92,
            14: 1.46,
            15: 2.36,
        }
        self.reset()

    def reset(self):
        self.accumulated_reward = 0
        self.gravity_counter = 0
        self.soft_drop = False
        self.left_pressed = False
        self.left_pressed_tick = 0
        self.right_pressed = False
        self.right_pressed_tick = 0
        self.last_movement = pygame.time.get_ticks()
        self.last_spawn = pygame.time.get_ticks()
        self.last_pause = pygame.time.get_ticks()

        self.paused = False
        self.game_over = False

        self.prev_line_cleared = 0
        self.move = Movement.PLACE_HOLDER
        self.frame_iteration = 0
        self.board = Board(self.screen, x=140, y=20, cell_size=32)
        self.hold_box = HoldBox(self.screen, x=4, y=84)
        self.next_blocks = NextBlocks(self.screen, self.board, x=460, y=84)
        self.active_block: Block = self.next_blocks.take_next()
        self.score_board = ScoreBoard(self.screen, x=300, y=740)
        self.draw()

    def play_step(self, action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self._move(action)

        # Lock delays and hard drop handling
        if pygame.time.get_ticks() - self.last_spawn > 20000 or pygame.time.get_ticks() - self.last_movement > max(500,
                                                                                                                   17.1 / self.get_gravity()):
            self.active_block.hard_drop()

        # Initialize or continue accumulating reward
        if not hasattr(self, 'accumulated_reward'):
            self.accumulated_reward = 0

        # Locking the block and checking for game over
        if self.active_block.locked:
            locked_out = self.board.add_block(self.active_block)

            # Check lines cleared before spawning new block
            lines_cleared = self.board.clear_lines()
            if lines_cleared != 0:
                # Add reward for line clearing
                self.score_board.add_cleared_lines(lines_cleared)
                self.accumulated_reward += self.score_board.score / 5

            self.active_block = self.next_blocks.take_next()
            blocked_out = self.active_block.is_blocked_out()

            self.last_movement = pygame.time.get_ticks()
            self.last_spawn = pygame.time.get_ticks()

            if locked_out or blocked_out:
                self.game_over = True
                # Add penalties for game over
                self.accumulated_reward -= 10
                self.accumulated_reward -= 0.1 * self.board.highest_column_height()
                self.accumulated_reward -= 0.5 * self.board.count_new_zeros()
                # Return final accumulated reward and reset for next game
                final_reward = self.accumulated_reward
                self.accumulated_reward = 0
                return final_reward, self.game_over, self.score_board.score


        if self.gravity_counter >= 1:
            gravity_floor = math.floor(self.gravity_counter)
            self.gravity_counter -= gravity_floor
            for i in range(gravity_floor):
                if self.active_block.move_down():
                    self.last_movement = pygame.time.get_ticks()

        if self.soft_drop:
            self.gravity_counter += self.get_gravity() * 20
        else:
            self.gravity_counter += self.get_gravity()

        # Right/left movement handling
        if self.left_pressed and (
                self.left_pressed_tick > 6 or self.left_pressed_tick == 0) and self.left_pressed_tick % 2 == 0:
            if self.active_block.move_left():
                self.last_movement = pygame.time.get_ticks()
            self.left_pressed_tick += 1
        if self.right_pressed and (
                self.right_pressed_tick > 6 or self.right_pressed_tick == 0) and self.right_pressed_tick % 2 == 0:
            if self.active_block.move_right():
                self.last_movement = pygame.time.get_ticks()
            self.right_pressed_tick += 1

        # UI update
        self._update_ui()
        self.clock.tick(FPS)

        return self.accumulated_reward, self.game_over, self.score_board.score

    def _update_ui(self):
        self.screen.fill("black")
        self.draw()
        pygame.display.update()

    def draw(self):
        self.board.draw()
        self.active_block.draw_ghost_piece()
        self.active_block.draw_on_board()
        self.hold_box.draw()
        self.next_blocks.draw()
        self.score_board.draw()

    def get_gravity(self):
        return self.gravity_levels[self.score_board.level]

    def hold(self):
        if self.hold_box.can_swap(self.active_block):
            self.active_block.reset()
            self.active_block = self.hold_box.swap(self.active_block)
            if self.active_block is None:
                self.active_block = self.next_blocks.take_next()
                self.last_spawn = pygame.time.get_ticks()

    def rotate_cw(self):
        rotated = self.active_block.rotate_cw()
        if rotated:
            self.last_movement = pygame.time.get_ticks()

    def rotate_ccw(self):
        rotated = self.active_block.rotate_ccw()
        if rotated:
            self.last_movement = pygame.time.get_ticks()

    def _move(self, action):
        # [soft drop, left, right, rotate cw, rotate ccw, hard drop, hold] <-- What agent can return as action.
        move_list = [Movement.SOFT_DROP, Movement.MOVE_LEFT, Movement.MOVE_RIGHT, Movement.ROTATE_CW,
                     Movement.ROTATE_CCW, Movement.HARD_DROP, Movement.HOLD]

        new_action = None

        if np.array_equal(action, [1, 0, 0, 0, 0, 0, 0]):
            new_action = move_list[0]
        elif np.array_equal(action, [0, 1, 0, 0, 0, 0, 0]):
            new_action = move_list[1]
        elif np.array_equal(action, [0, 0, 1, 0, 0, 0, 0]):
            new_action = move_list[2]
        elif np.array_equal(action, [0, 0, 0, 1, 0, 0, 0]):
            new_action = move_list[3]
        elif np.array_equal(action, [0, 0, 0, 0, 1, 0, 0]):
            new_action = move_list[4]
        elif np.array_equal(action, [0, 0, 0, 0, 0, 1, 0]):
            new_action = move_list[5]
        elif np.array_equal(action, [0, 0, 0, 0, 0, 0, 1]):
            new_action = move_list[6]

        self.move = new_action

        match self.move:
            case Movement.SOFT_DROP:
                self.soft_drop = True
            case Movement.MOVE_LEFT:
                self.left_pressed_tick = 0
                self.left_pressed = True
            case Movement.MOVE_RIGHT:
                self.right_pressed_tick = 0
                self.right_pressed = True
            case Movement.ROTATE_CW:
                self.rotate_cw()
            case Movement.ROTATE_CCW:
                self.rotate_ccw()
            case Movement.HARD_DROP:
                self.active_block.hard_drop()
            case Movement.HOLD:
                self.hold()

    def get_board_height(self):
        return self.board.get_height()

    def get_holes(self):
        return self.board.count_new_zeros()

    def get_next_states(self):
        """Get all possible final states for the current piece"""
        states = {}
        current_piece = self.active_block
        
        # Try all possible rotations for this piece type
        possible_rotations = self._get_possible_rotations(current_piece)
        
        for target_rotation in possible_rotations:
            # Create a temporary piece at each rotation
            temp_piece = self._create_temp_piece(current_piece, target_rotation)
            
            # Try all possible x positions for this rotation
            piece_grid = temp_piece.get_block_grid()
            min_x = min([x for y in range(len(piece_grid)) for x in range(len(piece_grid[0])) if piece_grid[y][x] > 0])
            max_x = max([x for y in range(len(piece_grid)) for x in range(len(piece_grid[0])) if piece_grid[y][x] > 0])
            
            # Try each valid x position
            for x_pos in range(-min_x, 10 - max_x):
                # Simulate dropping the piece at this position
                final_state = self._simulate_piece_placement(temp_piece, x_pos, target_rotation)
                if final_state is not None:
                    states[(x_pos, target_rotation)] = final_state
        
        return states
    
    def _get_possible_rotations(self, piece):
        """Get all valid rotations for a piece type"""
        if isinstance(piece, OBlock):  # O piece - no rotation needed
            return [0]
        elif isinstance(piece, IBlock):  # I piece - only 2 unique rotations
            return [0, 1]
        else:  # All other pieces - 4 rotations
            return [0, 1, 2, 3]
    
    def _create_temp_piece(self, original_piece, target_rotation):
        """Create a temporary piece with the specified rotation"""
        # Create new piece of same type
        piece_type = type(original_piece)
        temp_piece = piece_type(self.board, self.screen)
        temp_piece.rotation_index = target_rotation
        temp_piece.grid_x = 3  # Start at center
        temp_piece.grid_y = 0  # Start at top
        return temp_piece
    
    def _simulate_piece_placement(self, piece, x_pos, rotation):
        """Simulate placing a piece and return the resulting game state"""
        # Set piece position and rotation
        piece.grid_x = x_pos
        piece.grid_y = 0
        piece.rotation_index = rotation
        
        # Drop the piece to its final position
        while piece.can_move(piece.grid_x, piece.grid_y + 1, piece.get_block_grid()):
            piece.grid_y += 1
        
        # Check if placement is valid (piece must be at least partially visible)
        if piece.grid_y < 0:
            return None
        
        # Create a copy of the current board
        temp_board_grid = [row[:] for row in self.board.grid]
        
        # Add the piece to the temporary board
        block_grid = piece.get_block_grid()
        for y in range(len(block_grid)):
            for x in range(len(block_grid[0])):
                if block_grid[y][x] > 0:
                    board_y = piece.grid_y + y
                    board_x = piece.grid_x + x
                    if 0 <= board_y < 22 and 0 <= board_x < 10:
                        temp_board_grid[board_y][board_x] = block_grid[y][x]
        
        # Clear lines in the temporary board
        lines_cleared = 0
        for row_idx in range(len(temp_board_grid) - 1, -1, -1):
            if all(cell != 0 for cell in temp_board_grid[row_idx]):
                temp_board_grid.pop(row_idx)
                lines_cleared += 1
        
        # Add empty rows at the top
        for _ in range(lines_cleared):
            temp_board_grid.insert(0, [0] * 10)
        
        # Create a temporary game state for evaluation
        temp_game_state = {
            'board_grid': temp_board_grid,
            'lines_cleared': lines_cleared,
            'next_piece': self.next_blocks.next_blocks[0],
            'hold_piece': self.hold_box.block,
            'score': self.score_board.score + self._calculate_score_for_lines(lines_cleared)
        }
        
        return temp_game_state
    
    def _calculate_score_for_lines(self, lines_cleared):
        """Calculate score for clearing lines"""
        if lines_cleared == 0:
            return 0
        elif lines_cleared == 1:
            return 100 * self.score_board.level
        elif lines_cleared == 2:
            return 300 * self.score_board.level
        elif lines_cleared == 3:
            return 500 * self.score_board.level
        elif lines_cleared == 4:
            return 800 * self.score_board.level
        return 0
