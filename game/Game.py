import math
from typing import List

import pygame
from pygame.event import Event

from game.Controls import Controls
from game.Block import Block
from game.Board import Board
from game.HoldBox import HoldBox
from game.NextBlocks import NextBlocks
from game.ScoreBoard import ScoreBoard


class Game:
    def __init__(self, app, screen):
        self.app = app
        self.screen = screen
        self.font = pygame.font.Font("PressStart2P-Regular.ttf", 32)
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

        self.board = Board(self.screen, x=140, y=20, cell_size=32)
        self.hold_box = HoldBox(self.screen, x=4, y=84)
        self.next_blocks = NextBlocks(self.screen, self.board, x=460, y=84)
        self.active_block: Block = self.next_blocks.take_next()
        self.score_board = ScoreBoard(self.screen, x=300, y=740)

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

    def update(self, events: List[Event]):
        reward = 0
        if self.game_over:  # so we could just return game_over for done value.
            reward = -10
            return reward, self.game_over, self.score_board.score
        else:
            for event in events:
                if event.type == pygame.KEYDOWN:
                    if event.key == Controls.hold:
                        self.hold()
                    elif event.key == Controls.rotate_cw:
                        self.rotate_cw()
                    elif event.key == Controls.rotate_ccw:
                        self.rotate_ccw()
                    elif event.key == Controls.hard_drop:
                        self.active_block.hard_drop()
                    elif event.key == Controls.soft_drop:
                        self.soft_drop = True
                    elif event.key == Controls.move_left:
                        self.left_pressed_tick = 0
                        self.left_pressed = True
                    elif event.key == Controls.move_right:
                        self.right_pressed_tick = 0
                        self.right_pressed = True
                if event.type == pygame.KEYUP:
                    if event.key == Controls.soft_drop:
                        self.soft_drop = False
                    elif event.key == Controls.move_left:
                        self.left_pressed = False
                    elif event.key == Controls.move_right:
                        self.right_pressed = False

            # Lock delays
            if pygame.time.get_ticks() - self.last_spawn > 20000:
                self.active_block.hard_drop()

            if pygame.time.get_ticks() - self.last_movement > max(500, 17.1/self.get_gravity()):
                self.active_block.hard_drop()

            # Locking
            if self.active_block.locked:
                locked_out = self.board.add_block(self.active_block)
                reward -= 0.1 * self.board.highest_column_height()
                reward -= 1 * self.board.count_new_zeros()
                lines_cleared = self.board.clear_lines()
                # if agent clears more line than previous time, reward += 10
                if self.prev_line_cleared < lines_cleared:
                    reward += 10
                self.prev_line_cleared = lines_cleared  # <-- swap the value of lines
                self.active_block = self.next_blocks.take_next()
                blocked_out = self.active_block.is_blocked_out()
                self.last_movement = pygame.time.get_ticks()
                self.last_spawn = pygame.time.get_ticks()
                self.score_board.add_cleared_lines(lines_cleared)
                if locked_out or blocked_out:
                    reward = -10
                    self.game_over = True
                    pygame.mixer.music.pause()
                    return reward, self.game_over, self.score_board.score

            # gravity
            if self.gravity_counter >= 1:
                gravity_floor = math.floor(self.gravity_counter)
                self.gravity_counter -= gravity_floor
                for i in range(gravity_floor):
                    if self.active_block.move_down():
                        self.last_movement = pygame.time.get_ticks()

            if self.soft_drop:
                reward += 2
                self.gravity_counter += self.get_gravity()*20
            else:
                reward += 5
                self.gravity_counter += self.get_gravity()

            # right/left movement
            if self.left_pressed:
                if (self.left_pressed_tick > 6 or self.left_pressed_tick == 0) and self.left_pressed_tick % 2 == 0:
                    if self.active_block.move_left():
                        self.last_movement = pygame.time.get_ticks()
                self.left_pressed_tick += 1
            if self.right_pressed:
                if (self.right_pressed_tick > 6 or self.right_pressed_tick == 0) and self.right_pressed_tick % 2 == 0:
                    if self.active_block.move_right():
                        self.last_movement = pygame.time.get_ticks()
                self.right_pressed_tick += 1

            return reward, self.game_over, self.score_board.score

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
