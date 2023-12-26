import pygame


class Controls:
    name = "Guideline"
    move_left = pygame.K_LEFT
    move_right = pygame.K_RIGHT
    soft_drop = pygame.K_DOWN
    hard_drop = pygame.K_SPACE
    hold = pygame.K_c
    rotate_cw = pygame.K_UP
    rotate_ccw = pygame.K_z

    @staticmethod
    def change_controls():
        if Controls.name == "Guideline":
            Controls.name = "WASD"
            Controls.move_left = pygame.K_a
            Controls.move_right = pygame.K_d
            Controls.soft_drop = pygame.K_w
            Controls.hard_drop = pygame.K_s
            Controls.hold = pygame.K_LSHIFT
            Controls.rotate_cw = pygame.K_RIGHT
            Controls.rotate_ccw = pygame.K_LEFT
        else:
            Controls.name = "Guideline"
            Controls.move_left = pygame.K_LEFT
            Controls.move_right = pygame.K_RIGHT
            Controls.soft_drop = pygame.K_DOWN
            Controls.hard_drop = pygame.K_SPACE
            Controls.hold = pygame.K_c
            Controls.rotate_cw = pygame.K_UP
            Controls.rotate_ccw = pygame.K_z
