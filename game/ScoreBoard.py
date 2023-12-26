import pygame


class ScoreBoard:
    def __init__(self, screen, x, y):
        self.screen: pygame.Surface = screen
        self.x = x
        self.y = y
        self.score = 0
        self.level = 5
        self.lines_cleared = 0
        self.font = pygame.font.Font("PressStart2P-Regular.ttf", 24)
        self.score_text = self.font.render("Score", True, "white")
        self.level_text = self.font.render("Level", True, "white")

    def draw(self):
        score = self.font.render(str(self.score), True, "white")
        level = self.font.render(str(self.level), True, "white")
        self.screen.blit(self.score_text, (self.x-self.score_text.get_width()/2, self.y))
        self.screen.blit(score, (self.x-score.get_width()/2, self.y + 28))
        self.screen.blit(self.level_text, (self.x-self.score_text.get_width()/2, self.y+60))
        self.screen.blit(level, (self.x-level.get_width()/2, self.y + 88))

    def add_cleared_lines(self, line_count):
        if line_count == 1:
            self.score += 100*self.level
        elif line_count == 2:
            self.score += 300*self.level
        elif line_count == 3:
            self.score += 500*self.level
        elif line_count == 4:
            self.score += 800*self.level

        self.lines_cleared += line_count
        self.level = min((self.lines_cleared // 1200), 14) + 5
