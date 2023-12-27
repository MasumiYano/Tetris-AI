# Importing form library
import random

import torch
import numpy as np
from collections import deque

# Importing from folder
from game.game_main import TetrisAI
from game.Block import (IBlock, JBlock, LBlock, OBlock, SBlock, TBlock, ZBlock)
from model import LinearQNet, QTrainer
from helper import plot
from config import (MAX_MEMORY, BATCH_SIZE, LR, GAMMA)


def calculate_bumpiness(arr):
    differences = [abs(arr[i] - arr[i - 1]) for i in range(len(arr) - 1)]
    total_bumpiness = sum(differences)
    return total_bumpiness


def block_type(block):
    arr = [0] * 7
    if isinstance(block, IBlock):
        arr[0] = 1
    elif isinstance(block, JBlock):
        arr[1] = 1
    elif isinstance(block, LBlock):
        arr[2] = 1
    elif isinstance(block, OBlock):
        arr[3] = 1
    elif isinstance(block, SBlock):
        arr[4] = 1
    elif isinstance(block, TBlock):
        arr[5] = 1
    elif isinstance(block, ZBlock):
        arr[6] = 1

    return arr


def get_rotation(num):
    return [1 if i == num else 0 for i in range(4)]


def flatten_array(arr):
    flat_list = []
    for item in arr:
        if isinstance(item, list):
            flat_list.extend(flatten_array(item))
        else:
            flat_list.append(item)
    return flat_list


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = GAMMA
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = LinearQNet(33, 256, 7)
        self.trainer = QTrainer(self.model, learning_rate=LR, gamma=self.gamma)

    def get_state(self, game):
        height = game.get_board_height()
        holes = game.get_holes()
        bumpiness = calculate_bumpiness(height)
        current_piece = block_type(game.active_block)
        piece_rotation = get_rotation(game.active_block.rotation_index)
        x_coordinate = game.active_block.grid_x
        lines_cleared = game.prev_line_cleared
        total_score = game.score_board.score
        hold_piece_type = block_type(game.hold_box.block)

        state = [
            height, holes, bumpiness, current_piece,
            piece_rotation, x_coordinate, lines_cleared,
            total_score, hold_piece_type
        ]

        flatten_state = flatten_array(state)
        return np.array(flatten_state, dtype=int)

    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)

        else:
            mini_sample = self.memory

        states, actions, rewards, next_steps, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_steps, dones)

    def train_short_memory(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0, 0, 0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 6)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_score = []
    total_score = 0
    record = 0
    agent = Agent()
    game = TetrisAI()
    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)

        state_new = agent.get_state(game)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print(f'GAME: {agent.n_games}\nSCORE: {score}\nBEST SCORE: {record}\nREWARD: {reward}')

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_score.append(mean_score)
            plot(plot_scores, plot_mean_score)


if __name__ == '__main__':
    train()
