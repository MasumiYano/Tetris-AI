# Add imports at the top
from reward_system import calculate_board_features, calculate_tetris_reward

import torch
import numpy as np
from collections import deque
import random

# Importing from folder
from game.game_main import TetrisAI
from game.Block import (IBlock, JBlock, LBlock, OBlock, SBlock, TBlock, ZBlock)
from model import ConvQNet, QTrainer
from helper import plot
from config import (MAX_MEMORY, BATCH_SIZE, LR, GAMMA, HYPERPARAMETER_EXPLORATION_RATE)


def calculate_bumpiness(arr):
    differences = [abs(arr[i] - arr[i - 1]) for i in range(len(arr) - 1)]
    total_bumpiness = sum(differences)
    return total_bumpiness


def block_type(block):
    arr = [0] * 7
    if block is None:
        return arr
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


def create_board_tensor(game):
    """Create a 4-channel visual representation of the game state"""
    tensor = np.zeros((4, 22, 10), dtype=np.float32)
    
    # Channel 0: Board state (normalize block types to 0-1 range)
    board_grid = np.array(game.board.grid, dtype=np.float32)
    tensor[0] = board_grid / 7.0  # Normalize block types (1-7) to roughly 0-1
    
    # Channel 1: Current piece position
    current_piece_grid = add_piece_to_grid(game.active_block, game.board.grid)
    tensor[1] = current_piece_grid
    
    # Channel 2: Ghost piece position
    ghost_piece_grid = add_ghost_piece_to_grid(game.active_block, game.board.grid)
    tensor[2] = ghost_piece_grid
    
    # Channel 3: Boundaries (walls and floor)
    tensor[3] = create_boundary_mask()
    
    return tensor


def add_piece_to_grid(block, board_grid):
    """Add current piece to a copy of the board grid"""
    grid = np.zeros((22, 10), dtype=np.float32)
    block_grid = block.get_block_grid()
    
    for y in range(len(block_grid)):
        for x in range(len(block_grid[0])):
            if block_grid[y][x] > 0:
                grid_y = block.grid_y + y
                grid_x = block.grid_x + x
                if 0 <= grid_y < 22 and 0 <= grid_x < 10:
                    grid[grid_y][grid_x] = 1.0
    
    return grid


def add_ghost_piece_to_grid(block, board_grid):
    """Add ghost piece to a copy of the board grid"""
    grid = np.zeros((22, 10), dtype=np.float32)
    
    # Find ghost piece position
    ghost_y = None
    block_grid = block.get_block_grid()
    for i in range(1, 50):
        if not block.can_move(block.grid_x, block.grid_y + i, block_grid):
            ghost_y = block.grid_y + i - 1
            break
    
    if ghost_y is not None:
        for y in range(len(block_grid)):
            for x in range(len(block_grid[0])):
                if block_grid[y][x] > 0:
                    grid_y = ghost_y + y
                    grid_x = block.grid_x + x
                    if 0 <= grid_y < 22 and 0 <= grid_x < 10:
                        grid[grid_y][grid_x] = 1.0
    
    return grid


def create_boundary_mask():
    """Create boundary mask showing walls and floor"""
    mask = np.zeros((22, 10), dtype=np.float32)
    
    # Mark boundaries (this could be used to help the CNN understand game constraints)
    # For now, we'll mark the bottom row and maybe use it for edge detection
    mask[21, :] = 1.0  # Bottom boundary
    
    return mask


class Agent:
    def __init__(self, use_cnn=True):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = GAMMA
        self.memory = deque(maxlen=MAX_MEMORY)
        self.use_cnn = use_cnn
        
        # Track board features for reward calculation
        self.prev_board_features = None
        
        if use_cnn:
            # Use new CNN model with lower learning rate
            self.model = ConvQNet(additional_features_size=17, output_size=7)
            # Use CNN-specific learning rate if available, otherwise use main LR
            try:
                from config import CNN_LR
                lr = CNN_LR
            except ImportError:
                lr = LR
        else:
            # Use original linear model for comparison
            from model import LinearQNet
            self.model = LinearQNet(33, 590, 7)
            # Use Linear-specific learning rate if available, otherwise use main LR
            try:
                from config import LINEAR_LR
                lr = LINEAR_LR
            except ImportError:
                lr = LR
            
        self.trainer = QTrainer(self.model, learning_rate=lr, gamma=self.gamma)

    def get_state(self, game):
        """Get state representation - visual for CNN, feature-based for linear model"""
        if self.use_cnn:
            # Create 4-channel board tensor
            board_tensor = create_board_tensor(game)
            
            # Additional features that don't fit well in the visual representation
            next_piece = block_type(game.next_blocks.next_blocks[0])  # 7-dim
            hold_piece = block_type(game.hold_box.block)  # 7-dim
            game_context = [
                game.prev_line_cleared,
                game.score_board.score / 1000.0,  # Normalize score
                game.active_block.grid_x / 10.0   # Normalize x position
            ]  # 3-dim
            
            additional_features = np.array(next_piece + hold_piece + game_context, dtype=np.float32)
            
            return (board_tensor, additional_features)
        else:
            # Original feature-based state for linear model
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

    def get_reward(self, game, lines_cleared, game_over):
        """Calculate reward using improved reward shaping"""
        if self.use_cnn:
            # Calculate current board features
            curr_features = calculate_board_features(game.board.grid)
            
            if self.prev_board_features is None:
                # First state, no reward calculation
                self.prev_board_features = curr_features
                return 0
            
            # Calculate comprehensive reward
            reward = calculate_tetris_reward(
                self.prev_board_features, 
                curr_features, 
                lines_cleared, 
                game_over
            )
            
            # Update previous features
            self.prev_board_features = curr_features
            return reward
        else:
            # Original reward system for linear model
            reward = 0
            if lines_cleared > 0:
                reward += game.score_board.score / 5
            if game_over:
                reward -= 10
                reward -= 0.1 * game.board.highest_column_height()
                reward -= 0.5 * game.board.count_new_zeros()
            return reward

    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def evaluate_state(self, temp_game_state):
        """Evaluate a temporary game state using the CNN"""
        if not self.use_cnn:
            # For linear model, we'd need to implement feature extraction for temp state
            # For now, return random value as fallback
            return random.random()
        
        # Create board tensor from temporary state
        board_tensor = self._create_board_tensor_from_temp_state(temp_game_state)
        
        # Create additional features
        next_piece = block_type(temp_game_state['next_piece'])
        hold_piece = block_type(temp_game_state['hold_piece'])
        game_context = [
            temp_game_state['lines_cleared'],
            temp_game_state['score'] / 1000.0,
            5.0 / 10.0  # Default x position (middle)
        ]
        
        additional_features = np.array(next_piece + hold_piece + game_context, dtype=np.float32)
        
        # Convert to tensors and evaluate
        board_tensor = torch.tensor(board_tensor, dtype=torch.float).unsqueeze(0)
        additional_features = torch.tensor(additional_features, dtype=torch.float).unsqueeze(0)
        
        with torch.no_grad():
            prediction = self.model(board_tensor, additional_features)
            # Take max of the 7 outputs as state value
            state_value = torch.max(prediction).item()
        
        return state_value
    
    def _create_board_tensor_from_temp_state(self, temp_game_state):
        """Create board tensor from temporary game state"""
        tensor = np.zeros((4, 22, 10), dtype=np.float32)
        
        # Channel 0: Board state
        board_grid = np.array(temp_game_state['board_grid'], dtype=np.float32)
        tensor[0] = board_grid / 7.0
        
        # Channel 1: Next piece (we'll place it at default position for evaluation)
        # For simplicity, we'll leave this empty since we're evaluating the board after placement
        tensor[1] = np.zeros((22, 10), dtype=np.float32)
        
        # Channel 2: Ghost piece (empty for temp state)
        tensor[2] = np.zeros((22, 10), dtype=np.float32)
        
        # Channel 3: Boundaries
        tensor[3] = create_boundary_mask()
        
        return tensor

    def best_state(self, next_states):
        """Find the best state from all possible next states"""
        if not next_states:
            return None
        
        best_placement = None
        best_value = float('-inf')
        
        # Add some randomness during exploration
        if random.randint(0, 1000) < self.epsilon:
            return random.choice(list(next_states.keys()))
        
        # Evaluate each possible state
        for placement, temp_game_state in next_states.items():
            state_value = self.evaluate_state(temp_game_state)
            
            if state_value > best_value:
                best_value = state_value
                best_placement = placement
        
        return best_placement

    def get_action(self, state, game=None):
        # Update epsilon for exploration
        self.epsilon = max(10, HYPERPARAMETER_EXPLORATION_RATE - self.n_games * 0.5)
        
        # If game object is provided, use perfect information planning
        if game is not None:
            return self.get_perfect_action(game)
        
        # Fallback to original action selection
        final_move = [0, 0, 0, 0, 0, 0, 0]
        
        if random.randint(0, 1000) < self.epsilon:
            move = random.randint(0, 6)
            final_move[move] = 1
        else:
            if self.use_cnn:
                # Unpack state tuple for CNN
                board_tensor, additional_features = state
                
                # Convert to tensors
                board_tensor = torch.tensor(board_tensor, dtype=torch.float).unsqueeze(0)
                additional_features = torch.tensor(additional_features, dtype=torch.float).unsqueeze(0)
                
                # Get prediction from CNN model
                prediction = self.model(board_tensor, additional_features)
                move = torch.argmax(prediction).item()
                final_move[move] = 1
            else:
                # Original linear model approach
                state0 = torch.tensor(state, dtype=torch.float)
                prediction = self.model(state0)
                move = torch.argmax(prediction).item()
                final_move[move] = 1

        return final_move
    
    def get_perfect_action(self, game):
        """Use perfect information to find the best action sequence"""
        # Get all possible final states
        next_states = game.get_next_states()
        
        if not next_states:
            # Fallback to hard drop if no valid moves
            return [0, 0, 0, 0, 0, 1, 0]
        
        # Find the best placement
        best_placement = self.best_state(next_states)
        
        if best_placement is None:
            return [0, 0, 0, 0, 0, 1, 0]
        
        target_x, target_rotation = best_placement
        
        # Convert the target placement into action sequence
        return self._convert_placement_to_action(game, target_x, target_rotation)
    
    def _convert_placement_to_action(self, game, target_x, target_rotation):
        """Convert target placement to action sequence"""
        current_piece = game.active_block
        current_x = current_piece.grid_x
        current_rotation = current_piece.rotation_index
        
        # For simplicity, we'll just return the action that gets us closest to target
        # Priority: rotation first, then movement, then drop
        
        # Check if we need to rotate
        if current_rotation != target_rotation:
            # Calculate shortest rotation path
            rotation_diff = (target_rotation - current_rotation) % 4
            if rotation_diff == 1 or rotation_diff == -3:
                return [0, 0, 0, 1, 0, 0, 0]  # Rotate CW
            elif rotation_diff == 3 or rotation_diff == -1:
                return [0, 0, 0, 0, 1, 0, 0]  # Rotate CCW
            elif rotation_diff == 2:
                return [0, 0, 0, 1, 0, 0, 0]  # Rotate CW (will need another rotation next step)
        
        # Check if we need to move horizontally
        if current_x < target_x:
            return [0, 0, 1, 0, 0, 0, 0]  # Move right
        elif current_x > target_x:
            return [0, 1, 0, 0, 0, 0, 0]  # Move left
        
        # If position and rotation are correct, hard drop
        return [0, 0, 0, 0, 0, 1, 0]  # Hard drop


def train():
    plot_scores = []
    plot_mean_score = []
    total_score = 0
    record = 0
    
    # Set use_cnn=True for CNN model, use_cnn=False for original linear model
    agent = Agent(use_cnn=True)
    game = TetrisAI()
    
    print(f"Training with {'CNN' if agent.use_cnn else 'Linear'} model")
    print(f"Learning rate: {agent.trainer.learning_rate}")
    print(f"Batch size: {BATCH_SIZE}")
    print("Using Perfect Information Planning!")
    
    while True:
        state_old = agent.get_state(game)
        
        # Use perfect information planning to get action
        final_move = agent.get_action(state_old, game=game)
        
        # Store lines cleared before action
        lines_before = game.score_board.lines_cleared
        reward_raw, done, score = game.play_step(final_move)
        lines_after = game.score_board.lines_cleared
        lines_cleared = lines_after - lines_before
        
        # Calculate improved reward
        reward = agent.get_reward(game, lines_cleared, done)

        state_new = agent.get_state(game)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.prev_board_features = None  # Reset for new game
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print(f'GAME: {agent.n_games}\nSCORE: {score}\nBEST SCORE: {record}\nREWARD: {reward:.2f}\nEPSILON: {agent.epsilon:.1f}')

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_score.append(mean_score)
            plot(plot_scores, plot_mean_score)


if __name__ == '__main__':
    train()
