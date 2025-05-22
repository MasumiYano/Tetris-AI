import numpy as np

def calculate_board_features(board_grid):
    """Calculate advanced board features for reward shaping"""
    grid = np.array(board_grid)
    
    # Column heights
    heights = []
    for col in range(10):
        height = 0
        for row in range(22):
            if grid[row][col] != 0:
                height = 22 - row
                break
        heights.append(height)
    
    # Bumpiness (height differences between adjacent columns)
    bumpiness = sum(abs(heights[i] - heights[i+1]) for i in range(9))
    
    # Holes (empty cells with filled cells above them)
    holes = 0
    for col in range(10):
        block_found = False  # Fixed typo: was "block_fonud"
        for row in range(22):
            if grid[row][col] != 0:
                block_found = True  # Fixed typo: was "block_fonud"
            elif block_found and grid[row][col] == 0:
                holes += 1
    
    # Wells (deep single-column gaps)
    wells = 0
    for col in range(10):
        for row in range(21):  # Changed from 22 to 21 to avoid index out of bounds
            if grid[row][col] == 0:
                left_blocked = col == 0 or grid[row][col-1] != 0
                right_blocked = col == 9 or grid[row][col+1] != 0
                if left_blocked and right_blocked:
                    well_depth = 0
                    for r in range(row, 22):
                        if grid[r][col] == 0:
                            well_depth += 1
                        else:
                            break
                    wells += well_depth * (well_depth + 1) // 2  # Quadratic penalty for deep wells
                    break
    
    # Maximum height
    max_height = max(heights) if heights else 0
    
    # Height variance (prefer flat board)
    height_var = np.var(heights)
    
    return {
        'heights': heights,
        'bumpiness': bumpiness,
        'holes': holes,
        'wells': wells,
        'max_height': max_height,
        'height_variance': height_var
    }


def calculate_tetris_reward(prev_features, curr_features, lines_cleared, game_over):
    """Calculate comprehensive reward for Tetris gameplay"""
    reward = 0
    
    # Line clearing rewards (exponential for multiple lines)
    if lines_cleared > 0:
        line_rewards = {1: 40, 2: 100, 3: 300, 4: 1200}  # Tetris gets huge bonus
        reward += line_rewards.get(lines_cleared, 0)
    
    # Penalize creating holes
    holes_penalty = (curr_features['holes'] - prev_features['holes']) * -25
    reward += holes_penalty
    
    # Penalize increasing bumpiness
    bumpiness_penalty = (curr_features['bumpiness'] - prev_features['bumpiness']) * -5
    reward += bumpiness_penalty
    
    # Heavily penalize creating wells
    wells_penalty = (curr_features['wells'] - prev_features['wells']) * -10
    reward += wells_penalty
    
    # Penalize increasing max height
    height_penalty = (curr_features['max_height'] - prev_features['max_height']) * -2
    reward += height_penalty
    
    # Small reward for keeping board flat
    if curr_features['height_variance'] < prev_features['height_variance']:
        reward += 1
    
    # Game over penalty
    if game_over:
        reward -= 100  # Increased penalty
        reward -= curr_features['max_height'] * 5  # Extra penalty for high stacks
    
    return reward
