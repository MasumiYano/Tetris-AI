import numpy as np

def calculate_board_features(board_grid):
    grid = np.array(board_grid)

    heights = []
    for col in range(10):
        height = 0
        for row in range(22):
            if grid[row][col] != 0:
                height = 22 - row
                break
        heights.append(height)


    bumpiness = sum(abs(heights[i]-heights[i+1]) for i in range(9))

    holes = 0
    for col in range(10):
        block_fonud = False
        for row in range(22):
            if grid[row][col] != 0:
                block_found = True
            elif block_fonud and grid[row][col] == 0:
                holes += 1


    wells = 0
    for col in range(10):
        for row in range(22):
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
                    wells += well_depth * (well_depth + 1) // 2
                    break

    max_height = max(heights) if heights else 0

    height_var = np.var(heights)

    return {
            'heights': heights,
            'bumpiness': bumpiness,
            'holes': holes,
            'max_height': max_height,
            'height_variance': height_var
            }


def calculate_tetris_reward(prev_features, curr_features, lines_cleared, game_over):
    reward = 0

    if lines_cleared > 0:
        line_rewards = {1: 40, 2: 100, 3: 300, 4: 1200}
        reward += line_rewards.get(lines_cleared, 0)

    holes_penalty = (curr_features['holes'] - prev_features['holes']) * -25
    reward += holes_penalty

    bumpiness_penalty = (curr_features['bumpiness'] - prev_features['bumpiness']) * -5
    reward += bumpiness_penalty

    wells_penalty = (curr_features['wells'] - prev_features['wells']) * -10
    reward += wells_penalty

    heights_penalty = (curr_features['max_height'] - prev_features['max_height']) * -2
    reward += heights_penalty

    if curr_features['height_variance'] < prev_features['height_variance']:
        reward += 1

    if game_over:
        reward -= 100
        reward -= curr_features['max_height'] * 5

    return reward
