MAX_MEMORY = 100_000
BATCH_SIZE = 32 
LR = 0.001
FPS = 100
GAMMA = 0.99
HYPERPARAMETER_EXPLORATION_RATE = 800

CNN_LR = 0.0001
LINEAR_LR = 0.01

# New simple state configuration
USE_SIMPLE_STATE = True  # Toggle between complex and simple states
SIMPLE_STATE_SIZE = 4    # [lines_cleared, holes, bumpiness, sum_height]
SIMPLE_LR = 0.001        # Learning rate for simple model
